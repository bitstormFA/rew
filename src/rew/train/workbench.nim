## Workbench — opt-in training utilities for manual training loops.
##
## The Workbench is the "Fabric equivalent" — minimal API surface.
## You own the training loop; the Workbench provides device management,
## precision setup, gradient computation, distributed communication,
## checkpointing, and PRNG key management.

import std/[json, os, strutils]
import ../tensor
import ../device
import ../dtype
import ../dispatch
import ../pytree
import ../rng
from ../buffer import isBufferSet
from ../eager import transferToDevice, toHostBytes, toHostShards,
  fromHostByteShards, shardToMesh
import ../distributed
import ../serialize
import ../sharding
import ../autograd/transform
import ../data/dataset
import ../ops/distributed as distOps
import ../stablehlo/[ir, ops as shops]

type
  CompilePolicy* = object
    ## Controls how high-level typed steps are compiled.
    enabled*: bool
    donateParams*: bool

  CheckpointPolicy* = object
    ## Default checkpoint behavior for Runtime-owned save/load helpers.
    dir*: string

  Accelerator* = enum
    akCpu
    akCuda
    akRocm
    akTpu
    akAuto

  Precision* = enum
    prFloat32
    prFloat16
    prBFloat16
    prMixedF16
    prMixedBF16

  Workbench* = object
    accelerator*: Accelerator
    devices*: int
    precision*: Precision
    globalRank*: int
    worldSize*: int
    device*: Device
    dist*: DistributedContext
    mesh*: Mesh
    strategy*: ParallelPolicy
    processIndex*: int
    processCount*: int
    key*: Key
    compile*: CompilePolicy
    checkpoint*: CheckpointPolicy

  Runtime* = Workbench
    ## Public high-level runtime name. `Workbench` remains as a migration
    ## alias for user-owned loops.

# ---- PRNG key management -----------------------------------------------------

var globalSeed {.threadvar.}: int
var globalSeedSet {.threadvar.}: bool

proc seedEverything*(seed: int) =
  ## Sets a global seed for reproducibility. Call before `initWorkbench` to
  ## make the Workbench's internal key deterministic.
  globalSeed = seed
  globalSeedSet = true

proc initKeyFromGlobalSeed*(): Key =
  ## Returns a Key derived from the last `seedEverything` call.
  if globalSeedSet:
    initKey(uint64(globalSeed))
  else:
    initKey(0)

func nextKey*(wb: var Workbench): Key =
  ## Splits the Workbench's internal PRNG key and returns a fresh child key.
  ## Use this to get deterministic, reproducible random keys for each step.
  let keys = split(wb.key, 2)
  wb.key = keys[0]
  keys[1]

proc initWorkbench*(accelerator: Accelerator = akAuto; devices: int = 1;
    precision: Precision = prFloat32;
    strategy: ParallelPolicy = autoParallel()): Workbench =
  ## Creates a Workbench with the given accelerator, device count, and precision.
  ## v1: single-device only. `akCpu` forces CPU; other accelerators resolve
  ## via `defaultDevice()`.
  let dev = case accelerator
    of akCpu: cpu(0)
    else: defaultDevice()
  var localDevices = newSeq[Device](max(1, devices))
  for i in 0 ..< localDevices.len:
    localDevices[i] = initDevice(dev.target, i)
  let dist = initDistributed(localDevices)
  let globalDeviceCount = max(1, localDevices.len) *
    max(1, dist.process.processCount)
  let mesh = meshFromTopology(dist, ["data"], [globalDeviceCount])
  Workbench(
    accelerator: accelerator,
    devices: devices,
    precision: precision,
    globalRank: dist.process.rank,
    worldSize: dist.process.worldSize,
    device: dev,
    dist: dist,
    mesh: mesh,
    strategy: strategy,
    processIndex: dist.process.processIndex,
    processCount: dist.process.processCount,
    key: initKeyFromGlobalSeed(),
    compile: CompilePolicy(enabled: true, donateParams: false),
    checkpoint: CheckpointPolicy(dir: "checkpoints"),
  )

proc initRuntime*(accelerator: Accelerator = akAuto; devices: int = 1;
    precision: Precision = prFloat32;
    strategy: ParallelPolicy = autoParallel()): Runtime =
  ## Creates a high-level Runtime for typed training steps and manual loops.
  initWorkbench(accelerator, devices, precision, strategy)

# ---- setup (models) ----------------------------------------------------------

func shouldShardModelParameters(wb: Workbench): bool =
  if wb.mesh.meshSize <= 1:
    return false
  case wb.strategy.kind
  of ppkData:
    false
  of ppkZero:
    wb.strategy.zeroStage >= 3
  of ppkAuto, ppkTensor, ppkPipeline, ppkHybrid, ppkManual:
    true

func largestDivisibleDim(shape: openArray[int]; factor: int): int =
  result = -1
  var best = -1
  for i, dim in shape:
    if dim > best and factor > 0 and dim mod factor == 0:
      result = i
      best = dim

proc setupTensor(wb: Workbench; t: Tensor): Tensor =
  if not t.isEager or t.sharding.isPartitioned or t.sharding.isManual:
    return t
  if not wb.shouldShardModelParameters:
    return t
  let dim = largestDivisibleDim(t.shape, wb.mesh.meshSize)
  if dim < 0:
    return t
  var axes = newSeq[string](t.shape.len)
  axes[dim] = wb.mesh.axes[0]
  shardToMesh(t, wb.mesh, initPartitionSpec(axes), wb.processIndex)

proc setup*[T](wb: var Workbench; model: T): T =
  ## Sets up a model for training. Tensor/ZeRO-3/hybrid/auto strategies shard
  ## eager tensor leaves across the Workbench mesh when a divisible dimension
  ## exists, so parameter storage can exceed one device's memory.
  let localWb = wb
  treeMap(model, proc(t: Tensor): Tensor = setupTensor(localWb, t))

# ---- setup (data) ------------------------------------------------------------

func setup*[T](wb: var Workbench; pipe: Dataset[T]): Dataset[T] =
  ## Sets up a data pipeline for training. v1: no-op identity.
  ## Future phases will add automatic device transfer and distributed sampling.
  pipe

# ---- computeGrads ------------------------------------------------------------

proc computeGrads*[T](wb: Workbench;
    fn: proc(args: openArray[Tensor]): Tensor {.closure.}; params: T): T =
  ## Computes gradients of scalar-output `fn` w.r.t. `params`.
  ## Returns grads with the same pytree structure as `params`.
  ##
  ## Must be called inside a `withTrace` block (trace-mode gradient
  ## computation). For eager-mode, wrap the computation in `withTrace`.
  ##
  ## v1: thin wrapper over `grad` that handles pytree flatten/unflatten.
  ## Future phases add mixed-precision gradient scaling.
  let flatParams = treeFlatten(params)
  let flatGrads = grad(fn, flatParams)
  treeUnflatten(params, flatGrads)

# ---- Distributed communication ------------------------------------------------

func distributedSize(wb: Workbench): int =
  max(1, max(wb.worldSize, wb.mesh.meshSize))

proc requireTraceCollective(wb: Workbench; opName: string) =
  if wb.distributedSize > 1 and currentMode() != dmTrace:
    raise newException(TensorError,
      opName & ": multi-process collectives must run under jit/trace")

proc replicaGroup(wb: Workbench; src = 0): seq[int] =
  let size = wb.distributedSize
  if src < 0 or src >= size:
    raise newException(TensorError,
      "broadcast: source rank " & $src & " out of range for world size " &
        $size)
  result.add src
  for i in 0 ..< size:
    if i != src:
      result.add i

proc sumComputation(b: var ShBuilder;
    args: openArray[ShValueId]): seq[ShValueId] =
  @[shops.add(b, args[0], args[1])]

proc allGather*(wb: Workbench; t: Tensor): Tensor =
  ## Gathers tensors from all processes along the leading dimension.
  ## Single-process workbenches return `t` unchanged.
  if wb.distributedSize <= 1:
    return t
  wb.requireTraceCollective("allGather")
  var outShape = t.shape
  outShape[0] *= wb.distributedSize
  distOps.allGather([t], allGatherDim = 0, resultShapes = [outShape],
    replicaGroups = @[wb.replicaGroup()])[0]

proc allReduce*(wb: Workbench; t: Tensor): Tensor =
  ## Sums `t` across all processes. Single-process workbenches return `t`.
  if wb.distributedSize <= 1:
    return t
  wb.requireTraceCollective("allReduce")
  distOps.allReduce([t], replicaGroups = @[wb.replicaGroup()],
    computation = sumComputation)[0]

proc broadcast*(wb: Workbench; t: Tensor; src: int = 0): Tensor =
  ## Broadcasts `t` from rank `src`. Single-process workbenches return `t`.
  if wb.distributedSize <= 1:
    return t
  wb.requireTraceCollective("broadcast")
  distOps.collectiveBroadcast(t, replicaGroups = @[wb.replicaGroup(src)])

proc barrier*(wb: Workbench) =
  ## Synchronization hook. PJRT collectives are ordered through traced
  ## programs, so there is no host-side barrier in the single-process
  ## runtime.
  discard

func isGlobalZero*(wb: Workbench): bool =
  ## Returns true if this is the global rank-0 process.
  wb.globalRank == 0

# ---- Save / Load -------------------------------------------------------------

proc fromHostToDevice(d: Device; data: pointer; dt: DType;
    shape: openArray[int]): Tensor =
  ## Creates a device tensor from host data. Uses the eager transfer path.
  var dims = newSeq[int64](shape.len)
  for i, dim in shape: dims[i] = int64(dim)
  let h = transferToDevice(d, data, dt, dims)
  initEagerTensor(h, dt, shape, d)

func jsonInts(node: JsonNode): seq[int] =
  for item in node:
    result.add item.getInt()

proc loadShardBytes(path: string; item: JsonNode;
    layout: ShardLayout; dtype: DType): seq[byte] =
  for shard in item["shards"]:
    if shard["index"].getInt() != layout.index:
      continue
    let arr = loadNpy(path / shard["file"].getStr())
    if arr.dtype != dtype:
      raise newException(ValueError,
        "Workbench.load: shard dtype " & $arr.dtype &
          " does not match expected " & $dtype)
    if arr.shape != layout.localShape:
      raise newException(ValueError,
        "Workbench.load: shard shape " & $arr.shape &
          " does not match expected " & $layout.localShape)
    return arr.data
  raise newException(ValueError,
    "Workbench.load: checkpoint has no local shard " & $layout.index)

proc save*[T](wb: Workbench; path: string; state: T) =
  ## Saves training state to a directory as `.npy` files + `manifest.json`.
  ## `state` is any pytree (model params, optimizer state, etc.).
  ##
  ## Each tensor leaf is written as a separate `.npy` file; the manifest
  ## records the index, filename, dtype, and shape for reconstruction.
  createDir(path)
  let leaves = treeFlatten(state)
  var manifest = newJObject()
  var tlist = newJArray()
  for i, leaf in leaves:
    if leaf.buffer.isBufferSet and not leaf.sharding.isReplicated:
      var shardList = newJArray()
      for shard in toHostShards(leaf):
        let fname = "tensor_" & $i & "_shard_" & $shard.layout.index &
          ".npy"
        let arr = initNpyArray(leaf.dtype, shard.layout.localShape,
          shard.data)
        saveNpy(path / fname, arr)
        shardList.add(%* {
          "index": %shard.layout.index,
          "process": %shard.layout.process,
          "file": %fname,
          "shape": %shard.layout.localShape,
          "offsets": %shard.layout.offsets,
        })
      tlist.add(%* {
        "index": %i,
        "dtype": %($leaf.dtype),
        "shape": %leaf.shape,
        "sharded": %true,
        "sharding_key": %leaf.sharding.shardingKey,
        "shards": shardList,
      })
    else:
      let hostBytes = toHostBytes(leaf)
      let arr = initNpyArray(leaf.dtype, leaf.shape, hostBytes)
      let fname = "tensor_" & $i & ".npy"
      saveNpy(path / fname, arr)
      tlist.add(%* {"index": %i, "file": %fname,
                     "dtype": %($leaf.dtype), "shape": %leaf.shape,
                     "sharded": %false})
  manifest["tensors"] = tlist
  manifest["leaf_count"] = %leaves.len
  writeFile(path / "manifest.json", $manifest)

proc load*[T](wb: Workbench; path: string; prototype: T): T =
  ## Loads training state from a checkpoint directory created by `save`.
  ## `prototype` provides the pytree structure for reconstruction.
  ##
  ## v1: tensors are loaded onto `wb.device`. All dtypes supported by the
  ## `.npy` reader are handled.
  let manifestStr = readFile(path / "manifest.json")
  let manifest = parseJson(manifestStr)
  let leafCount = manifest["leaf_count"].getInt()
  var leaves = newSeq[Tensor](leafCount)
  let prototypeLeaves = treeFlatten(prototype)
  let tlist = manifest["tensors"]
  for i in 0 ..< tlist.len:
    let item = tlist[i]
    let idx = item["index"].getInt()
    if item.hasKey("sharded") and item["sharded"].getBool():
      if idx < 0 or idx >= prototypeLeaves.len:
        raise newException(ValueError,
          "Workbench.load: sharded checkpoint leaf has no prototype")
      let proto = prototypeLeaves[idx]
      if proto.sharding.isReplicated:
        raise newException(ValueError,
          "Workbench.load: sharded checkpoint requires a sharded prototype")
      let shape = jsonInts(item["shape"])
      leaves[idx] = fromHostByteShards(proto.dtype, shape, proto.sharding,
        proc(layout: ShardLayout; bytes: var seq[byte]) =
          bytes = loadShardBytes(path, item, layout, proto.dtype)
        , wb.processIndex)
      continue
    let fname = item["file"].getStr()
    let arr = loadNpy(path / fname)
    if arr.data.len > 0:
      leaves[idx] = fromHostToDevice(wb.device, unsafeAddr arr.data[0],
                                     arr.dtype, arr.shape)
    else:
      var dims = newSeq[int64](arr.shape.len)
      for i, dim in arr.shape: dims[i] = int64(dim)
      let h = transferToDevice(wb.device, nil, arr.dtype, dims, 0)
      leaves[idx] = initEagerTensor(h, arr.dtype, arr.shape, wb.device)
  treeUnflatten(prototype, leaves)
