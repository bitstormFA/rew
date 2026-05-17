## Runtime — execution context for typed steps and manual training loops.
##
## You own the training loop; the Runtime provides device management,
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

  Runtime* = object
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

# ---- PRNG key management -----------------------------------------------------

var globalSeed {.threadvar.}: int
var globalSeedSet {.threadvar.}: bool

proc seedEverything*(seed: int) =
  ## Sets a global seed for reproducibility. Call before `initRuntime` to
  ## make the Runtime's internal key deterministic.
  globalSeed = seed
  globalSeedSet = true

proc initKeyFromGlobalSeed*(): Key =
  ## Returns a Key derived from the last `seedEverything` call.
  if globalSeedSet:
    initKey(uint64(globalSeed))
  else:
    initKey(0)

func nextKey*(runtime: var Runtime): Key =
  ## Splits the Runtime's internal PRNG key and returns a fresh child key.
  ## Use this to get deterministic, reproducible random keys for each step.
  let keys = split(runtime.key, 2)
  runtime.key = keys[0]
  keys[1]

proc initRuntime*(accelerator: Accelerator = akAuto; devices: int = 1;
    precision: Precision = prFloat32;
    strategy: ParallelPolicy = autoParallel()): Runtime =
  ## Creates a Runtime with the given accelerator, device count, and precision.
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
  Runtime(
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

# ---- setup (models) ----------------------------------------------------------

func shouldShardModelParameters(runtime: Runtime): bool =
  if runtime.mesh.meshSize <= 1:
    return false
  case runtime.strategy.kind
  of ppkData:
    false
  of ppkZero:
    runtime.strategy.zeroStage >= 3
  of ppkAuto, ppkTensor, ppkPipeline, ppkHybrid, ppkManual:
    true

func largestDivisibleDim(shape: openArray[int]; factor: int): int =
  result = -1
  var best = -1
  for i, dim in shape:
    if dim > best and factor > 0 and dim mod factor == 0:
      result = i
      best = dim

proc setupTensor(runtime: Runtime; t: Tensor): Tensor =
  if not t.isEager or t.sharding.isPartitioned or t.sharding.isManual:
    return t
  if not runtime.shouldShardModelParameters:
    return t
  let dim = largestDivisibleDim(t.shape, runtime.mesh.meshSize)
  if dim < 0:
    return t
  var axes = newSeq[string](t.shape.len)
  axes[dim] = runtime.mesh.axes[0]
  shardToMesh(t, runtime.mesh, initPartitionSpec(axes), runtime.processIndex)

proc setup*[T](runtime: var Runtime; model: T): T =
  ## Sets up a model for training. Tensor/ZeRO-3/hybrid/auto strategies shard
  ## eager tensor leaves across the Runtime mesh when a divisible dimension
  ## exists, so parameter storage can exceed one device's memory.
  let localRuntime = runtime
  treeMap(model, proc(t: Tensor): Tensor = setupTensor(localRuntime, t))

# ---- setup (data) ------------------------------------------------------------

func setup*[T](runtime: var Runtime; pipe: Dataset[T]): Dataset[T] =
  ## Sets up a dataset pipeline for training. v1: no-op identity.
  ## Future phases will add automatic device transfer and distributed sampling.
  pipe

# ---- computeGrads ------------------------------------------------------------

proc computeGrads*[T](runtime: Runtime;
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

func distributedSize(runtime: Runtime): int =
  max(1, max(runtime.worldSize, runtime.mesh.meshSize))

proc requireTraceCollective(runtime: Runtime; opName: string) =
  if runtime.distributedSize > 1 and currentMode() != dmTrace:
    raise newException(TensorError,
      opName & ": multi-process collectives must run under jit/trace")

proc replicaGroup(runtime: Runtime; src = 0): seq[int] =
  let size = runtime.distributedSize
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

proc allGather*(runtime: Runtime; t: Tensor): Tensor =
  ## Gathers tensors from all processes along the leading dimension.
  ## Single-process runtimes return `t` unchanged.
  if runtime.distributedSize <= 1:
    return t
  runtime.requireTraceCollective("allGather")
  var outShape = t.shape
  outShape[0] *= runtime.distributedSize
  distOps.allGather([t], allGatherDim = 0, resultShapes = [outShape],
    replicaGroups = @[runtime.replicaGroup()])[0]

proc allReduce*(runtime: Runtime; t: Tensor): Tensor =
  ## Sums `t` across all processes. Single-process runtimes return `t`.
  if runtime.distributedSize <= 1:
    return t
  runtime.requireTraceCollective("allReduce")
  distOps.allReduce([t], replicaGroups = @[runtime.replicaGroup()],
    computation = sumComputation)[0]

proc broadcast*(runtime: Runtime; t: Tensor; src: int = 0): Tensor =
  ## Broadcasts `t` from rank `src`. Single-process runtimes return `t`.
  if runtime.distributedSize <= 1:
    return t
  runtime.requireTraceCollective("broadcast")
  distOps.collectiveBroadcast(t, replicaGroups = @[runtime.replicaGroup(src)])

proc barrier*(runtime: Runtime) =
  ## Synchronization hook. PJRT collectives are ordered through traced
  ## programs, so there is no host-side barrier in the single-process
  ## runtime.
  discard

func isGlobalZero*(runtime: Runtime): bool =
  ## Returns true if this is the global rank-0 process.
  runtime.globalRank == 0

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
        "Runtime.load: shard dtype " & $arr.dtype &
          " does not match expected " & $dtype)
    if arr.shape != layout.localShape:
      raise newException(ValueError,
        "Runtime.load: shard shape " & $arr.shape &
          " does not match expected " & $layout.localShape)
    return arr.data
  raise newException(ValueError,
    "Runtime.load: checkpoint has no local shard " & $layout.index)

proc save*[T](runtime: Runtime; path: string; state: T) =
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

proc load*[T](runtime: Runtime; path: string; prototype: T): T =
  ## Loads training state from a checkpoint directory created by `save`.
  ## `prototype` provides the pytree structure for reconstruction.
  ##
  ## v1: tensors are loaded onto `runtime.device`. All dtypes supported by the
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
          "Runtime.load: sharded checkpoint leaf has no prototype")
      let proto = prototypeLeaves[idx]
      if proto.sharding.isReplicated:
        raise newException(ValueError,
          "Runtime.load: sharded checkpoint requires a sharded prototype")
      let shape = jsonInts(item["shape"])
      leaves[idx] = fromHostByteShards(proto.dtype, shape, proto.sharding,
        proc(layout: ShardLayout; bytes: var seq[byte]) =
          bytes = loadShardBytes(path, item, layout, proto.dtype)
        , runtime.processIndex)
      continue
    let fname = item["file"].getStr()
    let arr = loadNpy(path / fname)
    if arr.data.len > 0:
      leaves[idx] = fromHostToDevice(runtime.device, unsafeAddr arr.data[0],
                                     arr.dtype, arr.shape)
    else:
      var dims = newSeq[int64](arr.shape.len)
      for i, dim in arr.shape: dims[i] = int64(dim)
      let h = transferToDevice(runtime.device, nil, arr.dtype, dims, 0)
      leaves[idx] = initEagerTensor(h, arr.dtype, arr.shape, runtime.device)
  treeUnflatten(prototype, leaves)
