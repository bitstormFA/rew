## Eager backend — multi-plugin PJRT client cache and `BufferHandle`
## adapter for the eager dispatch path.
##
## Uses `pjrt/registry` for per-Target API table and client lookup,
## enabling true multi-plugin usage (CPU + CUDA tensors in one process).
##
## Owns:
##  - A process-wide `PjrtClient` per `(Target, ordinal)`, materialized
##    lazily on first use via `pjrt/registry`.
##  - The `nimcall` releaser thunk that lets `BufferHandle` free its
##    underlying PJRT buffer via `registry.releaseBuffer`.
##  - The host<->device transfer adapter.
##  - Per-op compile-and-cache for eager dispatch.
##  - JIT module compilation and execution cache.

import std/os
import std/tables
import std/strutils
import ./buffer
import ./device
import ./dtype
import ./sharding
import ./tensor
import ./dispatch
import ./stablehlo/ir
import ./stablehlo/ops as shops
import ./stablehlo/text
import ./stablehlo/verify
import ./pjrt/[capi, client, loader, registry]
import ./binaries/cache
import ./binaries/target

export buffer.BufferHandle, buffer.BufferDonatedError
export buffer.requireLive, buffer.isLive, buffer.isDonated, buffer.markDonated
export buffer.isBufferSet, buffer.bufferCount, buffer.shardIndices
export device.Device, device.defaultDevice, device.setDefaultDevice
export device.cpu, device.cuda12, device.cuda13, device.rocm, device.metal,
  device.tpu
export client.PjrtClient, client.PjrtLoadedExecutable

type
  EagerError* = object of CatchableError
    ## Raised on host/device transfer or backend-init failures originating
    ## in the eager layer.

  HostShardLoader*[T] = proc(layout: ShardLayout; dst: var seq[T])
    {.closure.}
    ## Callback used by `fromHostShards` to fill one local shard without
    ## materialising the full global tensor.

  HostByteShardLoader* = proc(layout: ShardLayout; dst: var seq[byte])
    {.closure.}
    ## Byte-level variant used by checkpoint and dtype-erased loading code.

  HostTensorShard* = object
    ## Host copy of one local shard. Used by shard-aware checkpointing and
    ## diagnostics.
    layout*: ShardLayout
    dtype*: DType
    data*: seq[byte]

func stableHexHash(s: string): string

const
  ExecutableCacheEnvVar* = "REW_EXECUTABLE_CACHE"
    ## Set to `0`, `false`, `off`, or `no` to disable the persistent
    ## serialized PJRT executable cache.

proc executableCacheEnabled*(): bool =
  ## Returns whether rew should persist PJRT executables under
  ## `REW_CACHE_DIR`. The cache is best-effort: unsupported PJRT serialize
  ## and load calls fall back to normal compilation.
  let value = getEnv(ExecutableCacheEnvVar).strip().toLowerAscii()
  if value.len == 0:
    return true
  value notin ["0", "false", "off", "no"]

proc executableCacheDir*(): string =
  ## Directory used for serialized PJRT executables.
  executablesDir()

# ----- DType <-> PJRT element-type bridge ---------------------------------

func toPjrtBufferType*(dt: DType): PjrtBufferType =
  ## Maps a user-facing `DType` to the corresponding `PjrtBufferType`.
  case dt
  of dtBool: btPred
  of dtInt4, dtUint4, dtNF4:
    raise newException(TensorError,
      "toPjrtBufferType: 4-bit packed type " & dt.name &
        " is not natively supported by PJRT; use a byte-level buffer type")
  of dtInt8: btS8
  of dtInt16: btS16
  of dtInt32: btS32
  of dtInt64: btS64
  of dtUint8: btU8
  of dtUint16: btU16
  of dtUint32: btU32
  of dtUint64: btU64
  of dtFloat16: btF16
  of dtBFloat16: btBF16
  of dtFloat32: btF32
  of dtFloat64: btF64
  of dtComplex64: btC64
  of dtComplex128: btC128
  of dtFloat8E4M3Fn, dtFloat8E5M2:
    raise newException(TensorError,
      "toPjrtBufferType: float8 type " & dt.name &
        " is not natively supported by PJRT; use a byte-level buffer type")

# ----- Releaser thunk ---------------------------------------------------

proc pjrtReleaser(t: Target; raw: PjrtBufferRaw) {.nimcall, raises: [].} =
  ## `BufferReleaser` that frees `raw` against the registry API for `t`.
  registry.releaseBuffer(t, raw)

# ----- Client cache -----------------------------------------------------

proc ensureClient*(d: Device): PjrtClient {.raises: [CatchableError].} =
  ## Returns the cached `PjrtClient` for `d`, loading the plugin and
  ## creating the client on first call. Subsequent calls return the same
  ## reference.
  let key: DeviceKey = (d.target, d.ordinal)
  if hasClient(key):
    return lookupClient(key)

  var api: PjrtApiHandle
  if hasApi(d.target):
    api = lookupApi(d.target)
  else:
    try:
      api = loader.loadPlugin(d.target)
      pluginInitialize(api)
      registerApi(d.target, api)
    except PjrtError as e:
      raise newException(EagerError,
        "eager: failed to load PJRT plugin for device " & $d & ": " & e.msg)
    except CatchableError as e:
      raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: unexpected failure loading PJRT plugin for device " & $d &
        ": " & e.msg)

  let c =
    try: newPjrtClient(api)
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: PJRT client creation failed for device " & $d & ": " & e.msg)
  registerClient(key, c)
  c

proc clientApi*(d: Device): PjrtApiHandle
    {.raises: [CatchableError].} =
  ## Returns the API table backing `d`'s target. Materializes the client if
  ## needed.
  discard ensureClient(d)
  lookupApi(d.target)

proc selectDevice(c: PjrtClient; ordinal: int): PjrtDeviceHandle
    {.raises: [CatchableError].} =
  let devs =
    try: c.addressableDevices()
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: addressableDevices failed: " & e.msg)
  if ordinal < 0 or ordinal >= devs.len:
    raise newException(EagerError,
      "eager: device ordinal " & $ordinal & " out of range (have " &
      $devs.len & " addressable device(s))")
  devs[ordinal]

proc resolveDevice*(d: Device): PjrtDeviceHandle
    {.raises: [CatchableError].} =
  ## Returns the `PjrtDeviceHandle` for `d`, materializing the client first.
  selectDevice(ensureClient(d), d.ordinal)

proc addressableDeviceCountFor*(d: Device): int
    {.raises: [CatchableError].} =
  ## Returns the PJRT addressable-device count for `d`'s client.
  try:
    ensureClient(d).addressableDevices().len
  except CatchableError as e:
    raise e
  except Exception as e:
    raise newException(EagerError,
      "eager: addressableDeviceCountFor failed: " & e.msg)

proc defaultDeviceAssignmentFor*(d: Device;
    numReplicas, numPartitions: int): seq[int64]
    {.raises: [CatchableError].} =
  ## Bridge for higher layers: asks PJRT for the default flat
  ## `[replica, partition]` device assignment without exposing raw PJRT.
  let ints =
    try:
      ensureClient(d).defaultDeviceAssignment(numReplicas, numPartitions)
    except CatchableError as e:
      raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: defaultDeviceAssignmentFor failed: " & e.msg)
  result = newSeq[int64](ints.len)
  for i, value in ints:
    result[i] = int64 value

proc topologyFingerprintFor*(d: Device): string
    {.raises: [CatchableError].} =
  ## Stable hash of the client topology serialization for compile-cache keys.
  try:
    stableHexHash(ensureClient(d).topology().serialize())
  except CatchableError as e:
    raise e
  except Exception as e:
    raise newException(EagerError,
      "eager: topologyFingerprintFor failed: " & e.msg)

# ----- Host <-> device transfers -----------------------------------------

proc transferToDeviceRaw(d: Device; data: pointer; dt: DType;
    dims: openArray[int64]): PjrtBufferRaw =
  let c = ensureClient(d)
  let dev = selectDevice(c, d.ordinal)
  let pb = transferToDevice(c, dev, data, toPjrtBufferType(dt), dims)
  result = pb.raw
  pb.raw = nil.PjrtBufferRaw
  
proc transferToDevice*(d: Device; data: pointer; dt: DType;
    dims: openArray[int64]; sizeBytes: int = 0): BufferHandle =
  ## Copies `data` from host into a fresh device buffer on `d` and wraps
  ## the result in a `BufferHandle` whose releaser frees the buffer
  ## through the registry API on the last reference drop.
  let raw = transferToDeviceRaw(d, data, dt, dims)
  newBufferHandle(d.target, raw, pjrtReleaser, sizeBytes)

proc transferToHost*(d: Device; h: BufferHandle;
    dst: pointer; dstSize: int) =
  ## Copies the device buffer behind `h` into the host region.
  h.requireLive("eager.transferToHost")
  if h.isBufferSet:
    raise newException(EagerError,
      "eager.transferToHost: buffer set requires Tensor.toHost gather")
  let api = clientApi(d)
  let ev = bufferToHostBuffer(api, h.raw, dst, dstSize)
  if not ev.isNil:
    awaitEvent(api, ev, "eager.transferToHost:done")

proc transferRawToHost(d: Device; raw: PjrtBufferRaw;
    dst: pointer; dstSize: int) =
  let api = clientApi(d)
  let ev = bufferToHostBuffer(api, raw, dst, dstSize)
  if not ev.isNil:
    awaitEvent(api, ev, "eager.transferRawToHost:done")

proc shapeElementCount(shape: openArray[int]; opName: string): int =
  result = 1
  for d in shape:
    if d < 0:
      raise newException(EagerError,
        opName & ": shape contains negative dimension " & $d)
    result *= d

proc shapeDims(shape: openArray[int]): seq[int64] =
  result = newSeq[int64](shape.len)
  for i, d in shape:
    result[i] = int64(d)

func elementCount(shape: openArray[int]): int =
  result = 1
  for d in shape:
    result *= d

func rowMajorStrides(shape: openArray[int]): seq[int] =
  result = newSeq[int](shape.len)
  var stride = 1
  for i in countdown(shape.high, 0):
    result[i] = stride
    stride *= shape[i]

func meshAxisIndexLocal(mesh: Mesh; axis: string): int =
  for i, candidate in mesh.axes:
    if candidate == axis:
      return i
  -1

func meshAxisCoord(mesh: Mesh; linearIndex, axisIndex: int): int =
  var stride = 1
  for i in countdown(mesh.sizes.high, axisIndex + 1):
    stride *= mesh.sizes[i]
  (linearIndex div stride) mod mesh.sizes[axisIndex]

func shardShapeAndOffsets(t: Tensor; shardIndex: int):
    tuple[localShape, offsets: seq[int]] =
  result.localShape = t.shape
  result.offsets = newSeq[int](t.shape.len)
  if t.sharding.isReplicated:
    return
  let mesh = t.sharding.activeMesh()
  let spec = t.sharding.activeSpec()
  if shardIndex < 0 or shardIndex >= mesh.meshSize:
    raise newException(EagerError,
      "toHost: shard index " & $shardIndex &
        " is outside mesh size " & $mesh.meshSize)
  for dim, group in spec.axisGroups:
    if group.len == 0:
      continue
    var factor = 1
    var groupCoord = 0
    for axis in group:
      let axisIdx = meshAxisIndexLocal(mesh, axis)
      if axisIdx < 0:
        raise newException(EagerError,
          "toHost: sharding references unknown mesh axis '" & axis & "'")
      factor *= mesh.sizes[axisIdx]
      groupCoord = groupCoord * mesh.sizes[axisIdx] +
        meshAxisCoord(mesh, shardIndex, axisIdx)
    if t.shape[dim] mod factor != 0:
      raise newException(EagerError,
        "toHost: dimension " & $dim & " of shape " & $t.shape &
          " is not divisible by sharding factor " & $factor)
    result.localShape[dim] = t.shape[dim] div factor
    result.offsets[dim] = groupCoord * result.localShape[dim]

proc scatterShardToGlobal[T](src: openArray[T]; localShape: openArray[int];
    offsets: openArray[int]; globalShape: openArray[int]; dst: var seq[T]) =
  if src.len == 0:
    return
  let localStrides = rowMajorStrides(localShape)
  let globalStrides = rowMajorStrides(globalShape)
  for linear, value in src:
    var rest = linear
    var globalIndex = 0
    for dim in 0 ..< localShape.len:
      let coord =
        if localShape[dim] == 0: 0
        else:
          let c = rest div localStrides[dim]
          rest = rest mod localStrides[dim]
          c
      globalIndex += (offsets[dim] + coord) * globalStrides[dim]
    dst[globalIndex] = value

proc fromHostF32*(d: Device; data: openArray[float32];
    shape: openArray[int]): Tensor =
  ## Copies host `float32` data into a device tensor with `shape`.
  ## The shape product must equal `data.len`; host transfers are explicit
  ## and block until the plugin no longer reads the host buffer.
  let n = shapeElementCount(shape, "fromHostF32")
  if data.len != n:
    raise newException(EagerError,
      "fromHostF32: shape product " & $n &
        " != data length " & $data.len)
  var local = @data
  let ptrIn = if local.len == 0: nil else: addr local[0]
  let bytes = local.len * sizeof(float32)
  let h = transferToDevice(d, ptrIn, dtFloat32, shapeDims(shape), bytes)
  initEagerTensor(h, dtFloat32, shape, d)

proc fromHost*[T](d: Device; data: openArray[T];
    shape: openArray[int]): Tensor =
  ## Copies host scalar data into a device tensor with `shape`.
  ##
  ## `T` must be one of the native scalar types supported by `dtypeOf`.
  ## The shape product must equal `data.len`; host transfers are explicit
  ## and block until the plugin no longer reads the host buffer.
  let dt = dtypeOf(T)
  let n = shapeElementCount(shape, "fromHost")
  if data.len != n:
    raise newException(EagerError,
      "fromHost: shape product " & $n &
        " != data length " & $data.len)
  var local = @data
  let ptrIn = if local.len == 0: nil else: addr local[0]
  let bytes = local.len * sizeof(T)
  let h = transferToDevice(d, ptrIn, dt, shapeDims(shape), bytes)
  initEagerTensor(h, dt, shape, d)

proc copyGlobalBytesToShard(data: openArray[byte]; elemSize: int;
    globalShape: openArray[int]; layout: ShardLayout;
    dst: var seq[byte]) =
  let localElems = elementCount(layout.localShape)
  dst.setLen(localElems * elemSize)
  if dst.len == 0:
    return
  let localStrides = rowMajorStrides(layout.localShape)
  let globalStrides = rowMajorStrides(globalShape)
  for linear in 0 ..< localElems:
    var rest = linear
    var globalIndex = 0
    for dim in 0 ..< layout.localShape.len:
      let coord =
        if layout.localShape[dim] == 0: 0
        else:
          let c = rest div localStrides[dim]
          rest = rest mod localStrides[dim]
          c
      globalIndex += (layout.offsets[dim] + coord) * globalStrides[dim]
    copyMem(addr dst[linear * elemSize],
      unsafeAddr data[globalIndex * elemSize], elemSize)

proc scatterShardBytesToGlobal(data: openArray[byte]; elemSize: int;
    layout: ShardLayout; globalShape: openArray[int]; dst: var seq[byte]) =
  if data.len == 0:
    return
  let localElems = elementCount(layout.localShape)
  if data.len != localElems * elemSize:
    raise newException(EagerError,
      "scatterShardBytesToGlobal: shard " & $layout.index & " has " &
        $data.len & " byte(s), expected " & $(localElems * elemSize))
  let localStrides = rowMajorStrides(layout.localShape)
  let globalStrides = rowMajorStrides(globalShape)
  for linear in 0 ..< localElems:
    var rest = linear
    var globalIndex = 0
    for dim in 0 ..< layout.localShape.len:
      let coord =
        if layout.localShape[dim] == 0: 0
        else:
          let c = rest div localStrides[dim]
          rest = rest mod localStrides[dim]
          c
      globalIndex += (layout.offsets[dim] + coord) * globalStrides[dim]
    copyMem(addr dst[globalIndex * elemSize],
      unsafeAddr data[linear * elemSize], elemSize)

proc fromHostByteShards*(dtype: DType; shape: openArray[int];
    sharding: Sharding; loader: HostByteShardLoader;
    processIndex: int = -1): Tensor =
  ## Builds an eager global tensor from process-local host shards.
  ##
  ## The callback receives each local shard layout and fills exactly
  ## `prod(layout.localShape) * dtype.byteSize` bytes. Only those shards are
  ## transferred to devices, so no GPU ever receives the full global tensor.
  validateSharding(sharding, shape.len)
  let layouts = shardLayouts(shape, sharding, processIndex)
  if layouts.len == 0:
    raise newException(EagerError,
      "fromHostByteShards: no local shards for process " & $processIndex)
  var raws = newSeq[PjrtBufferRaw](layouts.len)
  var indices = newSeq[int](layouts.len)
  var firstDevice = layouts[0].device
  for i, layout in layouts:
    if layout.device.target != firstDevice.target:
      raise newException(EagerError,
        "fromHostByteShards: all shards must use the same PJRT target")
    var bytes: seq[byte] = @[]
    loader(layout, bytes)
    let expected = elementCount(layout.localShape) * dtype.byteSize
    if bytes.len != expected:
      raise newException(EagerError,
        "fromHostByteShards: loader returned " & $bytes.len &
          " byte(s) for shard " & $layout.index & ", expected " & $expected)
    let ptrIn = if bytes.len == 0: nil else: addr bytes[0]
    raws[i] = transferToDeviceRaw(layout.device, ptrIn, dtype,
      shapeDims(layout.localShape))
    indices[i] = layout.index
  let h = newBufferSetHandle(firstDevice.target, raws, pjrtReleaser,
    shardIndices = indices)
  initEagerTensor(h, dtype, shape, firstDevice, sharding)

proc fromHostShards*[T](shape: openArray[int]; sharding: Sharding;
    loader: HostShardLoader[T]; processIndex: int = -1): Tensor =
  ## Builds an eager global tensor from typed process-local host shards.
  ##
  ## This is the preferred path for loading oversized parameters: the callback
  ## fills one shard at a time, and only that shard is transferred to its
  ## target device.
  let dt = dtypeOf(T)
  fromHostByteShards(dt, shape, sharding,
    proc(layout: ShardLayout; bytes: var seq[byte]) =
      var values = newSeq[T](elementCount(layout.localShape))
      loader(layout, values)
      let expected = elementCount(layout.localShape)
      if values.len != expected:
        raise newException(EagerError,
          "fromHostShards: loader returned " & $values.len &
            " value(s) for shard " & $layout.index & ", expected " &
            $expected)
      bytes.setLen(values.len * sizeof(T))
      if bytes.len > 0:
        copyMem(addr bytes[0], addr values[0], bytes.len)
    , processIndex)

proc fromHostSharded*[T](data: openArray[T]; shape: openArray[int];
    sharding: Sharding; processIndex: int = -1): Tensor =
  ## Splits a host tensor into device shards and transfers each shard.
  ##
  ## This still materialises the full tensor on the host, but never on one GPU.
  ## For checkpoint streaming and truly huge tensors, prefer `fromHostShards`.
  let n = shapeElementCount(shape, "fromHostSharded")
  if data.len != n:
    raise newException(EagerError,
      "fromHostSharded: shape product " & $n &
        " != data length " & $data.len)
  let dt = dtypeOf(T)
  var bytes = newSeq[byte](data.len * sizeof(T))
  if bytes.len > 0:
    copyMem(addr bytes[0], unsafeAddr data[0], bytes.len)
  let globalShape = @shape
  fromHostByteShards(dt, globalShape, sharding,
    proc(layout: ShardLayout; shardBytes: var seq[byte]) =
      copyGlobalBytesToShard(bytes, dt.byteSize, globalShape, layout,
        shardBytes)
    , processIndex)

proc zerosSharded*(shape: openArray[int]; dtype: DType;
    sharding: Sharding; processIndex: int = -1): Tensor =
  ## Creates a sharded zero tensor without allocating the global value on host
  ## or on any single device.
  fromHostByteShards(dtype, shape, sharding,
    proc(layout: ShardLayout; bytes: var seq[byte]) =
      bytes = newSeq[byte](elementCount(layout.localShape) * dtype.byteSize)
    , processIndex)

proc zerosLikeEager*(t: Tensor): Tensor =
  ## Eager zero tensor preserving the input tensor's shape, dtype, device, and
  ## sharding/storage distribution.
  t.requireEager("zerosLikeEager")
  if t.buffer.isBufferSet and not t.sharding.isReplicated:
    let indices = t.buffer.shardIndices()
    var raws = newSeq[PjrtBufferRaw](indices.len)
    for i, shardIndex in indices:
      let layout = shardLayout(t.shape, t.sharding.activeMesh(),
        t.sharding.activeSpec(), shardIndex)
      let bytes = newSeq[byte](elementCount(layout.localShape) *
        t.dtype.byteSize)
      let ptrIn = if bytes.len == 0: nil else: unsafeAddr bytes[0]
      raws[i] = transferToDeviceRaw(layout.device, ptrIn, t.dtype,
        shapeDims(layout.localShape))
    let h = newBufferSetHandle(t.device.target, raws, pjrtReleaser,
      shardIndices = indices)
    initEagerTensor(h, t.dtype, t.shape, t.device, t.sharding)
  else:
    let bytes = newSeq[byte](t.numElements * t.dtype.byteSize)
    let ptrIn = if bytes.len == 0: nil else: unsafeAddr bytes[0]
    let h = transferToDevice(t.device, ptrIn, t.dtype, shapeDims(t.shape),
      bytes.len)
    initEagerTensor(h, t.dtype, t.shape, t.device, t.sharding)

proc fromHost*[T](data: openArray[T]; shape: openArray[int];
    device: Device = defaultDevice()): Tensor =
  ## Copies host scalar data into a tensor on `device`.
  fromHost(device, data, shape)

proc scalarF32*(d: Device; value: float32): Tensor =
  ## Copies a host `float32` scalar into a 0-d device tensor.
  fromHostF32(d, [value], [])

proc scalar*[T](d: Device; value: T): Tensor =
  ## Copies a native scalar into a 0-d device tensor.
  fromHost(d, [value], [])

proc scalar*[T](value: T; device: Device = defaultDevice()): Tensor =
  ## Copies a native scalar into a 0-d tensor on `device`.
  scalar(device, value)

proc toHost*[T](t: Tensor; valueType: typedesc[T]): seq[T] =
  ## Copies an eager tensor from device to host as a `seq[T]`.
  ##
  ## `T` must match `t.dtype` exactly. Use `astype` explicitly before
  ## transferring when a dtype conversion is desired.
  t.requireEager("toHost")
  let dt = dtypeOf(T)
  if t.dtype != dt:
    raise newException(EagerError,
      "toHost: requested " & dt.name & " but tensor dtype is " &
        t.dtype.name)
  result = newSeq[T](t.numElements)
  if result.len == 0:
    return
  if t.buffer.isBufferSet:
    if t.sharding.isReplicated:
      transferRawToHost(t.device, t.buffer.raws[0], addr result[0],
        result.len * sizeof(T))
      return
    let mesh = t.sharding.activeMesh()
    if t.buffer.raws.len != mesh.meshSize:
      raise newException(EagerError,
        "toHost: global gather requires all " & $mesh.meshSize &
          " mesh shard(s) to be addressable in this process; have " &
          $t.buffer.raws.len)
    let indices = t.buffer.shardIndices()
    for rawPos, raw in t.buffer.raws:
      let shardIndex = indices[rawPos]
      let layout = shardShapeAndOffsets(t, shardIndex)
      var shard = newSeq[T](elementCount(layout.localShape))
      if shard.len > 0:
        transferRawToHost(t.device, raw, addr shard[0],
          shard.len * sizeof(T))
      scatterShardToGlobal(shard, layout.localShape, layout.offsets,
        t.shape, result)
  else:
    transferToHost(t.device, t.buffer, addr result[0],
      result.len * sizeof(T))

proc transferRawToHostBytes(d: Device; raw: PjrtBufferRaw;
    byteSize: int): seq[byte] =
  result = newSeq[byte](byteSize)
  if byteSize > 0:
    transferRawToHost(d, raw, addr result[0], byteSize)

proc toHostShards*(t: Tensor): seq[HostTensorShard] =
  ## Transfers each locally-addressable shard to host bytes.
  ##
  ## Unlike `toHost`, this never gathers a global sharded tensor. It is the
  ## checkpointing and diagnostics path for model states larger than one
  ## device.
  t.requireEager("toHostShards")
  if t.buffer.isBufferSet:
    if t.sharding.isReplicated:
      let layout = ShardLayout(index: 0, process: 0, device: t.device,
        globalShape: t.shape, localShape: t.shape,
        offsets: newSeq[int](t.shape.len))
      return @[HostTensorShard(layout: layout, dtype: t.dtype,
        data: transferRawToHostBytes(t.device, t.buffer.raws[0],
          t.numElements * t.dtype.byteSize))]
    let mesh = t.sharding.activeMesh()
    let spec = t.sharding.activeSpec()
    let indices = t.buffer.shardIndices()
    result = newSeq[HostTensorShard](t.buffer.raws.len)
    for rawPos, raw in t.buffer.raws:
      let layout = shardLayout(t.shape, mesh, spec, indices[rawPos])
      let byteSize = elementCount(layout.localShape) * t.dtype.byteSize
      result[rawPos] = HostTensorShard(layout: layout, dtype: t.dtype,
        data: transferRawToHostBytes(layout.device, raw, byteSize))
  else:
    let layout = ShardLayout(index: 0, process: 0, device: t.device,
      globalShape: t.shape, localShape: t.shape,
      offsets: newSeq[int](t.shape.len))
    result = @[HostTensorShard(layout: layout, dtype: t.dtype,
      data: transferRawToHostBytes(t.device, t.buffer.raw,
        t.numElements * t.dtype.byteSize))]

proc toHostBytes*(t: Tensor): seq[byte] =
  ## Transfers an eager tensor to host bytes, gathering sharded tensors when
  ## all global shards are addressable in this process.
  t.requireEager("toHostBytes")
  result = newSeq[byte](t.numElements * t.dtype.byteSize)
  if result.len == 0:
    return
  if t.buffer.isBufferSet:
    if t.sharding.isReplicated:
      transferRawToHost(t.device, t.buffer.raws[0], addr result[0],
        result.len)
      return
    let mesh = t.sharding.activeMesh()
    if t.buffer.raws.len != mesh.meshSize:
      raise newException(EagerError,
        "toHostBytes: global gather requires all " & $mesh.meshSize &
          " mesh shard(s) to be addressable in this process; have " &
          $t.buffer.raws.len)
    let shards = toHostShards(t)
    for shard in shards:
      scatterShardBytesToGlobal(shard.data, t.dtype.byteSize,
        shard.layout, t.shape, result)
  else:
    transferToHost(t.device, t.buffer, addr result[0], result.len)

proc item*[T](t: Tensor; valueType: typedesc[T]): T =
  ## Copies a 0-d eager tensor to the host and returns it as a scalar.
  ##
  ## Raises `TensorError` when `t` is not scalar. Use `toHost(T)` for
  ## non-scalar tensors.
  if t.shape.len != 0:
    raise newException(TensorError,
      "item: expected a 0-d tensor, got shape " & $t.shape)
  let values = toHost(t, T)
  values[0]

proc `to`*(t: Tensor; d: Device): Tensor =
  ## Explicitly copies an eager tensor to `d`.
  ##
  ## Cross-device movement in rew is never implicit; this proc is the
  ## public mover. Moving to the current device is a metadata-preserving
  ## no-op.
  if t.device == d:
    return t
  t.requireEager("to")
  let byteSize = t.numElements * t.dtype.byteSize
  var host = toHostBytes(t)
  let hostPtr = if host.len == 0: nil else: addr host[0]
  let h = transferToDevice(d, hostPtr, t.dtype, shapeDims(t.shape), byteSize)
  initEagerTensor(h, t.dtype, t.shape, d, t.sharding)

proc `to`*(t: Tensor; sharding: Sharding; processIndex: int = -1): Tensor =
  ## Explicitly moves an eager tensor into the storage layout described by
  ## `sharding`. Metadata-only annotations use `withSharding`; this proc owns
  ## real host/device movement and creates one device buffer per local shard.
  t.requireEager("to(sharding)")
  validateSharding(sharding, t.shape.len)
  if not t.buffer.isBufferSet and sharding.isReplicated:
    return t.withSharding(sharding)
  if t.buffer.isBufferSet and
      t.sharding.shardingKey == sharding.shardingKey:
    return t
  let globalBytes = toHostBytes(t)
  fromHostByteShards(t.dtype, t.shape, sharding,
    proc(layout: ShardLayout; shardBytes: var seq[byte]) =
      copyGlobalBytesToShard(globalBytes, t.dtype.byteSize, t.shape, layout,
        shardBytes)
    , processIndex)

proc `to`*(t: Tensor; mesh: Mesh; spec: PartitionSpec;
    processIndex: int = -1): Tensor =
  ## Explicitly moves an eager tensor to a mesh-partitioned layout.
  t.to(initPartitioned(mesh, spec), processIndex)

proc shardToMesh*(t: Tensor; mesh: Mesh; spec: PartitionSpec;
    processIndex: int = -1): Tensor =
  ## Verbose alias for `t.to(mesh, spec)` used at parameter-loading sites
  ## where sharded residency is more readable than generic movement.
  t.to(mesh, spec, processIndex)

# ----- Per-op StableHLO emission and execution cache --------------------

type
  EagerOpKind = enum
    eokBinary
    eokUnary
    eokComplex
    eokReal
    eokImag
    eokReshape
    eokTranspose
    eokReverse
    eokConvert
    eokBitcastConvert
    eokIsFinite
    eokReducePrecision
    eokReduce
    eokDot
    eokDotGeneral
    eokBroadcastTo
    eokConv2d
    eokMaxPool2d
    eokBatchNormInference
    eokBatchNormTraining
    eokBatchNormGrad
    eokCholesky
    eokGetDimensionSize
    eokPad
    eokBroadcast
    eokDynamicSlice
    eokDynamicUpdateSlice
    eokIota
    eokReplicaId
    eokPartitionId
    eokSetDimensionSize
    eokDynamicReshape
    eokDynamicPad
    eokDynamicIota
    eokRealDynamicSlice
    eokDynamicBroadcastInDim
    eokFft
    eokTriangularSolve
    eokEinsum
    eokUnaryEinsum
    eokTorchIndexSelect
    eokClamp
    eokConcatenate
    eokSlice
    eokSelect
    eokCompare

  EagerOpSpec = object
    kind: EagerOpKind
    name: string

var
  eagerExecCache: Table[string, PjrtLoadedExecutable]
    ## Compiled executables keyed by `target|op|dtype|shape|attrs`.
  eagerOpSpecs: Table[string, EagerOpSpec]

const eagerOps: array[47, EagerOpSpec] = [
  EagerOpSpec(kind: eokBinary, name: "add"),
  EagerOpSpec(kind: eokBinary, name: "sub"),
  EagerOpSpec(kind: eokBinary, name: "mul"),
  EagerOpSpec(kind: eokBinary, name: "divide"),
  EagerOpSpec(kind: eokBinary, name: "maximum"),
  EagerOpSpec(kind: eokBinary, name: "minimum"),
  EagerOpSpec(kind: eokBinary, name: "atan2"),
  EagerOpSpec(kind: eokBinary, name: "power"),
  EagerOpSpec(kind: eokBinary, name: "remainder"),
  EagerOpSpec(kind: eokComplex, name: "complex"),
  EagerOpSpec(kind: eokBinary, name: "bitwiseAnd"),
  EagerOpSpec(kind: eokBinary, name: "bitwiseOr"),
  EagerOpSpec(kind: eokBinary, name: "bitwiseXor"),
  EagerOpSpec(kind: eokBinary, name: "shiftLeft"),
  EagerOpSpec(kind: eokBinary, name: "shiftRightArithmetic"),
  EagerOpSpec(kind: eokBinary, name: "shiftRightLogical"),
  EagerOpSpec(kind: eokUnary,  name: "neg"),
  EagerOpSpec(kind: eokUnary,  name: "abs"),
  EagerOpSpec(kind: eokUnary,  name: "exp"),
  EagerOpSpec(kind: eokUnary,  name: "log"),
  EagerOpSpec(kind: eokUnary,  name: "sqrt"),
  EagerOpSpec(kind: eokUnary,  name: "tanh"),
  EagerOpSpec(kind: eokUnary,  name: "cbrt"),
  EagerOpSpec(kind: eokUnary,  name: "ceil"),
  EagerOpSpec(kind: eokUnary,  name: "expm1"),
  EagerOpSpec(kind: eokUnary,  name: "floor"),
  EagerOpSpec(kind: eokUnary,  name: "log1p"),
  EagerOpSpec(kind: eokUnary,  name: "logistic"),
  EagerOpSpec(kind: eokUnary,  name: "tan"),
  EagerOpSpec(kind: eokUnary,  name: "sign"),
  EagerOpSpec(kind: eokUnary,  name: "roundNearestAfz"),
  EagerOpSpec(kind: eokUnary,  name: "roundNearestEven"),
  EagerOpSpec(kind: eokUnary,  name: "bitwiseNot"),
  EagerOpSpec(kind: eokUnary,  name: "countLeadingZeros"),
  EagerOpSpec(kind: eokUnary,  name: "popcnt"),
  EagerOpSpec(kind: eokUnary,  name: "optimizationBarrier"),
  EagerOpSpec(kind: eokConvert, name: "astype"),
  EagerOpSpec(kind: eokBitcastConvert, name: "bitcastConvert"),
  EagerOpSpec(kind: eokIsFinite, name: "isFinite"),
  EagerOpSpec(kind: eokReal, name: "real"),
  EagerOpSpec(kind: eokImag, name: "imag"),
  EagerOpSpec(kind: eokReducePrecision, name: "reducePrecision"),
  EagerOpSpec(kind: eokReshape,    name: "reshape"),
  EagerOpSpec(kind: eokTranspose,  name: "transpose"),
  EagerOpSpec(kind: eokReverse,    name: "reverse"),
  EagerOpSpec(kind: eokDot,        name: "dot"),
  EagerOpSpec(kind: eokDotGeneral, name: "dotGeneral"),
]

const eagerExtraOps: array[41, EagerOpSpec] = [
  EagerOpSpec(kind: eokReduce,       name: "reduceSum"),
  EagerOpSpec(kind: eokReduce,       name: "reduceMax"),
  EagerOpSpec(kind: eokReduce,       name: "reduceMin"),
  EagerOpSpec(kind: eokReduce,       name: "reduceProd"),
  EagerOpSpec(kind: eokReduce,       name: "all"),
  EagerOpSpec(kind: eokReduce,       name: "any"),
  EagerOpSpec(kind: eokBroadcastTo,  name: "broadcastTo"),
  EagerOpSpec(kind: eokConv2d,       name: "conv2d"),
  EagerOpSpec(kind: eokMaxPool2d,    name: "maxPool2d"),
  EagerOpSpec(kind: eokBatchNormInference, name: "batchNormInference"),
  EagerOpSpec(kind: eokBatchNormTraining, name: "batchNormTraining"),
  EagerOpSpec(kind: eokBatchNormGrad, name: "batchNormGrad"),
  EagerOpSpec(kind: eokCholesky, name: "cholesky"),
  EagerOpSpec(kind: eokGetDimensionSize, name: "getDimensionSize"),
  EagerOpSpec(kind: eokPad, name: "pad"),
  EagerOpSpec(kind: eokBroadcast, name: "broadcast"),
  EagerOpSpec(kind: eokDynamicSlice, name: "dynamicSlice"),
  EagerOpSpec(kind: eokDynamicUpdateSlice, name: "dynamicUpdateSlice"),
  EagerOpSpec(kind: eokIota, name: "iota"),
  EagerOpSpec(kind: eokReplicaId, name: "replicaId"),
  EagerOpSpec(kind: eokPartitionId, name: "partitionId"),
  EagerOpSpec(kind: eokSetDimensionSize, name: "setDimensionSize"),
  EagerOpSpec(kind: eokDynamicReshape, name: "dynamicReshape"),
  EagerOpSpec(kind: eokDynamicPad, name: "dynamicPad"),
  EagerOpSpec(kind: eokDynamicIota, name: "dynamicIota"),
  EagerOpSpec(kind: eokRealDynamicSlice, name: "realDynamicSlice"),
  EagerOpSpec(kind: eokDynamicBroadcastInDim,
    name: "dynamicBroadcastInDim"),
  EagerOpSpec(kind: eokFft, name: "fft"),
  EagerOpSpec(kind: eokTriangularSolve, name: "triangularSolve"),
  EagerOpSpec(kind: eokEinsum, name: "einsum"),
  EagerOpSpec(kind: eokUnaryEinsum, name: "unaryEinsum"),
  EagerOpSpec(kind: eokTorchIndexSelect, name: "torchIndexSelect"),
  EagerOpSpec(kind: eokUnary, name: "sine"),
  EagerOpSpec(kind: eokUnary, name: "cosine"),
  EagerOpSpec(kind: eokUnary, name: "rsqrt"),
  EagerOpSpec(kind: eokUnary, name: "stopGradient"),
  EagerOpSpec(kind: eokClamp, name: "clamp"),
  EagerOpSpec(kind: eokConcatenate, name: "concat"),
  EagerOpSpec(kind: eokSlice, name: "slice"),
  EagerOpSpec(kind: eokSelect, name: "select"),
  EagerOpSpec(kind: eokCompare, name: "compare"),
]

proc ensureEagerOpSpecs() =
  if eagerOpSpecs.len > 0:
    return
  for spec in eagerOps:
    eagerOpSpecs[spec.name] = spec
  for spec in eagerExtraOps:
    eagerOpSpecs[spec.name] = spec

proc findOpSpec(name: string): EagerOpSpec =
  ensureEagerOpSpecs()
  if eagerOpSpecs.hasKey(name):
    return eagerOpSpecs[name]
  raise newException(EagerError,
    "eager backend: unsupported op '" & name & "'")

# ----- Attribute parsing --------------------------------------------------

proc findAttr(attrs: openArray[(string, string)]; key: string): string
    {.raises: [EagerError].} =
  for (k, v) in attrs:
    if k == key: return v
  raise newException(EagerError,
    "eager: required attr '" & key & "' missing")

proc parseDType(s: string): DType {.raises: [EagerError].} =
  for dt in DType:
    if s == dt.name or s == $dt:
      return dt
  raise newException(EagerError,
    "eager: unknown dtype attr '" & s & "'")

proc parseFftType(s: string): FftType {.raises: [EagerError].} =
  case s
  of "FFT": ftFft
  of "IFFT": ftIfft
  of "RFFT": ftRfft
  of "IRFFT": ftIrfft
  else:
    raise newException(EagerError,
      "eager: unknown fft_type attr '" & s & "'")

proc parseTransposeKind(s: string): TransposeKind {.raises: [EagerError].} =
  case s
  of "NO_TRANSPOSE": tkNoTranspose
  of "TRANSPOSE": tkTranspose
  of "ADJOINT": tkAdjoint
  else:
    raise newException(EagerError,
      "eager: unknown transpose_a attr '" & s & "'")

proc parseIntAttr(s: string): int {.raises: [EagerError].} =
  try:
    parseInt(s)
  except ValueError:
    raise newException(EagerError,
      "eager: malformed int attr '" & s & "'")

proc parseDeviceAttr(attrs: openArray[(string, string)]): Device
    {.raises: [EagerError].} =
  try:
    parseDevice(findAttr(attrs, "device"))
  except DeviceError as e:
    raise newException(EagerError, "eager: " & e.msg)

proc parseIntList(s: string): seq[int] {.raises: [EagerError].} =
  var t = s.strip()
  if t.len > 0 and t[0] == '@': t = t[1 .. ^1]
  if t.len < 2 or t[0] != '[' or t[^1] != ']':
    raise newException(EagerError,
      "eager: malformed int-list attr (expected '[a,b,c]', got '" & s & "')")
  let body = t[1 ..< t.len - 1].strip()
  if body.len == 0: return @[]
  result = @[]
  for part in body.split(','):
    let p = part.strip()
    try: result.add parseInt(p)
    except ValueError:
      raise newException(EagerError,
        "eager: malformed int in attr '" & s & "'")

proc parsePadding(s: string): seq[array[2, int]] {.raises: [EagerError].} =
  let t = s.strip()
  if t.len < 2 or t[0] != '[' or t[^1] != ']':
    raise newException(EagerError,
      "eager: malformed padding attr (expected '[[lo,hi],...]', got '" &
        s & "')")
  let body = t[1 ..< t.len - 1].strip()
  result = @[]
  if body.len == 0: return result
  var i = 0
  while i < body.len:
    while i < body.len and (body[i] == ',' or body[i] == ' '): inc i
    if i >= body.len: break
    if body[i] != '[':
      raise newException(EagerError,
        "eager: malformed padding pair start in '" & s & "'")
    let close = body.find(']', i)
    if close < 0:
      raise newException(EagerError,
        "eager: malformed padding pair end in '" & s & "'")
    let pair = body[i + 1 ..< close]
    let parts = pair.split(',')
    if parts.len != 2:
      raise newException(EagerError,
        "eager: padding pair must have 2 ints in '" & s & "'")
    var lo, hi: int
    try:
      lo = parseInt(parts[0].strip())
      hi = parseInt(parts[1].strip())
    except ValueError:
      raise newException(EagerError,
        "eager: malformed int in padding pair of '" & s & "'")
    result.add [lo, hi]
    i = close + 1

# ----- Output-shape helpers ---------------------------------------------

proc reducedShape(inShape: openArray[int]; dims: openArray[int]): seq[int] =
  var keep = newSeq[bool](inShape.len)
  for i in 0 ..< inShape.len: keep[i] = true
  for d in dims:
    if d >= 0 and d < inShape.len: keep[d] = false
  result = @[]
  for i, d in inShape:
    if keep[i]: result.add d

proc transposedShape(inShape: openArray[int]; perm: openArray[int]): seq[int] =
  result = newSeq[int](perm.len)
  for i, p in perm:
    result[i] = inShape[p]

proc dotGeneralShape(lShape, rShape: openArray[int];
    lhsBatching, rhsBatching, lhsContracting, rhsContracting:
        openArray[int]): seq[int] =
  var lUsed = newSeq[bool](lShape.len)
  var rUsed = newSeq[bool](rShape.len)
  for d in lhsBatching: lUsed[d] = true
  for d in rhsBatching: rUsed[d] = true
  for d in lhsContracting: lUsed[d] = true
  for d in rhsContracting: rUsed[d] = true
  result = @[]
  for d in lhsBatching: result.add lShape[d]
  for i in 0 ..< lShape.len:
    if not lUsed[i]: result.add lShape[i]
  for i in 0 ..< rShape.len:
    if not rUsed[i]: result.add rShape[i]

# ----- Constants ---------------------------------------------------------

proc float32Bytes(v: float32): seq[byte] =
  let bits = cast[uint32](v)
  result = newSeq[byte](4)
  result[0] = byte(bits and 0xFF'u32)
  result[1] = byte((bits shr 8) and 0xFF'u32)
  result[2] = byte((bits shr 16) and 0xFF'u32)
  result[3] = byte((bits shr 24) and 0xFF'u32)

proc reductionInit(op: string; dt: DType): seq[byte]
    {.raises: [EagerError].} =
  case op
  of "reduceSum", "reduceMax", "reduceMin", "reduceProd":
    if dt != dtFloat32:
      raise newException(EagerError,
        "eager '" & op & "': v1 supports only float32 (got " & $dt & ")")
    case op
    of "reduceSum": float32Bytes(0.0'f32)
    of "reduceMax": float32Bytes(NegInf.float32)
    of "reduceMin": float32Bytes(Inf.float32)
    else: float32Bytes(1.0'f32)
  of "all", "any":
    if dt != dtBool:
      raise newException(EagerError,
        "eager '" & op & "': expected bool tensor (got " & $dt & ")")
    if op == "all": @[1'u8] else: @[0'u8]
  else:
    raise newException(EagerError,
      "eager: unknown reduction op '" & op & "'")

proc reduceOpBody(b: var ShBuilder; op: string;
    lhs, rhs: ShValueId): ShValueId =
  case op
  of "reduceSum": shops.add(b, lhs, rhs)
  of "reduceMax": shops.maximum(b, lhs, rhs)
  of "reduceMin": shops.minimum(b, lhs, rhs)
  of "reduceProd": shops.mul(b, lhs, rhs)
  of "all": shops.andOp(b, lhs, rhs)
  of "any": shops.orOp(b, lhs, rhs)
  else: raise newException(ValueError, "eager: unknown reduction '" & op & "'")

# ----- Output-shape derivation -------------------------------------------

proc deriveOutputShape(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): seq[int]
    {.raises: [EagerError].} =
  case spec.kind
  of eokBinary, eokUnary:
    @(operands[0].shape)
  of eokComplex, eokReal, eokImag:
    @(operands[0].shape)
  of eokConvert, eokIsFinite:
    @(operands[0].shape)
  of eokReducePrecision:
    @(operands[0].shape)
  of eokBitcastConvert:
    parseIntList(findAttr(attrs, "output_shape"))
  of eokReshape:
    parseIntList(findAttr(attrs, "shape"))
  of eokTranspose:
    transposedShape(operands[0].shape, parseIntList(findAttr(attrs, "permutation")))
  of eokReverse:
    @(operands[0].shape)
  of eokReduce:
    reducedShape(operands[0].shape, parseIntList(findAttr(attrs, "dims")))
  of eokDot:
    try:
      shops.dotOutputShape(operands[0].shape, operands[1].shape)
    except ShBuilderError as e:
      raise newException(EagerError, "eager dot: " & e.msg)
  of eokDotGeneral:
    dotGeneralShape(operands[0].shape, operands[1].shape,
      parseIntList(findAttr(attrs, "lhs_batching")),
      parseIntList(findAttr(attrs, "rhs_batching")),
      parseIntList(findAttr(attrs, "lhs_contracting")),
      parseIntList(findAttr(attrs, "rhs_contracting")))
  of eokBroadcastTo:
    parseIntList(findAttr(attrs, "output_shape"))
  of eokConv2d:
    let strides = parseIntList(findAttr(attrs, "strides"))
    let padPairs = parsePadding(findAttr(attrs, "padding"))
    let dilation = parseIntList(findAttr(attrs, "dilation"))
    if strides.len != 2 or padPairs.len != 2 or dilation.len != 2:
      raise newException(EagerError,
        "eager conv2d: strides/padding/dilation must have 2 entries each")
    var stridesArr: array[2, int] = [strides[0], strides[1]]
    var dilArr: array[2, int] = [dilation[0], dilation[1]]
    var padArr: array[2, array[2, int]] = [
      [padPairs[0][0], padPairs[0][1]],
      [padPairs[1][0], padPairs[1][1]],
    ]
    let dims = nhwcOIHWConvDims(2)
    try:
      convolutionOutputShape(operands[0].shape, operands[1].shape,
        stridesArr, padArr, [1, 1], dilArr, dims, 1, 1)
    except ShBuilderError as e:
      raise newException(EagerError, "eager conv2d: " & e.msg)
  of eokMaxPool2d:
    let kernel = parseIntList(findAttr(attrs, "kernel_size"))
    let strides = parseIntList(findAttr(attrs, "strides"))
    let padPairs = parsePadding(findAttr(attrs, "padding"))
    if kernel.len != 2 or strides.len != 2 or padPairs.len != 2:
      raise newException(EagerError,
        "eager maxPool2d: kernel/strides/padding must have 2 entries each")
    let windowDims = [1, kernel[0], kernel[1], 1]
    let windowStrides = [1, strides[0], strides[1], 1]
    let pad: array[4, array[2, int]] =
      [[0, 0],
       [padPairs[0][0], padPairs[0][1]],
       [padPairs[1][0], padPairs[1][1]],
       [0, 0]]
    let dilations = [1, 1, 1, 1]
    try:
      reduceWindowOutputShape(operands[0].shape, windowDims, windowStrides,
        pad, dilations, dilations)
    except ShBuilderError as e:
      raise newException(EagerError, "eager maxPool2d: " & e.msg)
  of eokBatchNormInference:
    @(operands[0].shape)
  of eokBatchNormTraining, eokBatchNormGrad:
    raise newException(EagerError,
      "eager: multi-output op '" & spec.name &
        "' must use deriveOutputShapes")
  of eokCholesky:
    @(operands[0].shape)
  of eokGetDimensionSize:
    @[]
  of eokPad:
    let lows = parseIntList(findAttr(attrs, "edge_padding_low"))
    let highs = parseIntList(findAttr(attrs, "edge_padding_high"))
    let interiors = parseIntList(findAttr(attrs, "interior_padding"))
    try:
      shops.padOutputShape(operands[0].shape, lows, highs, interiors)
    except ShBuilderError as e:
      raise newException(EagerError, "eager pad: " & e.msg)
  of eokBroadcast:
    let sizes = parseIntList(findAttr(attrs, "broadcast_sizes"))
    var outShape = sizes
    outShape.add operands[0].shape
    outShape
  of eokDynamicSlice:
    parseIntList(findAttr(attrs, "slice_sizes"))
  of eokDynamicUpdateSlice:
    @(operands[0].shape)
  of eokIota:
    parseIntList(findAttr(attrs, "shape"))
  of eokReplicaId, eokPartitionId:
    @[]
  of eokSetDimensionSize:
    @(operands[0].shape)
  of eokDynamicReshape, eokDynamicPad:
    parseIntList(findAttr(attrs, "result_shape"))
  of eokDynamicIota, eokRealDynamicSlice:
    parseIntList(findAttr(attrs, "result_shape"))
  of eokDynamicBroadcastInDim:
    parseIntList(findAttr(attrs, "result_shape"))
  of eokFft:
    let fftType = parseFftType(findAttr(attrs, "fft_type"))
    let fftLength = parseIntList(findAttr(attrs, "fft_length"))
    try:
      shops.fftOutputType(operands[0].tensorTypeOf, fftType, fftLength).shape
    except ShBuilderError as e:
      raise newException(EagerError, "eager fft: " & e.msg)
  of eokTriangularSolve:
    try:
      shops.triangularSolveOutputShape(operands[0].tensorTypeOf,
        operands[1].tensorTypeOf, findAttr(attrs, "left_side") == "true")
    except ShBuilderError as e:
      raise newException(EagerError, "eager triangularSolve: " & e.msg)
  of eokEinsum, eokUnaryEinsum:
    parseIntList(findAttr(attrs, "output_shape"))
  of eokTorchIndexSelect:
    try:
      shops.torchIndexSelectOutputShape(operands[0].shape, operands[1].shape,
        parseIntAttr(findAttr(attrs, "dim")),
        parseIntAttr(findAttr(attrs, "batch_dims")))
    except ShBuilderError as e:
      raise newException(EagerError, "eager torchIndexSelect: " & e.msg)
  of eokClamp:
    @(operands[1].shape)
  of eokConcatenate:
    let dim = parseIntAttr(findAttr(attrs, "dimension"))
    var outShape = @(operands[0].shape)
    if dim < 0 or dim >= outShape.len:
      raise newException(EagerError,
        "eager concat: dimension " & $dim & " out of range for rank " &
          $outShape.len)
    for i in 1 ..< operands.len:
      outShape[dim] += operands[i].shape[dim]
    outShape
  of eokSlice:
    let startIndices = parseIntList(findAttr(attrs, "start_indices"))
    let limitIndices = parseIntList(findAttr(attrs, "limit_indices"))
    let strides = parseIntList(findAttr(attrs, "strides"))
    let rank = operands[0].shape.len
    if startIndices.len != rank or limitIndices.len != rank or strides.len != rank:
      raise newException(EagerError,
        "eager slice: index arrays length must match operand rank " & $rank)
    var outShape = newSeq[int](rank)
    for i in 0 ..< rank:
      outShape[i] = (limitIndices[i] - startIndices[i] + strides[i] - 1) div strides[i]
    outShape
  of eokSelect:
    @(operands[0].shape)
  of eokCompare:
    @(operands[0].shape)

proc deriveOutputDType(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): DType
    {.raises: [EagerError].} =
  case spec.kind
  of eokComplex:
    operands[0].dtype.complexDType
  of eokReal, eokImag:
    operands[0].dtype.complexPartDType
  of eokConvert, eokBitcastConvert:
    parseDType(findAttr(attrs, "dtype"))
  of eokIsFinite:
    dtBool
  of eokGetDimensionSize:
    dtInt32
  of eokIota:
    parseDType(findAttr(attrs, "dtype"))
  of eokDynamicIota:
    parseDType(findAttr(attrs, "dtype"))
  of eokFft:
    let fftType = parseFftType(findAttr(attrs, "fft_type"))
    let fftLength = parseIntList(findAttr(attrs, "fft_length"))
    try:
      shops.fftOutputType(operands[0].tensorTypeOf, fftType, fftLength).dtype
    except ShBuilderError as e:
      raise newException(EagerError, "eager fft: " & e.msg)
  of eokReplicaId, eokPartitionId:
    dtUint32
  of eokSelect:
    operands[1].dtype
  of eokCompare:
    dtBool
  of eokClamp:
    operands[1].dtype
  else:
    operands[0].dtype

proc deriveOutputShapes(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): seq[seq[int]]
    {.raises: [EagerError].} =
  case spec.kind
  of eokBatchNormTraining, eokBatchNormGrad:
    let featureIndex = parseIntAttr(findAttr(attrs, "feature_index"))
    if featureIndex < 0 or featureIndex >= operands[0].shape.len:
      raise newException(EagerError,
        "eager '" & spec.name & "': feature_index " & $featureIndex &
          " out of range for rank " & $operands[0].shape.len)
    let featureShape = @[operands[0].shape[featureIndex]]
    @[(@operands[0].shape), featureShape, featureShape]
  else:
    @[deriveOutputShape(spec, operands, attrs)]

proc deriveOutputDTypes(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): seq[DType]
    {.raises: [EagerError].} =
  case spec.kind
  of eokBatchNormTraining, eokBatchNormGrad:
    @[operands[0].dtype, operands[0].dtype, operands[0].dtype]
  else:
    @[deriveOutputDType(spec, operands, attrs)]

# ----- Module emission ---------------------------------------------------

proc emitElementwise(b: var ShBuilder; op: string;
    lhs, rhs: ShValueId): ShValueId =
  case op
  of "add":     shops.add(b, lhs, rhs)
  of "sub":     shops.sub(b, lhs, rhs)
  of "mul":     shops.mul(b, lhs, rhs)
  of "divide":  shops.divide(b, lhs, rhs)
  of "maximum": shops.maximum(b, lhs, rhs)
  of "minimum": shops.minimum(b, lhs, rhs)
  of "atan2":   shops.atan2(b, lhs, rhs)
  of "power":   shops.power(b, lhs, rhs)
  of "remainder": shops.remainder(b, lhs, rhs)
  of "bitwiseAnd": shops.andOp(b, lhs, rhs)
  of "bitwiseOr": shops.orOp(b, lhs, rhs)
  of "bitwiseXor": shops.xorOp(b, lhs, rhs)
  of "shiftLeft": shops.shiftLeft(b, lhs, rhs)
  of "shiftRightArithmetic": shops.shiftRightArithmetic(b, lhs, rhs)
  of "shiftRightLogical": shops.shiftRightLogical(b, lhs, rhs)
  else:
    raise newException(EagerError,
      "emitElementwise: unsupported op '" & op & "'")

proc emitUnary(b: var ShBuilder; op: string; x: ShValueId): ShValueId =
  case op
  of "neg":  shops.neg(b, x)
  of "abs":  shops.abs(b, x)
  of "exp":  shops.exponential(b, x)
  of "log":  shops.log(b, x)
  of "sqrt": shops.sqrt(b, x)
  of "tanh": shops.tanh(b, x)
  of "cbrt": shops.cbrt(b, x)
  of "ceil": shops.ceil(b, x)
  of "expm1": shops.exponentialMinusOne(b, x)
  of "floor": shops.floor(b, x)
  of "log1p": shops.logPlusOne(b, x)
  of "logistic": shops.logistic(b, x)
  of "tan": shops.tan(b, x)
  of "sign": shops.sign(b, x)
  of "roundNearestAfz": shops.roundNearestAfz(b, x)
  of "roundNearestEven": shops.roundNearestEven(b, x)
  of "bitwiseNot": shops.notOp(b, x)
  of "countLeadingZeros": shops.countLeadingZeros(b, x)
  of "popcnt": shops.popcnt(b, x)
  of "cosine": shops.cosine(b, x)
  of "rsqrt": shops.rsqrt(b, x)
  of "sine": shops.sine(b, x)
  of "optimizationBarrier": shops.optimizationBarrier(b, x)
  of "stopGradient": shops.optimizationBarrier(b, x)
  else:
    raise newException(EagerError,
      "emitUnary: unsupported op '" & op & "'")

proc emitBody(b: var ShBuilder; spec: EagerOpSpec;
    operands: openArray[Tensor]; attrs: openArray[(string, string)];
    argIds: openArray[ShValueId]): seq[ShValueId] =
  case spec.kind
  of eokBinary:
    @[emitElementwise(b, spec.name, argIds[0], argIds[1])]
  of eokUnary:
    @[emitUnary(b, spec.name, argIds[0])]
  of eokComplex:
    @[shops.complexOp(b, argIds[0], argIds[1])]
  of eokReal:
    @[shops.real(b, argIds[0])]
  of eokImag:
    @[shops.imag(b, argIds[0])]
  of eokConvert:
    @[shops.convert(b, argIds[0], parseDType(findAttr(attrs, "dtype")))]
  of eokBitcastConvert:
    @[shops.bitcastConvert(b, argIds[0],
      parseDType(findAttr(attrs, "dtype")),
      parseIntList(findAttr(attrs, "output_shape")))]
  of eokIsFinite:
    @[shops.isFinite(b, argIds[0])]
  of eokReducePrecision:
    @[shops.reducePrecision(b, argIds[0],
      parseIntAttr(findAttr(attrs, "exponent_bits")),
      parseIntAttr(findAttr(attrs, "mantissa_bits")))]
  of eokReshape:
    let outShape = parseIntList(findAttr(attrs, "shape"))
    @[shops.reshape(b, argIds[0], outShape)]
  of eokTranspose:
    let perm = parseIntList(findAttr(attrs, "permutation"))
    @[shops.transpose(b, argIds[0], perm)]
  of eokReverse:
    let dims = parseIntList(findAttr(attrs, "dimensions"))
    @[shops.reverse(b, argIds[0], dims)]
  of eokReduce:
    let dims = parseIntList(findAttr(attrs, "dims"))
    let initBytes = reductionInit(spec.name, operands[0].dtype)
    let initId = b.constant(operands[0].dtype, [], initBytes)
    let opName = spec.name
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      reduceOpBody(b, opName, lhs, rhs)
    @[b.reduce(argIds[0], initId, dims, body)]
  of eokDot:
    @[shops.dot(b, argIds[0], argIds[1])]
  of eokDotGeneral:
    @[shops.dotGeneral(b, argIds[0], argIds[1],
      parseIntList(findAttr(attrs, "lhs_batching")),
      parseIntList(findAttr(attrs, "rhs_batching")),
      parseIntList(findAttr(attrs, "lhs_contracting")),
      parseIntList(findAttr(attrs, "rhs_contracting")))]
  of eokBroadcastTo:
    @[shops.broadcastInDim(b, argIds[0],
      parseIntList(findAttr(attrs, "output_shape")),
      parseIntList(findAttr(attrs, "broadcast_dimensions")))]
  of eokConv2d:
    let strides = parseIntList(findAttr(attrs, "strides"))
    let padPairs = parsePadding(findAttr(attrs, "padding"))
    let dilation = parseIntList(findAttr(attrs, "dilation"))
    var stridesArr: array[2, int] = [strides[0], strides[1]]
    var dilArr: array[2, int] = [dilation[0], dilation[1]]
    var padArr: array[2, array[2, int]] = [
      [padPairs[0][0], padPairs[0][1]],
      [padPairs[1][0], padPairs[1][1]],
    ]
    let dims = nhwcOIHWConvDims(2)
    @[shops.convolution(b, argIds[0], argIds[1],
      stridesArr, padArr, [1, 1], dilArr, dims, 1, 1)]
  of eokMaxPool2d:
    let kernel = parseIntList(findAttr(attrs, "kernel_size"))
    let strides = parseIntList(findAttr(attrs, "strides"))
    let padPairs = parsePadding(findAttr(attrs, "padding"))
    let windowDims = [1, kernel[0], kernel[1], 1]
    let windowStrides = [1, strides[0], strides[1], 1]
    let pad: array[4, array[2, int]] =
      [[0, 0],
       [padPairs[0][0], padPairs[0][1]],
       [padPairs[1][0], padPairs[1][1]],
       [0, 0]]
    let dilations = [1, 1, 1, 1]
    let initBytes = float32Bytes(NegInf.float32)
    let initId = b.constant(operands[0].dtype, [], initBytes)
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      shops.maximum(b, lhs, rhs)
    @[b.reduceWindow(argIds[0], initId, windowDims, windowStrides,
      pad, dilations, dilations, body)]
  of eokBatchNormInference:
    @[shops.batchNormInference(b, argIds[0], argIds[1], argIds[2],
      argIds[3], argIds[4], parseFloat(findAttr(attrs, "epsilon")).float32,
      parseIntAttr(findAttr(attrs, "feature_index")))]
  of eokBatchNormTraining:
    shops.batchNormTraining(b, argIds[0], argIds[1], argIds[2],
      parseFloat(findAttr(attrs, "epsilon")).float32,
      parseIntAttr(findAttr(attrs, "feature_index")))
  of eokBatchNormGrad:
    shops.batchNormGrad(b, argIds[0], argIds[1], argIds[2],
      argIds[3], argIds[4], parseFloat(findAttr(attrs, "epsilon")).float32,
      parseIntAttr(findAttr(attrs, "feature_index")))
  of eokCholesky:
    @[shops.cholesky(b, argIds[0], findAttr(attrs, "lower") == "true")]
  of eokGetDimensionSize:
    @[shops.getDimensionSize(b, argIds[0],
      parseIntAttr(findAttr(attrs, "dimension")))]
  of eokPad:
    @[shops.pad(b, argIds[0], argIds[1],
      parseIntList(findAttr(attrs, "edge_padding_low")),
      parseIntList(findAttr(attrs, "edge_padding_high")),
      parseIntList(findAttr(attrs, "interior_padding")))]
  of eokBroadcast:
    @[shops.broadcast(b, argIds[0],
      parseIntList(findAttr(attrs, "broadcast_sizes")))]
  of eokDynamicSlice:
    @[shops.dynamicSlice(b, argIds[0], argIds.toOpenArray(1, argIds.high),
      parseIntList(findAttr(attrs, "slice_sizes")))]
  of eokDynamicUpdateSlice:
    @[shops.dynamicUpdateSlice(b, argIds[0], argIds[1],
      argIds.toOpenArray(2, argIds.high))]
  of eokIota:
    @[shops.iota(b, parseDType(findAttr(attrs, "dtype")),
      parseIntList(findAttr(attrs, "shape")),
      parseIntAttr(findAttr(attrs, "dimension")))]
  of eokReplicaId:
    @[shops.replicaId(b)]
  of eokPartitionId:
    @[shops.partitionId(b)]
  of eokSetDimensionSize:
    @[shops.setDimensionSize(b, argIds[0], argIds[1],
      parseIntAttr(findAttr(attrs, "dimension")))]
  of eokDynamicReshape:
    @[shops.dynamicReshape(b, argIds[0], argIds[1],
      parseIntList(findAttr(attrs, "result_shape")))]
  of eokDynamicPad:
    @[shops.dynamicPad(b, argIds[0], argIds[1], argIds[2],
      argIds[3], argIds[4], parseIntList(findAttr(attrs, "result_shape")))]
  of eokDynamicIota:
    @[shops.dynamicIota(b, parseDType(findAttr(attrs, "dtype")),
      argIds[0], parseIntList(findAttr(attrs, "result_shape")),
      parseIntAttr(findAttr(attrs, "dimension")))]
  of eokRealDynamicSlice:
    @[shops.realDynamicSlice(b, argIds[0], argIds[1], argIds[2],
      argIds[3], parseIntList(findAttr(attrs, "result_shape")))]
  of eokDynamicBroadcastInDim:
    @[shops.dynamicBroadcastInDim(b, argIds[0], argIds[1],
      parseIntList(findAttr(attrs, "result_shape")),
      parseIntList(findAttr(attrs, "broadcast_dimensions")),
      parseIntList(findAttr(attrs, "known_expanding_dimensions")),
      parseIntList(findAttr(attrs, "known_nonexpanding_dimensions")))]
  of eokFft:
    @[shops.fft(b, argIds[0], parseFftType(findAttr(attrs, "fft_type")),
      parseIntList(findAttr(attrs, "fft_length")))]
  of eokTriangularSolve:
    @[shops.triangularSolve(b, argIds[0], argIds[1],
      findAttr(attrs, "left_side") == "true",
      findAttr(attrs, "lower") == "true",
      findAttr(attrs, "unit_diagonal") == "true",
      parseTransposeKind(findAttr(attrs, "transpose_a")))]
  of eokEinsum:
    @[shops.einsum(b, argIds[0], argIds[1],
      findAttr(attrs, "einsum_config"),
      parseIntList(findAttr(attrs, "output_shape")))]
  of eokUnaryEinsum:
    @[shops.unaryEinsum(b, argIds[0],
      findAttr(attrs, "einsum_config"),
      parseIntList(findAttr(attrs, "output_shape")))]
  of eokTorchIndexSelect:
    @[shops.torchIndexSelect(b, argIds[0], argIds[1],
      parseIntAttr(findAttr(attrs, "dim")),
      parseIntAttr(findAttr(attrs, "batch_dims")))]
  of eokClamp:
    @[shops.clamp(b, argIds[0], argIds[1], argIds[2])]
  of eokConcatenate:
    @[shops.concatenate(b, argIds,
      parseIntAttr(findAttr(attrs, "dimension")))]
  of eokSlice:
    @[shops.slice(b, argIds[0],
      parseIntList(findAttr(attrs, "start_indices")),
      parseIntList(findAttr(attrs, "limit_indices")),
      parseIntList(findAttr(attrs, "strides")))]
  of eokSelect:
    @[shops.select(b, argIds[0], argIds[1], argIds[2])]
  of eokCompare:
    @[shops.compare(b, argIds[0], argIds[1],
      findAttr(attrs, "comparison_direction"))]

# ----- Cache key + compile path -----------------------------------------

proc shapeKey(shape: openArray[int]): string =
  result = "["
  for i, d in shape:
    if i > 0: result.add ','
    result.add $d
  result.add ']'

func stableHexHash(s: string): string =
  ## FNV-1a over the full cache material. `std/hash` is intentionally not
  ## used here because its output is not a cross-process persistence key.
  const
    offset = 14695981039346656037'u64
    prime = 1099511628211'u64
    digits = "0123456789abcdef"
  var h = offset
  for ch in s:
    h = (h xor uint64(ord(ch))) * prime
  result = newString(16)
  for i in countdown(15, 0):
    result[i] = digits[int(h and 0x0f'u64)]
    h = h shr 4

proc clientCacheKey(c: PjrtClient): string =
  ## Best-effort plugin identity for persistent executable cache keys.
  result = "pjrt"
  try:
    result = c.platformName() & "@" & c.platformVersion()
  except CatchableError:
    discard
  except Exception:
    discard

proc executableCachePath(c: PjrtClient; device: Device; fullKey: string;
    moduleText: string; compileOptions: string = ""): string =
  let material =
    "target=" & $device & "\n" &
    "client=" & clientCacheKey(c) & "\n" &
    "key=" & fullKey & "\n" &
    "compile_options=" & compileOptions & "\n" &
    "module=" & moduleText
  executablesDir() / (stableHexHash(material) & ".pjrt")

proc removeStaleExecutable(path: string) =
  try:
    if fileExists(path):
      removeFile(path)
  except OSError:
    discard

proc loadPersistentExecutable(c: PjrtClient; path: string;
    compileOptions: string = ""):
    PjrtLoadedExecutable =
  if not executableCacheEnabled():
    return nil
  try:
    if fileExists(path):
      let bytes = readFile(path)
      if bytes.len > 0:
        return c.loadExecutable(bytes, compileOptions)
      removeStaleExecutable(path)
  except CatchableError:
    removeStaleExecutable(path)
  except Exception:
    removeStaleExecutable(path)
  nil

proc storePersistentExecutable(exe: PjrtLoadedExecutable; path: string) =
  if not executableCacheEnabled() or exe.isNil:
    return
  try:
    let bytes = exe.serialize()
    if bytes.len > 0:
      createDir(parentDir(path))
      writeFile(path, bytes)
  except CatchableError:
    discard
  except Exception:
    discard

proc compileWithPersistentCache(c: PjrtClient; device: Device;
    fullKey: string; moduleText: string; context: string;
    compileOptions: string = ""):
    PjrtLoadedExecutable {.raises: [CatchableError].} =
  let path = executableCachePath(c, device, fullKey, moduleText,
    compileOptions)
  result = loadPersistentExecutable(c, path, compileOptions)
  if not result.isNil:
    return
  result =
    try: c.compile(moduleText, compileOptions = compileOptions)
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError, context & " PJRT compile failed: " & e.msg)
  storePersistentExecutable(result, path)

proc signatureKey(op: string; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): string =
  if operands.len > 0:
    result = $operands[0].device & "|" & op
  else:
    result = findAttr(attrs, "device") & "|" & op
  for t in operands:
    result.add '|'
    result.add $t.dtype
    result.add ':'
    result.add shapeKey(t.shape)
  for (k, v) in attrs:
    result.add "|@"
    result.add k
    result.add '='
    result.add v

proc compileOpModule(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)];
    d: Device; cacheKey: string): PjrtLoadedExecutable
    {.raises: [CatchableError].} =
  var inTypes = newSeq[ShTensorType](operands.len)
  for i, t in operands:
    inTypes[i] = initTensorType(t.dtype, t.shape)
  var b = initBuilder("rew_eager_" & spec.name)
  let argIds = b.beginFunc("main", inTypes, [])
  let outIds =
    try: emitBody(b, spec, operands, attrs, argIds)
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: emit failed for op '" & spec.name & "': " & e.msg)
  var outTypes = newSeq[ShTensorType](outIds.len)
  for i, outId in outIds:
    outTypes[i] = b.getType(outId)
  setCurrentOutputTypes(b, outTypes)
  b.returnOp(outIds)
  b.endFunc()
  let m = b.build()
  verify(m)
  var mlir: string
  try:
    mlir = emitText(m)
  except CatchableError as e:
    raise e
  except Exception as e:
    raise newException(EagerError,
      "eager: StableHLO emitText failed for op '" & spec.name & "': " & e.msg)
  let c = ensureClient(d)
  compileWithPersistentCache(c, d, cacheKey, mlir,
    "eager op '" & spec.name & "'")

proc executableFor(spec: EagerOpSpec; operands: openArray[Tensor];
    attrs: openArray[(string, string)];
    d: Device): PjrtLoadedExecutable {.raises: [CatchableError].} =
  let key = signatureKey(spec.name, operands, attrs)
  if eagerExecCache.hasKey(key):
    return eagerExecCache[key]
  let exe = compileOpModule(spec, operands, attrs, d, key)
  eagerExecCache[key] = exe
  exe

proc allNonDonatable(count: int): seq[int64] =
  ## PJRT's execute options accept the complement of donated inputs.
  result = newSeq[int64](count)
  for i in 0 ..< count:
    result[i] = int64(i)

proc nonDonatableInputs(count: int; donateIdx: openArray[int]): seq[int64] =
  var donated = newSeq[bool](count)
  for idx in donateIdx:
    if idx >= 0 and idx < count:
      donated[idx] = true
  for i in 0 ..< count:
    if not donated[i]:
      result.add int64(i)

proc localShardIndices(inputs: openArray[Tensor]; deviceCount: int): seq[int] =
  result = newSeq[int](deviceCount)
  for i in 0 ..< deviceCount:
    result[i] = i
  for t in inputs:
    if t.buffer.raws.len == deviceCount:
      let indices = t.buffer.shardIndices()
      if indices.len == deviceCount:
        return indices

func shardingNeedsBufferSet(sharding: Sharding): bool =
  if sharding.isReplicated:
    false
  else:
    sharding.activeMesh().meshSize > 1

proc clearEagerCache*() =
  ## Drops all cached compiled executables.
  eagerExecCache.clear()

proc executeEagerOp(op: string; operands: openArray[Tensor];
    attrs: openArray[(string, string)]): seq[Tensor]
    {.nimcall, raises: [CatchableError].} =
  let spec = findOpSpec(op)
  let d =
    if operands.len > 0:
      operands[0].device
    else:
      parseDeviceAttr(attrs)
  for i in 1 ..< operands.len:
    if operands[i].device != d:
      raise newException(EagerError,
        "eager '" & op & "': cross-device operands forbidden")
  for t in operands:
    requireEager(t, "eager '" & op & "'")
  let outShapes = deriveOutputShapes(spec, operands, attrs)
  let outDtypes = deriveOutputDTypes(spec, operands, attrs)
  let exe = executableFor(spec, operands, attrs, d)
  let dev = resolveDevice(d)
  let api = lookupApi(d.target)

  var pbInputs = newSeq[PjrtBuffer](operands.len)
  for i, t in operands:
    pbInputs[i] = newPjrtBuffer(api, t.buffer.raw)
  defer:
    for pb in pbInputs:
      pb.raw = nil.PjrtBufferRaw

  let outBuffers =
    try:
      exe.execute(dev, pbInputs,
        PjrtRunOptions(
          nonDonatableInputIndices: allNonDonatable(pbInputs.len),
          callLocation: "rew:eager:" & op))
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "eager: PJRT execute failed for op '" & op & "': " & e.msg)
  if outBuffers.len != outShapes.len:
    raise newException(EagerError,
      "eager '" & op & "': PJRT returned " & $outBuffers.len &
        " output(s); op expected " & $outShapes.len)
  result = newSeq[Tensor](outBuffers.len)
  for i, pb in outBuffers:
    let raw = pb.raw
    pb.raw = nil.PjrtBufferRaw
    let h = newBufferHandle(d.target, raw, pjrtReleaser)
    let sharding =
      if operands.len > 0: operands[0].sharding else: initReplicated()
    result[i] = initEagerTensor(h, outDtypes[i], outShapes[i], d,
                                sharding)

proc installEagerBackend*() =
  ## Registers the PJRT-backed eager dispatch backend with the dispatcher.
  ## Idempotent.
  setEagerBackend(executeEagerOp)

# ----- Jit module execution ---------------------------------------------

var jitExecCache: Table[string, PjrtLoadedExecutable]

proc clearJitCache*() =
  ## Drops all cached jit-compiled executables.
  jitExecCache.clear()

proc executeJit*(jitName: string; cacheKey: string; moduleText: string;
    inputs: openArray[Tensor]; outDtypes: openArray[DType];
    outShapes: openArray[seq[int]];
    donateIdx: openArray[int];
    outShardings: openArray[Sharding] = []): seq[Tensor]
    {.raises: [CatchableError].} =
  doAssert inputs.len > 0, "executeJit: zero inputs"
  doAssert outDtypes.len == outShapes.len,
    "executeJit: outDtypes / outShapes length mismatch"
  doAssert outShardings.len == 0 or outShardings.len == outDtypes.len,
    "executeJit: outShardings length mismatch"
  let d = inputs[0].device
  for i in 1 ..< inputs.len:
    if inputs[i].device != d:
      raise newException(EagerError,
        "jit '" & jitName & "': cross-device inputs forbidden")
  for t in inputs:
    requireEager(t, "jit '" & jitName & "'")
  for idx in donateIdx:
    if idx < 0 or idx >= inputs.len:
      raise newException(EagerError,
        "jit '" & jitName & "': donation index " & $idx & " out of range")
  let fullKey = $d & "::" & jitName & "::" & cacheKey & "::" &
    stableHexHash(moduleText)
  var exe: PjrtLoadedExecutable
  if jitExecCache.hasKey(fullKey):
    exe = jitExecCache[fullKey]
  else:
    let c = ensureClient(d)
    exe = compileWithPersistentCache(c, d, fullKey, moduleText,
      "jit '" & jitName & "'")
    jitExecCache[fullKey] = exe
  let dev = resolveDevice(d)
  let api = lookupApi(d.target)
  var pbInputs = newSeq[PjrtBuffer](inputs.len)
  for i, t in inputs:
    pbInputs[i] = newPjrtBuffer(api, t.buffer.raw)
  defer:
    for pb in pbInputs:
      pb.raw = nil.PjrtBufferRaw
  let outBuffers =
    try:
      exe.execute(dev, pbInputs,
        PjrtRunOptions(
          nonDonatableInputIndices:
            nonDonatableInputs(inputs.len, donateIdx),
          callLocation: "rew:jit:" & jitName))
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "jit '" & jitName & "' PJRT execute failed: " & e.msg)
  if outBuffers.len != outDtypes.len:
    raise newException(EagerError,
      "jit '" & jitName & "': PJRT returned " & $outBuffers.len &
        " output(s); trace expected " & $outDtypes.len)
  for idx in donateIdx:
    inputs[idx].buffer.markDonated("jit '" & jitName & "'")
  result = newSeq[Tensor](outBuffers.len)
  for i, pb in outBuffers:
    let raw = pb.raw
    pb.raw = nil.PjrtBufferRaw
    let h = newBufferHandle(d.target, raw, pjrtReleaser)
    let sharding =
      if outShardings.len > 0: outShardings[i] else: inputs[0].sharding
    result[i] = initEagerTensor(h, outDtypes[i], outShapes[i], d,
                                sharding)

proc executeDistributedJit*(jitName: string; cacheKey: string;
    moduleText: string; inputs: openArray[Tensor];
    outDtypes: openArray[DType]; outShapes: openArray[seq[int]];
    donateIdx: openArray[int]; outShardings: openArray[Sharding];
    devices: openArray[Device]; compileOptions: string = "";
    launchId: int = 0; taskIds: openArray[int] = [];
    incarnationIds: openArray[int64] = []): seq[Tensor]
    {.raises: [CatchableError].} =
  ## Distributed `jit` bridge used by the high-level distributed module.
  ## Single-device execution is identical to `executeJit`; multi-device
  ## execution consumes/produces buffer sets ordered by executable addressable
  ## device order.
  doAssert inputs.len > 0, "executeDistributedJit: zero inputs"
  doAssert outDtypes.len == outShapes.len,
    "executeDistributedJit: outDtypes / outShapes length mismatch"
  doAssert outShardings.len == 0 or outShardings.len == outDtypes.len,
    "executeDistributedJit: outShardings length mismatch"
  if devices.len == 0:
    raise newException(EagerError,
      "distributed jit '" & jitName & "': no devices in parallel plan")
  let d = devices[0]
  for device in devices:
    if device.target != d.target:
      raise newException(EagerError,
        "distributed jit '" & jitName &
          "': all devices must use the same PJRT target")
  for t in inputs:
    requireEager(t, "distributed jit '" & jitName & "'")
    if t.device.target != d.target:
      raise newException(EagerError,
        "distributed jit '" & jitName &
          "': input target does not match plan target")
  for idx in donateIdx:
    if idx < 0 or idx >= inputs.len:
      raise newException(EagerError,
        "distributed jit '" & jitName & "': donation index " & $idx &
          " out of range")

  let fullKey = $d & "::dist::" & jitName & "::" & cacheKey & "::" &
    stableHexHash(moduleText) & "::" & stableHexHash(compileOptions)
  var exe: PjrtLoadedExecutable
  if jitExecCache.hasKey(fullKey):
    exe = jitExecCache[fullKey]
  else:
    let c = ensureClient(d)
    exe = compileWithPersistentCache(c, d, fullKey, moduleText,
      "distributed jit '" & jitName & "'", compileOptions)
    jitExecCache[fullKey] = exe
  let api = lookupApi(d.target)
  let runOptions = PjrtRunOptions(
    nonDonatableInputIndices: nonDonatableInputs(inputs.len, donateIdx),
    launchId: launchId,
    callLocation: "rew:dist_jit:" & jitName,
    taskIds: @taskIds,
    incarnationIds: @incarnationIds)

  if devices.len == 1:
    let dev = resolveDevice(d)
    var pbInputs = newSeq[PjrtBuffer](inputs.len)
    for i, t in inputs:
      let raw =
        if t.buffer.raws.len > 0: t.buffer.raws[0]
        else: t.buffer.raw
      pbInputs[i] = newPjrtBuffer(api, raw)
    defer:
      for pb in pbInputs:
        pb.raw = nil.PjrtBufferRaw
    let outBuffers =
      try:
        exe.execute(dev, pbInputs, runOptions)
      except CatchableError as e: raise e
      except Exception as e:
        raise newException(EagerError,
          "distributed jit '" & jitName & "' PJRT execute failed: " & e.msg)
    if outBuffers.len != outDtypes.len:
      raise newException(EagerError,
        "distributed jit '" & jitName & "': PJRT returned " &
          $outBuffers.len & " output(s); trace expected " & $outDtypes.len)
    for idx in donateIdx:
      inputs[idx].buffer.markDonated("distributed jit '" & jitName & "'")
    result = newSeq[Tensor](outBuffers.len)
    let shardIndices = localShardIndices(inputs, 1)
    for i, pb in outBuffers:
      let raw = pb.raw
      pb.raw = nil.PjrtBufferRaw
      let sharding =
        if outShardings.len > 0: outShardings[i] else: inputs[0].sharding
      let h =
        if inputs[0].buffer.isBufferSet or shardingNeedsBufferSet(sharding):
          newBufferSetHandle(d.target, [raw], pjrtReleaser,
            shardIndices = shardIndices)
        else:
          newBufferHandle(d.target, raw, pjrtReleaser)
      result[i] = initEagerTensor(h, outDtypes[i], outShapes[i], d,
                                  sharding)
    return

  var pbRows = newSeq[seq[PjrtBuffer]](devices.len)
  for devIdx in 0 ..< devices.len:
    pbRows[devIdx] = newSeq[PjrtBuffer](inputs.len)
  for argIdx, t in inputs:
    if t.buffer.raws.len != devices.len:
      raise newException(EagerError,
        "distributed jit '" & jitName & "': input " & $argIdx &
          " is not a buffer set for " & $devices.len & " device(s)")
    for devIdx, raw in t.buffer.raws:
      pbRows[devIdx][argIdx] = newPjrtBuffer(api, raw)
  defer:
    for row in pbRows.mitems:
      for pb in row.mitems:
        pb.raw = nil.PjrtBufferRaw

  var outRows =
    try:
      exe.execute(pbRows, runOptions)
    except CatchableError as e: raise e
    except Exception as e:
      raise newException(EagerError,
        "distributed jit '" & jitName &
          "' multi-device PJRT execute failed: " & e.msg)
  if outRows.len != devices.len:
    raise newException(EagerError,
      "distributed jit '" & jitName & "': PJRT returned outputs for " &
        $outRows.len & " device(s); expected " & $devices.len)
  for row in outRows:
    if row.len != outDtypes.len:
      raise newException(EagerError,
        "distributed jit '" & jitName & "': PJRT returned " & $row.len &
          " output(s) per device; trace expected " & $outDtypes.len)
  for idx in donateIdx:
    inputs[idx].buffer.markDonated("distributed jit '" & jitName & "'")

  result = newSeq[Tensor](outDtypes.len)
  let shardIndices = localShardIndices(inputs, devices.len)
  for outIdx in 0 ..< outDtypes.len:
    var raws = newSeq[PjrtBufferRaw](devices.len)
    for devIdx in 0 ..< devices.len:
      raws[devIdx] = outRows[devIdx][outIdx].raw
      outRows[devIdx][outIdx].raw = nil.PjrtBufferRaw
    let h = newBufferSetHandle(d.target, raws, pjrtReleaser,
      shardIndices = shardIndices)
    let sharding =
      if outShardings.len > 0: outShardings[outIdx] else: inputs[0].sharding
    result[outIdx] = initEagerTensor(h, outDtypes[outIdx],
      outShapes[outIdx], d, sharding)
