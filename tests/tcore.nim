## Core smoke, dtype/device metadata, sharding, buffer lifecycle, and eager
## host-transfer checks. Kept together because each case imports the same
## broad runtime surface and the assertions are independent.

import std/[os, strutils]
import rew
import rew/xla
import rew/[buffer, device, dtype, eager]
import rew/binaries/target
import rew/pjrt/[capi, loader]

type EnvSnapshot = tuple[exists: bool, value: string]

var releaseCount {.threadvar.}: int

proc saveEnv(key: string): EnvSnapshot =
  (exists: existsEnv(key), value: getEnv(key))

proc restoreEnv(key: string; snapshot: EnvSnapshot) =
  if snapshot.exists:
    putEnv(key, snapshot.value)
  else:
    delEnv(key)

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "tcore: eager round-trip skipped - ", e.msg
    false

proc fakeReleaser(t: Target; raw: PjrtBufferRaw) {.nimcall, raises: [].} =
  inc releaseCount

template withCount(body: untyped): int =
  releaseCount = 0
  body
  releaseCount

block version_present:
  doAssert RewVersion.len > 0

block dtype_sizes:
  doAssert dtBool.byteSize == 1
  doAssert dtInt8.byteSize == 1
  doAssert dtInt32.byteSize == 4
  doAssert dtFloat32.byteSize == 4
  doAssert dtFloat64.byteSize == 8
  doAssert dtComplex64.byteSize == 8
  doAssert dtComplex128.byteSize == 16
  doAssert dtBFloat16.byteSize == 2
  doAssert dtFloat16.byteSize == 2

block dtype_bit_widths:
  doAssert dtBool.bitWidth == 1
  doAssert dtInt8.bitWidth == 8
  doAssert dtInt32.bitWidth == 32
  doAssert dtFloat16.bitWidth == 16
  doAssert dtFloat32.bitWidth == 32
  doAssert dtFloat64.bitWidth == 64
  doAssert dtComplex64.bitWidth == 64
  doAssert dtComplex128.bitWidth == 128

block dtype_names:
  doAssert dtFloat32.name == "float32"
  doAssert dtBFloat16.name == "bfloat16"
  doAssert dtBool.name == "bool"
  doAssert dtComplex64.name == "complex64"
  doAssert dtComplex128.name == "complex128"

block dtype_classification:
  doAssert dtFloat32.isFloat
  doAssert not dtFloat32.isSignedInt
  doAssert dtInt32.isSignedInt
  doAssert not dtInt32.isFloat
  doAssert dtUint32.isUnsignedInt
  doAssert not dtUint32.isSignedInt
  doAssert not dtBool.isFloat
  doAssert not dtBool.isSignedInt
  doAssert dtComplex64.isComplex
  doAssert not dtComplex64.isFloat
  doAssert dtComplex64.complexPartDType == dtFloat32
  doAssert dtFloat64.complexDType == dtComplex128

block dtype_of_native:
  doAssert dtypeOf(float32) == dtFloat32
  doAssert dtypeOf(float64) == dtFloat64
  doAssert dtypeOf(int32) == dtInt32
  doAssert dtypeOf(uint8) == dtUint8
  doAssert dtypeOf(bool) == dtBool

block device_construct_eq:
  let a = cpu(0)
  let b = cpu(0)
  let c = cuda12(0)
  doAssert a == b
  doAssert a != c
  doAssert $a == "cpu:0"

block device_parse_ok:
  doAssert parseDevice("cpu") == cpu(0)
  doAssert parseDevice("cuda12:1") == cuda12(1)
  doAssert parseDevice("metal:0") == metal(0)
  doAssert parseDevice("  tpu:3 ") == tpu(3)

block device_parse_errors:
  for bad in ["", "  ", "cpu:abc", "a:b:c"]:
    var raised = false
    try:
      discard parseDevice(bad)
    except DeviceError:
      raised = true
    doAssert raised, "expected DeviceError for " & bad

block device_negative_ordinals_raise:
  doAssertRaises(DeviceError):
    discard cpu(-1)
  doAssertRaises(DeviceError):
    discard initDevice(tCpu, -2)
  doAssertRaises(DeviceError):
    discard parseDevice("cpu:-1")

block require_same_device_ok:
  let d = cpu(0)
  requireSameDevice(d, d, "add")

block require_same_device_raises:
  var raised = false
  try:
    requireSameDevice(cpu(0), cuda12(0), "matmul")
  except DeviceError:
    raised = true
  doAssert raised

block default_target_env_override:
  let oldTarget = saveEnv("REW_TARGET")
  let oldDevice = defaultDevice()
  try:
    putEnv("REW_TARGET", "cpu")
    resetDefaultTarget()
    setDefaultDevice(cpu(7))
    doAssert defaultDevice() == cpu(7)
  finally:
    restoreEnv("REW_TARGET", oldTarget)
    setDefaultDevice(oldDevice)
    resetDefaultTarget()

block sharding_default:
  let s = initReplicated()
  doAssert s.kind == skReplicated
  doAssert s.isReplicated

block mesh_and_partition_spec:
  let devices = [cpu(0), cpu(1), cpu(2), cpu(3)]
  let mesh = initMesh("data_model", ["data", "model"], [2, 2], devices,
                      [0, 0, 1, 1])
  doAssert mesh.meshSize == 4
  doAssert mesh.containsAxis("data")
  doAssert not mesh.containsAxis("missing")
  doAssert $mesh == "data_model[data=2, model=2]"

  let spec = initPartitionSpec(["data", ""])
  validatePartitionSpec(mesh, spec, 2)
  doAssert $spec == "(data, *)"

  let grouped = initPartitionSpecGroups([@["data", "model"], @[]])
  validatePartitionSpec(mesh, grouped, 2)
  doAssert $grouped == "(data+model, *)"

block malformed_meshes_and_specs_raise:
  doAssertRaises(ValueError):
    discard initMesh("bad", ["x", "x"], [2, 2])
  doAssertRaises(ValueError):
    discard initMesh("bad", ["x", "y"], [2, 2], [cpu(0), cpu(1)])

  let mesh = initMesh("m", ["x"], [2])
  doAssertRaises(ValueError):
    validatePartitionSpec(mesh, initPartitionSpec(["y"]), 1)
  doAssertRaises(ValueError):
    validatePartitionSpec(mesh, initPartitionSpec(["x", "x"]), 2)

block tensor_constructor_edges_raise:
  doAssertRaises(TensorModeError):
    discard initTraceTensor(ShValueId(0), dtFloat32, [2], cpu(0))
  doAssertRaises(TensorError):
    discard initTraceTensor(ShValueId(41), dtFloat32, [-1], cpu(0))
  doAssertRaises(TensorError):
    discard initEagerTensor(nil.BufferHandle, dtFloat32, [1], cpu(0))

block tensor_annotation_is_metadata_only:
  let mesh = initMesh("m", ["x"], [2])
  let spec = initPartitionSpec(["x"])
  let t = initTraceTensor(ShValueId(1), dtFloat32, [8], cpu(0))
  let sharded = t.shard(mesh, spec)
  doAssert sharded.sharding.isPartitioned
  doAssert sharded.traceId == t.traceId
  doAssert shardingKey(sharded.sharding) == "partitioned:m[x=2](x)"
  doAssert sharded.replicate().sharding.isReplicated

block sharding_key_includes_device_assignment:
  let spec = initPartitionSpec(["x"])
  let a = initPartitioned(initMesh("m", ["x"], [2], [cpu(0), cpu(1)]), spec)
  let b = initPartitioned(initMesh("m", ["x"], [2], [cpu(2), cpu(3)]), spec)
  doAssert shardingKey(a) != shardingKey(b)
  doAssert "@devices=cpu:0,cpu:1" in shardingKey(a)

block manual_sharding:
  let mesh = initMesh("manual_mesh", ["x"], [2])
  let spec = initPartitionSpec(["x"])
  let t = initTraceTensor(ShValueId(2), dtFloat32, [8], cpu(0))
  let manual = t.manualShard(mesh, spec)
  doAssert manual.sharding.isManual
  doAssert $manual.sharding == "manual(manual_mesh[x=2], (x))"

block shape_composite_edge_errors:
  let t = initTraceTensor(ShValueId(3), dtFloat32, [2, 3], cpu(0))
  doAssertRaises(TensorError):
    discard split(t, [1, -1, 3], 1)
  doAssertRaises(TensorError):
    discard chunk(t, 4, 0)
  doAssertRaises(TensorError):
    discard unbind(t, 2)
  doAssertRaises(TensorError):
    discard roll(t, [1], [3])
  doAssertRaises(TensorError):
    discard rot90(t, dims = [0, 0])

block shardy_rendering:
  let mesh = initMesh("mesh", ["x", "y"], [2, 2],
                      [cpu(0), cpu(1), cpu(2), cpu(3)])
  let spec = initPartitionSpecGroups([@["x"], @["y"]])
  let sharding = initPartitioned(mesh, spec)
  doAssert shardyMeshOp(mesh) ==
    "sdy.mesh @mesh = <[\"x\"=2, \"y\"=2], device_ids=[0, 1, 2, 3]>"
  doAssert shardyTensorSharding(sharding, 2) ==
    "<@mesh, [{\"x\"}, {\"y\"}]>"
  doAssert shardyPerValueAttr([sharding, initReplicated()], [2, 0]) ==
    "#sdy.sharding_per_value<[<@mesh, [{\"x\"}, {\"y\"}]>, <>]>"

block missing_backend_raises:
  var raised = false
  try:
    discard loadPluginByPath("/this/path/does/not/exist.so")
  except PjrtError as e:
    raised = true
    doAssert "Could not dlopen" in e.msg
  doAssert raised

block executable_cache_env_control:
  let old = saveEnv(ExecutableCacheEnvVar)
  try:
    putEnv(ExecutableCacheEnvVar, "0")
    doAssert not executableCacheEnabled()
    putEnv(ExecutableCacheEnvVar, "false")
    doAssert not executableCacheEnabled()
    putEnv(ExecutableCacheEnvVar, "1")
    doAssert executableCacheEnabled()
  finally:
    restoreEnv(ExecutableCacheEnvVar, old)

block executable_cache_dir_uses_rew_cache:
  let old = saveEnv("REW_CACHE_DIR")
  try:
    putEnv("REW_CACHE_DIR", "/tmp/rew_eager_cache_test")
    doAssert executableCacheDir().endsWith("executables")
    doAssert "/tmp/rew_eager_cache_test" in executableCacheDir()
  finally:
    restoreEnv("REW_CACHE_DIR", old)

block release_on_last_drop:
  let n = withCount:
    block:
      let h = newBufferHandle(tCpu,
        cast[PjrtBufferRaw](cast[pointer](0xdeadbeef)),
        fakeReleaser, sizeBytes = 64)
      doAssert h.isLive
      doAssert not h.isDonated
      doAssert h.sizeBytes == 64
  doAssert n == 1, "expected exactly one release, got " & $n

block alias_releases_once:
  let n = withCount:
    block:
      let h = newBufferHandle(tCpu,
        cast[PjrtBufferRaw](cast[pointer](0x1)),
        fakeReleaser)
      let alias = h
      doAssert alias.isLive
      doAssert h.isLive
  doAssert n == 1, "alias should not double-release; got " & $n

block donated_releases_on_last_drop:
  let n = withCount:
    block:
      let h = newBufferHandle(tCpu,
        cast[PjrtBufferRaw](cast[pointer](0x2)),
        fakeReleaser)
      markDonated(h, "jit:trainStep")
      doAssert h.isDonated
      doAssert not h.isLive
  doAssert n == 1, "donated buffer should release its PJRT wrapper"

block buffer_set_releases_each_raw:
  let n = withCount:
    block:
      let h = newBufferSetHandle(tCpu, [
        cast[PjrtBufferRaw](cast[pointer](0x21)),
        cast[PjrtBufferRaw](cast[pointer](0x22)),
        cast[PjrtBufferRaw](cast[pointer](0x23)),
      ], fakeReleaser)
      doAssert h.isLive
      doAssert h.isBufferSet
      doAssert h.bufferCount == 3
      doAssert h.shardIndices == @[0, 1, 2]
  doAssert n == 3, "buffer set should release every raw buffer; got " & $n

block buffer_set_preserves_global_shard_indices:
  let n = withCount:
    block:
      let h = newBufferSetHandle(tCpu, [
        cast[PjrtBufferRaw](cast[pointer](0x31)),
        cast[PjrtBufferRaw](cast[pointer](0x32)),
        cast[PjrtBufferRaw](cast[pointer](0x33)),
      ], fakeReleaser, shardIndices = [3, 5, 7])
      doAssert h.isBufferSet
      doAssert h.shardIndices == @[3, 5, 7]
  doAssert n == 3, "indexed buffer set should release every raw buffer"

block use_after_donate_raises:
  let h = newBufferHandle(tCpu,
    cast[PjrtBufferRaw](cast[pointer](0x3)),
    fakeReleaser)
  markDonated(h, "jit:trainStep")
  var raised = false
  try:
    requireLive(h, "matmul")
  except BufferDonatedError as e:
    raised = true
    doAssert "matmul" in e.msg
    doAssert "jit:trainStep" in e.msg
  doAssert raised

block nil_buffer_guards_are_safe:
  let h = nil.BufferHandle
  doAssert not h.isLive
  doAssert not h.isDonated
  doAssert h.bufferCount == 0
  doAssertRaises(ValueError):
    requireLive(h, "nilGuard")
  doAssertRaises(ValueError):
    markDonated(h, "nilGuard")

block live_buffer_passes_guard:
  let h = newBufferHandle(tCpu,
    cast[PjrtBufferRaw](cast[pointer](0x4)),
    fakeReleaser)
  requireLive(h, "add")

block transfer_round_trip_or_skip:
  if canLoadCpu():
    let d = cpu(0)
    setDefaultDevice(d)
    doAssert defaultDevice() == d

    var src: array[5, float32] =
      [0.5'f32, 1.5'f32, 2.5'f32, 3.5'f32, 4.5'f32]
    let dims: array[1, int64] = [5'i64]
    let h = transferToDevice(d, addr src[0], dtFloat32, dims,
                             sizeBytes = sizeof(src))
    doAssert h.isLive
    doAssert h.sizeBytes == sizeof(src)

    var dst: array[5, float32]
    transferToHost(d, h, addr dst[0], sizeof(dst))
    for i in 0 ..< src.len:
      doAssert dst[i] == src[i]

block public_host_transfer_round_trip_or_skip:
  if canLoadCpu():
    let d = cpu(0)
    setDefaultDevice(d)

    let xf = fromHost(d, [1.25'f32, 2.5'f32, 3.75'f32], [3])
    doAssert xf.dtype == dtFloat32
    doAssert xf.shape == @[3]
    doAssert xf.toHost(float32) == @[1.25'f32, 2.5'f32, 3.75'f32]

    let xi = fromHost(d, [1'i32, 2'i32, 3'i32, 4'i32], [2, 2])
    doAssert xi.dtype == dtInt32
    doAssert xi.toHost(int32) == @[1'i32, 2'i32, 3'i32, 4'i32]

    let xb = fromHost(d, [true, false, true], [3])
    doAssert xb.dtype == dtBool
    doAssert xb.toHost(bool) == @[true, false, true]

    let xc = constantF32([2, 2], [1'f32, 2'f32, 3'f32, 4'f32], d)
    doAssert xc.dtype == dtFloat32
    doAssert xc.shape == @[2, 2]
    doAssert xc.toHost(float32) == @[1'f32, 2'f32, 3'f32, 4'f32]

    let xiLit = scalarI32(17'i32, d)
    doAssert xiLit.shape.len == 0
    doAssert xiLit.item(int32) == 17'i32

    let xbLit = scalarBool(true, d)
    doAssert xbLit.shape.len == 0
    doAssert xbLit.item(bool)

    let s = scalar(d, 42'i32)
    doAssert s.shape.len == 0
    doAssert s.item(int32) == 42'i32
    doAssert s.to(d).device == d

block public_host_transfer_errors:
  var shapeRaised = false
  try:
    discard fromHost(cpu(0), [1.0'f32, 2.0'f32], [3])
  except EagerError:
    shapeRaised = true
  doAssert shapeRaised

  var literalRaised = false
  try:
    discard constant(dtFloat32, [2], f32Bytes([1'f32]))
  except TensorError:
    literalRaised = true
  doAssert literalRaised

  if canLoadCpu():
    let d = cpu(0)
    let x = fromHost(d, [1.0'f32, 2.0'f32], [2])

    var dtypeRaised = false
    try:
      discard x.toHost(int32)
    except EagerError:
      dtypeRaised = true
    doAssert dtypeRaised

    var itemRaised = false
    try:
      discard x.item(float32)
    except TensorError:
      itemRaised = true
    doAssert itemRaised

block like_factories_trace_metadata:
  withTrace ctx, "like_factories", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let z = zerosLike(inputs[0])
    let o = onesLike(inputs[0])
    let f = fullLike(inputs[0], 7'f32)
    doAssert z.shape == inputs[0].shape
    doAssert o.shape == inputs[0].shape
    doAssert f.shape == inputs[0].shape
    doAssert z.dtype == inputs[0].dtype
    doAssert o.device == inputs[0].device
    ctx.traceReturn([z, o, f])

block interpolate_trace_metadata:
  withTrace ctx, "interpolate_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[1, 2, 2, 1], @[1, 2, 2, 1]])
    let linear = interpolate(inputs[0], [3, 3], ipBilinear)
    let cubic = interpolate(inputs[0], [3, 3], ipBicubic)
    let linear1d = interpolate(reshape(inputs[0], [4]), [6], ipBilinear)
    doAssert linear.shape == @[1, 3, 3, 1]
    doAssert cubic.shape == @[1, 3, 3, 1]
    doAssert linear1d.shape == @[6]

    var dtypeRaised = false
    try:
      discard interpolate(inputs[1], [3, 3], ipBilinear)
    except TensorError:
      dtypeRaised = true
    doAssert dtypeRaised
    ctx.traceReturn([linear, cubic, linear1d])

block fold_trace_metadata:
  withTrace ctx, "fold_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 1, 3, 3]])
    let patches = unfold(inputs[0], [2, 2], stride = [1, 1],
      padding = [1, 1], dilation = [1, 1])
    doAssert patches.shape == @[1, 4, 16]
    let folded = fold(patches, [3, 3], [2, 2], stride = [1, 1],
      padding = [1, 1], dilation = [1, 1])
    doAssert folded.shape == @[1, 1, 3, 3]
    ctx.traceReturn([patches, folded])

block grid_sample_trace_metadata:
  withTrace ctx, "grid_sample_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtInt32],
      @[@[1, 1, 2, 2], @[1, 2, 2, 2], @[1, 2, 2, 2]])
    let linear = gridSample(inputs[0], inputs[1], gsBilinear,
      alignCorners = true)
    let nearest = gridSample(inputs[0], inputs[1], gsNearest,
      alignCorners = true)
    doAssert linear.shape == @[1, 1, 2, 2]
    doAssert nearest.shape == @[1, 1, 2, 2]

    var dtypeRaised = false
    try:
      discard gridSample(inputs[0], inputs[2], gsBilinear)
    except TensorError:
      dtypeRaised = true
    doAssert dtypeRaised
    ctx.traceReturn([linear, nearest])

block max_pool_indices_trace_metadata:
  withTrace ctx, "max_pool_indices_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 3, 3, 1]])
    let pooled = maxPool2dWithIndices(inputs[0], [2, 2], [1, 1])
    doAssert pooled.values.shape == @[1, 2, 2, 1]
    doAssert pooled.indices.shape == @[1, 2, 2, 1]
    doAssert pooled.indices.dtype == dtInt32
    let unpooled = maxUnpool2d(pooled.values, pooled.indices, [3, 3])
    doAssert unpooled.shape == @[1, 3, 3, 1]
    ctx.traceReturn([pooled.values, pooled.indices, unpooled])

block ctc_loss_trace_metadata:
  withTrace ctx, "ctc_loss_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32, dtInt32, dtInt32],
      @[@[2, 1, 2], @[1, 1], @[1], @[1]])
    let loss = ctcLoss(inputs[0], inputs[1], inputs[2], inputs[3])
    doAssert loss.shape.len == 0
    ctx.traceReturn([loss])

block tensor_product_trace_metadata:
  withTrace ctx, "tensor_product_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2, 1], @[2, 1]])
    let tp = TensorProduct(
      weights: param(constantF32([1, 1], [2.0'f32])),
      cgCoeffs: buffer(constantF32([1, 1, 1, 1], [1.0'f32])),
      inIrreps: @[0],
      outIrreps: @[0],
      sharedIrreps: @[0],
      totalChannels: 1,
      outChannels: 1,
    )
    let y = tp.forward(inputs[0], inputs[1])
    doAssert y.shape == @[2, 1]
    ctx.traceReturn([y])

block qlora_trace_metadata:
  withTrace ctx, "qlora_trace", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2]])
    let layer = initQloraLinearFromF32(initKey(8u64),
      [1.0'f32, 0.0'f32, 0.0'f32, 1.0'f32], 2, 2,
      bias = [0.5'f32, -1.0'f32], rank = 1, alpha = 1.0'f32,
      groupSize = 3)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 2]
    ctx.traceReturn([y])

block donated_buffer_raises_on_transfer:
  proc noopReleaser(t: Target; raw: PjrtBufferRaw)
      {.nimcall, raises: [].} = discard
  let h = newBufferHandle(tCpu, nil.PjrtBufferRaw, noopReleaser)
  h.markDonated("tcore.donated_test")
  var raised = false
  try:
    transferToHost(cpu(0), h, nil, 0)
  except BufferDonatedError:
    raised = true
  doAssert raised

echo "tcore: OK"
