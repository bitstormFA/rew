## Smoke test for the PJRT plugin loader + initial typed wrappers
## (`pluginInitialize`, `pluginAttributes`, `apiVersion`).
##
## A real PJRT plugin (`pjrt_c_api_cpu_plugin.so` / `.dylib` / `.dll`) is
## not bundled with the repo, so when the loader can't find one the test
## skips the plugin-dependent checks rather than failing. This keeps CI
## green on hosts without a vendored plugin while still exercising the
## ABI/wrapper code as soon as one is present (locally or via
## `REW_PJRT_PLUGIN_PATH`).

import std/strutils
import rew/pjrt/[capi, client, loader]
import rew/binaries/target

template optionalPjrt(label: string; body: untyped) =
  try:
    body
  except PjrtError as e:
    echo "tpjrt: optional ", label, " skipped — ", e.msg

proc tryLoadCpu(api: var PjrtApiHandle): bool =
  try:
    api = loadPlugin(tCpu)
    pluginInitialize(api)
    true
  except PjrtError as e:
    echo "tpjrt: skipped — ", e.msg
    false

block plugin_load_or_skip:
  var api: PjrtApiHandle
  var loaded = false
  try:
    api = loadPlugin(tCpu)
    loaded = true
  except PjrtError as e:
    doAssert e.msg.contains("PJRT plugin") or e.msg.contains("Could not")

  if not loaded:
    echo "tpjrt: skipped — no CPU PJRT plugin found on this host"
  else:
    let (major, minor) = apiVersion(api)
    doAssert major >= 0
    doAssert minor >= 0
    echo "tpjrt: plugin api version ", major, ".", minor,
         " (struct_size=", apiStructSize(api), ")"

    pluginInitialize(api)

    let attrs = pluginAttributes(api)
    echo "tpjrt: plugin advertises ", attrs.len, " attribute(s)"
    for (k, v) in attrs:
      echo "  - ", k, " = ", v

block error_destroy_on_nil_is_noop:
  ## `destroyError` and `errorMessage` must tolerate a nil error and a
  ## nil API handle so `checkErr` can be called from teardown paths.
  let api = PjrtApiHandle(nil)
  destroyError(api, PjrtErrorRaw(nil))
  doAssert errorMessage(api, PjrtErrorRaw(nil)) == ""
  doAssert errorCode(api, PjrtErrorRaw(nil)) == pecOk
  doAssert errorPayloads(api, PjrtErrorRaw(nil)).len == 0
  checkErr(api, PjrtErrorRaw(nil), "noop ctx")

block sized_string_and_api_field_guards:
  let raw = "a\0b"
  doAssert copySizedString(cast[cstring](unsafeAddr raw[0]), csize_t raw.len) ==
    raw

  var apiObj = PjrtApi(
    structSize: csize_t(offsetof(PjrtApi, fnDeviceClearMemoryStats) +
      sizeof(pointer)),
    fnDeviceClearMemoryStats: cast[pointer](1))
  let api = cast[PjrtApiHandle](addr apiObj)
  doAssert apiFunctionAvailable(api, fnDeviceClearMemoryStats)
  doAssert not apiFunctionAvailable(api, fnExecutableParameterMemoryKinds)

block client_create_destroy_or_skip:
  ## Creates a `PjrtClient` and inspects its addressable devices. The
  ## destructor releases the client when the ref leaves scope.
  var api: PjrtApiHandle
  if not tryLoadCpu(api): break client_create_destroy_or_skip

  let c = newPjrtClient(api)
  doAssert not c.handle.isNil

  optionalPjrt "client metadata":
    doAssert c.platformName().len > 0
    doAssert c.processIndex() >= 0
    discard c.platformVersion()
    let allDevs = c.devices()
    doAssert allDevs.len > 0

  let devs = c.addressableDevices()
  doAssert devs.len > 0
  echo "tpjrt: client has ", devs.len, " addressable device(s)"
  for i, d in devs:
    let kind = c.deviceKind(d)
    let id = c.deviceId(d)
    echo "  - device[", i, "] kind=", kind, " id=", id
    doAssert kind.len > 0
    doAssert id >= 0
    optionalPjrt "device metadata":
      doAssert c.isAddressable(d)
      doAssert c.deviceProcessIndex(d) >= 0
      doAssert c.deviceToString(d).len > 0
      doAssert c.deviceDebugString(d).len > 0
      discard c.deviceAttributes(d)
      let localId = c.localHardwareId(d)
      doAssert localId >= -1
      let byId = c.lookupDevice(id)
      doAssert cast[pointer](byId) == cast[pointer](d)
      if localId >= 0:
        let byLocal = c.lookupAddressableDevice(localId)
        doAssert cast[pointer](byLocal) == cast[pointer](d)

  optionalPjrt "memory metadata":
    let memories = c.addressableMemories()
    doAssert memories.len > 0
    let mem = c.defaultMemory(devs[0])
    doAssert not mem.isNil
    doAssert c.memoryKind(mem).len > 0
    doAssert c.memoryId(mem) >= 0
    discard c.memoryKindId(mem)
    doAssert c.memoryToString(mem).len > 0
    doAssert c.memoryDebugString(mem).len > 0
    doAssert c.memoryAddressableByDevices(mem).len > 0
    doAssert c.deviceMemories(devs[0]).len > 0
    discard c.memoryStats(devs[0])

  optionalPjrt "topology metadata":
    let topo = c.topology()
    doAssert not topo.raw.isNil
    doAssert topo.platformName().len > 0
    discard topo.platformVersion()
    doAssert topo.deviceDescriptions().len > 0
    discard topo.attributes()
    discard topo.fingerprint()
    let encoded = topo.serialize()
    if encoded.len > 0:
      let decoded = c.deserializeTopology(encoded)
      doAssert decoded.platformName().len > 0

block event_helpers_nil_safe:
  ## `awaitEvent`/`destroyEvent` are no-ops on a nil event so destructors
  ## can call them unconditionally.
  let api = PjrtApiHandle(nil)
  destroyEvent(api, PjrtEventRaw(nil))
  awaitEvent(api, PjrtEventRaw(nil), "noop ctx")

block buffer_destroy_nil_safe:
  ## `bufferDestroy` must be a no-op on nil so the `PjrtBuffer` destructor
  ## can run safely on partially-constructed wrappers.
  let api = PjrtApiHandle(nil)
  bufferDestroy(api, PjrtBufferRaw(nil))

block buffer_round_trip_or_skip:
  ## Round-trips a small float32 array host → device → host through a
  ## live PJRT plugin. Verifies dimensions/element type metadata along the
  ## way. Skips cleanly when no CPU plugin is installed.
  var api: PjrtApiHandle
  if not tryLoadCpu(api): break buffer_round_trip_or_skip

  let c = newPjrtClient(api)
  let devs = c.addressableDevices()
  doAssert devs.len > 0
  let dev = devs[0]

  var src: array[6, float32] = [1.0'f32, -2.0'f32, 3.5'f32, 0.0'f32,
    42.0'f32, -7.25'f32]
  let dims: array[2, int64] = [2'i64, 3'i64]
  let buf = c.transferToDevice(dev, addr src[0], btF32, dims)
  doAssert not buf.raw.isNil

  doAssert buf.elementType() == btF32
  doAssert buf.dimensions() == @[2'i64, 3'i64]
  doAssert buf.onDeviceSizeInBytes() >= int(sizeof(src))

  var dst: array[6, float32]
  buf.transferToHost(addr dst[0], sizeof(dst))
  for i in 0 ..< src.len:
    doAssert dst[i] == src[i]
  echo "tpjrt: buffer round-trip OK (", src.len, " float32 elements)"

block compile_execute_or_skip:
  ## End-to-end smoke: compile a trivial StableHLO `add` graph and execute
  ## it against the CPU plugin. Verifies the compile + execute + readback
  ## pipeline. Skips cleanly when no plugin is installed.
  var api: PjrtApiHandle
  if not tryLoadCpu(api): break compile_execute_or_skip

  let c = newPjrtClient(api)
  let devs = c.addressableDevices()
  doAssert devs.len > 0
  let dev = devs[0]

  const mlirSrc = """
module @add_mod {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
"""
  let exe = c.compile(mlirSrc)
  doAssert not exe.raw.isNil
  doAssert exe.numOutputs() == 1
  optionalPjrt "executable metadata":
    discard exe.name()
    doAssert exe.numReplicas() >= 1
    doAssert exe.numPartitions() >= 1
    discard exe.generatedCodeSizeInBytes()
    let outTypes = exe.outputElementTypes()
    doAssert outTypes.len == 1
    doAssert outTypes[0] == btF32
    doAssert exe.outputDimensions() == @[@[3'i64]]
    discard exe.costAnalysis()
    discard exe.compiledMemoryStats()
    discard exe.outputMemoryKinds()
    discard exe.parameterMemoryKinds(2)
    discard exe.fingerprint()
    doAssert exe.addressableDevices().len > 0
    discard exe.addressableDeviceLogicalIds()
    discard exe.deviceAssignment()
    discard exe.compileOptions()
    discard exe.optimizedProgram()

  var a: array[3, float32] = [1.0'f32, 2.0'f32, 3.0'f32]
  var b: array[3, float32] = [10.0'f32, 20.0'f32, 30.0'f32]
  let dims: array[1, int64] = [3'i64]
  let bufA = c.transferToDevice(dev, addr a[0], btF32, dims)
  let bufB = c.transferToDevice(dev, addr b[0], btF32, dims)

  let outs = exe.execute(dev, [bufA, bufB])
  doAssert outs.len == 1
  let bufOut = outs[0]
  doAssert bufOut.dimensions() == @[3'i64]

  var got: array[3, float32]
  bufOut.transferToHost(addr got[0], sizeof(got))
  doAssert got[0] == 11.0'f32
  doAssert got[1] == 22.0'f32
  doAssert got[2] == 33.0'f32
  echo "tpjrt: compile+execute round-trip OK (", got, ")"

  optionalPjrt "executable serialize/load":
    let encoded = exe.serialize()
    doAssert encoded.len > 0
    let loaded = c.loadExecutable(encoded)
    doAssert loaded.numOutputs() == 1
    let outs2 = loaded.execute(dev, [bufA, bufB])
    doAssert outs2.len == 1
    var got2: array[3, float32]
    outs2[0].transferToHost(addr got2[0], sizeof(got2))
    doAssert got2[0] == 11.0'f32
    doAssert got2[1] == 22.0'f32
    doAssert got2[2] == 33.0'f32

  optionalPjrt "loaded executable delete":
    let disposable = c.compile(mlirSrc)
    doAssert not disposable.isDeleted()
    disposable.delete()
    discard disposable.isDeleted()
