## High-level PJRT client wrapper.
##
## `PjrtClient` is a `ref object` that owns a `PjrtClientHandle` and
## releases it via `=destroy` (one PJRT_Client_Destroy call per Nim
## reference graph). Holding on to a `PjrtClient` keeps both the client
## and the loaded plugin's API table alive for the lifetime of the
## reference.
##
## The client is intentionally bound to a single `PjrtApiHandle`; using
## a client across plugins is undefined.

import ./capi
export capi

type
  PjrtClientObj* = object
    ## Underlying value type. Keep `PjrtClient` (the `ref`) as the
    ## user-facing alias.
    api*: PjrtApiHandle
    handle*: PjrtClientHandle

  PjrtClient* = ref PjrtClientObj
    ## RAII wrapper around `PjrtClientHandle`. The destructor calls
    ## `clientDestroy` exactly once when the last reference is dropped.
    ## Single-process / single-plugin only in v1.

proc `=destroy`(c: PjrtClientObj) =
  ## ARC/ORC hook — releases the underlying PJRT client. Errors thrown by
  ## the plugin during shutdown are swallowed so destructors stay
  ## `raises: []` clean (PJRT shutdown errors are diagnostic-only).
  if c.handle.isNil or c.api.isNil:
    return
  try:
    clientDestroy(c.api, c.handle)
  except CatchableError:
    discard
  except Exception:
    discard

proc newPjrtClient*(api: PjrtApiHandle): PjrtClient =
  ## Calls `PJRT_Client_Create` and returns a `PjrtClient` that will
  ## destroy the handle when garbage-collected.
  let h = clientCreate(api)
  result = PjrtClient(api: api, handle: h)

proc addressableDevices*(c: PjrtClient): seq[PjrtDeviceHandle] =
  ## Returns the devices addressable from this client. Handles remain
  ## valid until `c` is destroyed.
  clientAddressableDevices(c.api, c.handle)

proc devices*(c: PjrtClient): seq[PjrtDeviceHandle] =
  ## Returns all devices visible to this client, including non-addressable
  ## devices in multi-host runtimes.
  clientDevices(c.api, c.handle)

proc platformName*(c: PjrtClient): string =
  ## Returns the PJRT platform name (for example `cpu`, `gpu`, or `tpu`).
  clientPlatformName(c.api, c.handle)

proc processIndex*(c: PjrtClient): int =
  ## Returns the global process index for this client.
  clientProcessIndex(c.api, c.handle)

proc platformVersion*(c: PjrtClient): string =
  ## Returns platform-specific runtime version details.
  clientPlatformVersion(c.api, c.handle)

proc lookupDevice*(c: PjrtClient; id: int): PjrtDeviceHandle =
  ## Looks up a visible device by PJRT global id.
  clientLookupDevice(c.api, c.handle, id)

proc lookupAddressableDevice*(c: PjrtClient;
    localHardwareId: int): PjrtDeviceHandle =
  ## Looks up an addressable device by local hardware id.
  clientLookupAddressableDevice(c.api, c.handle, localHardwareId)

proc addressableMemories*(c: PjrtClient): seq[PjrtMemoryRaw] =
  ## Returns memory spaces this client can transfer to and from.
  clientAddressableMemories(c.api, c.handle)

proc defaultDeviceAssignment*(c: PjrtClient;
    numReplicas, numPartitions: int): seq[int] =
  ## Returns the flat default `[replica, partition]` device assignment.
  clientDefaultDeviceAssignment(c.api, c.handle, numReplicas, numPartitions)

proc updateGlobalProcessInfo*(c: PjrtClient;
    infos: openArray[PjrtProcessInfo]) =
  ## Updates global process state metadata for multi-host collective runtimes.
  clientUpdateGlobalProcessInfo(c.api, c.handle, infos)

proc deviceKind*(c: PjrtClient; dev: PjrtDeviceHandle): string =
  ## Convenience: returns the device-kind string for `dev`.
  let desc = deviceDescription(c.api, dev)
  descriptionKind(c.api, desc)

proc deviceId*(c: PjrtClient; dev: PjrtDeviceHandle): int =
  ## Convenience: returns the device id for `dev`.
  let desc = deviceDescription(c.api, dev)
  descriptionId(c.api, desc)

proc deviceProcessIndex*(c: PjrtClient; dev: PjrtDeviceHandle): int =
  ## Returns the process index owning `dev`.
  let desc = deviceDescription(c.api, dev)
  descriptionProcessIndex(c.api, desc)

proc deviceDebugString*(c: PjrtClient; dev: PjrtDeviceHandle): string =
  ## Returns a verbose device debug string.
  let desc = deviceDescription(c.api, dev)
  descriptionDebugString(c.api, desc)

proc deviceToString*(c: PjrtClient; dev: PjrtDeviceHandle): string =
  ## Returns a concise user-facing device string.
  let desc = deviceDescription(c.api, dev)
  descriptionToString(c.api, desc)

proc deviceAttributes*(c: PjrtClient; dev: PjrtDeviceHandle):
    seq[tuple[name, value: string]] =
  ## Returns device-description attributes as owned strings.
  let desc = deviceDescription(c.api, dev)
  descriptionAttributes(c.api, desc)

proc isAddressable*(c: PjrtClient; dev: PjrtDeviceHandle): bool =
  ## Returns true when this client can execute work on `dev`.
  deviceIsAddressable(c.api, dev)

proc localHardwareId*(c: PjrtClient; dev: PjrtDeviceHandle): int =
  ## Returns the local hardware id for `dev`, or -1 if undefined.
  deviceLocalHardwareId(c.api, dev)

proc deviceMemories*(c: PjrtClient; dev: PjrtDeviceHandle):
    seq[PjrtMemoryRaw] =
  ## Returns memory spaces addressable by `dev`.
  deviceAddressableMemories(c.api, dev)

proc defaultMemory*(c: PjrtClient; dev: PjrtDeviceHandle): PjrtMemoryRaw =
  ## Returns the default memory space for `dev`.
  deviceDefaultMemory(c.api, dev)

proc memoryStats*(c: PjrtClient; dev: PjrtDeviceHandle):
    PjrtDeviceMemoryStats =
  ## Returns allocator diagnostics for `dev` where the plugin supports it.
  deviceMemoryStats(c.api, dev)

proc clearMemoryStats*(c: PjrtClient; dev: PjrtDeviceHandle) =
  ## Clears peak-memory diagnostics for `dev` where the plugin supports it.
  deviceClearMemoryStats(c.api, dev)

proc memoryId*(c: PjrtClient; memory: PjrtMemoryRaw): int =
  ## Returns the unique PJRT id of `memory`.
  capi.memoryId(c.api, memory)

proc memoryKind*(c: PjrtClient; memory: PjrtMemoryRaw): string =
  ## Returns the platform-specific memory kind.
  capi.memoryKind(c.api, memory)

proc memoryKindId*(c: PjrtClient; memory: PjrtMemoryRaw): int =
  ## Returns the platform-specific memory-kind id.
  capi.memoryKindId(c.api, memory)

proc memoryDebugString*(c: PjrtClient; memory: PjrtMemoryRaw): string =
  ## Returns the verbose debug string for `memory`.
  capi.memoryDebugString(c.api, memory)

proc memoryToString*(c: PjrtClient; memory: PjrtMemoryRaw): string =
  ## Returns the concise user-facing string for `memory`.
  capi.memoryToString(c.api, memory)

proc memoryAddressableByDevices*(c: PjrtClient; memory: PjrtMemoryRaw):
    seq[PjrtDeviceHandle] =
  ## Returns devices that can address `memory`.
  capi.memoryAddressableByDevices(c.api, memory)

# ----- Topology wrapper -------------------------------------------------

type
  PjrtTopologyObj* = object
    ## Non-owning or owning wrapper around `PJRT_TopologyDescription`.
    api*: PjrtApiHandle
    raw*: PjrtTopologyDescriptionRaw
    owned*: bool

  PjrtTopology* = ref PjrtTopologyObj
    ## RAII wrapper for topology descriptions. Topologies returned by a
    ## client are borrowed (`owned = false`); deserialized topologies are
    ## destroyed when this wrapper is collected.

proc `=destroy`(t: PjrtTopologyObj) =
  if t.owned and not t.raw.isNil and not t.api.isNil:
    try:
      topologyDescriptionDestroy(t.api, t.raw)
    except CatchableError:
      discard
    except Exception:
      discard

proc topology*(c: PjrtClient): PjrtTopology =
  ## Returns the client-owned topology description.
  PjrtTopology(api: c.api, raw: clientTopologyDescription(c.api, c.handle),
    owned: false)

proc deserializeTopology*(c: PjrtClient; bytes: string): PjrtTopology =
  ## Deserializes a caller-owned topology description.
  PjrtTopology(api: c.api, raw: topologyDeserialize(c.api, bytes),
    owned: true)

proc platformName*(t: PjrtTopology): string =
  ## Returns the topology platform name.
  topologyPlatformName(t.api, t.raw)

proc platformVersion*(t: PjrtTopology): string =
  ## Returns topology platform version details.
  topologyPlatformVersion(t.api, t.raw)

proc deviceDescriptions*(t: PjrtTopology): seq[PjrtDeviceDescriptionRaw] =
  ## Returns all device descriptions in the topology.
  topologyDeviceDescriptions(t.api, t.raw)

proc serialize*(t: PjrtTopology): string =
  ## Serializes the topology for cache keys or transfer.
  topologySerialize(t.api, t.raw)

proc attributes*(t: PjrtTopology): seq[tuple[name, value: string]] =
  ## Returns topology attributes as owned strings.
  topologyAttributes(t.api, t.raw)

proc fingerprint*(t: PjrtTopology): uint64 =
  ## Returns the topology fingerprint.
  topologyFingerprint(t.api, t.raw)

# ----- Buffer wrapper ---------------------------------------------------

type
  PjrtBufferObj* = object
    ## Owning value type for a PJRT device buffer. Keep `PjrtBuffer`
    ## (the `ref`) as the user-facing alias.
    api*: PjrtApiHandle
    raw*: PjrtBufferRaw

  PjrtBuffer* = ref PjrtBufferObj
    ## RAII wrapper around a raw PJRT buffer. The destructor calls
    ## `PJRT_Buffer_Destroy` exactly once when the last reference is
    ## dropped. The eager backend layer (Phase 7c.5) will instead route
    ## device buffers through `BufferHandle`; this wrapper exists for
    ## standalone PJRT-layer code, tests, and tooling.

proc `=destroy`(b: PjrtBufferObj) =
  ## ARC/ORC hook — releases the device buffer. Errors are swallowed so
  ## the hook stays `raises: []` clean.
  if b.raw.isNil or b.api.isNil:
    return
  try:
    bufferDestroy(b.api, b.raw)
  except CatchableError:
    discard
  except Exception:
    discard

proc newPjrtBuffer*(api: PjrtApiHandle; raw: PjrtBufferRaw): PjrtBuffer =
  ## Wraps an already-allocated `raw` buffer for ownership tracking.
  ## Used by `transferToDevice` and (later) the executable-execute path.
  PjrtBuffer(api: api, raw: raw)

proc dimensions*(b: PjrtBuffer): seq[int64] =
  ## Returns the logical dimensions of `b`.
  bufferDimensions(b.api, b.raw)

proc elementType*(b: PjrtBuffer): PjrtBufferType =
  ## Returns the PJRT element type of `b`.
  bufferElementType(b.api, b.raw)

proc onDeviceSizeInBytes*(b: PjrtBuffer): int =
  ## Returns the on-device size of `b` in bytes.
  bufferOnDeviceSizeInBytes(b.api, b.raw)

proc transferToDevice*(c: PjrtClient; device: PjrtDeviceHandle;
    data: pointer; bufferType: PjrtBufferType;
    dims: openArray[int64]): PjrtBuffer =
  ## Copies `data` from host into a fresh device buffer on `device`.
  ## Blocks (via the returned event) until the host buffer is no longer
  ## read by the plugin, so callers may free `data` immediately on return.
  let (raw, ev) = clientBufferFromHostBuffer(
    c.api, c.handle, device, data, bufferType, dims)
  if not ev.isNil:
    awaitEvent(c.api, ev, "PJRT_Client_BufferFromHostBuffer:done")
  newPjrtBuffer(c.api, raw)

proc transferToHost*(b: PjrtBuffer; dst: pointer; dstSize: int) =
  ## Copies `b` into the host region `dst`/`dstSize`, blocking until the
  ## device→host transfer completes.
  let ev = bufferToHostBuffer(b.api, b.raw, dst, dstSize)
  if not ev.isNil:
    awaitEvent(b.api, ev, "PJRT_Buffer_ToHostBuffer:done")

# ----- Compile + Execute ------------------------------------------------

type
  PjrtLoadedExecutableObj* = object
    ## Owning value type for a loaded executable. Use `PjrtLoadedExecutable`
    ## (the `ref`) as the user-facing alias.
    api*: PjrtApiHandle
    raw*: PjrtLoadedExecutableRaw

  PjrtLoadedExecutable* = ref PjrtLoadedExecutableObj
    ## RAII wrapper around a `PJRT_LoadedExecutable`. The destructor calls
    ## `PJRT_LoadedExecutable_Destroy` exactly once when the last reference
    ## is dropped.

  PjrtRunOptions* = object
    ## High-level execution options for `PjrtLoadedExecutable.execute`.
    nonDonatableInputIndices*: seq[int64]
      ## Inputs PJRT must not donate even if the compiled program may alias.
    launchId*: int
      ## Optional multi-device launch id used by PJRT runtimes.
    callLocation*: string
      ## Optional source-location label passed to the plugin for diagnostics.
    taskIds*: seq[int]
      ## Optional global task ids for multi-slice/multi-host launches.
    incarnationIds*: seq[int64]
      ## Optional task incarnation ids paired with `taskIds`.

proc `=destroy`(e: PjrtLoadedExecutableObj) =
  ## ARC/ORC hook — releases the loaded executable. Errors are swallowed
  ## so the destructor stays `raises: []` clean.
  if e.raw.isNil or e.api.isNil:
    return
  try:
    loadedExecutableDestroy(e.api, e.raw)
  except CatchableError:
    discard
  except Exception:
    discard

proc compile*(c: PjrtClient; code: string;
    format: string = "mlir";
    compileOptions: string = ""): PjrtLoadedExecutable =
  ## Compiles `code` into a loaded executable on `c`. `format` defaults to
  ## `"mlir"` (StableHLO text or bytecode); pass `"hlo"` for serialized HLO.
  ## `compileOptions` is a serialized `xla.CompileOptionsProto`; an empty
  ## string requests plugin defaults.
  let raw = clientCompile(c.api, c.handle, code, format, compileOptions)
  PjrtLoadedExecutable(api: c.api, raw: raw)

proc numOutputs*(e: PjrtLoadedExecutable): int =
  ## Returns the number of outputs produced per device. Internally fetches
  ## a fresh `PJRT_Executable` view, queries it, and frees it.
  let view = loadedExecutableGetExecutable(e.api, e.raw)
  try:
    result = executableNumOutputs(e.api, view)
  finally:
    executableDestroy(e.api, view)

template withExecutableView(e: PjrtLoadedExecutable; viewName: untyped;
    body: untyped): untyped =
  let viewName = loadedExecutableGetExecutable(e.api, e.raw)
  try:
    body
  finally:
    executableDestroy(e.api, viewName)

proc name*(e: PjrtLoadedExecutable): string =
  ## Returns the executable name reported by PJRT.
  withExecutableView(e, view):
    result = executableName(e.api, view)

proc numReplicas*(e: PjrtLoadedExecutable): int =
  ## Returns the number of replicas compiled into `e`.
  withExecutableView(e, view):
    result = executableNumReplicas(e.api, view)

proc numPartitions*(e: PjrtLoadedExecutable): int =
  ## Returns the number of partitions compiled into `e`.
  withExecutableView(e, view):
    result = executableNumPartitions(e.api, view)

proc generatedCodeSizeInBytes*(e: PjrtLoadedExecutable): int64 =
  ## Returns the generated-code size reported by PJRT.
  withExecutableView(e, view):
    result = executableSizeOfGeneratedCodeInBytes(e.api, view)

proc fingerprint*(e: PjrtLoadedExecutable): string =
  ## Returns the modern executable fingerprint, falling back to the deprecated
  ## loaded-executable slot when older plugins expose only that variant.
  try:
    withExecutableView(e, view):
      result = executableFingerprint(e.api, view)
  except PjrtError:
    result = loadedExecutableFingerprint(e.api, e.raw)

proc costAnalysis*(e: PjrtLoadedExecutable): seq[tuple[name, value: string]] =
  ## Returns plugin-specific executable cost-analysis properties.
  withExecutableView(e, view):
    result = executableCostAnalysis(e.api, view)

proc compiledMemoryStats*(e: PjrtLoadedExecutable): PjrtCompiledMemoryStats =
  ## Returns compile-time memory estimates for `e`.
  withExecutableView(e, view):
    result = executableCompiledMemoryStats(e.api, view)

proc outputElementTypes*(e: PjrtLoadedExecutable): seq[PjrtBufferType] =
  ## Returns each output's element type.
  withExecutableView(e, view):
    result = executableOutputElementTypes(e.api, view)

proc outputDimensions*(e: PjrtLoadedExecutable): seq[seq[int64]] =
  ## Returns each output's dimensions.
  withExecutableView(e, view):
    let n = executableNumOutputs(e.api, view)
    result = executableOutputDimensions(e.api, view, n)

proc parameterMemoryKinds*(e: PjrtLoadedExecutable;
    numParameters: int): seq[string] =
  ## Returns memory kind strings for executable parameters.
  withExecutableView(e, view):
    result = executableParameterMemoryKinds(e.api, view, numParameters)

proc outputMemoryKinds*(e: PjrtLoadedExecutable): seq[string] =
  ## Returns memory kind strings for executable outputs.
  withExecutableView(e, view):
    let n = executableNumOutputs(e.api, view)
    result = executableOutputMemoryKinds(e.api, view, n)

proc serialize*(e: PjrtLoadedExecutable): string =
  ## Serializes the unloaded executable payload so it can be cached or loaded
  ## later by the same platform/runtime version.
  withExecutableView(e, view):
    result = executableSerialize(e.api, view)

proc compileOptions*(e: PjrtLoadedExecutable): string =
  ## Returns the serialized CompileOptions used to build `e`.
  withExecutableView(e, view):
    result = executableCompileOptions(e.api, view)

proc optimizedProgram*(e: PjrtLoadedExecutable; format = "hlo"):
    tuple[code, format: string] =
  ## Returns the optimized program bytes and actual format reported by PJRT.
  withExecutableView(e, view):
    result = executableOptimizedProgram(e.api, view, format)

proc addressableDevices*(e: PjrtLoadedExecutable): seq[PjrtDeviceHandle] =
  ## Returns devices this executable can run on.
  loadedExecutableAddressableDevices(e.api, e.raw)

proc addressableDeviceLogicalIds*(e: PjrtLoadedExecutable):
    seq[PjrtLogicalDeviceIds] =
  ## Returns logical `(replica, partition)` ids matching addressable devices.
  loadedExecutableAddressableDeviceLogicalIds(e.api, e.raw)

proc deviceAssignment*(e: PjrtLoadedExecutable): string =
  ## Returns serialized `DeviceAssignmentProto` bytes.
  loadedExecutableDeviceAssignment(e.api, e.raw)

proc delete*(e: PjrtLoadedExecutable) =
  ## Deletes the underlying runtime object while keeping `e` destroyable.
  loadedExecutableDelete(e.api, e.raw)

proc isDeleted*(e: PjrtLoadedExecutable): bool =
  ## Returns true after `delete(e)`.
  loadedExecutableIsDeleted(e.api, e.raw)

proc loadExecutable*(c: PjrtClient; serializedExecutable: string;
    compileOptions: string = ""): PjrtLoadedExecutable =
  ## Deserializes and loads an executable produced by `serialize`.
  let raw = executableDeserializeAndLoad(
    c.api, c.handle, serializedExecutable, compileOptions)
  PjrtLoadedExecutable(api: c.api, raw: raw)

proc execute*(e: PjrtLoadedExecutable; device: PjrtDeviceHandle;
    inputs: openArray[PjrtBuffer]; options: PjrtRunOptions):
    seq[PjrtBuffer] =
  ## Single-device synchronous execute of `e` on `device`. The returned
  ## buffers are wrapped for ownership tracking — destroying them releases
  ## the underlying device memory.
  var rawInputs = newSeq[PjrtBufferRaw](inputs.len)
  for i, b in inputs:
    rawInputs[i] = b.raw
  let n = e.numOutputs()
  let outRaws = loadedExecutableExecute(e.api, e.raw, device, rawInputs, n,
    options.nonDonatableInputIndices, options.launchId,
    options.callLocation)
  result = newSeq[PjrtBuffer](outRaws.len)
  for i, raw in outRaws:
    result[i] = newPjrtBuffer(e.api, raw)

proc execute*(e: PjrtLoadedExecutable; inputs: openArray[seq[PjrtBuffer]];
    options: PjrtRunOptions): seq[seq[PjrtBuffer]] =
  ## Multi-device synchronous execute. The outer input dimension is device,
  ## the inner dimension is argument. PJRT picks the executable's addressable
  ## devices because the C wrapper passes `execute_device = nil`.
  var rawInputs = newSeq[seq[PjrtBufferRaw]](inputs.len)
  for i, row in inputs:
    rawInputs[i] = newSeq[PjrtBufferRaw](row.len)
    for j, b in row:
      rawInputs[i][j] = b.raw
  let n = e.numOutputs()
  let outRaws = loadedExecutableExecuteMulti(e.api, e.raw, rawInputs, n,
    options.nonDonatableInputIndices, options.launchId,
    options.callLocation, options.taskIds, options.incarnationIds)
  result = newSeq[seq[PjrtBuffer]](outRaws.len)
  for i, row in outRaws:
    result[i] = newSeq[PjrtBuffer](row.len)
    for j, raw in row:
      result[i][j] = newPjrtBuffer(e.api, raw)

proc execute*(e: PjrtLoadedExecutable; device: PjrtDeviceHandle;
    inputs: openArray[PjrtBuffer]): seq[PjrtBuffer] =
  ## Single-device synchronous execute with all inputs protected from
  ## runtime donation.
  var nonDonatable = newSeq[int64](inputs.len)
  for i in 0 ..< inputs.len:
    nonDonatable[i] = int64(i)
  e.execute(device, inputs,
    PjrtRunOptions(nonDonatableInputIndices: nonDonatable))
