## Raw PJRT C API surface — opaque handles, error type, and the entry-point
## function-pointer typedef.
##
## **This module is the *only* place in the codebase that touches the PJRT
## C API directly.** Everything outside `src/rew/pjrt/` consumes the typed
## wrappers in `loader.nim` and the higher layers above. The
## `check_layer_imports` lint enforces this boundary.
##
## The `PjrtApi` object below mirrors the upstream `PJRT_Api` struct from
## `xla/pjrt/c/pjrt_c_api.h` (PJRT_API_MAJOR=0, PJRT_API_MINOR=107). Every
## function-pointer field is declared as `pointer` because the wire ABI for
## any C function pointer is a single pointer-sized slot — typed proc
## signatures are introduced lazily via `cast[T]` at the call sites in the
## typed wrappers further down. This keeps the struct layout obviously
## correct (one `pointer` per slot, in the upstream order) while still
## letting us add typed wrappers for individual functions one-by-one as the
## higher layers grow.
##
## When the upstream header gains new trailing fields, append them at the
## end of `PjrtApi`. Older plugins set `struct_size` to a smaller value;
## use `apiStructSize` and the per-call `safeFn` accessor before invoking
## any newly-added field.

type
  PjrtExtensionBaseRaw* = distinct pointer
    ## Opaque pointer to the head of a `PJRT_Extension_Base` chain (we never
    ## traverse extensions in v1).

  PjrtApiVersion* {.bycopy.} = object
    ## ABI-stable copy of `PJRT_Api_Version` embedded inside `PjrtApi`.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    majorVersion*: cint
    minorVersion*: cint

  PjrtApi* {.bycopy.} = object
    ## ABI-equivalent layout of the upstream `PJRT_Api` struct. Every
    ## function-pointer field is declared as `pointer`; cast to the
    ## appropriate typed proc inside a wrapper before calling.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    pjrtApiVersion*: PjrtApiVersion

    fnErrorDestroy*, fnErrorMessage*, fnErrorGetCode*: pointer
    fnPluginInitialize*, fnPluginAttributes*: pointer
    fnEventDestroy*, fnEventIsReady*, fnEventError*, fnEventAwait*,
      fnEventOnReady*: pointer
    fnClientCreate*, fnClientDestroy*, fnClientPlatformName*,
      fnClientProcessIndex*, fnClientPlatformVersion*, fnClientDevices*,
      fnClientAddressableDevices*, fnClientLookupDevice*,
      fnClientLookupAddressableDevice*, fnClientAddressableMemories*,
      fnClientCompile*, fnClientDefaultDeviceAssignment*,
      fnClientBufferFromHostBuffer*: pointer
    fnDeviceDescriptionId*, fnDeviceDescriptionProcessIndex*,
      fnDeviceDescriptionAttributes*, fnDeviceDescriptionKind*,
      fnDeviceDescriptionDebugString*, fnDeviceDescriptionToString*: pointer
    fnDeviceGetDescription*, fnDeviceIsAddressable*,
      fnDeviceLocalHardwareId*, fnDeviceAddressableMemories*,
      fnDeviceDefaultMemory*, fnDeviceMemoryStats*: pointer
    fnMemoryId*, fnMemoryKind*, fnMemoryDebugString*, fnMemoryToString*,
      fnMemoryAddressableByDevices*: pointer
    fnExecutableDestroy*, fnExecutableName*, fnExecutableNumReplicas*,
      fnExecutableNumPartitions*, fnExecutableNumOutputs*,
      fnExecutableSizeOfGeneratedCodeInBytes*,
      fnExecutableGetCostAnalysis*, fnExecutableOutputMemoryKinds*,
      fnExecutableOptimizedProgram*, fnExecutableSerialize*: pointer
    fnLoadedExecutableDestroy*, fnLoadedExecutableGetExecutable*,
      fnLoadedExecutableAddressableDevices*, fnLoadedExecutableDelete*,
      fnLoadedExecutableIsDeleted*, fnLoadedExecutableExecute*,
      fnExecutableDeserializeAndLoad*, fnLoadedExecutableFingerprint*: pointer
    fnBufferDestroy*, fnBufferElementType*, fnBufferDimensions*,
      fnBufferUnpaddedDimensions*, fnBufferDynamicDimensionIndices*,
      fnBufferGetMemoryLayout*, fnBufferOnDeviceSizeInBytes*,
      fnBufferDevice*, fnBufferMemory*, fnBufferDelete*, fnBufferIsDeleted*,
      fnBufferCopyToDevice*, fnBufferToHostBuffer*, fnBufferIsOnCpu*,
      fnBufferReadyEvent*, fnBufferUnsafePointer*,
      fnBufferIncreaseExternalReferenceCount*,
      fnBufferDecreaseExternalReferenceCount*,
      fnBufferOpaqueDeviceMemoryDataPointer*: pointer
    fnCopyToDeviceStreamDestroy*, fnCopyToDeviceStreamAddChunk*,
      fnCopyToDeviceStreamTotalBytes*, fnCopyToDeviceStreamGranuleSize*,
      fnCopyToDeviceStreamCurrentBytes*: pointer
    fnTopologyDescriptionCreate*, fnTopologyDescriptionDestroy*,
      fnTopologyDescriptionPlatformName*,
      fnTopologyDescriptionPlatformVersion*,
      fnTopologyDescriptionGetDeviceDescriptions*,
      fnTopologyDescriptionSerialize*,
      fnTopologyDescriptionAttributes*: pointer
    fnCompile*: pointer
    fnExecutableOutputElementTypes*, fnExecutableOutputDimensions*: pointer
    fnBufferCopyToMemory*: pointer
    fnClientCreateViewOfDeviceBuffer*: pointer
    fnExecutableFingerprint*: pointer
    fnClientTopologyDescription*: pointer
    fnExecutableGetCompiledMemoryStats*: pointer
    fnMemoryKindId*: pointer
    fnExecuteContextCreate*, fnExecuteContextDestroy*: pointer
    fnBufferCopyRawToHost*: pointer
    fnAsyncHostToDeviceTransferManagerDestroy*,
      fnAsyncHostToDeviceTransferManagerTransferData*: pointer
    fnClientCreateBuffersForAsyncHostToDevice*: pointer
    fnAsyncHostToDeviceTransferManagerRetrieveBuffer*,
      fnAsyncHostToDeviceTransferManagerDevice*,
      fnAsyncHostToDeviceTransferManagerBufferCount*,
      fnAsyncHostToDeviceTransferManagerBufferSize*,
      fnAsyncHostToDeviceTransferManagerSetBufferError*,
      fnAsyncHostToDeviceTransferManagerAddMetadata*: pointer
    fnClientDmaMap*, fnClientDmaUnmap*: pointer
    fnClientCreateUninitializedBuffer*: pointer
    fnClientUpdateGlobalProcessInfo*: pointer
    fnTopologyDescriptionDeserialize*: pointer
    fnClientCreateAliasBuffer*, fnClientFulfillAliasBuffer*: pointer
    fnLoadedExecutableGetDeviceAssignment*: pointer
    fnClientCreateErrorBuffer*: pointer
    fnAsyncHostToDeviceTransferManagerTransferLiteral*: pointer
    fnBufferCopyRawToHostFuture*: pointer
    fnDevicePoisonExecution*, fnDeviceCreateAsyncTrackingEvent*,
      fnAsyncTrackingEventDestroy*: pointer
    fnExecutableGetCompileOptions*: pointer
    fnBufferDonateWithControlDependency*: pointer
    fnEventCreate*, fnEventSet*: pointer
    fnDeviceGetAttributes*: pointer
    fnClientLoad*: pointer
    fnLoadedExecutableAddressableDeviceLogicalIds*: pointer
    fnBufferBitcast*: pointer
    fnErrorForEachPayload*: pointer
    fnTopologyDescriptionFingerprint*: pointer
    fnExecutableParameterMemoryKinds*: pointer
    fnDeviceClearMemoryStats*: pointer

  PjrtApiHandle* = distinct pointer
    ## Pointer to the `PjrtApi` table returned by a plugin's `GetPjrtApi`
    ## symbol. Use `apiPtr` to dereference, and the typed wrappers below
    ## (`apiVersion`, `pluginInitialize`, `pluginAttributes`, …) for
    ## individual functions. Outside `src/rew/pjrt/` only the high-level
    ## wrappers are intended.

  PjrtClientHandle* = distinct pointer
    ## Opaque pointer to a PJRT client (per-process per-plugin instance).

  PjrtBufferRaw* = distinct pointer
    ## Opaque pointer to a PJRT device buffer. Owned by `BufferHandle`
    ## (`src/rew/buffer.nim`); never freed manually.

  PjrtDeviceHandle* = distinct pointer
    ## Opaque pointer to a PJRT device descriptor.

  PjrtMemoryRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_Memory`. Memory handles are owned by a
    ## client/device/topology and must not be destroyed by callers.

  PjrtTopologyDescriptionRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_TopologyDescription`. Handles returned by a
    ## client are client-owned; handles created/deserialized through the
    ## topology APIs are caller-owned and must be destroyed.

  PjrtErrorRaw* = distinct pointer
    ## Opaque pointer to a PJRT error object returned by a failing call.
    ## Always destroyed and translated into a `PjrtError` exception by the
    ## wrapper that originated the call.

  PjrtSerializedExecutableRaw* = distinct pointer
    ## Backing allocation for serialized executable bytes returned by PJRT.
    ## It is freed by the deleter function returned in the same args struct.

  PjrtSerializedCompileOptionsRaw* = distinct pointer
    ## Backing allocation for serialized compile-options bytes returned by
    ## PJRT. It is freed by the matching deleter.

  PjrtSerializedTopologyRaw* = distinct pointer
    ## Backing allocation for serialized topology bytes returned by PJRT. It
    ## is freed by the matching deleter.

  PjrtDeviceAssignmentSerializedRaw* = distinct pointer
    ## Backing allocation for serialized `DeviceAssignmentProto` bytes. It is
    ## freed by the matching deleter.

  PjrtErrorCode* {.size: sizeof(cint).} = enum
    ## Mirrors `PJRT_Error_Code`.
    pecOk = 0
    pecCancelled = 1
    pecUnknown = 2
    pecInvalidArgument = 3
    pecDeadlineExceeded = 4
    pecNotFound = 5
    pecAlreadyExists = 6
    pecPermissionDenied = 7
    pecResourceExhausted = 8
    pecFailedPrecondition = 9
    pecAborted = 10
    pecOutOfRange = 11
    pecUnimplemented = 12
    pecInternal = 13
    pecUnavailable = 14
    pecDataLoss = 15
    pecUnauthenticated = 16

  PjrtError* = object of CatchableError
    ## Exception type for every PJRT-originated failure. Carries the plugin's
    ## error message verbatim plus the wrapper-side context that triggered
    ## the call.
    pluginCode*: int  ## Plugin-reported error code, or 0 if unknown.
    payloads*: seq[tuple[key, value: string]]
      ## Optional typed status payloads attached by newer PJRT plugins.

  GetPjrtApiProc* = proc (): PjrtApiHandle {.cdecl, gcsafe.}
    ## Type of the C symbol every PJRT plugin must export under the name
    ## `GetPjrtApi`. The loader resolves this symbol and calls it once per
    ## plugin to obtain the API table.

const
  GetPjrtApiSymbol* = "GetPjrtApi"
    ## Name of the entry-point symbol exported by every PJRT plugin.

  DefaultCompileOptionsProto* = "\x1a\x04\x20\x01\x28\x01"
    ## Serialized `xla.CompileOptionsProto` for single-device execution:
    ## `executable_build_options` (field 3) containing `num_replicas = 1`
    ## (field 4) and `num_partitions = 1` (field 5).

func isNil*(h: PjrtApiHandle): bool {.borrow.}
func isNil*(h: PjrtClientHandle): bool {.borrow.}
func isNil*(h: PjrtBufferRaw): bool {.borrow.}
func isNil*(h: PjrtDeviceHandle): bool {.borrow.}
func isNil*(h: PjrtMemoryRaw): bool {.borrow.}
func isNil*(h: PjrtTopologyDescriptionRaw): bool {.borrow.}
func isNil*(h: PjrtErrorRaw): bool {.borrow.}
func isNil*(h: PjrtSerializedExecutableRaw): bool {.borrow.}
func isNil*(h: PjrtSerializedCompileOptionsRaw): bool {.borrow.}
func isNil*(h: PjrtSerializedTopologyRaw): bool {.borrow.}
func isNil*(h: PjrtDeviceAssignmentSerializedRaw): bool {.borrow.}

func apiPtr*(h: PjrtApiHandle): ptr PjrtApi {.inline.} =
  ## Casts the opaque handle to a typed `ptr PjrtApi`. The pointer is owned
  ## by the loaded shared library and lives until the library is unloaded;
  ## callers must never free it.
  cast[ptr PjrtApi](h)

proc raisePjrt*(msg: string; pluginCode: int = 0) {.noreturn, noinline.} =
  ## Raises a `PjrtError`. Centralised so call sites never construct the
  ## exception inline (matches the `nim-error-handling` skill's "one private
  ## helper for shared failure modes" pattern).
  var e = newException(PjrtError, msg)
  e.pluginCode = pluginCode
  raise e

proc raisePjrt*(msg: string; code: PjrtErrorCode;
    payloads: seq[tuple[key, value: string]]) {.noreturn, noinline.} =
  ## Raises a `PjrtError` with a typed PJRT status code and optional payloads.
  var e = newException(PjrtError, msg)
  e.pluginCode = ord(code)
  e.payloads = payloads
  raise e

# ----- Args structs for the wrappers we currently use -------------------

type
  PjrtErrorDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    error*: PjrtErrorRaw

  PjrtErrorMessageArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    error*: PjrtErrorRaw
    message*: cstring        ## out — owned by the error
    messageSize*: csize_t    ## out

  PjrtErrorGetCodeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    error*: PjrtErrorRaw
    code*: PjrtErrorCode      ## out

  PjrtErrorPayloadVisitor* =
    proc (key: cstring; keySize: csize_t; value: cstring;
      valueSize: csize_t; userArg: pointer) {.cdecl.}

  PjrtErrorForEachPayloadArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    error*: PjrtErrorRaw
    visitor*: PjrtErrorPayloadVisitor
    userArg*: pointer

  PjrtPluginInitializeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw

  PjrtNamedValueKind* {.size: sizeof(cint).} = enum
    nvString = 0
    nvInt64 = 1
    nvInt64List = 2
    nvFloat = 3
    nvBool = 4

  PjrtNamedValuePayload* {.union, bycopy.} = object
    ## Mirrors the anonymous union inside `PJRT_NamedValue`.
    stringValue*: cstring
    int64Value*: int64
    int64ArrayValue*: ptr UncheckedArray[int64]
    floatValue*: float32
    boolValue*: bool

  PjrtNamedValue* {.bycopy.} = object
    ## Mirrors the upstream `PJRT_NamedValue`. The C struct is a tagged
    ## union; `payload` preserves the union's pointer alignment so
    ## `valueSize` lands at the same offset as the C header.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    name*: cstring
    nameSize*: csize_t
    kind*: PjrtNamedValueKind
    payload*: PjrtNamedValuePayload
    valueSize*: csize_t

  PjrtPluginAttributesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    attributes*: ptr UncheckedArray[PjrtNamedValue]  ## out
    numAttributes*: csize_t                          ## out

  PjrtClientCreateArgs* {.bycopy.} = object
    ## Mirrors `PJRT_Client_Create_Args`. We pass nil for every kv-store /
    ## kv-try-get callback in v1 (single-process only).
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    createOptions*: ptr PjrtNamedValue   ## may be nil
    numOptions*: csize_t
    kvGetCallback*: pointer
    kvGetUserArg*: pointer
    kvPutCallback*: pointer
    kvPutUserArg*: pointer
    client*: PjrtClientHandle            ## out
    kvTryGetCallback*: pointer
    kvTryGetUserArg*: pointer

  PjrtClientDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle

  PjrtClientPlatformNameArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    platformName*: cstring        ## out — owned by client
    platformNameSize*: csize_t    ## out

  PjrtClientProcessIndexArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    processIndex*: cint           ## out

  PjrtClientPlatformVersionArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    platformVersion*: cstring       ## out — owned by client
    platformVersionSize*: csize_t   ## out

  PjrtClientTopologyDescriptionArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    topology*: PjrtTopologyDescriptionRaw  ## out — owned by client

  PjrtClientDevicesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    devices*: ptr UncheckedArray[PjrtDeviceHandle]  ## out
    numDevices*: csize_t                            ## out

  PjrtClientAddressableDevicesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    addressableDevices*: ptr UncheckedArray[PjrtDeviceHandle]  ## out
    numAddressableDevices*: csize_t                            ## out

  PjrtClientLookupDeviceArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    id*: cint
    device*: PjrtDeviceHandle  ## out — owned by client

  PjrtClientLookupAddressableDeviceArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    localHardwareId*: cint
    addressableDevice*: PjrtDeviceHandle  ## out — owned by client

  PjrtProcessState* {.size: sizeof(cint).} = enum
    ppsUnspecified = 0
    ppsUninitialized = 1
    ppsDisconnected = 2
    ppsConnected = 3
    ppsError = 4

  PjrtProcessInfo* {.bycopy.} = object
    structSize*: csize_t
    taskId*: cint
    incarnationId*: uint64
    state*: PjrtProcessState
    errorCode*: cint
    errorMessage*: cstring
    errorMessageSize*: csize_t

  PjrtClientUpdateGlobalProcessInfoArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    processInfos*: ptr PjrtProcessInfo
    numProcessInfos*: csize_t

  PjrtClientAddressableMemoriesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    addressableMemories*: ptr UncheckedArray[PjrtMemoryRaw]  ## out
    numAddressableMemories*: csize_t                         ## out

  PjrtClientDefaultDeviceAssignmentArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    numReplicas*: cint
    numPartitions*: cint
    defaultAssignmentSize*: csize_t
    defaultAssignment*: ptr cint  ## caller-allocated in/out

  PjrtEventRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_Event` (async completion handle).

  PjrtEventDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    event*: PjrtEventRaw

  PjrtEventAwaitArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    event*: PjrtEventRaw

  PjrtDeviceDescriptionRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_DeviceDescription`.

  PjrtDeviceGetDescriptionArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    deviceDescription*: PjrtDeviceDescriptionRaw  ## out

  PjrtDeviceDescriptionKindArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    deviceKind*: cstring        ## out — owned by description
    deviceKindSize*: csize_t    ## out

  PjrtDeviceDescriptionIdArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    id*: cint                   ## out

  PjrtDeviceDescriptionProcessIndexArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    processIndex*: cint         ## out

  PjrtDeviceDescriptionDebugStringArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    debugString*: cstring       ## out — owned by description
    debugStringSize*: csize_t   ## out

  PjrtDeviceDescriptionToStringArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    toString*: cstring          ## out — owned by description
    toStringSize*: csize_t      ## out

  PjrtDeviceDescriptionAttributesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    deviceDescription*: PjrtDeviceDescriptionRaw
    numAttributes*: csize_t                         ## out
    attributes*: ptr UncheckedArray[PjrtNamedValue] ## out

  PjrtDeviceIsAddressableArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    isAddressable*: bool        ## out

  PjrtDeviceLocalHardwareIdArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    localHardwareId*: cint      ## out

  PjrtDeviceAddressableMemoriesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    memories*: ptr UncheckedArray[PjrtMemoryRaw]  ## out
    numMemories*: csize_t                         ## out

  PjrtDeviceDefaultMemoryArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    memory*: PjrtMemoryRaw       ## out — owned by device/client

  PjrtDeviceMemoryStatsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle
    bytesInUse*: int64
    peakBytesInUse*: int64
    peakBytesInUseIsSet*: bool
    numAllocs*: int64
    numAllocsIsSet*: bool
    largestAllocSize*: int64
    largestAllocSizeIsSet*: bool
    bytesLimit*: int64
    bytesLimitIsSet*: bool
    bytesReserved*: int64
    bytesReservedIsSet*: bool
    peakBytesReserved*: int64
    peakBytesReservedIsSet*: bool
    bytesReservableLimit*: int64
    bytesReservableLimitIsSet*: bool
    largestFreeBlockBytes*: int64
    largestFreeBlockBytesIsSet*: bool
    poolBytes*: int64
    poolBytesIsSet*: bool
    peakPoolBytes*: int64
    peakPoolBytesIsSet*: bool

  PjrtDeviceMemoryStats* = object
    ## Device allocator diagnostics returned by `PJRT_Device_MemoryStats`.
    ## Optional fields are paired with `has*` booleans because not every
    ## platform reports every statistic.
    bytesInUse*: int64
    peakBytesInUse*: int64
    hasPeakBytesInUse*: bool
    numAllocs*: int64
    hasNumAllocs*: bool
    largestAllocSize*: int64
    hasLargestAllocSize*: bool
    bytesLimit*: int64
    hasBytesLimit*: bool
    bytesReserved*: int64
    hasBytesReserved*: bool
    peakBytesReserved*: int64
    hasPeakBytesReserved*: bool
    bytesReservableLimit*: int64
    hasBytesReservableLimit*: bool
    largestFreeBlockBytes*: int64
    hasLargestFreeBlockBytes*: bool
    poolBytes*: int64
    hasPoolBytes*: bool
    peakPoolBytes*: int64
    hasPeakPoolBytes*: bool

  PjrtDeviceClearMemoryStatsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    device*: PjrtDeviceHandle

  PjrtMemoryIdArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    id*: cint                    ## out

  PjrtMemoryKindArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    kind*: cstring               ## out — owned by memory
    kindSize*: csize_t           ## out

  PjrtMemoryKindIdArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    kindId*: cint                ## out

  PjrtMemoryDebugStringArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    debugString*: cstring        ## out — owned by memory
    debugStringSize*: csize_t    ## out

  PjrtMemoryToStringArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    toString*: cstring           ## out — owned by memory
    toStringSize*: csize_t       ## out

  PjrtMemoryAddressableByDevicesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    memory*: PjrtMemoryRaw
    devices*: ptr UncheckedArray[PjrtDeviceHandle]  ## out
    numDevices*: csize_t                            ## out

  PjrtBufferType* {.size: sizeof(cint).} = enum
    ## Mirrors `PJRT_Buffer_Type`. Only the dtypes we currently use need
    ## stable ordinals; the rest are listed for completeness.
    btInvalid = 0
    btPred    = 1
    btS8      = 2
    btS16     = 3
    btS32     = 4
    btS64     = 5
    btU8      = 6
    btU16     = 7
    btU32     = 8
    btU64     = 9
    btF16     = 10
    btF32     = 11
    btF64     = 12
    btBF16    = 13
    btC64     = 14
    btC128    = 15

  PjrtHostBufferSemantics* {.size: sizeof(cint).} = enum
    ## Mirrors `PJRT_HostBufferSemantics`.
    hbsImmutableOnlyDuringCall      = 0
    hbsImmutableUntilTransferDone   = 1
    hbsImmutableZeroCopy            = 2
    hbsMutableZeroCopy              = 3

  PjrtClientBufferFromHostBufferArgs* {.bycopy.} = object
    ## Mirrors `PJRT_Client_BufferFromHostBuffer_Args`. We always pass nil
    ## for `byteStrides`, `memory`, and `deviceLayout` (dense major-to-minor
    ## layout, default memory of `device`). Two outputs: `doneWithHostBuffer`
    ## (event) and `buffer`.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    data*: pointer
    bufferType*: PjrtBufferType
    dims*: ptr int64
    numDims*: csize_t
    byteStrides*: ptr int64        ## may be nil → dense layout
    numByteStrides*: csize_t
    hostBufferSemantics*: PjrtHostBufferSemantics
    device*: PjrtDeviceHandle
    memory*: pointer               ## PjrtMemory*; nil → default memory
    deviceLayout*: pointer         ## PjrtBufferMemoryLayout*; nil → dense
    doneWithHostBuffer*: PjrtEventRaw   ## out
    buffer*: PjrtBufferRaw              ## out

  PjrtBufferDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    buffer*: PjrtBufferRaw

  PjrtBufferToHostBufferArgs* {.bycopy.} = object
    ## Mirrors `PJRT_Buffer_ToHostBuffer_Args`. Two-pass usage: first call
    ## with `dst = nil` to learn `dstSize`; then allocate and call again.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    src*: PjrtBufferRaw
    hostLayout*: pointer        ## nil → use src layout
    dst*: pointer               ## in/out
    dstSize*: csize_t           ## in/out
    event*: PjrtEventRaw        ## out

  PjrtBufferElementTypeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    buffer*: PjrtBufferRaw
    `type`*: PjrtBufferType     ## out

  PjrtBufferDimensionsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    buffer*: PjrtBufferRaw
    dims*: ptr int64            ## out — owned by buffer
    numDims*: csize_t           ## out

  PjrtBufferOnDeviceSizeInBytesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    buffer*: PjrtBufferRaw
    onDeviceSizeInBytes*: csize_t   ## out

  # ----- Compile + Execute -----

  PjrtExecutableRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_Executable`.

  PjrtLoadedExecutableRaw* = distinct pointer
    ## Opaque pointer to a `PJRT_LoadedExecutable`.

  PjrtProgram* {.bycopy.} = object
    ## Mirrors `PJRT_Program`. `format` is one of `"hlo"`,
    ## `"hlo_with_config"`, or `"mlir"`.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    code*: cstring               ## in/out
    codeSize*: csize_t
    format*: cstring
    formatSize*: csize_t
    memory*: PjrtMemoryRaw

  PjrtClientCompileArgs* {.bycopy.} = object
    ## Mirrors `PJRT_Client_Compile_Args`. We always pass a serialized
    ## (or empty) `CompileOptionsProto` via `compileOptions`/`compileOptionsSize`.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    program*: ptr PjrtProgram
    compileOptions*: cstring
    compileOptionsSize*: csize_t
    executable*: PjrtLoadedExecutableRaw   ## out

  PjrtExecutableDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw

  PjrtLoadedExecutableDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw

  PjrtLoadedExecutableGetExecutableArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    loadedExecutable*: PjrtLoadedExecutableRaw
    executable*: PjrtExecutableRaw           ## out

  PjrtDeviceAssignmentDeleter* =
    proc (assignment: PjrtDeviceAssignmentSerializedRaw) {.cdecl.}

  PjrtLoadedExecutableGetDeviceAssignmentArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    serializedBytes*: cstring                         ## out
    serializedBytesSize*: csize_t                     ## out
    serializedDeviceAssignment*: PjrtDeviceAssignmentSerializedRaw
    serializedDeviceAssignmentDeleter*: PjrtDeviceAssignmentDeleter

  PjrtExecutableNameArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    executableName*: cstring     ## out — owned by executable
    executableNameSize*: csize_t ## out

  PjrtExecutableNumReplicasArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numReplicas*: csize_t        ## out

  PjrtExecutableNumPartitionsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numPartitions*: csize_t      ## out

  PjrtLogicalDeviceIds* {.bycopy.} = object
    replica*: cint
    partition*: cint

  PjrtLoadedExecutableAddressableDevicesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    addressableDevices*: ptr UncheckedArray[PjrtDeviceHandle]  ## out
    numAddressableDevices*: csize_t                            ## out

  PjrtLoadedExecutableAddressableDeviceLogicalIdsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    addressableDeviceLogicalIds*: ptr UncheckedArray[PjrtLogicalDeviceIds] ## out
    numAddressableDeviceLogicalIds*: csize_t ## out

  PjrtExecutableOptimizedProgramArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    program*: ptr PjrtProgram    ## in/out

  PjrtLoadedExecutableDeleteArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw

  PjrtLoadedExecutableIsDeletedArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    isDeleted*: bool             ## out

  PjrtExecutableNumOutputsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numOutputs*: csize_t                     ## out

  PjrtExecutableSizeOfGeneratedCodeInBytesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    sizeInBytes*: int64          ## out

  PjrtExecutableFingerprintArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    executableFingerprint*: cstring       ## out — owned by executable
    executableFingerprintSize*: csize_t   ## out

  PjrtExecutableGetCostAnalysisArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numProperties*: csize_t                         ## out
    properties*: ptr UncheckedArray[PjrtNamedValue] ## out

  PjrtExecutableGetCompiledMemoryStatsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    generatedCodeSizeInBytes*: int64
    argumentSizeInBytes*: int64
    outputSizeInBytes*: int64
    aliasSizeInBytes*: int64
    tempSizeInBytes*: int64
    hostGeneratedCodeSizeInBytes*: int64
    hostArgumentSizeInBytes*: int64
    hostOutputSizeInBytes*: int64
    hostAliasSizeInBytes*: int64
    hostTempSizeInBytes*: int64
    peakMemoryInBytes*: int64
    totalSizeInBytes*: int64
    totalAllocationBytes*: int64
    indefiniteAllocations*: int64
    peakUnpaddedHeapBytes*: int64

  PjrtCompiledMemoryStats* = object
    ## Compile-time memory estimate for an executable.
    generatedCodeSizeInBytes*: int64
    argumentSizeInBytes*: int64
    outputSizeInBytes*: int64
    aliasSizeInBytes*: int64
    tempSizeInBytes*: int64
    hostGeneratedCodeSizeInBytes*: int64
    hostArgumentSizeInBytes*: int64
    hostOutputSizeInBytes*: int64
    hostAliasSizeInBytes*: int64
    hostTempSizeInBytes*: int64
    peakMemoryInBytes*: int64
    totalSizeInBytes*: int64
    totalAllocationBytes*: int64
    indefiniteAllocations*: int64
    peakUnpaddedHeapBytes*: int64

  PjrtExecutableOutputElementTypesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    outputTypes*: ptr UncheckedArray[PjrtBufferType]  ## out
    numOutputTypes*: csize_t                          ## out

  PjrtExecutableOutputDimensionsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numOutputs*: csize_t
    dims*: ptr UncheckedArray[int64]     ## out
    dimSizes*: ptr UncheckedArray[csize_t] ## out

  PjrtExecutableParameterMemoryKindsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numParameters*: csize_t
    memoryKinds*: ptr UncheckedArray[cstring]       ## out
    memoryKindSizes*: ptr UncheckedArray[csize_t]   ## out

  PjrtExecutableOutputMemoryKindsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    numOutputs*: csize_t
    memoryKinds*: ptr UncheckedArray[cstring]       ## out
    memoryKindSizes*: ptr UncheckedArray[csize_t]   ## out

  PjrtSerializedExecutableDeleter* =
    proc (executable: PjrtSerializedExecutableRaw) {.cdecl.}

  PjrtExecutableSerializeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    serializedBytes*: cstring
    serializedBytesSize*: csize_t
    serializedExecutable*: PjrtSerializedExecutableRaw
    serializedExecutableDeleter*: PjrtSerializedExecutableDeleter

  PjrtSerializedCompileOptionsDeleter* =
    proc (options: PjrtSerializedCompileOptionsRaw) {.cdecl.}

  PjrtExecutableGetCompileOptionsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtExecutableRaw
    serializedBytes*: cstring
    serializedBytesSize*: csize_t
    serializedCompileOptions*: PjrtSerializedCompileOptionsRaw
    serializedCompileOptionsDeleter*: PjrtSerializedCompileOptionsDeleter

  PjrtExecutableDeserializeAndLoadArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    client*: PjrtClientHandle
    serializedExecutable*: cstring
    serializedExecutableSize*: csize_t
    loadedExecutable*: PjrtLoadedExecutableRaw
    overriddenSerializedCompileOptions*: cstring
    overriddenSerializedCompileOptionsSize*: csize_t

  PjrtLoadedExecutableFingerprintArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    executableFingerprint*: cstring       ## out — owned by executable
    executableFingerprintSize*: csize_t   ## out

  PjrtExecuteOptions* {.bycopy.} = object
    ## Mirrors `PJRT_ExecuteOptions`. We zero-initialize for default
    ## single-device execution; only `structSize` needs to be set.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    sendCallbacks*: pointer        ## PJRT_SendCallbackInfo**
    recvCallbacks*: pointer        ## PJRT_RecvCallbackInfo**
    numSendOps*: csize_t
    numRecvOps*: csize_t
    launchId*: cint
    nonDonatableInputIndices*: ptr int64
    numNonDonatableInputIndices*: csize_t
    context*: pointer              ## PJRT_ExecuteContext*
    callLocation*: cstring
    numTasks*: csize_t
    taskIds*: ptr cint
    incarnationIds*: ptr int64
    multiSliceConfig*: pointer
    useMajorToMinorDataLayoutForCallbacks*: bool

  PjrtLoadedExecutableExecuteArgs* {.bycopy.} = object
    ## Mirrors `PJRT_LoadedExecutable_Execute_Args`. For single-device
    ## execution we set `numDevices = 1`, `executeDevice` to the target
    ## device, and the outer dimension of `argumentLists`/`outputLists` is 1.
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    executable*: PjrtLoadedExecutableRaw
    options*: ptr PjrtExecuteOptions
    argumentLists*: ptr ptr PjrtBufferRaw    ## [numDevices][numArgs]
    numDevices*: csize_t
    numArgs*: csize_t
    outputLists*: ptr ptr PjrtBufferRaw      ## [numDevices][numOutputs] — caller-allocated
    deviceCompleteEvents*: ptr PjrtEventRaw  ## optional
    executeDevice*: PjrtDeviceHandle

  PjrtTopologyDescriptionDestroyArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw

  PjrtTopologyDescriptionPlatformVersionArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    platformVersion*: cstring      ## out — owned by topology
    platformVersionSize*: csize_t  ## out

  PjrtTopologyDescriptionPlatformNameArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    platformName*: cstring         ## out — owned by topology
    platformNameSize*: csize_t     ## out

  PjrtTopologyDescriptionGetDeviceDescriptionsArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    descriptions*: ptr UncheckedArray[PjrtDeviceDescriptionRaw] ## out
    numDescriptions*: csize_t                                   ## out

  PjrtSerializedTopologyDeleter* =
    proc (topology: PjrtSerializedTopologyRaw) {.cdecl.}

  PjrtTopologyDescriptionSerializeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    serializedBytes*: cstring
    serializedBytesSize*: csize_t
    serializedTopology*: PjrtSerializedTopologyRaw
    serializedTopologyDeleter*: PjrtSerializedTopologyDeleter

  PjrtTopologyDescriptionDeserializeArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    serializedTopology*: cstring
    serializedTopologySize*: csize_t
    topology*: PjrtTopologyDescriptionRaw ## out — caller-owned

  PjrtTopologyDescriptionAttributesArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    attributes*: ptr UncheckedArray[PjrtNamedValue] ## out
    numAttributes*: csize_t                         ## out

  PjrtTopologyDescriptionFingerprintArgs* {.bycopy.} = object
    structSize*: csize_t
    extensionStart*: PjrtExtensionBaseRaw
    topology*: PjrtTopologyDescriptionRaw
    fingerprint*: uint64 ## out

func isNil*(h: PjrtEventRaw): bool {.borrow.}
func isNil*(h: PjrtDeviceDescriptionRaw): bool {.borrow.}
func isNil*(h: PjrtExecutableRaw): bool {.borrow.}
func isNil*(h: PjrtLoadedExecutableRaw): bool {.borrow.}

# ----- Typed proc aliases for the function-pointer slots we cast to -----

type
  PjrtFnErrorDestroy = proc (args: ptr PjrtErrorDestroyArgs) {.cdecl.}
  PjrtFnErrorMessage = proc (args: ptr PjrtErrorMessageArgs) {.cdecl.}
  PjrtFnErrorGetCode =
    proc (args: ptr PjrtErrorGetCodeArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnErrorForEachPayload =
    proc (args: ptr PjrtErrorForEachPayloadArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnPluginInitialize =
    proc (args: ptr PjrtPluginInitializeArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnPluginAttributes =
    proc (args: ptr PjrtPluginAttributesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientCreate =
    proc (args: ptr PjrtClientCreateArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientDestroy =
    proc (args: ptr PjrtClientDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientPlatformName =
    proc (args: ptr PjrtClientPlatformNameArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientProcessIndex =
    proc (args: ptr PjrtClientProcessIndexArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientPlatformVersion =
    proc (args: ptr PjrtClientPlatformVersionArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientTopologyDescription =
    proc (args: ptr PjrtClientTopologyDescriptionArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientDevices =
    proc (args: ptr PjrtClientDevicesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientAddressableDevices =
    proc (args: ptr PjrtClientAddressableDevicesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientLookupDevice =
    proc (args: ptr PjrtClientLookupDeviceArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientLookupAddressableDevice =
    proc (args: ptr PjrtClientLookupAddressableDeviceArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientUpdateGlobalProcessInfo =
    proc (args: ptr PjrtClientUpdateGlobalProcessInfoArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnClientAddressableMemories =
    proc (args: ptr PjrtClientAddressableMemoriesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientDefaultDeviceAssignment =
    proc (args: ptr PjrtClientDefaultDeviceAssignmentArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnEventDestroy =
    proc (args: ptr PjrtEventDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnEventAwait =
    proc (args: ptr PjrtEventAwaitArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceGetDescription =
    proc (args: ptr PjrtDeviceGetDescriptionArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceDescriptionKind =
    proc (args: ptr PjrtDeviceDescriptionKindArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceDescriptionId =
    proc (args: ptr PjrtDeviceDescriptionIdArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceDescriptionProcessIndex =
    proc (args: ptr PjrtDeviceDescriptionProcessIndexArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnDeviceDescriptionDebugString =
    proc (args: ptr PjrtDeviceDescriptionDebugStringArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnDeviceDescriptionToString =
    proc (args: ptr PjrtDeviceDescriptionToStringArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceDescriptionAttributes =
    proc (args: ptr PjrtDeviceDescriptionAttributesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceIsAddressable =
    proc (args: ptr PjrtDeviceIsAddressableArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceLocalHardwareId =
    proc (args: ptr PjrtDeviceLocalHardwareIdArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceAddressableMemories =
    proc (args: ptr PjrtDeviceAddressableMemoriesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceDefaultMemory =
    proc (args: ptr PjrtDeviceDefaultMemoryArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceMemoryStats =
    proc (args: ptr PjrtDeviceMemoryStatsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnDeviceClearMemoryStats =
    proc (args: ptr PjrtDeviceClearMemoryStatsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryId =
    proc (args: ptr PjrtMemoryIdArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryKind =
    proc (args: ptr PjrtMemoryKindArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryKindId =
    proc (args: ptr PjrtMemoryKindIdArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryDebugString =
    proc (args: ptr PjrtMemoryDebugStringArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryToString =
    proc (args: ptr PjrtMemoryToStringArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnMemoryAddressableByDevices =
    proc (args: ptr PjrtMemoryAddressableByDevicesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientBufferFromHostBuffer =
    proc (args: ptr PjrtClientBufferFromHostBufferArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnBufferDestroy =
    proc (args: ptr PjrtBufferDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnBufferToHostBuffer =
    proc (args: ptr PjrtBufferToHostBufferArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnBufferElementType =
    proc (args: ptr PjrtBufferElementTypeArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnBufferDimensions =
    proc (args: ptr PjrtBufferDimensionsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnBufferOnDeviceSizeInBytes =
    proc (args: ptr PjrtBufferOnDeviceSizeInBytesArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnClientCompile =
    proc (args: ptr PjrtClientCompileArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableDestroy =
    proc (args: ptr PjrtExecutableDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableDestroy =
    proc (args: ptr PjrtLoadedExecutableDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableGetExecutable =
    proc (args: ptr PjrtLoadedExecutableGetExecutableArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableGetDeviceAssignment =
    proc (args: ptr PjrtLoadedExecutableGetDeviceAssignmentArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnExecutableName =
    proc (args: ptr PjrtExecutableNameArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableNumReplicas =
    proc (args: ptr PjrtExecutableNumReplicasArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableNumPartitions =
    proc (args: ptr PjrtExecutableNumPartitionsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableAddressableDevices =
    proc (args: ptr PjrtLoadedExecutableAddressableDevicesArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnLoadedExecutableAddressableDeviceLogicalIds =
    proc (args: ptr PjrtLoadedExecutableAddressableDeviceLogicalIdsArgs):
      PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableOptimizedProgram =
    proc (args: ptr PjrtExecutableOptimizedProgramArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableDelete =
    proc (args: ptr PjrtLoadedExecutableDeleteArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableIsDeleted =
    proc (args: ptr PjrtLoadedExecutableIsDeletedArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableNumOutputs =
    proc (args: ptr PjrtExecutableNumOutputsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableSizeOfGeneratedCodeInBytes =
    proc (args: ptr PjrtExecutableSizeOfGeneratedCodeInBytesArgs):
      PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableFingerprint =
    proc (args: ptr PjrtExecutableFingerprintArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableGetCostAnalysis =
    proc (args: ptr PjrtExecutableGetCostAnalysisArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableGetCompiledMemoryStats =
    proc (args: ptr PjrtExecutableGetCompiledMemoryStatsArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnExecutableOutputElementTypes =
    proc (args: ptr PjrtExecutableOutputElementTypesArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnExecutableOutputDimensions =
    proc (args: ptr PjrtExecutableOutputDimensionsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableParameterMemoryKinds =
    proc (args: ptr PjrtExecutableParameterMemoryKindsArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnExecutableOutputMemoryKinds =
    proc (args: ptr PjrtExecutableOutputMemoryKindsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableSerialize =
    proc (args: ptr PjrtExecutableSerializeArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableGetCompileOptions =
    proc (args: ptr PjrtExecutableGetCompileOptionsArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnExecutableDeserializeAndLoad =
    proc (args: ptr PjrtExecutableDeserializeAndLoadArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableFingerprint =
    proc (args: ptr PjrtLoadedExecutableFingerprintArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnLoadedExecutableExecute =
    proc (args: ptr PjrtLoadedExecutableExecuteArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnTopologyDescriptionDestroy =
    proc (args: ptr PjrtTopologyDescriptionDestroyArgs): PjrtErrorRaw {.cdecl.}
  PjrtFnTopologyDescriptionPlatformVersion =
    proc (args: ptr PjrtTopologyDescriptionPlatformVersionArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnTopologyDescriptionPlatformName =
    proc (args: ptr PjrtTopologyDescriptionPlatformNameArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnTopologyDescriptionGetDeviceDescriptions =
    proc (args: ptr PjrtTopologyDescriptionGetDeviceDescriptionsArgs):
      PjrtErrorRaw {.cdecl.}
  PjrtFnTopologyDescriptionSerialize =
    proc (args: ptr PjrtTopologyDescriptionSerializeArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnTopologyDescriptionDeserialize =
    proc (args: ptr PjrtTopologyDescriptionDeserializeArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnTopologyDescriptionAttributes =
    proc (args: ptr PjrtTopologyDescriptionAttributesArgs): PjrtErrorRaw
      {.cdecl.}
  PjrtFnTopologyDescriptionFingerprint =
    proc (args: ptr PjrtTopologyDescriptionFingerprintArgs): PjrtErrorRaw
      {.cdecl.}

proc copySizedString*(data: cstring; size: csize_t): string =
  ## Copies a PJRT `(char*, size)` pair into a Nim string. PJRT uses sized
  ## byte strings for serialized protos and not all of them are
  ## null-terminated.
  if data.isNil or size == 0:
    return ""
  result = newString(int size)
  copyMem(addr result[0], cast[pointer](data), int size)

proc namedValueName*(nv: PjrtNamedValue): string =
  ## Copies the name of a `PJRT_NamedValue`.
  copySizedString(nv.name, nv.nameSize)

proc namedValueValue*(nv: PjrtNamedValue): string =
  ## Formats a `PJRT_NamedValue` as a string for diagnostics and public
  ## metadata helpers. Int64-list values are rendered losslessly.
  case nv.kind
  of nvString:
    result = copySizedString(nv.payload.stringValue, nv.valueSize)
  of nvInt64:
    result = $nv.payload.int64Value
  of nvInt64List:
    let p = nv.payload.int64ArrayValue
    if p.isNil or nv.valueSize == 0:
      return "[]"
    result = "["
    for i in 0 ..< int nv.valueSize:
      if i > 0: result.add ", "
      result.add $p[i]
    result.add "]"
  of nvFloat:
    result = $nv.payload.floatValue
  of nvBool:
    result = $nv.payload.boolValue

proc namedValuesToStrings*(values: ptr UncheckedArray[PjrtNamedValue];
    len: int): seq[tuple[name, value: string]] =
  ## Copies an array of `PJRT_NamedValue` into owned Nim strings.
  result = newSeqOfCap[tuple[name, value: string]](len)
  if values.isNil: return
  for i in 0 ..< len:
    let nv = values[i]
    result.add((namedValueName(nv), namedValueValue(nv)))

template apiFunctionAvailable*(api: PjrtApiHandle; field: untyped): bool =
  ## True when `field` is inside the plugin-reported `PJRT_Api.struct_size`
  ## and the corresponding function pointer is non-nil. Use before calling
  ## newer trailing PJRT APIs against older plugins.
  ((not (api).isNil) and
    int(apiPtr(api).structSize) >= offsetof(PjrtApi, field) + sizeof(pointer) and
    not apiPtr(api).field.isNil)

template requireApiFn(api: PjrtApiHandle; field: untyped; fnType: typedesc;
    cName: string): untyped =
  block:
    if (api).isNil:
      raisePjrt(cName & ": nil PJRT API handle")
    const fieldEnd = offsetof(PjrtApi, field) + sizeof(pointer)
    if int(apiPtr(api).structSize) < fieldEnd:
      raisePjrt(cName & " is not available in this PJRT API table " &
        "(struct_size=" & $int(apiPtr(api).structSize) & ", needs >= " &
        $fieldEnd & ")")
    let raw = apiPtr(api).field
    if raw.isNil:
      raisePjrt("PJRT plugin does not export " & cName)
    cast[fnType](raw)

# ----- Error helpers ----------------------------------------------------

proc destroyError*(api: PjrtApiHandle; err: PjrtErrorRaw) =
  ## Calls the plugin's `PJRT_Error_Destroy` on `err`. Safe to call with a
  ## nil `err` (no-op).
  if err.isNil: return
  let fn = cast[PjrtFnErrorDestroy](apiPtr(api).fnErrorDestroy)
  var args = PjrtErrorDestroyArgs(
    structSize: csize_t sizeof(PjrtErrorDestroyArgs),
    error: err)
  fn(addr args)

proc errorMessage*(api: PjrtApiHandle; err: PjrtErrorRaw): string =
  ## Returns the message attached to `err`, or `""` if `err` is nil. Does
  ## not destroy `err`.
  if err.isNil: return ""
  let fn = cast[PjrtFnErrorMessage](apiPtr(api).fnErrorMessage)
  var args = PjrtErrorMessageArgs(
    structSize: csize_t sizeof(PjrtErrorMessageArgs),
    error: err)
  fn(addr args)
  if args.message.isNil or args.messageSize == 0: ""
  else: $cast[cstring](args.message)

proc errorCode*(api: PjrtApiHandle; err: PjrtErrorRaw): PjrtErrorCode =
  ## Returns the plugin-reported PJRT error code. Older plugins without the
  ## slot are treated as `pecUnknown` so error translation still succeeds.
  if err.isNil: return pecOk
  if not apiFunctionAvailable(api, fnErrorGetCode):
    return pecUnknown
  let fn = cast[PjrtFnErrorGetCode](apiPtr(api).fnErrorGetCode)
  var args = PjrtErrorGetCodeArgs(
    structSize: csize_t sizeof(PjrtErrorGetCodeArgs),
    error: err)
  let nested = fn(addr args)
  if not nested.isNil:
    destroyError(api, nested)
    return pecUnknown
  args.code

proc payloadVisitor(key: cstring; keySize: csize_t; value: cstring;
    valueSize: csize_t; userArg: pointer) {.cdecl.} =
  let payloadSeq = cast[ptr seq[tuple[key, value: string]]](userArg)
  payloadSeq[].add((
    copySizedString(key, keySize),
    copySizedString(value, valueSize)))

proc errorPayloads*(api: PjrtApiHandle; err: PjrtErrorRaw):
    seq[tuple[key, value: string]] =
  ## Copies all typed status payloads attached to `err`, if the plugin
  ## supports `PJRT_Error_ForEachPayload`.
  result = @[]
  if err.isNil or not apiFunctionAvailable(api, fnErrorForEachPayload):
    return
  let fn = cast[PjrtFnErrorForEachPayload](
    apiPtr(api).fnErrorForEachPayload)
  var args = PjrtErrorForEachPayloadArgs(
    structSize: csize_t sizeof(PjrtErrorForEachPayloadArgs),
    error: err,
    visitor: payloadVisitor,
    userArg: addr result)
  let nested = fn(addr args)
  if not nested.isNil:
    destroyError(api, nested)

proc checkErr*(api: PjrtApiHandle; err: PjrtErrorRaw; ctx: string) =
  ## Translates a `PjrtErrorRaw` returned by a PJRT call into a
  ## `PjrtError` exception. No-op when `err` is nil. Always destroys
  ## `err` before raising.
  if err.isNil: return
  let msg = errorMessage(api, err)
  let code = errorCode(api, err)
  let payloads = errorPayloads(api, err)
  destroyError(api, err)
  raisePjrt(ctx & ": " & msg, code, payloads)

# ----- Versioning + plugin lifecycle ------------------------------------

func apiVersion*(api: PjrtApiHandle): tuple[major, minor: int] {.inline.} =
  ## Reads the `pjrt_api_version` field of the API table. Useful for
  ## branching on plugin compatibility before calling later trailing fields.
  let v = apiPtr(api).pjrtApiVersion
  (int v.majorVersion, int v.minorVersion)

func apiStructSize*(api: PjrtApiHandle): int {.inline.} =
  ## Returns the `struct_size` the plugin reported for its API table. Use
  ## with `offsetof(PjrtApi, fnX)` to verify a given trailing field is
  ## present before invoking it.
  int apiPtr(api).structSize

proc pluginInitialize*(api: PjrtApiHandle) =
  ## Calls `PJRT_Plugin_Initialize`. Must be invoked exactly once per
  ## plugin handle before any other API call.
  let fn = cast[PjrtFnPluginInitialize](apiPtr(api).fnPluginInitialize)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Plugin_Initialize")
  var args = PjrtPluginInitializeArgs(
    structSize: csize_t sizeof(PjrtPluginInitializeArgs))
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Plugin_Initialize")

proc pluginAttributes*(api: PjrtApiHandle): seq[tuple[name, value: string]] =
  ## Returns the plugin attributes as `(name, value)` strings. Non-string
  ## values are formatted as `<int64=…>`, `<bool=…>`, etc. for diagnostic
  ## use; this v1 helper is intended for `echo`/logging, not parsing.
  let fn = cast[PjrtFnPluginAttributes](apiPtr(api).fnPluginAttributes)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Plugin_Attributes")
  var args = PjrtPluginAttributesArgs(
    structSize: csize_t sizeof(PjrtPluginAttributesArgs))
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Plugin_Attributes")
  let n = int args.numAttributes
  result = namedValuesToStrings(args.attributes, n)

# ----- Event helpers ----------------------------------------------------

proc destroyEvent*(api: PjrtApiHandle; ev: PjrtEventRaw) =
  ## Calls `PJRT_Event_Destroy` on `ev`. Safe with a nil event.
  if ev.isNil: return
  let fn = cast[PjrtFnEventDestroy](apiPtr(api).fnEventDestroy)
  var args = PjrtEventDestroyArgs(
    structSize: csize_t sizeof(PjrtEventDestroyArgs),
    event: ev)
  let err = fn(addr args)
  # Destroy errors are intentionally silenced — callers in destruction
  # paths cannot observe them.
  if not err.isNil: destroyError(api, err)

proc awaitEvent*(api: PjrtApiHandle; ev: PjrtEventRaw; ctx: string) =
  ## Blocks until `ev` is ready, then translates any reported error into
  ## a `PjrtError`. Always destroys `ev` afterwards.
  if ev.isNil: return
  let fn = cast[PjrtFnEventAwait](apiPtr(api).fnEventAwait)
  var args = PjrtEventAwaitArgs(
    structSize: csize_t sizeof(PjrtEventAwaitArgs),
    event: ev)
  let err = fn(addr args)
  destroyEvent(api, ev)
  checkErr(api, err, ctx)

# ----- Client lifecycle -------------------------------------------------

proc clientCreate*(api: PjrtApiHandle): PjrtClientHandle =
  ## Creates a new PJRT client with no extra options and no kv-store
  ## callbacks (single-process use). Caller is responsible for calling
  ## `clientDestroy` exactly once.
  let fn = cast[PjrtFnClientCreate](apiPtr(api).fnClientCreate)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Client_Create")
  var args = PjrtClientCreateArgs(
    structSize: csize_t sizeof(PjrtClientCreateArgs))
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_Create")
  args.client

proc clientDestroy*(api: PjrtApiHandle; client: PjrtClientHandle) =
  ## Shuts down `client`. Safe with a nil client. Errors from destroy
  ## are translated into `PjrtError`.
  if client.isNil: return
  let fn = cast[PjrtFnClientDestroy](apiPtr(api).fnClientDestroy)
  var args = PjrtClientDestroyArgs(
    structSize: csize_t sizeof(PjrtClientDestroyArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_Destroy")

proc clientAddressableDevices*(api: PjrtApiHandle;
    client: PjrtClientHandle): seq[PjrtDeviceHandle] =
  ## Returns the list of devices this client can submit work to. The
  ## returned handles are owned by the client and remain valid until
  ## `clientDestroy` is called.
  let fn = cast[PjrtFnClientAddressableDevices](
    apiPtr(api).fnClientAddressableDevices)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Client_AddressableDevices")
  var args = PjrtClientAddressableDevicesArgs(
    structSize: csize_t sizeof(PjrtClientAddressableDevicesArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_AddressableDevices")
  let n = int args.numAddressableDevices
  result = newSeqOfCap[PjrtDeviceHandle](n)
  for i in 0 ..< n:
    result.add args.addressableDevices[i]

proc clientPlatformName*(api: PjrtApiHandle;
    client: PjrtClientHandle): string =
  ## Returns the PJRT platform name for `client`.
  let fn = requireApiFn(api, fnClientPlatformName, PjrtFnClientPlatformName,
    "PJRT_Client_PlatformName")
  var args = PjrtClientPlatformNameArgs(
    structSize: csize_t sizeof(PjrtClientPlatformNameArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_PlatformName")
  copySizedString(args.platformName, args.platformNameSize)

proc clientProcessIndex*(api: PjrtApiHandle;
    client: PjrtClientHandle): int =
  ## Returns the global process index of `client`.
  let fn = requireApiFn(api, fnClientProcessIndex, PjrtFnClientProcessIndex,
    "PJRT_Client_ProcessIndex")
  var args = PjrtClientProcessIndexArgs(
    structSize: csize_t sizeof(PjrtClientProcessIndexArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_ProcessIndex")
  int args.processIndex

proc clientPlatformVersion*(api: PjrtApiHandle;
    client: PjrtClientHandle): string =
  ## Returns platform-specific version details for `client`.
  let fn = requireApiFn(api, fnClientPlatformVersion,
    PjrtFnClientPlatformVersion, "PJRT_Client_PlatformVersion")
  var args = PjrtClientPlatformVersionArgs(
    structSize: csize_t sizeof(PjrtClientPlatformVersionArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_PlatformVersion")
  copySizedString(args.platformVersion, args.platformVersionSize)

proc clientTopologyDescription*(api: PjrtApiHandle;
    client: PjrtClientHandle): PjrtTopologyDescriptionRaw =
  ## Returns the client-owned topology description for `client`.
  let fn = requireApiFn(api, fnClientTopologyDescription,
    PjrtFnClientTopologyDescription, "PJRT_Client_TopologyDescription")
  var args = PjrtClientTopologyDescriptionArgs(
    structSize: csize_t sizeof(PjrtClientTopologyDescriptionArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_TopologyDescription")
  args.topology

proc clientDevices*(api: PjrtApiHandle;
    client: PjrtClientHandle): seq[PjrtDeviceHandle] =
  ## Returns all devices visible to the runtime, including non-addressable
  ## devices on multi-host platforms.
  let fn = requireApiFn(api, fnClientDevices, PjrtFnClientDevices,
    "PJRT_Client_Devices")
  var args = PjrtClientDevicesArgs(
    structSize: csize_t sizeof(PjrtClientDevicesArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_Devices")
  result = newSeqOfCap[PjrtDeviceHandle](int args.numDevices)
  for i in 0 ..< int args.numDevices:
    result.add args.devices[i]

proc clientLookupDevice*(api: PjrtApiHandle; client: PjrtClientHandle;
    id: int): PjrtDeviceHandle =
  ## Looks up a visible device by global PJRT id.
  let fn = requireApiFn(api, fnClientLookupDevice, PjrtFnClientLookupDevice,
    "PJRT_Client_LookupDevice")
  var args = PjrtClientLookupDeviceArgs(
    structSize: csize_t sizeof(PjrtClientLookupDeviceArgs),
    client: client,
    id: cint id)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_LookupDevice")
  args.device

proc clientLookupAddressableDevice*(api: PjrtApiHandle;
    client: PjrtClientHandle; localHardwareId: int): PjrtDeviceHandle =
  ## Looks up an addressable device by local hardware id.
  let fn = requireApiFn(api, fnClientLookupAddressableDevice,
    PjrtFnClientLookupAddressableDevice,
    "PJRT_Client_LookupAddressableDevice")
  var args = PjrtClientLookupAddressableDeviceArgs(
    structSize: csize_t sizeof(PjrtClientLookupAddressableDeviceArgs),
    client: client,
    localHardwareId: cint localHardwareId)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_LookupAddressableDevice")
  args.addressableDevice

proc clientAddressableMemories*(api: PjrtApiHandle;
    client: PjrtClientHandle): seq[PjrtMemoryRaw] =
  ## Returns all memories the client can directly transfer to/from.
  let fn = requireApiFn(api, fnClientAddressableMemories,
    PjrtFnClientAddressableMemories, "PJRT_Client_AddressableMemories")
  var args = PjrtClientAddressableMemoriesArgs(
    structSize: csize_t sizeof(PjrtClientAddressableMemoriesArgs),
    client: client)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_AddressableMemories")
  result = newSeqOfCap[PjrtMemoryRaw](int args.numAddressableMemories)
  for i in 0 ..< int args.numAddressableMemories:
    result.add args.addressableMemories[i]

proc clientDefaultDeviceAssignment*(api: PjrtApiHandle;
    client: PjrtClientHandle; numReplicas, numPartitions: int): seq[int] =
  ## Computes the default replica/partition to device assignment as a flat
  ## row-major `[replica, partition]` array of device ids.
  if numReplicas <= 0 or numPartitions <= 0:
    raisePjrt("PJRT_Client_DefaultDeviceAssignment: replica and partition " &
      "counts must be positive")
  let fn = requireApiFn(api, fnClientDefaultDeviceAssignment,
    PjrtFnClientDefaultDeviceAssignment,
    "PJRT_Client_DefaultDeviceAssignment")
  var raw = newSeq[cint](numReplicas * numPartitions)
  var args = PjrtClientDefaultDeviceAssignmentArgs(
    structSize: csize_t sizeof(PjrtClientDefaultDeviceAssignmentArgs),
    client: client,
    numReplicas: cint numReplicas,
    numPartitions: cint numPartitions,
    defaultAssignmentSize: csize_t raw.len,
    defaultAssignment: addr raw[0])
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_DefaultDeviceAssignment")
  result = newSeq[int](raw.len)
  for i, v in raw:
    result[i] = int v

proc clientUpdateGlobalProcessInfo*(api: PjrtApiHandle;
    client: PjrtClientHandle; infos: openArray[PjrtProcessInfo]) =
  ## Updates multi-host process state metadata used by collective runtimes.
  let fn = requireApiFn(api, fnClientUpdateGlobalProcessInfo,
    PjrtFnClientUpdateGlobalProcessInfo,
    "PJRT_Client_UpdateGlobalProcessInfo")
  var local = @infos
  for i in 0 ..< local.len:
    local[i].structSize = csize_t sizeof(PjrtProcessInfo)
  var args = PjrtClientUpdateGlobalProcessInfoArgs(
    structSize: csize_t sizeof(PjrtClientUpdateGlobalProcessInfoArgs),
    client: client,
    processInfos:
      if local.len == 0: nil
      else: cast[ptr PjrtProcessInfo](addr local[0]),
    numProcessInfos: csize_t local.len)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_UpdateGlobalProcessInfo")

# ----- Device description helpers ---------------------------------------

proc deviceDescription*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): PjrtDeviceDescriptionRaw =
  ## Returns the description handle for `dev`. The description is owned
  ## by the device's client and must not be freed by the caller.
  let fn = cast[PjrtFnDeviceGetDescription](
    apiPtr(api).fnDeviceGetDescription)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Device_GetDescription")
  var args = PjrtDeviceGetDescriptionArgs(
    structSize: csize_t sizeof(PjrtDeviceGetDescriptionArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_GetDescription")
  args.deviceDescription

proc descriptionKind*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): string =
  ## Returns the platform-specific kind string of `desc`
  ## (e.g. `"cpu"`, `"Tesla V100-SXM2-16GB"`).
  let fn = cast[PjrtFnDeviceDescriptionKind](
    apiPtr(api).fnDeviceDescriptionKind)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_DeviceDescription_Kind")
  var args = PjrtDeviceDescriptionKindArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionKindArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_Kind")
  if args.deviceKind.isNil or args.deviceKindSize == 0: ""
  else: $cast[cstring](args.deviceKind)

proc descriptionId*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): int =
  ## Returns the unique device id of `desc`.
  let fn = cast[PjrtFnDeviceDescriptionId](
    apiPtr(api).fnDeviceDescriptionId)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_DeviceDescription_Id")
  var args = PjrtDeviceDescriptionIdArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionIdArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_Id")
  int args.id

proc descriptionProcessIndex*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): int =
  ## Returns the process index owning `desc`.
  let fn = requireApiFn(api, fnDeviceDescriptionProcessIndex,
    PjrtFnDeviceDescriptionProcessIndex,
    "PJRT_DeviceDescription_ProcessIndex")
  var args = PjrtDeviceDescriptionProcessIndexArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionProcessIndexArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_ProcessIndex")
  int args.processIndex

proc descriptionDebugString*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): string =
  ## Returns the verbose debug string for `desc`.
  let fn = requireApiFn(api, fnDeviceDescriptionDebugString,
    PjrtFnDeviceDescriptionDebugString,
    "PJRT_DeviceDescription_DebugString")
  var args = PjrtDeviceDescriptionDebugStringArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionDebugStringArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_DebugString")
  copySizedString(args.debugString, args.debugStringSize)

proc descriptionToString*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): string =
  ## Returns the user-facing string for `desc`.
  let fn = requireApiFn(api, fnDeviceDescriptionToString,
    PjrtFnDeviceDescriptionToString, "PJRT_DeviceDescription_ToString")
  var args = PjrtDeviceDescriptionToStringArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionToStringArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_ToString")
  copySizedString(args.toString, args.toStringSize)

proc descriptionAttributes*(api: PjrtApiHandle;
    desc: PjrtDeviceDescriptionRaw): seq[tuple[name, value: string]] =
  ## Returns device-description attributes as owned strings.
  let fn = requireApiFn(api, fnDeviceDescriptionAttributes,
    PjrtFnDeviceDescriptionAttributes, "PJRT_DeviceDescription_Attributes")
  var args = PjrtDeviceDescriptionAttributesArgs(
    structSize: csize_t sizeof(PjrtDeviceDescriptionAttributesArgs),
    deviceDescription: desc)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_DeviceDescription_Attributes")
  namedValuesToStrings(args.attributes, int args.numAttributes)

proc deviceIsAddressable*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): bool =
  ## Returns whether `dev` can execute work from this client.
  let fn = requireApiFn(api, fnDeviceIsAddressable, PjrtFnDeviceIsAddressable,
    "PJRT_Device_IsAddressable")
  var args = PjrtDeviceIsAddressableArgs(
    structSize: csize_t sizeof(PjrtDeviceIsAddressableArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_IsAddressable")
  args.isAddressable

proc deviceLocalHardwareId*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): int =
  ## Returns the device-local hardware id, or -1 if the plugin leaves it
  ## undefined.
  let fn = requireApiFn(api, fnDeviceLocalHardwareId,
    PjrtFnDeviceLocalHardwareId, "PJRT_Device_LocalHardwareId")
  var args = PjrtDeviceLocalHardwareIdArgs(
    structSize: csize_t sizeof(PjrtDeviceLocalHardwareIdArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_LocalHardwareId")
  int args.localHardwareId

proc deviceAddressableMemories*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): seq[PjrtMemoryRaw] =
  ## Returns the memory spaces addressable by `dev`.
  let fn = requireApiFn(api, fnDeviceAddressableMemories,
    PjrtFnDeviceAddressableMemories, "PJRT_Device_AddressableMemories")
  var args = PjrtDeviceAddressableMemoriesArgs(
    structSize: csize_t sizeof(PjrtDeviceAddressableMemoriesArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_AddressableMemories")
  result = newSeqOfCap[PjrtMemoryRaw](int args.numMemories)
  for i in 0 ..< int args.numMemories:
    result.add args.memories[i]

proc deviceDefaultMemory*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): PjrtMemoryRaw =
  ## Returns the default memory space for buffers processed by `dev`.
  let fn = requireApiFn(api, fnDeviceDefaultMemory, PjrtFnDeviceDefaultMemory,
    "PJRT_Device_DefaultMemory")
  var args = PjrtDeviceDefaultMemoryArgs(
    structSize: csize_t sizeof(PjrtDeviceDefaultMemoryArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_DefaultMemory")
  args.memory

proc deviceMemoryStats*(api: PjrtApiHandle;
    dev: PjrtDeviceHandle): PjrtDeviceMemoryStats =
  ## Returns allocator diagnostics for `dev`. Some platforms report
  ## `pecUnimplemented`; that is translated to `PjrtError` like any other
  ## PJRT failure.
  let fn = requireApiFn(api, fnDeviceMemoryStats, PjrtFnDeviceMemoryStats,
    "PJRT_Device_MemoryStats")
  var args = PjrtDeviceMemoryStatsArgs(
    structSize: csize_t sizeof(PjrtDeviceMemoryStatsArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_MemoryStats")
  PjrtDeviceMemoryStats(
    bytesInUse: args.bytesInUse,
    peakBytesInUse: args.peakBytesInUse,
    hasPeakBytesInUse: args.peakBytesInUseIsSet,
    numAllocs: args.numAllocs,
    hasNumAllocs: args.numAllocsIsSet,
    largestAllocSize: args.largestAllocSize,
    hasLargestAllocSize: args.largestAllocSizeIsSet,
    bytesLimit: args.bytesLimit,
    hasBytesLimit: args.bytesLimitIsSet,
    bytesReserved: args.bytesReserved,
    hasBytesReserved: args.bytesReservedIsSet,
    peakBytesReserved: args.peakBytesReserved,
    hasPeakBytesReserved: args.peakBytesReservedIsSet,
    bytesReservableLimit: args.bytesReservableLimit,
    hasBytesReservableLimit: args.bytesReservableLimitIsSet,
    largestFreeBlockBytes: args.largestFreeBlockBytes,
    hasLargestFreeBlockBytes: args.largestFreeBlockBytesIsSet,
    poolBytes: args.poolBytes,
    hasPoolBytes: args.poolBytesIsSet,
    peakPoolBytes: args.peakPoolBytes,
    hasPeakPoolBytes: args.peakPoolBytesIsSet)

proc deviceClearMemoryStats*(api: PjrtApiHandle; dev: PjrtDeviceHandle) =
  ## Clears allocator peak-memory diagnostics for `dev` where supported.
  let fn = requireApiFn(api, fnDeviceClearMemoryStats,
    PjrtFnDeviceClearMemoryStats, "PJRT_Device_ClearMemoryStats")
  var args = PjrtDeviceClearMemoryStatsArgs(
    structSize: csize_t sizeof(PjrtDeviceClearMemoryStatsArgs),
    device: dev)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Device_ClearMemoryStats")

proc memoryId*(api: PjrtApiHandle; memory: PjrtMemoryRaw): int =
  ## Returns the unique PJRT memory id.
  let fn = requireApiFn(api, fnMemoryId, PjrtFnMemoryId, "PJRT_Memory_Id")
  var args = PjrtMemoryIdArgs(
    structSize: csize_t sizeof(PjrtMemoryIdArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_Id")
  int args.id

proc memoryKind*(api: PjrtApiHandle; memory: PjrtMemoryRaw): string =
  ## Returns the platform-dependent memory kind string.
  let fn = requireApiFn(api, fnMemoryKind, PjrtFnMemoryKind,
    "PJRT_Memory_Kind")
  var args = PjrtMemoryKindArgs(
    structSize: csize_t sizeof(PjrtMemoryKindArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_Kind")
  copySizedString(args.kind, args.kindSize)

proc memoryKindId*(api: PjrtApiHandle; memory: PjrtMemoryRaw): int =
  ## Returns the platform-dependent memory kind id.
  let fn = requireApiFn(api, fnMemoryKindId, PjrtFnMemoryKindId,
    "PJRT_Memory_KindId")
  var args = PjrtMemoryKindIdArgs(
    structSize: csize_t sizeof(PjrtMemoryKindIdArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_KindId")
  int args.kindId

proc memoryDebugString*(api: PjrtApiHandle; memory: PjrtMemoryRaw): string =
  ## Returns the verbose debug string for `memory`.
  let fn = requireApiFn(api, fnMemoryDebugString, PjrtFnMemoryDebugString,
    "PJRT_Memory_DebugString")
  var args = PjrtMemoryDebugStringArgs(
    structSize: csize_t sizeof(PjrtMemoryDebugStringArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_DebugString")
  copySizedString(args.debugString, args.debugStringSize)

proc memoryToString*(api: PjrtApiHandle; memory: PjrtMemoryRaw): string =
  ## Returns the user-facing string for `memory`.
  let fn = requireApiFn(api, fnMemoryToString, PjrtFnMemoryToString,
    "PJRT_Memory_ToString")
  var args = PjrtMemoryToStringArgs(
    structSize: csize_t sizeof(PjrtMemoryToStringArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_ToString")
  copySizedString(args.toString, args.toStringSize)

proc memoryAddressableByDevices*(api: PjrtApiHandle;
    memory: PjrtMemoryRaw): seq[PjrtDeviceHandle] =
  ## Returns the devices that can address `memory`.
  let fn = requireApiFn(api, fnMemoryAddressableByDevices,
    PjrtFnMemoryAddressableByDevices, "PJRT_Memory_AddressableByDevices")
  var args = PjrtMemoryAddressableByDevicesArgs(
    structSize: csize_t sizeof(PjrtMemoryAddressableByDevicesArgs),
    memory: memory)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Memory_AddressableByDevices")
  result = newSeqOfCap[PjrtDeviceHandle](int args.numDevices)
  for i in 0 ..< int args.numDevices:
    result.add args.devices[i]

# ----- Buffer transfer & lifecycle --------------------------------------

proc bufferDestroy*(api: PjrtApiHandle; buf: PjrtBufferRaw) =
  ## Calls `PJRT_Buffer_Destroy`. No-op if `api` or `buf` is nil so it can
  ## be used unconditionally from destructors.
  if api.isNil or buf.isNil: return
  let fn = cast[PjrtFnBufferDestroy](apiPtr(api).fnBufferDestroy)
  if cast[pointer](fn).isNil: return
  var args = PjrtBufferDestroyArgs(
    structSize: csize_t sizeof(PjrtBufferDestroyArgs),
    buffer: buf)
  let err = fn(addr args)
  if not err.isNil: destroyError(api, err)

proc bufferElementType*(api: PjrtApiHandle;
    buf: PjrtBufferRaw): PjrtBufferType =
  ## Returns `PJRT_Buffer_Type` of `buf`.
  let fn = cast[PjrtFnBufferElementType](apiPtr(api).fnBufferElementType)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Buffer_ElementType")
  var args = PjrtBufferElementTypeArgs(
    structSize: csize_t sizeof(PjrtBufferElementTypeArgs),
    buffer: buf)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Buffer_ElementType")
  args.`type`

proc bufferDimensions*(api: PjrtApiHandle;
    buf: PjrtBufferRaw): seq[int64] =
  ## Returns the logical dimensions of `buf` as a copied seq.
  let fn = cast[PjrtFnBufferDimensions](apiPtr(api).fnBufferDimensions)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Buffer_Dimensions")
  var args = PjrtBufferDimensionsArgs(
    structSize: csize_t sizeof(PjrtBufferDimensionsArgs),
    buffer: buf)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Buffer_Dimensions")
  result = newSeq[int64](int args.numDims)
  if args.numDims > 0 and not args.dims.isNil:
    let src = cast[ptr UncheckedArray[int64]](args.dims)
    for i in 0 ..< int args.numDims:
      result[i] = src[i]

proc bufferOnDeviceSizeInBytes*(api: PjrtApiHandle;
    buf: PjrtBufferRaw): int =
  ## Returns the on-device size of `buf` in bytes.
  let fn = cast[PjrtFnBufferOnDeviceSizeInBytes](
    apiPtr(api).fnBufferOnDeviceSizeInBytes)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Buffer_OnDeviceSizeInBytes")
  var args = PjrtBufferOnDeviceSizeInBytesArgs(
    structSize: csize_t sizeof(PjrtBufferOnDeviceSizeInBytesArgs),
    buffer: buf)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Buffer_OnDeviceSizeInBytes")
  int args.onDeviceSizeInBytes

proc clientBufferFromHostBuffer*(api: PjrtApiHandle;
    client: PjrtClientHandle; device: PjrtDeviceHandle;
    data: pointer; bufferType: PjrtBufferType;
    dims: openArray[int64]): tuple[buffer: PjrtBufferRaw,
    doneEvent: PjrtEventRaw] =
  ## Copies `data` (host) into a fresh device buffer on `device`. Always
  ## uses `hbsImmutableUntilTransferDone`, dense major-to-minor layout, and
  ## the device's default memory. Returns the new buffer and the event that
  ## fires once the host buffer is no longer read.
  let fn = cast[PjrtFnClientBufferFromHostBuffer](
    apiPtr(api).fnClientBufferFromHostBuffer)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Client_BufferFromHostBuffer")
  var dimsCopy = @dims
  let dimsPtr =
    if dimsCopy.len == 0: nil
    else: cast[ptr int64](addr dimsCopy[0])
  var args = PjrtClientBufferFromHostBufferArgs(
    structSize: csize_t sizeof(PjrtClientBufferFromHostBufferArgs),
    client: client,
    data: data,
    bufferType: bufferType,
    dims: dimsPtr,
    numDims: csize_t dimsCopy.len,
    byteStrides: nil,
    numByteStrides: 0,
    hostBufferSemantics: hbsImmutableUntilTransferDone,
    device: device,
    memory: nil,
    deviceLayout: nil)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_BufferFromHostBuffer")
  (args.buffer, args.doneWithHostBuffer)

proc bufferToHostBuffer*(api: PjrtApiHandle;
    buf: PjrtBufferRaw; dst: pointer; dstSize: int): PjrtEventRaw =
  ## Initiates a device→host copy of `buf` into `dst` (`dstSize` bytes).
  ## Returns the event that fires when the copy completes.
  let fn = cast[PjrtFnBufferToHostBuffer](apiPtr(api).fnBufferToHostBuffer)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Buffer_ToHostBuffer")
  var args = PjrtBufferToHostBufferArgs(
    structSize: csize_t sizeof(PjrtBufferToHostBufferArgs),
    src: buf,
    hostLayout: nil,
    dst: dst,
    dstSize: csize_t dstSize)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Buffer_ToHostBuffer")
  args.event

# ----- Compile + Execute ------------------------------------------------

proc clientCompile*(api: PjrtApiHandle; client: PjrtClientHandle;
    code: string; format: string = "mlir";
    compileOptions: string = ""): PjrtLoadedExecutableRaw =
  ## Compiles `code` (in `format`, default `"mlir"`) into a loaded
  ## executable on `client`. `compileOptions` is a serialized
  ## `xla.CompileOptionsProto` — pass `""` for plugin defaults
  ## (single replica, single partition, default device assignment).
  let fn = cast[PjrtFnClientCompile](apiPtr(api).fnClientCompile)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Client_Compile")
  var codeCopy = code
  var fmtCopy = format
  var optsCopy =
    if compileOptions.len == 0: DefaultCompileOptionsProto
    else: compileOptions
  var program = PjrtProgram(
    structSize: csize_t sizeof(PjrtProgram),
    code:
      if codeCopy.len == 0: nil
      else: cast[cstring](addr codeCopy[0]),
    codeSize: csize_t codeCopy.len,
    format:
      if fmtCopy.len == 0: nil
      else: cast[cstring](addr fmtCopy[0]),
    formatSize: csize_t fmtCopy.len)
  var args = PjrtClientCompileArgs(
    structSize: csize_t sizeof(PjrtClientCompileArgs),
    client: client,
    program: addr program,
    compileOptions:
      if optsCopy.len == 0: nil
      else: cast[cstring](addr optsCopy[0]),
    compileOptionsSize: csize_t optsCopy.len)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Client_Compile")
  args.executable

proc executableDestroy*(api: PjrtApiHandle; exe: PjrtExecutableRaw) =
  ## Calls `PJRT_Executable_Destroy`. No-op if `api` or `exe` is nil.
  if api.isNil or exe.isNil: return
  let fn = cast[PjrtFnExecutableDestroy](apiPtr(api).fnExecutableDestroy)
  if cast[pointer](fn).isNil: return
  var args = PjrtExecutableDestroyArgs(
    structSize: csize_t sizeof(PjrtExecutableDestroyArgs),
    executable: exe)
  let err = fn(addr args)
  if not err.isNil: destroyError(api, err)

proc loadedExecutableDestroy*(api: PjrtApiHandle;
    exe: PjrtLoadedExecutableRaw) =
  ## Calls `PJRT_LoadedExecutable_Destroy`. No-op if `api` or `exe` is nil.
  if api.isNil or exe.isNil: return
  let fn = cast[PjrtFnLoadedExecutableDestroy](
    apiPtr(api).fnLoadedExecutableDestroy)
  if cast[pointer](fn).isNil: return
  var args = PjrtLoadedExecutableDestroyArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableDestroyArgs),
    executable: exe)
  let err = fn(addr args)
  if not err.isNil: destroyError(api, err)

proc loadedExecutableGetExecutable*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): PjrtExecutableRaw =
  ## Returns a fresh `PjrtExecutable` for `loaded`. Caller owns the
  ## returned value and must free it via `executableDestroy`.
  let fn = cast[PjrtFnLoadedExecutableGetExecutable](
    apiPtr(api).fnLoadedExecutableGetExecutable)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_LoadedExecutable_GetExecutable")
  var args = PjrtLoadedExecutableGetExecutableArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableGetExecutableArgs),
    loadedExecutable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_GetExecutable")
  args.executable

proc executableNumOutputs*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): int =
  ## Returns the number of outputs produced by `exe` per device.
  let fn = cast[PjrtFnExecutableNumOutputs](
    apiPtr(api).fnExecutableNumOutputs)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_Executable_NumOutputs")
  var args = PjrtExecutableNumOutputsArgs(
    structSize: csize_t sizeof(PjrtExecutableNumOutputsArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_NumOutputs")
  int args.numOutputs

proc executableName*(api: PjrtApiHandle; exe: PjrtExecutableRaw): string =
  ## Returns the executable name.
  let fn = requireApiFn(api, fnExecutableName, PjrtFnExecutableName,
    "PJRT_Executable_Name")
  var args = PjrtExecutableNameArgs(
    structSize: csize_t sizeof(PjrtExecutableNameArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_Name")
  copySizedString(args.executableName, args.executableNameSize)

proc executableNumReplicas*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): int =
  ## Returns the number of replicas compiled into `exe`.
  let fn = requireApiFn(api, fnExecutableNumReplicas,
    PjrtFnExecutableNumReplicas, "PJRT_Executable_NumReplicas")
  var args = PjrtExecutableNumReplicasArgs(
    structSize: csize_t sizeof(PjrtExecutableNumReplicasArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_NumReplicas")
  int args.numReplicas

proc executableNumPartitions*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): int =
  ## Returns the number of partitions compiled into `exe`.
  let fn = requireApiFn(api, fnExecutableNumPartitions,
    PjrtFnExecutableNumPartitions, "PJRT_Executable_NumPartitions")
  var args = PjrtExecutableNumPartitionsArgs(
    structSize: csize_t sizeof(PjrtExecutableNumPartitionsArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_NumPartitions")
  int args.numPartitions

proc executableSizeOfGeneratedCodeInBytes*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): int64 =
  ## Returns the generated-code size reported by PJRT.
  let fn = requireApiFn(api, fnExecutableSizeOfGeneratedCodeInBytes,
    PjrtFnExecutableSizeOfGeneratedCodeInBytes,
    "PJRT_Executable_SizeOfGeneratedCodeInBytes")
  var args = PjrtExecutableSizeOfGeneratedCodeInBytesArgs(
    structSize: csize_t sizeof(PjrtExecutableSizeOfGeneratedCodeInBytesArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_SizeOfGeneratedCodeInBytes")
  args.sizeInBytes

proc executableFingerprint*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): string =
  ## Returns a compiler/plugin fingerprint for `exe` where supported.
  let fn = requireApiFn(api, fnExecutableFingerprint,
    PjrtFnExecutableFingerprint, "PJRT_Executable_Fingerprint")
  var args = PjrtExecutableFingerprintArgs(
    structSize: csize_t sizeof(PjrtExecutableFingerprintArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_Fingerprint")
  copySizedString(args.executableFingerprint, args.executableFingerprintSize)

proc executableCostAnalysis*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): seq[tuple[name, value: string]] =
  ## Returns plugin-specific cost-analysis properties as owned strings.
  let fn = requireApiFn(api, fnExecutableGetCostAnalysis,
    PjrtFnExecutableGetCostAnalysis, "PJRT_Executable_GetCostAnalysis")
  var args = PjrtExecutableGetCostAnalysisArgs(
    structSize: csize_t sizeof(PjrtExecutableGetCostAnalysisArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_GetCostAnalysis")
  namedValuesToStrings(args.properties, int args.numProperties)

proc executableCompiledMemoryStats*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): PjrtCompiledMemoryStats =
  ## Returns compile-time memory estimates for `exe`.
  let fn = requireApiFn(api, fnExecutableGetCompiledMemoryStats,
    PjrtFnExecutableGetCompiledMemoryStats,
    "PJRT_Executable_GetCompiledMemoryStats")
  var args = PjrtExecutableGetCompiledMemoryStatsArgs(
    structSize: csize_t sizeof(PjrtExecutableGetCompiledMemoryStatsArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_GetCompiledMemoryStats")
  PjrtCompiledMemoryStats(
    generatedCodeSizeInBytes: args.generatedCodeSizeInBytes,
    argumentSizeInBytes: args.argumentSizeInBytes,
    outputSizeInBytes: args.outputSizeInBytes,
    aliasSizeInBytes: args.aliasSizeInBytes,
    tempSizeInBytes: args.tempSizeInBytes,
    hostGeneratedCodeSizeInBytes: args.hostGeneratedCodeSizeInBytes,
    hostArgumentSizeInBytes: args.hostArgumentSizeInBytes,
    hostOutputSizeInBytes: args.hostOutputSizeInBytes,
    hostAliasSizeInBytes: args.hostAliasSizeInBytes,
    hostTempSizeInBytes: args.hostTempSizeInBytes,
    peakMemoryInBytes: args.peakMemoryInBytes,
    totalSizeInBytes: args.totalSizeInBytes,
    totalAllocationBytes: args.totalAllocationBytes,
    indefiniteAllocations: args.indefiniteAllocations,
    peakUnpaddedHeapBytes: args.peakUnpaddedHeapBytes)

proc executableOutputElementTypes*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): seq[PjrtBufferType] =
  ## Returns the element type of each output.
  let fn = requireApiFn(api, fnExecutableOutputElementTypes,
    PjrtFnExecutableOutputElementTypes,
    "PJRT_Executable_OutputElementTypes")
  var args = PjrtExecutableOutputElementTypesArgs(
    structSize: csize_t sizeof(PjrtExecutableOutputElementTypesArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_OutputElementTypes")
  result = newSeqOfCap[PjrtBufferType](int args.numOutputTypes)
  for i in 0 ..< int args.numOutputTypes:
    result.add args.outputTypes[i]

proc executableOutputDimensions*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw; numOutputs: int): seq[seq[int64]] =
  ## Returns the dimensions of each output. `numOutputs` should come from
  ## `executableNumOutputs`.
  let fn = requireApiFn(api, fnExecutableOutputDimensions,
    PjrtFnExecutableOutputDimensions, "PJRT_Executable_OutputDimensions")
  var args = PjrtExecutableOutputDimensionsArgs(
    structSize: csize_t sizeof(PjrtExecutableOutputDimensionsArgs),
    executable: exe,
    numOutputs: csize_t numOutputs)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_OutputDimensions")
  result = newSeq[seq[int64]](numOutputs)
  var offset = 0
  for i in 0 ..< numOutputs:
    let rank = int args.dimSizes[i]
    result[i] = newSeq[int64](rank)
    for j in 0 ..< rank:
      result[i][j] = args.dims[offset + j]
    offset += rank

proc copyMemoryKindStrings(kinds: ptr UncheckedArray[cstring];
    sizes: ptr UncheckedArray[csize_t]; len: int): seq[string] =
  result = newSeqOfCap[string](len)
  if kinds.isNil or sizes.isNil: return
  for i in 0 ..< len:
    result.add copySizedString(kinds[i], sizes[i])

proc executableParameterMemoryKinds*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw; numParameters: int): seq[string] =
  ## Returns the memory kind selected for each executable parameter.
  let fn = requireApiFn(api, fnExecutableParameterMemoryKinds,
    PjrtFnExecutableParameterMemoryKinds,
    "PJRT_Executable_ParameterMemoryKinds")
  var args = PjrtExecutableParameterMemoryKindsArgs(
    structSize: csize_t sizeof(PjrtExecutableParameterMemoryKindsArgs),
    executable: exe,
    numParameters: csize_t numParameters)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_ParameterMemoryKinds")
  copyMemoryKindStrings(args.memoryKinds, args.memoryKindSizes, numParameters)

proc executableOutputMemoryKinds*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw; numOutputs: int): seq[string] =
  ## Returns the memory kind selected for each executable output.
  let fn = requireApiFn(api, fnExecutableOutputMemoryKinds,
    PjrtFnExecutableOutputMemoryKinds, "PJRT_Executable_OutputMemoryKinds")
  var args = PjrtExecutableOutputMemoryKindsArgs(
    structSize: csize_t sizeof(PjrtExecutableOutputMemoryKindsArgs),
    executable: exe,
    numOutputs: csize_t numOutputs)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_OutputMemoryKinds")
  copyMemoryKindStrings(args.memoryKinds, args.memoryKindSizes, numOutputs)

proc executableSerialize*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): string =
  ## Serializes `exe` to plugin-specific bytes and frees PJRT's backing
  ## allocation after copying.
  let fn = requireApiFn(api, fnExecutableSerialize, PjrtFnExecutableSerialize,
    "PJRT_Executable_Serialize")
  var args = PjrtExecutableSerializeArgs(
    structSize: csize_t sizeof(PjrtExecutableSerializeArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_Serialize")
  result = copySizedString(args.serializedBytes, args.serializedBytesSize)
  if args.serializedExecutableDeleter != nil and
      not args.serializedExecutable.isNil:
    args.serializedExecutableDeleter(args.serializedExecutable)

proc executableCompileOptions*(api: PjrtApiHandle;
    exe: PjrtExecutableRaw): string =
  ## Returns the serialized CompileOptions used to build `exe`.
  let fn = requireApiFn(api, fnExecutableGetCompileOptions,
    PjrtFnExecutableGetCompileOptions, "PJRT_Executable_GetCompileOptions")
  var args = PjrtExecutableGetCompileOptionsArgs(
    structSize: csize_t sizeof(PjrtExecutableGetCompileOptionsArgs),
    executable: exe)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_GetCompileOptions")
  result = copySizedString(args.serializedBytes, args.serializedBytesSize)
  if args.serializedCompileOptionsDeleter != nil and
      not args.serializedCompileOptions.isNil:
    args.serializedCompileOptionsDeleter(args.serializedCompileOptions)

proc executableOptimizedProgram*(api: PjrtApiHandle; exe: PjrtExecutableRaw;
    format = "hlo"): tuple[code, format: string] =
  ## Returns the optimized program bytes in a requested format. PJRT uses a
  ## two-pass API: first ask for size, then provide caller-owned storage.
  let fn = requireApiFn(api, fnExecutableOptimizedProgram,
    PjrtFnExecutableOptimizedProgram, "PJRT_Executable_OptimizedProgram")
  var fmt = format
  var program = PjrtProgram(
    structSize: csize_t sizeof(PjrtProgram),
    code: nil,
    codeSize: 0,
    format: if fmt.len == 0: nil else: cast[cstring](addr fmt[0]),
    formatSize: csize_t fmt.len,
    memory: PjrtMemoryRaw(nil))
  var args = PjrtExecutableOptimizedProgramArgs(
    structSize: csize_t sizeof(PjrtExecutableOptimizedProgramArgs),
    executable: exe,
    program: addr program)
  var err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_OptimizedProgram:size")
  var storage = newString(int program.codeSize)
  if storage.len > 0:
    program.code = cast[cstring](addr storage[0])
  err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_OptimizedProgram")
  (copySizedString(program.code, program.codeSize),
    copySizedString(program.format, program.formatSize))

proc executableDeserializeAndLoad*(api: PjrtApiHandle;
    client: PjrtClientHandle; serializedExecutable: string;
    compileOptions: string = ""): PjrtLoadedExecutableRaw =
  ## Loads a serialized executable produced by `executableSerialize`.
  let fn = requireApiFn(api, fnExecutableDeserializeAndLoad,
    PjrtFnExecutableDeserializeAndLoad,
    "PJRT_Executable_DeserializeAndLoad")
  var bytes = serializedExecutable
  var opts = compileOptions
  var args = PjrtExecutableDeserializeAndLoadArgs(
    structSize: csize_t sizeof(PjrtExecutableDeserializeAndLoadArgs),
    client: client,
    serializedExecutable:
      if bytes.len == 0: nil else: cast[cstring](addr bytes[0]),
    serializedExecutableSize: csize_t bytes.len,
    overriddenSerializedCompileOptions:
      if opts.len == 0: nil else: cast[cstring](addr opts[0]),
    overriddenSerializedCompileOptionsSize: csize_t opts.len)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_Executable_DeserializeAndLoad")
  args.loadedExecutable

proc loadedExecutableAddressableDevices*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): seq[PjrtDeviceHandle] =
  ## Returns devices this loaded executable can run on.
  let fn = requireApiFn(api, fnLoadedExecutableAddressableDevices,
    PjrtFnLoadedExecutableAddressableDevices,
    "PJRT_LoadedExecutable_AddressableDevices")
  var args = PjrtLoadedExecutableAddressableDevicesArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableAddressableDevicesArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_AddressableDevices")
  result = newSeqOfCap[PjrtDeviceHandle](int args.numAddressableDevices)
  for i in 0 ..< int args.numAddressableDevices:
    result.add args.addressableDevices[i]

proc loadedExecutableAddressableDeviceLogicalIds*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): seq[PjrtLogicalDeviceIds] =
  ## Returns `(replica, partition)` ids matching addressable executable
  ## devices.
  let fn = requireApiFn(api, fnLoadedExecutableAddressableDeviceLogicalIds,
    PjrtFnLoadedExecutableAddressableDeviceLogicalIds,
    "PJRT_LoadedExecutable_AddressableDeviceLogicalIds")
  var args = PjrtLoadedExecutableAddressableDeviceLogicalIdsArgs(
    structSize:
      csize_t sizeof(PjrtLoadedExecutableAddressableDeviceLogicalIdsArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_AddressableDeviceLogicalIds")
  result = newSeqOfCap[PjrtLogicalDeviceIds](
    int args.numAddressableDeviceLogicalIds)
  for i in 0 ..< int args.numAddressableDeviceLogicalIds:
    result.add args.addressableDeviceLogicalIds[i]

proc loadedExecutableDelete*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw) =
  ## Deletes the underlying runtime object while leaving the wrapper object
  ## valid for `isDeleted`/destroy.
  let fn = requireApiFn(api, fnLoadedExecutableDelete,
    PjrtFnLoadedExecutableDelete, "PJRT_LoadedExecutable_Delete")
  var args = PjrtLoadedExecutableDeleteArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableDeleteArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_Delete")

proc loadedExecutableIsDeleted*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): bool =
  ## Returns true after `loadedExecutableDelete`.
  let fn = requireApiFn(api, fnLoadedExecutableIsDeleted,
    PjrtFnLoadedExecutableIsDeleted, "PJRT_LoadedExecutable_IsDeleted")
  var args = PjrtLoadedExecutableIsDeletedArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableIsDeletedArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_IsDeleted")
  args.isDeleted

proc loadedExecutableFingerprint*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): string =
  ## Deprecated PJRT fingerprint helper kept for plugins that still expose
  ## only the loaded-executable variant.
  let fn = requireApiFn(api, fnLoadedExecutableFingerprint,
    PjrtFnLoadedExecutableFingerprint, "PJRT_LoadedExecutable_Fingerprint")
  var args = PjrtLoadedExecutableFingerprintArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableFingerprintArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_Fingerprint")
  copySizedString(args.executableFingerprint, args.executableFingerprintSize)

proc loadedExecutableDeviceAssignment*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw): string =
  ## Returns the serialized `DeviceAssignmentProto` for `loaded`.
  let fn = requireApiFn(api, fnLoadedExecutableGetDeviceAssignment,
    PjrtFnLoadedExecutableGetDeviceAssignment,
    "PJRT_LoadedExecutable_GetDeviceAssignment")
  var args = PjrtLoadedExecutableGetDeviceAssignmentArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableGetDeviceAssignmentArgs),
    executable: loaded)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_GetDeviceAssignment")
  result = copySizedString(args.serializedBytes, args.serializedBytesSize)
  if args.serializedDeviceAssignmentDeleter != nil and
      not args.serializedDeviceAssignment.isNil:
    args.serializedDeviceAssignmentDeleter(args.serializedDeviceAssignment)

proc loadedExecutableExecute*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw; execDevice: PjrtDeviceHandle;
    inputs: openArray[PjrtBufferRaw];
    numOutputs: int;
    nonDonatableInputIndices: openArray[int64] = [];
    launchId: int = 0;
    callLocation: string = ""): seq[PjrtBufferRaw] =
  ## Single-device synchronous execute. Builds the 1-row argument and
  ## output lists required by the C API, calls `PJRT_LoadedExecutable_Execute`,
  ## and returns the output buffer pointers. The caller is responsible for
  ## destroying the output buffers (the Nim `PjrtBuffer` wrapper does this).
  let fn = cast[PjrtFnLoadedExecutableExecute](
    apiPtr(api).fnLoadedExecutableExecute)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_LoadedExecutable_Execute")

  var argRow = @inputs
  let argRowPtr =
    if argRow.len == 0: nil
    else: cast[ptr PjrtBufferRaw](addr argRow[0])
  var argLists: array[1, ptr PjrtBufferRaw] = [argRowPtr]

  result = newSeq[PjrtBufferRaw](numOutputs)
  let outRowPtr =
    if numOutputs == 0: nil
    else: cast[ptr PjrtBufferRaw](addr result[0])
  var outLists: array[1, ptr PjrtBufferRaw] = [outRowPtr]

  var nonDonatable = @nonDonatableInputIndices
  var location = callLocation
  var opts = PjrtExecuteOptions(
    structSize: csize_t sizeof(PjrtExecuteOptions),
    launchId: cint launchId,
    nonDonatableInputIndices:
      if nonDonatable.len == 0: nil
      else: cast[ptr int64](addr nonDonatable[0]),
    numNonDonatableInputIndices: csize_t nonDonatable.len,
    callLocation:
      if location.len == 0: nil
      else: cstring(location))
  var args = PjrtLoadedExecutableExecuteArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableExecuteArgs),
    executable: loaded,
    options: addr opts,
    argumentLists: cast[ptr ptr PjrtBufferRaw](addr argLists[0]),
    numDevices: 1,
    numArgs: csize_t inputs.len,
    outputLists: cast[ptr ptr PjrtBufferRaw](addr outLists[0]),
    deviceCompleteEvents: nil,
    executeDevice: execDevice)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_Execute")

proc loadedExecutableExecuteMulti*(api: PjrtApiHandle;
    loaded: PjrtLoadedExecutableRaw;
    deviceInputs: openArray[seq[PjrtBufferRaw]];
    numOutputs: int;
    nonDonatableInputIndices: openArray[int64] = [];
    launchId: int = 0;
    callLocation: string = "";
    taskIds: openArray[int] = [];
    incarnationIds: openArray[int64] = []): seq[seq[PjrtBufferRaw]] =
  ## Multi-device synchronous execute. `deviceInputs[d][a]` is the `a`th
  ## argument buffer for addressable device `d`; PJRT chooses devices from
  ## the loaded executable because `execute_device` is nil.
  let fn = cast[PjrtFnLoadedExecutableExecute](
    apiPtr(api).fnLoadedExecutableExecute)
  if cast[pointer](fn).isNil:
    raisePjrt("PJRT plugin does not export PJRT_LoadedExecutable_Execute")
  if deviceInputs.len == 0:
    raisePjrt("PJRT_LoadedExecutable_Execute requires at least one device")
  if taskIds.len > 0 and incarnationIds.len > 0 and
      taskIds.len != incarnationIds.len:
    raisePjrt("PJRT execute task_ids and incarnation_ids length mismatch")

  let numArgs = deviceInputs[0].len
  var argRows = newSeq[seq[PjrtBufferRaw]](deviceInputs.len)
  var argLists = newSeq[ptr PjrtBufferRaw](deviceInputs.len)
  for i, row in deviceInputs:
    if row.len != numArgs:
      raisePjrt("PJRT multi-device execute input rows must have equal length")
    argRows[i] = @row
    argLists[i] =
      if argRows[i].len == 0: nil
      else: cast[ptr PjrtBufferRaw](addr argRows[i][0])

  result = newSeq[seq[PjrtBufferRaw]](deviceInputs.len)
  var outLists = newSeq[ptr PjrtBufferRaw](deviceInputs.len)
  for i in 0 ..< result.len:
    result[i] = newSeq[PjrtBufferRaw](numOutputs)
    outLists[i] =
      if numOutputs == 0: nil
      else: cast[ptr PjrtBufferRaw](addr result[i][0])

  var nonDonatable = @nonDonatableInputIndices
  var location = callLocation
  var taskIdsC = newSeq[cint](taskIds.len)
  for i, id in taskIds:
    taskIdsC[i] = cint id
  var incarnationIdsLocal = @incarnationIds
  let taskCount =
    if taskIdsC.len > 0: taskIdsC.len
    else: incarnationIdsLocal.len
  var opts = PjrtExecuteOptions(
    structSize: csize_t sizeof(PjrtExecuteOptions),
    launchId: cint launchId,
    nonDonatableInputIndices:
      if nonDonatable.len == 0: nil
      else: cast[ptr int64](addr nonDonatable[0]),
    numNonDonatableInputIndices: csize_t nonDonatable.len,
    callLocation:
      if location.len == 0: nil
      else: cstring(location),
    numTasks: csize_t taskCount,
    taskIds:
      if taskIdsC.len == 0: nil
      else: cast[ptr cint](addr taskIdsC[0]),
    incarnationIds:
      if incarnationIdsLocal.len == 0: nil
      else: cast[ptr int64](addr incarnationIdsLocal[0]))
  var args = PjrtLoadedExecutableExecuteArgs(
    structSize: csize_t sizeof(PjrtLoadedExecutableExecuteArgs),
    executable: loaded,
    options: addr opts,
    argumentLists: cast[ptr ptr PjrtBufferRaw](addr argLists[0]),
    numDevices: csize_t deviceInputs.len,
    numArgs: csize_t numArgs,
    outputLists: cast[ptr ptr PjrtBufferRaw](addr outLists[0]),
    deviceCompleteEvents: nil,
    executeDevice: nil.PjrtDeviceHandle)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_LoadedExecutable_Execute")

# ----- Topology description helpers -------------------------------------

proc topologyDescriptionDestroy*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw) =
  ## Destroys a caller-owned topology description. Client-owned topologies
  ## returned by `clientTopologyDescription` must not be passed here.
  if api.isNil or topology.isNil: return
  let fn = requireApiFn(api, fnTopologyDescriptionDestroy,
    PjrtFnTopologyDescriptionDestroy, "PJRT_TopologyDescription_Destroy")
  var args = PjrtTopologyDescriptionDestroyArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionDestroyArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_Destroy")

proc topologyPlatformName*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): string =
  ## Returns the topology platform name.
  let fn = requireApiFn(api, fnTopologyDescriptionPlatformName,
    PjrtFnTopologyDescriptionPlatformName,
    "PJRT_TopologyDescription_PlatformName")
  var args = PjrtTopologyDescriptionPlatformNameArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionPlatformNameArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_PlatformName")
  copySizedString(args.platformName, args.platformNameSize)

proc topologyPlatformVersion*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): string =
  ## Returns topology platform version details.
  let fn = requireApiFn(api, fnTopologyDescriptionPlatformVersion,
    PjrtFnTopologyDescriptionPlatformVersion,
    "PJRT_TopologyDescription_PlatformVersion")
  var args = PjrtTopologyDescriptionPlatformVersionArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionPlatformVersionArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_PlatformVersion")
  copySizedString(args.platformVersion, args.platformVersionSize)

proc topologyDeviceDescriptions*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): seq[PjrtDeviceDescriptionRaw] =
  ## Returns all device descriptions in `topology`.
  let fn = requireApiFn(api, fnTopologyDescriptionGetDeviceDescriptions,
    PjrtFnTopologyDescriptionGetDeviceDescriptions,
    "PJRT_TopologyDescription_GetDeviceDescriptions")
  var args = PjrtTopologyDescriptionGetDeviceDescriptionsArgs(
    structSize:
      csize_t sizeof(PjrtTopologyDescriptionGetDeviceDescriptionsArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_GetDeviceDescriptions")
  result = newSeqOfCap[PjrtDeviceDescriptionRaw](int args.numDescriptions)
  for i in 0 ..< int args.numDescriptions:
    result.add args.descriptions[i]

proc topologySerialize*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): string =
  ## Serializes `topology` for cache keys or cross-process transfer.
  let fn = requireApiFn(api, fnTopologyDescriptionSerialize,
    PjrtFnTopologyDescriptionSerialize,
    "PJRT_TopologyDescription_Serialize")
  var args = PjrtTopologyDescriptionSerializeArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionSerializeArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_Serialize")
  result = copySizedString(args.serializedBytes, args.serializedBytesSize)
  if args.serializedTopologyDeleter != nil and
      not args.serializedTopology.isNil:
    args.serializedTopologyDeleter(args.serializedTopology)

proc topologyDeserialize*(api: PjrtApiHandle;
    serializedTopology: string): PjrtTopologyDescriptionRaw =
  ## Deserializes a caller-owned topology description.
  let fn = requireApiFn(api, fnTopologyDescriptionDeserialize,
    PjrtFnTopologyDescriptionDeserialize,
    "PJRT_TopologyDescription_Deserialize")
  var bytes = serializedTopology
  var args = PjrtTopologyDescriptionDeserializeArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionDeserializeArgs),
    serializedTopology:
      if bytes.len == 0: nil else: cast[cstring](addr bytes[0]),
    serializedTopologySize: csize_t bytes.len)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_Deserialize")
  args.topology

proc topologyAttributes*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): seq[tuple[name, value: string]] =
  ## Returns platform-specific topology attributes.
  let fn = requireApiFn(api, fnTopologyDescriptionAttributes,
    PjrtFnTopologyDescriptionAttributes,
    "PJRT_TopologyDescription_Attributes")
  var args = PjrtTopologyDescriptionAttributesArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionAttributesArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_Attributes")
  namedValuesToStrings(args.attributes, int args.numAttributes)

proc topologyFingerprint*(api: PjrtApiHandle;
    topology: PjrtTopologyDescriptionRaw): uint64 =
  ## Returns the topology fingerprint.
  let fn = requireApiFn(api, fnTopologyDescriptionFingerprint,
    PjrtFnTopologyDescriptionFingerprint,
    "PJRT_TopologyDescription_Fingerprint")
  var args = PjrtTopologyDescriptionFingerprintArgs(
    structSize: csize_t sizeof(PjrtTopologyDescriptionFingerprintArgs),
    topology: topology)
  let err = fn(addr args)
  checkErr(api, err, "PJRT_TopologyDescription_Fingerprint")
  args.fingerprint
