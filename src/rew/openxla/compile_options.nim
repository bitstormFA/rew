## Tiny protobuf encoder for PJRT/XLA CompileOptions.
##
## rew intentionally avoids generated protobuf code and external dependencies.
## This module encodes only the stable fields needed for distributed PJRT
## compilation: replica/partition counts, device assignment, SPMD/Shardy
## toggles, process metadata, and auto-SPMD mesh metadata.

type
  CompileOptionsSpec* = object
    ## Subset of `xla.CompileOptionsProto` used by rew.
    numReplicas*: int
    numPartitions*: int
    deviceAssignment*: seq[int64]
      ## Flat `[replica, partition]` physical device ids.
    useSpmdPartitioning*: bool
    useAutoSpmdPartitioning*: bool
    useShardyPartitioner*: bool
    processIndex*: int
      ## Encoded when `processCount > 0`.
    processCount*: int
      ## Encoded when positive.
    sliceSize*: int
      ## Encoded when positive.
    autoSpmdMeshShape*: seq[int64]
    autoSpmdMeshIds*: seq[int64]

func writeVarint(dst: var string; value: uint64) =
  var v = value
  while v >= 0x80'u64:
    dst.add char((v and 0x7f'u64) or 0x80'u64)
    v = v shr 7
  dst.add char(v)

func writeKey(dst: var string; field: int; wire: int) =
  writeVarint(dst, uint64((field shl 3) or wire))

func writeInt64Field(dst: var string; field: int; value: int64) =
  writeKey(dst, field, 0)
  writeVarint(dst, uint64(value))

func writeInt32Field(dst: var string; field: int; value: int) =
  writeKey(dst, field, 0)
  writeVarint(dst, uint64(value))

func writeBoolField(dst: var string; field: int; value: bool) =
  if value:
    writeKey(dst, field, 0)
    writeVarint(dst, 1)

func writeBytesField(dst: var string; field: int; bytes: string) =
  if bytes.len == 0:
    return
  writeKey(dst, field, 2)
  writeVarint(dst, uint64(bytes.len))
  dst.add bytes

func encodePackedInt64(values: openArray[int64]): string =
  for value in values:
    writeVarint(result, uint64(value))

func writePackedInt64Field(dst: var string; field: int;
    values: openArray[int64]) =
  if values.len == 0:
    return
  writeBytesField(dst, field, encodePackedInt64(values))

func encodeComputationDevice(replicaDeviceIds: openArray[int64]): string =
  ## `DeviceAssignmentProto.ComputationDevice`.
  writePackedInt64Field(result, 1, replicaDeviceIds)

func encodeDeviceAssignment*(numReplicas, numPartitions: int;
    flatDeviceIds: openArray[int64]): string =
  ## Encodes `xla.DeviceAssignmentProto`.
  ##
  ## `flatDeviceIds` is ordered `[replica, partition]`, matching PJRT's
  ## default device-assignment wrapper. The protobuf stores one computation
  ## entry per partition, each containing all replica device ids.
  if numReplicas <= 0 or numPartitions <= 0:
    raise newException(ValueError,
      "encodeDeviceAssignment: replica and partition counts must be positive")
  let expected = numReplicas * numPartitions
  if flatDeviceIds.len != expected:
    raise newException(ValueError,
      "encodeDeviceAssignment: expected " & $expected &
        " device id(s), got " & $flatDeviceIds.len)
  writeInt32Field(result, 1, numReplicas)
  writeInt32Field(result, 2, numPartitions)
  for partition in 0 ..< numPartitions:
    var replicaIds = newSeq[int64](numReplicas)
    for replica in 0 ..< numReplicas:
      replicaIds[replica] = flatDeviceIds[replica * numPartitions + partition]
    writeBytesField(result, 3, encodeComputationDevice(replicaIds))

func initCompileOptionsSpec*(numReplicas: int = 1;
    numPartitions: int = 1;
    deviceAssignment: openArray[int64] = [];
    useSpmdPartitioning: bool = false;
    useAutoSpmdPartitioning: bool = false;
    useShardyPartitioner: bool = false;
    processIndex: int = 0;
    processCount: int = 0;
    sliceSize: int = 0;
    autoSpmdMeshShape: openArray[int64] = [];
    autoSpmdMeshIds: openArray[int64] = []): CompileOptionsSpec =
  if numReplicas <= 0 or numPartitions <= 0:
    raise newException(ValueError,
      "initCompileOptionsSpec: replica and partition counts must be positive")
  CompileOptionsSpec(
    numReplicas: numReplicas,
    numPartitions: numPartitions,
    deviceAssignment: @deviceAssignment,
    useSpmdPartitioning: useSpmdPartitioning,
    useAutoSpmdPartitioning: useAutoSpmdPartitioning,
    useShardyPartitioner: useShardyPartitioner,
    processIndex: processIndex,
    processCount: processCount,
    sliceSize: sliceSize,
    autoSpmdMeshShape: @autoSpmdMeshShape,
    autoSpmdMeshIds: @autoSpmdMeshIds,
  )

func encodeExecutableBuildOptions*(spec: CompileOptionsSpec): string =
  ## Encodes `xla.ExecutableBuildOptionsProto`.
  writeInt64Field(result, 4, int64 spec.numReplicas)
  writeInt64Field(result, 5, int64 spec.numPartitions)
  writeBoolField(result, 6, spec.useSpmdPartitioning)
  writeBoolField(result, 7, spec.useAutoSpmdPartitioning)
  if spec.deviceAssignment.len > 0:
    writeBytesField(result, 9, encodeDeviceAssignment(spec.numReplicas,
      spec.numPartitions, spec.deviceAssignment))
  writePackedInt64Field(result, 16, spec.autoSpmdMeshShape)
  writePackedInt64Field(result, 17, spec.autoSpmdMeshIds)
  writeBoolField(result, 19, spec.useShardyPartitioner)
  if spec.processCount > 0:
    writeInt64Field(result, 22, int64 spec.processIndex)
    writeInt64Field(result, 23, int64 spec.processCount)
  if spec.sliceSize > 0:
    writeInt64Field(result, 26, int64 spec.sliceSize)

func encodeCompileOptions*(spec: CompileOptionsSpec): string =
  ## Encodes `xla.CompileOptionsProto`.
  writeBytesField(result, 3, encodeExecutableBuildOptions(spec))

func hexBytes*(bytes: string): string =
  ## Debug helper used by tests for exact protobuf-byte assertions.
  const Hex = "0123456789abcdef"
  for ch in bytes:
    let b = ord(ch)
    result.add Hex[(b shr 4) and 0xf]
    result.add Hex[b and 0xf]
