## Distributed/topology scalar ops.

import ../tensor
import ../dtype
import ../device
import ../dispatch
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker

proc replicaId*(device: Device = defaultDevice()): Tensor {.rewOp.} =
  ## Return the current replica id as a scalar uint32 tensor.
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.replicaId(ctx.builder)
    result = initTraceTensor(id, dtUint32, [], ctx.device)
    recordTraceOp("replicaId", [], result)
  of dmEager:
    let outs = dispatchEager("replicaId", [], [("device", $device)])
    doAssert outs.len == 1, "replicaId: eager backend returned wrong arity"
    result = outs[0]

proc partitionId*(device: Device = defaultDevice()): Tensor {.rewOp.} =
  ## Return the current partition id as a scalar uint32 tensor.
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.partitionId(ctx.builder)
    result = initTraceTensor(id, dtUint32, [], ctx.device)
    recordTraceOp("partitionId", [], result)
  of dmEager:
    let outs = dispatchEager("partitionId", [], [("device", $device)])
    doAssert outs.len == 1, "partitionId: eager backend returned wrong arity"
    result = outs[0]

# ---- collective ops ------------------------------------------------------

proc allGather*(operands: openArray[Tensor]; allGatherDim: int;
    resultShapes: openArray[seq[int]];
    replicaGroups: openArray[seq[int]] = @[@[]];
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): seq[Tensor] {.rewOp.} =
  ## Gather values from all replicas along `allGatherDim`.
  case currentMode()
  of dmTrace:
    for t in operands: requireTrace(t, "allGather")
    let ctx = currentTraceContext()
    var inIds: seq[ShValueId] = @[]
    for t in operands: inIds.add t.traceId
    let outIds = shops.allGather(ctx.builder, inIds, allGatherDim,
      resultShapes, replicaGroups, channelHandle, useGlobalDeviceIds)
    result = newSeq[Tensor](outIds.len)
    for i, id in outIds:
      result[i] = initTraceTensor(id, operands[i].dtype,
        resultShapes[i], operands[i].device, operands[i].sharding)
    recordTraceOp("allGather", operands, result[0])
  of dmEager:
    raise newException(TensorError,
      "allGather: only supported in trace/jit mode")

proc allReduce*(operands: openArray[Tensor];
    replicaGroups: openArray[seq[int]] = @[@[]];
    computation: ShArgRegionBuilder;
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): seq[Tensor] {.rewOp.} =
  ## Reduce values across all replicas using `computation`.
  case currentMode()
  of dmTrace:
    for t in operands: requireTrace(t, "allReduce")
    let ctx = currentTraceContext()
    var inIds: seq[ShValueId] = @[]
    for t in operands: inIds.add t.traceId
    let outIds = shops.allReduce(ctx.builder, inIds, replicaGroups,
      computation, channelHandle, useGlobalDeviceIds)
    result = newSeq[Tensor](outIds.len)
    for i, id in outIds:
      result[i] = initTraceTensor(id, operands[i].dtype,
        operands[i].shape, operands[i].device, operands[i].sharding)
    recordTraceOp("allReduce", operands, result[0])
  of dmEager:
    raise newException(TensorError,
      "allReduce: only supported in trace/jit mode")

proc reduceScatter*(operand: Tensor; scatterDimension: int;
    resultShape: openArray[int];
    replicaGroups: openArray[seq[int]] = @[@[]];
    computation: ShArgRegionBuilder;
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): Tensor {.rewOp.} =
  ## Reduce and scatter values across replicas.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "reduceScatter")
    let ctx = currentTraceContext()
    let outId = shops.reduceScatter(ctx.builder, operand.traceId,
      scatterDimension, resultShape, replicaGroups, computation,
      channelHandle, useGlobalDeviceIds)
    result = initTraceTensor(outId, operand.dtype, resultShape,
      operand.device, operand.sharding)
    recordTraceOp("reduceScatter", [operand], result)
  of dmEager:
    raise newException(TensorError,
      "reduceScatter: only supported in trace/jit mode")

proc allToAll*(operands: openArray[Tensor];
    splitDimension, concatDimension, splitCount: int;
    resultShapes: openArray[seq[int]];
    replicaGroups: openArray[seq[int]] = @[@[]];
    channelHandle: ChannelHandle = NoChannelHandle): seq[Tensor] {.rewOp.} =
  ## Exchange data across replicas via all-to-all.
  case currentMode()
  of dmTrace:
    for t in operands: requireTrace(t, "allToAll")
    let ctx = currentTraceContext()
    var inIds: seq[ShValueId] = @[]
    for t in operands: inIds.add t.traceId
    let outIds = shops.allToAll(ctx.builder, inIds, splitDimension,
      concatDimension, splitCount, resultShapes, replicaGroups,
      channelHandle)
    result = newSeq[Tensor](outIds.len)
    for i, id in outIds:
      result[i] = initTraceTensor(id, operands[i].dtype,
        resultShapes[i], operands[i].device, operands[i].sharding)
    recordTraceOp("allToAll", operands, result[0])
  of dmEager:
    raise newException(TensorError,
      "allToAll: only supported in trace/jit mode")

proc collectiveBroadcast*(operand: Tensor;
    replicaGroups: openArray[seq[int]] = @[@[]];
    channelHandle: ChannelHandle = NoChannelHandle): Tensor {.rewOp.} =
  ## Broadcast a value from one replica to all others.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "collectiveBroadcast")
    let ctx = currentTraceContext()
    let outId = shops.collectiveBroadcast(ctx.builder, operand.traceId,
      replicaGroups, channelHandle)
    result = initTraceTensor(outId, operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("collectiveBroadcast", [operand], result)
  of dmEager:
    raise newException(TensorError,
      "collectiveBroadcast: only supported in trace/jit mode")

proc collectivePermute*(operand: Tensor;
    sourceTargetPairs: openArray[array[2, int]];
    channelHandle: ChannelHandle = NoChannelHandle): Tensor {.rewOp.} =
  ## Permute data across replicas according to `sourceTargetPairs`.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "collectivePermute")
    let ctx = currentTraceContext()
    let outId = shops.collectivePermute(ctx.builder, operand.traceId,
      sourceTargetPairs, channelHandle)
    result = initTraceTensor(outId, operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("collectivePermute", [operand], result)
  of dmEager:
    raise newException(TensorError,
      "collectivePermute: only supported in trace/jit mode")
