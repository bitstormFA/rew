## Segment reduction ops — scatter-based per-segment aggregation.
##
## These are the core primitive for GNN message aggregation, implementing
## `segmentSum`, `segmentMax`, and `segmentMean` via `stablehlo.scatter`
## with the appropriate update computation (add / maximum).

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker
import ./literal
import ./shape
import ./linalg
import ./arith

proc segmentDimNumbers*(dataRank: int): ScatterDimensionNumbers =
  ## Builds consistent `ScatterDimensionNumbers` for segment operations.
  ##
  ## For data of rank R, the accumulator has shape `[numSegments, ...]`,
  ## where dim 0 is the scatter dimension indexed by segment IDs.
  ## `updateWindowDims` covers all trailing dims beyond the element axis.
  var updateWindowDims: seq[int] = @[]
  for i in 1 ..< dataRank:
    updateWindowDims.add i
  ScatterDimensionNumbers(
    updateWindowDims: updateWindowDims,
    insertedWindowDims: @[0],
    inputBatchingDims: @[],
    scatterIndicesBatchingDims: @[],
    scatterDimsToOperandDims: @[0],
    indexVectorDim: 1,
  )

proc segmentSum*(data, indices: Tensor; numSegments: int): Tensor {.rewOp.} =
  ## Segment-sum reduction: sum `data` values sharing the same `indices` value.
  ##
  ## `data` shape `[E, F1, F2, ...]`, `indices` shape `[E]` (int32),
  ## result shape `[numSegments, F1, F2, ...]`.
  if data.shape.len == 0:
    raise newException(TensorError,
      "segmentSum: data must have rank >= 1")
  if indices.shape.len != 1:
    raise newException(TensorError,
      "segmentSum: indices must be rank 1, got rank " & $indices.shape.len)
  if not (indices.dtype.isSignedInt or indices.dtype.isUnsignedInt):
    raise newException(TensorError,
      "segmentSum: indices must be integer, got " & $indices.dtype)
  if data.shape[0] != indices.shape[0]:
    raise newException(TensorError,
      "segmentSum: data dim 0 (" & $data.shape[0] &
        ") != indices dim 0 (" & $indices.shape[0] & ")")
  if numSegments <= 0:
    raise newException(TensorError,
      "segmentSum: numSegments must be > 0, got " & $numSegments)
  case currentMode()
  of dmTrace:
    requireTrace(data, "segmentSum")
    requireTrace(indices, "segmentSum")
    let ctx = currentTraceContext()
    # Build scatter indices as [E, 1].
    let scatterIdx = unsqueeze(indices, 1)
    # Build zero-filled accumulator of shape [numSegments, F1, ...].
    var accShape = @[numSegments]
    var accSize = numSegments
    for i in 1 ..< data.shape.len:
      accShape.add data.shape[i]
      accSize *= data.shape[i]
    let accData = newSeq[float32](accSize)
    let acc = constantF32(accShape, accData)
    # Scatter via add update computation.
    let dims = segmentDimNumbers(data.shape.len)
    let results = shops.scatter(ctx.builder, @[acc.traceId],
      scatterIdx.traceId, @[data.traceId], dims,
      proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
        @[b.add(xs[0], xs[1])])
    result = initTraceTensor(results[0], data.dtype, accShape,
      data.device, data.sharding)
    recordTraceOp("segmentSum", [data, indices], result,
      @[("numSegments", @[numSegments])])
  of dmEager:
    raise newException(TensorError,
      "segmentSum: only supported in trace/jit mode")

proc segmentMax*(data, indices: Tensor; numSegments: int): Tensor {.rewOp.} =
  ## Segment-max reduction: max of `data` values per segment.
  ##
  ## `data` shape `[E, F1, F2, ...]`, `indices` shape `[E]` (int32),
  ## result shape `[numSegments, F1, F2, ...]`.
  if data.shape.len == 0:
    raise newException(TensorError,
      "segmentMax: data must have rank >= 1")
  if indices.shape.len != 1:
    raise newException(TensorError,
      "segmentMax: indices must be rank 1, got rank " & $indices.shape.len)
  if not (indices.dtype.isSignedInt or indices.dtype.isUnsignedInt):
    raise newException(TensorError,
      "segmentMax: indices must be integer, got " & $indices.dtype)
  if data.shape[0] != indices.shape[0]:
    raise newException(TensorError,
      "segmentMax: data dim 0 (" & $data.shape[0] &
        ") != indices dim 0 (" & $indices.shape[0] & ")")
  if numSegments <= 0:
    raise newException(TensorError,
      "segmentMax: numSegments must be > 0, got " & $numSegments)
  case currentMode()
  of dmTrace:
    requireTrace(data, "segmentMax")
    requireTrace(indices, "segmentMax")
    let ctx = currentTraceContext()
    let scatterIdx = unsqueeze(indices, 1)
    var accShape = @[numSegments]
    var accSize = numSegments
    for i in 1 ..< data.shape.len:
      accShape.add data.shape[i]
      accSize *= data.shape[i]
    # For max reduction, initialise with the lowest representable float32.
    var initData = newSeq[float32](accSize)
    for i in 0 ..< accSize:
      initData[i] = float32.low
    let acc = constantF32(accShape, initData)
    let dims = segmentDimNumbers(data.shape.len)
    let results = shops.scatter(ctx.builder, @[acc.traceId],
      scatterIdx.traceId, @[data.traceId], dims,
      proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
        @[b.maximum(xs[0], xs[1])])
    result = initTraceTensor(results[0], data.dtype, accShape,
      data.device, data.sharding)
    recordTraceOp("segmentMax", [data, indices], result,
      @[("numSegments", @[numSegments])])
  of dmEager:
    raise newException(TensorError,
      "segmentMax: only supported in trace/jit mode")

proc segmentMean*(data, indices: Tensor; numSegments: int): Tensor =
  ## Segment-mean reduction: mean of `data` values per segment.
  ##
  ## Composite: `segmentSum(data) / segmentSum(ones)`.
  let summed = segmentSum(data, indices, numSegments)
  # Build a ones tensor matching the first dim of data.
  var onesShape = @[data.shape[0]]
  var onesData = newSeq[float32](data.shape[0])
  for i in 0 ..< onesData.len:
    onesData[i] = 1'f32
  let ones = constantF32(onesShape, onesData)
  let counts = segmentSum(ones, indices, numSegments)
  # Broadcast count to match summed shape. If summed is 1D, counts is
  # already `[numSegments]`. If ND, unsqueeze count to `[numSegments, 1]`
  # then broadcast to full summed shape.
  if summed.shape.len == 1:
    divide(summed, counts)
  else:
    let countBc = unsqueeze(counts, 1)
    var bcDims: seq[int] = @[]
    for d in 0 ..< summed.shape.len:
      bcDims.add d
    let countFull = broadcastTo(countBc, summed.shape, bcDims)
    divide(summed, countFull)
