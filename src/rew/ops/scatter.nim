## Scatter ops: `scatter`, `select_and_scatter`.

import ../tensor
import ../dispatch
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

proc scatter*(inputs: openArray[Tensor]; scatterIndices: Tensor;
    updates: openArray[Tensor]; dims: ScatterDimensionNumbers;
    updateComputation: ShArgRegionBuilder;
    indicesAreSorted = false; uniqueIndices = false): seq[Tensor] {.rewOp.} =
  ## Scatters `updates` into `inputs` at positions given by `scatterIndices`,
  ## combining values with `updateComputation`.
  if inputs.len == 0 or inputs.len != updates.len:
    raise newException(TensorError,
      "scatter: inputs and updates must have the same non-zero length")
  case currentMode()
  of dmTrace:
    for t in inputs: requireTrace(t, "scatter")
    requireTrace(scatterIndices, "scatter")
    for t in updates: requireTrace(t, "scatter")
    let ctx = currentTraceContext()
    var inIds: seq[ShValueId]
    for t in inputs: inIds.add t.traceId
    var upIds: seq[ShValueId]
    for t in updates: upIds.add t.traceId
    let ids = shops.scatter(ctx.builder, inIds, scatterIndices.traceId,
      upIds, dims, updateComputation, indicesAreSorted, uniqueIndices)
    result = newSeq[Tensor](ids.len)
    for i, id in ids:
      result[i] = initTraceTensor(id, inputs[i].dtype, inputs[i].shape,
        inputs[i].device, inputs[i].sharding)
    var allOperands: seq[Tensor]
    allOperands.add inputs
    allOperands.add scatterIndices
    allOperands.add updates
    recordTraceOp("scatter", allOperands, result[0])
  of dmEager:
    raise newException(TensorError,
      "scatter: only supported in trace/jit mode")

proc selectAndScatter*(operand, source, initValue: Tensor;
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    selectComputation, scatterComputation: ShArgRegionBuilder): Tensor
    {.rewOp.} =
  ## Window-based select-and-scatter. `initValue` must be a 0-rank tensor
  ## with the same dtype as `operand`.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "select_and_scatter")
    requireTrace(source, "select_and_scatter")
    requireTrace(initValue, "select_and_scatter")
    let ctx = currentTraceContext()
    let id = shops.selectAndScatter(ctx.builder, operand.traceId,
      source.traceId, initValue.traceId, windowDimensions, windowStrides,
      padding, selectComputation, scatterComputation)
    result = initTraceTensor(id, operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("selectAndScatter", [operand, source, initValue], result)
  of dmEager:
    raise newException(TensorError,
      "select_and_scatter: only supported in trace/jit mode")
