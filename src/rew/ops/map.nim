## Map op: element-wise map with a computation region.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

proc mapOp*(operand: Tensor; outputDtype: DType; dimensions: openArray[int];
    computation: proc (b: var ShBuilder; arg: ShValueId): ShValueId
      {.closure.}): Tensor {.rewOp.} =
  ## Applies `computation` element-wise to `operand` over the given
  ## `dimensions`. The result has `outputDtype` and the same shape as
  ## `operand`.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "mapOp")
    let ctx = currentTraceContext()
    let ids = shops.mapOp(ctx.builder, [operand.traceId], [outputDtype],
      dimensions,
      proc(b: var ShBuilder; args: openArray[ShValueId]): seq[ShValueId] =
        result = @[computation(b, args[0])])
    result = initTraceTensor(ids[0], outputDtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("mapOp", [operand], result,
      @[("dimensions", @dimensions)])
  of dmEager:
    raise newException(TensorError,
      "mapOp: only supported in trace/jit mode")
