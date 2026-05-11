## Element-wise ternary op `select` (like `torch.where`).

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

proc select*(pred, onTrue, onFalse: Tensor): Tensor {.rewOp.} =
  ## Element-wise conditional selection: returns `onTrue` where `pred` is
  ## true, `onFalse` otherwise. `pred` must be a boolean tensor;
  ## `onTrue`/`onFalse` must agree on dtype and shape.
  if pred.dtype != dtBool:
    raise newException(TensorError,
      "select: pred must be bool, got " & $pred.dtype)
  case currentMode()
  of dmTrace:
    requireTrace(pred, "select")
    let ctx = currentTraceContext()
    let id = shops.select(ctx.builder, pred.traceId, onTrue.traceId,
      onFalse.traceId)
    result = initTraceTensor(id, onTrue.dtype, onTrue.shape, onTrue.device,
      onTrue.sharding)
    recordTraceOp("select", [pred, onTrue, onFalse], result)
  of dmEager:
    requireEager(pred, "select")
    let outs = dispatchEager("select", [pred, onTrue, onFalse])
    doAssert outs.len == 1, "select: eager backend returned wrong arity"
    result = outs[0]
