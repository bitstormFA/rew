## Element-wise comparison op.
##
## Produces a boolean tensor of the same shape as the operands.
## `direction` must be one of ``CompareDirections``: "LT", "LE", "GT",
## "GE", "EQ", "NE".

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

const CompareDirections* = ["LT", "LE", "GT", "GE", "EQ", "NE"]

proc compare*(lhs, rhs: Tensor; direction: string): Tensor {.rewOp.} =
  ## Element-wise comparison. `direction` must be one of "LT", "LE", "GT",
  ## "GE", "EQ", "NE". Returns a boolean tensor with the same shape as the
  ## operands.
  if direction notin CompareDirections:
    raise newException(TensorError,
      "compare: invalid direction '" & direction &
        "'; must be one of LT, LE, GT, GE, EQ, NE")
  case currentMode()
  of dmTrace:
    requireTrace(lhs, "compare")
    requireTrace(rhs, "compare")
    let ctx = currentTraceContext()
    let id = shops.compare(ctx.builder, lhs.traceId, rhs.traceId, direction)
    result = initTraceTensor(id, dtBool, lhs.shape, lhs.device, lhs.sharding)
    recordTraceOp("compare", [lhs, rhs], result)
  of dmEager:
    requireEager(lhs, "compare")
    requireEager(rhs, "compare")
    let outs = dispatchEager("compare", [lhs, rhs],
      [("comparison_direction", direction)])
    doAssert outs.len == 1, "compare: eager backend returned wrong arity"
    result = outs[0]
