## Element-wise arithmetic — binary `add`, `sub`, `mul`, `div`, `max`,
## `min`, `atan2`, `power`, `remainder`, bitwise/shift binary ops, and
## the unary `neg`.
##
## Every op is dispatcher-aware: in trace mode it emits the matching
## StableHLO op into the active builder and returns a trace tensor; in
## eager mode it routes through `dispatchEager`, which calls the
## registered execution backend (Phase 7). Each op carries the `{.rewOp.}`
## marker so the vjp-coverage lint can verify a matching
## `registerVjp(...)` exists in `src/rew/autograd/registry.nim`.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

# ---- shape / dtype helpers ------------------------------------------------

proc requireSameShapeDtype*(a, b: Tensor; op: string) =
  ## Public so sibling op modules (`unary.nim`, `shape.nim`) can share it.
  if a.dtype != b.dtype:
    raise newException(TensorError,
      op & ": dtype mismatch (" & $a.dtype & " vs " & $b.dtype & ")")
  if a.shape != b.shape:
    raise newException(TensorError,
      op & ": shape mismatch")

# ---- emitters -------------------------------------------------------------

template binaryOpImpl(opName, traceCall: untyped): untyped {.dirty.} =
  ## Internal: shared body for the element-wise binary ops. Captures `a`,
  ## `b`, and emits an SSA op via `traceCall` in trace mode.
  requireSameMode(a, b, opName)
  requireSameDevice(a, b, opName)
  requireSameShapeDtype(a, b, opName)
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = traceCall(ctx.builder, a.traceId, b.traceId)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp(opName, [a, b], result)
  of dmEager:
    let outs = dispatchEager(opName, [a, b])
    doAssert outs.len == 1, opName & ": eager backend returned wrong arity"
    result = outs[0]

proc add*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise addition. Operands must agree on dtype, shape, device,
  ## and dispatch mode.
  binaryOpImpl("add", shops.add)

proc sub*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise subtraction. Same constraints as `add`.
  binaryOpImpl("sub", shops.sub)

proc mul*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise multiplication. Same constraints as `add`.
  binaryOpImpl("mul", shops.mul)

proc divide*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise division. Same constraints as `add`.
  binaryOpImpl("divide", shops.divide)

proc maximum*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise maximum. Same constraints as `add`.
  binaryOpImpl("maximum", shops.maximum)

proc minimum*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise minimum. Same constraints as `add`.
  binaryOpImpl("minimum", shops.minimum)

proc atan2*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise two-argument arctangent. Same constraints as `add`.
  binaryOpImpl("atan2", shops.atan2)

proc power*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise exponentiation, `a` raised to `b`.
  binaryOpImpl("power", shops.power)

proc remainder*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise remainder after division.
  binaryOpImpl("remainder", shops.remainder)

proc complex*(real, imag: Tensor): Tensor {.rewOp.} =
  ## Builds a complex tensor from same-shaped real and imaginary tensors.
  requireSameMode(real, imag, "complex")
  requireSameDevice(real, imag, "complex")
  requireSameShapeDtype(real, imag, "complex")
  if real.dtype notin {dtFloat32, dtFloat64}:
    raise newException(TensorError,
      "complex: operands must be float32 or float64, got " & $real.dtype)
  let outDType = real.dtype.complexDType
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.complexOp(ctx.builder, real.traceId, imag.traceId)
    result = initTraceTensor(id, outDType, real.shape, real.device,
      real.sharding)
    recordTraceOp("complex", [real, imag], result)
  of dmEager:
    let outs = dispatchEager("complex", [real, imag])
    doAssert outs.len == 1, "complex: eager backend returned wrong arity"
    result = outs[0]

proc bitwiseAnd*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise bitwise/logical AND. Operands must agree on dtype,
  ## shape, device, and dispatch mode.
  binaryOpImpl("bitwiseAnd", shops.andOp)

proc bitwiseOr*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise bitwise/logical OR. Same constraints as `bitwiseAnd`.
  binaryOpImpl("bitwiseOr", shops.orOp)

proc bitwiseXor*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise bitwise/logical XOR. Same constraints as `bitwiseAnd`.
  binaryOpImpl("bitwiseXor", shops.xorOp)

proc shiftLeft*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise left shift. Same constraints as `bitwiseAnd`.
  binaryOpImpl("shiftLeft", shops.shiftLeft)

proc shiftRightArithmetic*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise arithmetic right shift. Same constraints as `bitwiseAnd`.
  binaryOpImpl("shiftRightArithmetic", shops.shiftRightArithmetic)

proc shiftRightLogical*(a, b: Tensor): Tensor {.rewOp.} =
  ## Element-wise logical right shift. Same constraints as `bitwiseAnd`.
  binaryOpImpl("shiftRightLogical", shops.shiftRightLogical)

proc neg*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise negation.
  case currentMode()
  of dmTrace:
    requireTrace(a, "neg")
    let ctx = currentTraceContext()
    let id = shops.neg(ctx.builder, a.traceId)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("neg", [a], result)
  of dmEager:
    requireEager(a, "neg")
    let outs = dispatchEager("neg", [a])
    doAssert outs.len == 1, "neg: eager backend returned wrong arity"
    result = outs[0]
