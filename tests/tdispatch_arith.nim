## Phase 3 — Tensor + dispatch + arith ops, end-to-end via trace mode.
##
## Eager execution requires a PJRT plugin, so we exercise:
##   * the value-type tensor (construction, predicates, mode guards)
##   * the dispatcher state machine (mode switching, error paths)
##   * the trace path of every `{.rewOp.}` arith op (StableHLO output)
##   * the no-eager-backend error path (Phase 7 will fill this in)

import rew
import rew/pjrt/capi
import rew/binaries/target
import std/strutils

let TestDevice = cpu(0)

var dispatchReleaseTouched {.threadvar.}: int

proc noopReleaser(t: Target; raw: PjrtBufferRaw) {.nimcall, raises: [].} =
  inc dispatchReleaseTouched

template fakeBuffer(addrLit: int): BufferHandle =
  newBufferHandle(tCpu, cast[PjrtBufferRaw](cast[pointer](addrLit)), noopReleaser)

# --- Tensor value type -----------------------------------------------------

block tensor_eager_predicates:
  let h = fakeBuffer(0xdead0001)
  let t = initEagerTensor(h, dtFloat32, [2, 2], TestDevice)
  doAssert t.isEager
  doAssert not t.isTrace
  doAssert t.numElements == 4
  doAssert t.shape == @[2, 2]
  doAssert t.tensorTypeOf == initTensorType(dtFloat32, [2, 2])
  doAssert t.shValueTypeOf == initValueType(initTensorType(dtFloat32, [2, 2]))
  doAssert t.valueTypeOf.kind == vkTensor
  doAssert t.valueTypeOf.element.dtype == dtFloat32
  doAssert t.valueTypeOf.dims[1].size == 2

block tensor_trace_predicates:
  let t = initTraceTensor(ShValueId(7), dtInt32, [3], TestDevice)
  doAssert t.isTrace
  doAssert not t.isEager
  doAssert t.traceId == ShValueId(7)
  doAssert t.buffer.isNil

block tensor_mode_guards:
  let h = fakeBuffer(0xdead0002)
  let eager = initEagerTensor(h, dtFloat32, [2], TestDevice)
  let trace = initTraceTensor(ShValueId(1), dtFloat32, [2], TestDevice)
  doAssertRaises(TensorModeError):
    requireTrace(eager, "test")
  doAssertRaises(TensorModeError):
    requireEager(trace, "test")
  doAssertRaises(TensorModeError):
    requireSameMode(eager, trace, "test")

# --- Dispatch mode ---------------------------------------------------------

block dispatch_mode_default_is_eager:
  doAssert currentMode() == dmEager
  doAssert currentTraceContext().isNil

block dispatch_no_eager_backend_raises:
  clearEagerBackend()
  let h = fakeBuffer(0xdead0003)
  let a = initEagerTensor(h, dtFloat32, [2], TestDevice)
  let b = initEagerTensor(h, dtFloat32, [2], TestDevice)
  doAssertRaises(NoEagerBackendError):
    discard add(a, b)

block dispatch_eager_backend_round_trip:
  ## Inject a fake backend, dispatch through it, then clear it.
  let fake: EagerBackend = proc(op: string; operands: openArray[Tensor];
      attrs: openArray[(string, string)]): seq[Tensor] {.nimcall.} =
    discard attrs
    doAssert op == "add"
    doAssert operands.len == 2
    @[operands[0]]
  setEagerBackend(fake)
  defer: clearEagerBackend()
  let h = fakeBuffer(0xdead0004)
  let a = initEagerTensor(h, dtFloat32, [2], TestDevice)
  let b = initEagerTensor(h, dtFloat32, [2], TestDevice)
  let r = add(a, b)
  doAssert r.isEager
  doAssert r.shape == @[2]

# --- Trace path of arith ops ----------------------------------------------

block trace_emits_add_module:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[3], @[3]])
    let r = add(inputs[0], inputs[1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.add %v1, %v2" in text, text
  doAssert "tensor<3xf32>" in text, text
  doAssert "func.return" in text, text

block trace_emits_chained_ops:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2, 2], @[2, 2]])
    let s = sub(inputs[0], inputs[1])
    let m = mul(s, inputs[0])
    let n = neg(m)
    ctx.traceReturn([n])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.subtract" in text
  doAssert "stablehlo.multiply" in text
  doAssert "stablehlo.negate" in text
  doAssert "tensor<2x2xf32>" in text

block trace_restores_mode_on_exit:
  doAssert currentMode() == dmEager
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1]])
    doAssert currentMode() == dmTrace
    ctx.traceReturn([inputs[0]])
  doAssert currentMode() == dmEager

block trace_dtype_mismatch_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32], @[@[2], @[2]])
    doAssertRaises(TensorError):
      discard add(inputs[0], inputs[1])
    # close the function so endTrace doesn't leak open state
    ctx.traceReturn([inputs[0]])

block trace_shape_mismatch_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2], @[3]])
    doAssertRaises(TensorError):
      discard add(inputs[0], inputs[1])
    ctx.traceReturn([inputs[0]])

# --- VJP registry ----------------------------------------------------------

block vjp_registry_has_arith_entries:
  doAssert hasVjp("add")
  doAssert hasVjp("sub")
  doAssert hasVjp("mul")
  doAssert hasVjp("neg")

block vjp_duplicate_registration_raises:
  doAssertRaises(VjpRegistryError):
    registerVjp("add")
