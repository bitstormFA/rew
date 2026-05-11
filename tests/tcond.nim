## Phase 7b.1 \u2014 in-graph control flow: `cond`.

import rew
import std/strutils

let TestDevice = cpu(0)

block cond_lowers_to_stablehlo_if:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[3], @[3]])
    let pred = scalarBool(true)
    let res = cond(pred,
      proc(): Tensor = add(inputs[0], inputs[1]),
      proc(): Tensor = sub(inputs[0], inputs[1]))
    doAssert res.shape == @[3]
    doAssert res.dtype == dtFloat32
    ctx.traceReturn([res])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.if" in text
  doAssert "stablehlo.add" in text
  doAssert "stablehlo.subtract" in text

block cond_multi_output:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    let pred = scalarBool(false)
    let outs = condN(pred,
      proc(): seq[Tensor] = @[add(inputs[0], inputs[0]), neg(inputs[0])],
      proc(): seq[Tensor] = @[sub(inputs[0], inputs[0]), inputs[0]])
    doAssert outs.len == 2
    doAssert outs[0].shape == @[2]
    doAssert outs[1].shape == @[2]
    ctx.traceReturn(outs)
  let m = ctx.builder.build()
  verify(m)

block cond_branch_shape_mismatch:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2], @[3]])
    let pred = scalarBool(true)
    doAssertRaises(CondError):
      discard cond(pred,
        proc(): Tensor = inputs[0],
        proc(): Tensor = inputs[1])
    ctx.traceReturn(@[inputs[0]])

block cond_predicate_must_be_bool:
  withTrace ctx, "main", TestDevice:
    let scalarShape: seq[int] = @[]
    let inputs = ctx.traceInputs(@[dtFloat32], @[scalarShape])
    doAssertRaises(CondError):
      discard cond(inputs[0],
        proc(): Tensor = inputs[0],
        proc(): Tensor = inputs[0])
    ctx.traceReturn(inputs)

block cond_outside_trace_raises:
  let fakePred = initTraceTensor(ShValueId(1), dtBool, @[], TestDevice)
  let fakeArg = initTraceTensor(ShValueId(2), dtFloat32, @[2], TestDevice)
  doAssertRaises(CondError):
    discard cond(fakePred,
      proc(): Tensor = fakeArg,
      proc(): Tensor = fakeArg)
