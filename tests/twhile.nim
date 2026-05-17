## Phase 7b.2 — in-graph control flow: `compare`, `whileLoop`, `fori`.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block compare_emits_stablehlo_compare:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[3], @[3]])
    let pred = compare(inputs[0], inputs[1], "LT")
    doAssert pred.dtype == dtBool
    doAssert pred.shape == @[3]
    ctx.traceReturn([inputs[0]])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.compare LT" in text

block compare_invalid_direction_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2], @[2]])
    doAssertRaises(TensorError):
      discard compare(inputs[0], inputs[1], "FOO")
    ctx.traceReturn([inputs[0]])

block while_loop_lowers_to_stablehlo_while:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    let zero = scalarI32(0'i32)
    let limit = scalarI32(3'i32)
    let outs = whileLoop(@[zero, inputs[0]],
      proc(carry: openArray[Tensor]): Tensor =
        compare(carry[0], limit, "LT"),
      proc(carry: openArray[Tensor]): seq[Tensor] =
        @[add(carry[0], scalarI32(1'i32)),
          add(carry[1], inputs[0])])
    doAssert outs.len == 2
    doAssert outs[0].dtype == dtInt32
    doAssert outs[1].shape == @[2]
    ctx.traceReturn(outs)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.while" in text
  doAssert "stablehlo.compare" in text

block fori_threads_carry:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let outs = fori(0'i32, 4'i32, @[inputs[0]],
      proc(i: Tensor; carry: openArray[Tensor]): seq[Tensor] =
        doAssert i.dtype == dtInt32
        doAssert i.shape == @[]
        @[add(carry[0], inputs[0])])
    doAssert outs.len == 1
    doAssert outs[0].shape == @[3]
    doAssert outs[0].dtype == dtFloat32
    ctx.traceReturn(outs)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.while" in text

block while_loop_carry_mismatch_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    doAssertRaises(CondError):
      discard whileLoop(@[inputs[0]],
        proc(carry: openArray[Tensor]): Tensor =
          compare(carry[0], carry[0], "LT"),
        proc(carry: openArray[Tensor]): seq[Tensor] =
          # Returns wrong shape.
          @[add(carry[0], carry[0]), carry[0]])
    ctx.traceReturn([inputs[0]])

block while_loop_outside_trace_raises:
  let fakeArg = initTraceTensor(ShValueId(1), dtFloat32, @[2], TestDevice)
  doAssertRaises(CondError):
    discard whileLoop(@[fakeArg],
      proc(carry: openArray[Tensor]): Tensor = fakeArg,
      proc(carry: openArray[Tensor]): seq[Tensor] = @[fakeArg])
