## Phase 5c \u2014 dot_general / matmul / broadcast_in_dim, IR + dispatch.

import rew
import std/strutils

let TestDevice = cpu(0)

# ---- builder + verifier + text -------------------------------------------

block builder_matmul:
  var b = initBuilder("m")
  let f32_2x3 = initTensorType(dtFloat32, [2, 3])
  let f32_3x4 = initTensorType(dtFloat32, [3, 4])
  let f32_2x4 = initTensorType(dtFloat32, [2, 4])
  let args = b.beginFunc("main", [f32_2x3, f32_3x4], [f32_2x4])
  let r = b.matmul(args[0], args[1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text, text
  doAssert "contracting_dims = [1] x [0]" in text, text
  doAssert "tensor<2x4xf32>" in text, text

block builder_dot_general_with_batching:
  var b = initBuilder("m")
  let lhs = initTensorType(dtFloat32, [4, 2, 3])
  let rhs = initTensorType(dtFloat32, [4, 3, 5])
  let outTy = initTensorType(dtFloat32, [4, 2, 5])
  let args = b.beginFunc("main", [lhs, rhs], [outTy])
  let r = b.dotGeneral(args[0], args[1], [0], [0], [2], [1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "batching_dims = [0] x [0]" in text, text
  doAssert "contracting_dims = [2] x [1]" in text, text
  doAssert "tensor<4x2x5xf32>" in text, text

block builder_dot_general_dim_size_mismatch:
  var b = initBuilder()
  let lhs = initTensorType(dtFloat32, [2, 3])
  let rhs = initTensorType(dtFloat32, [4, 5])
  let args = b.beginFunc("main", [lhs, rhs], [])
  doAssertRaises(ShBuilderError):
    discard b.dotGeneral(args[0], args[1], [], [], [1], [0])

block builder_broadcast_in_dim:
  var b = initBuilder("m")
  let inTy = initTensorType(dtFloat32, [3])
  let outTy = initTensorType(dtFloat32, [2, 3])
  let args = b.beginFunc("main", [inTy], [outTy])
  let r = b.broadcastInDim(args[0], [2, 3], [1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.broadcast_in_dim" in text, text
  doAssert "dims = [1]" in text, text
  doAssert "(tensor<3xf32>) -> tensor<2x3xf32>" in text, text

block builder_broadcast_size_mismatch:
  var b = initBuilder()
  let inTy = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [inTy], [])
  doAssertRaises(ShBuilderError):
    discard b.broadcastInDim(args[0], [2, 5], [1])

# ---- dispatch trace mode -------------------------------------------------

block trace_matmul:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 3], @[3, 4]])
    let r = matmul(inputs[0], inputs[1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "tensor<2x4xf32>" in text

block trace_matmul_inner_dim_mismatch:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 3], @[5, 4]])
    doAssertRaises(TensorError):
      discard matmul(inputs[0], inputs[1])
    ctx.traceReturn([inputs[0]])

block trace_broadcast_to:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let r = broadcastTo(inputs[0], [2, 3], [1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.broadcast_in_dim" in text

block trace_broadcast_then_add:
  ## Realistic pattern: bias broadcast onto a matmul output.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 3], @[3, 4], @[4]])
    let xw = matmul(inputs[0], inputs[1])
    let biasB = broadcastTo(inputs[2], [2, 4], [1])
    let r = add(xw, biasB)
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.broadcast_in_dim" in text
  doAssert "stablehlo.add" in text

block phase5c_vjp_registry:
  for op in ["matmul", "dotGeneral", "broadcastTo"]:
    doAssert hasVjp(op), "missing vjp for " & op
