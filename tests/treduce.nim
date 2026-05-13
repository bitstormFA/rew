## Phase 5b — `stablehlo.reduce` IR + verifier + text emitter, plus the
## `reduceSum`/`reduceMax`/`reduceMin` dispatch ops in trace mode.

import rew
import std/strutils

let TestDevice = cpu(0)

# ---- builder + verifier + text -------------------------------------------

block builder_reduce_sum_text:
  var b = initBuilder("m")
  let f32_2x3 = initTensorType(dtFloat32, [2, 3])
  let f32_3 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32_2x3], [f32_3])
  let zero = b.constant(dtFloat32, [], @[0'u8, 0, 0, 0])
  let r = b.reduce(args[0], zero, [0],
    proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reduce(" in text, text
  doAssert "across dimensions = [0]" in text, text
  doAssert "reducer(" in text, text
  doAssert "stablehlo.add" in text, text
  doAssert "stablehlo.return" in text, text
  doAssert "tensor<f32>" in text, text  # element type used in reducer

block builder_reduce_multi_dim:
  var b = initBuilder("m")
  let f32_2x3x4 = initTensorType(dtFloat32, [2, 3, 4])
  let f32_3 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32_2x3x4], [f32_3])
  let init = b.constant(dtFloat32, [], @[0'u8, 0, 0, 0])
  let r = b.reduce(args[0], init, [0, 2],
    proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "across dimensions = [0, 2]" in text, text
  doAssert "tensor<2x3x4xf32>" in text, text
  doAssert "-> tensor<3xf32>" in text, text

block builder_reduce_init_must_be_scalar:
  var b = initBuilder()
  let f32_2 = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main", [f32_2, f32_2], [])
  doAssertRaises(ShBuilderError):
    discard b.reduce(args[0], args[1], [0],
      proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))

block builder_reduce_dim_out_of_range:
  var b = initBuilder()
  let f32_2x3 = initTensorType(dtFloat32, [2, 3])
  let args = b.beginFunc("main", [f32_2x3], [])
  let init = b.constant(dtFloat32, [], @[0'u8, 0, 0, 0])
  doAssertRaises(ShBuilderError):
    discard b.reduce(args[0], init, [5],
      proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))

block builder_reduce_dim_repeated:
  var b = initBuilder()
  let f32_2x3 = initTensorType(dtFloat32, [2, 3])
  let args = b.beginFunc("main", [f32_2x3], [])
  let init = b.constant(dtFloat32, [], @[0'u8, 0, 0, 0])
  doAssertRaises(ShBuilderError):
    discard b.reduce(args[0], init, [0, 0],
      proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))

block stablehlo_return_outside_region_raises:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main", [f32], [])
  doAssertRaises(ShBuilderError):
    b.stablehloReturn([args[0]])

# ---- dispatch trace mode -------------------------------------------------

block trace_reduce_sum:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let r = reduceSum(inputs[0], [0])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reduce(" in text
  doAssert "stablehlo.add" in text
  doAssert "across dimensions = [0]" in text

block trace_reduce_max:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let r = reduceMax(inputs[0], [0])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.maximum" in text
  doAssert "tensor<f32>" in text  # output is rank-0

block trace_reduce_min:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 2]])
    let r = reduceMin(inputs[0], [1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.minimum" in text
  doAssert "-> tensor<2xf32>" in text

block trace_reduce_prod:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let r = reduceProd(inputs[0], [-1])
    doAssert r.shape == @[2]
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.multiply" in text
  doAssert "across dimensions = [1]" in text

block trace_bool_reductions:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtBool], @[@[2, 3]])
    let a = all(inputs[0], [1])
    let b = any(inputs[0], [0], keepdims = true)
    doAssert a.dtype == dtBool
    doAssert a.shape == @[2]
    doAssert b.dtype == dtBool
    doAssert b.shape == @[1, 3]
    ctx.traceReturn([a, b])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.and" in text
  doAssert "stablehlo.or" in text

block trace_reduction_composite_dim_normalization:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let lse = logSumExp(inputs[0], [-1])
    let v = variance(inputs[0], [-1])
    doAssert lse.shape == @[2]
    doAssert v.shape == @[2]
    ctx.traceReturn([lse, v])
  let m = ctx.builder.build()
  verify(m)

block trace_reduction_composite_api_errors:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let vector = initTraceTensor(ShValueId(99), dtFloat32, [3], TestDevice)
    doAssertRaises(TensorError):
      discard cumsum(inputs[0], 2)
    doAssertRaises(TensorError):
      discard norm(inputs[0], p = 0'f32)
    doAssertRaises(TensorError):
      discard cov(vector)
    ctx.traceReturn([inputs[0]])

block trace_reduce_window:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 4, 1]])
    let init = constantF32([], @[0'f32])
    let r = reduceWindow(inputs[0], init,
      [1, 2, 2, 1], [1, 2, 2, 1],
      [[0, 0], [0, 0], [0, 0], [0, 0]],
      [1, 1, 1, 1], [1, 1, 1, 1],
      proc(b: var ShBuilder; x, y: ShValueId): ShValueId = b.add(x, y))
    doAssert r.shape == @[1, 2, 2, 1]
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reduce_window" in text

block reduce_dim_validation_in_dispatch:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    doAssertRaises(TensorError):
      discard reduceSum(inputs[0], [])
    doAssertRaises(TensorError):
      discard reduceSum(inputs[0], [5])
    doAssertRaises(TensorError):
      discard reduceSum(inputs[0], [0, 0])
    ctx.traceReturn([inputs[0]])

block reduce_negative_dim_validation:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let r = reduceSum(inputs[0], [-1])
    doAssert r.shape == @[2]
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)

block reduce_rejects_non_float32:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtInt32], @[@[3]])
    doAssertRaises(TensorError):
      discard reduceSum(inputs[0], [0])
    ctx.traceReturn([inputs[0]])

block bool_reduction_rejects_non_bool:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    doAssertRaises(TensorError):
      discard all(inputs[0], 0)
    doAssertRaises(TensorError):
      discard any(inputs[0], [0])
    ctx.traceReturn([inputs[0]])

block phase5b_vjp_registry:
  for op in ["reduceSum", "reduceMax", "reduceMin", "reduceProd"]:
    doAssert hasVjp(op), "missing vjp for " & op
  for op in ["all", "any"]:
    doAssert hasNoGradient(op), "missing no-grad policy for " & op
