## Phase 4 — high-level op coverage via the dispatcher (trace mode).

import rew
import std/strutils

let TestDevice = cpu(0)

template assertContains(haystack, needle: string) =
  doAssert needle in haystack,
    "expected " & needle.escape & " in:\n" & haystack

block trace_binary_extras:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2], @[2]])
    let d = divide(inputs[0], inputs[1])
    let mx = maximum(d, inputs[0])
    let mn = minimum(mx, inputs[1])
    ctx.traceReturn([mn])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.divide"
  assertContains text, "stablehlo.maximum"
  assertContains text, "stablehlo.minimum"

block trace_openxla_binary_math:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2], @[2]])
    let a = atan2(inputs[0], inputs[1])
    let p = power(a, inputs[0])
    let r = remainder(p, inputs[1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["atan2", "power", "remainder"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_bitwise:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtUint32, dtUint32], @[@[4], @[4]])
    let a = bitwiseAnd(inputs[0], inputs[1])
    let o = bitwiseOr(a, inputs[0])
    let x = bitwiseXor(o, inputs[1])
    let n = bitwiseNot(x)
    ctx.traceReturn([n])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["and", "or", "xor", "not"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_shifts:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtUint32, dtUint32], @[@[4], @[4]])
    let l = shiftLeft(inputs[0], inputs[1])
    let a = shiftRightArithmetic(l, inputs[1])
    let r = shiftRightLogical(a, inputs[1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["shift_left", "shift_right_arithmetic",
                 "shift_right_logical"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_integer_bit_counts:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtUint32], @[@[4]])
    let clz = countLeadingZeros(inputs[0])
    let pc = popcnt(clz)
    ctx.traceReturn([pc])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["count_leading_zeros", "popcnt"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_optimization_barrier:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let r = optimizationBarrier(inputs[0])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.optimization_barrier"

block trace_openxla_type_changing_unary:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let finite = isFinite(inputs[0])
    let converted = astype(inputs[0], dtInt32)
    let bits = bitcastConvert(inputs[0], dtInt32, [4])
    doAssert finite.dtype == dtBool
    doAssert converted.dtype == dtInt32
    doAssert bits.dtype == dtInt32
    ctx.traceReturn([finite, converted, bits])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["is_finite", "convert", "bitcast_convert"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_complex_real_imag:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[4], @[4]])
    let c = complex(inputs[0], inputs[1])
    let r = real(c)
    let i = imag(c)
    doAssert c.dtype == dtComplex64
    doAssert r.dtype == dtFloat32
    doAssert i.dtype == dtFloat32
    ctx.traceReturn([c, r, i])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.complex"
  assertContains text, "stablehlo.real"
  assertContains text, "stablehlo.imag"

block trace_openxla_reduce_precision:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let r = reducePrecision(inputs[0], 8, 23)
    doAssert r.dtype == dtFloat32
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.reduce_precision"

block trace_select_and_scatter_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[3, 3], @[2, 2], @[]])
    let y = selectAndScatter(inputs[0], inputs[1], inputs[2],
      [2, 2], [1, 1], [[0, 0], [0, 0]],
      proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
        @[b.compare(xs[0], xs[1], "GE")],
      proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
        @[b.add(xs[0], xs[1])])
    doAssert y.shape == @[3, 3]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.select_and_scatter"

block trace_openxla_batch_norm_inference:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(
      @[dtFloat32, dtFloat32, dtFloat32, dtFloat32, dtFloat32],
      @[@[4], @[4], @[4], @[4], @[4]])
    let r = batchNormInference(inputs[0], inputs[1], inputs[2],
      inputs[3], inputs[4], 0.0'f32, 0)
    doAssert r.dtype == dtFloat32
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.batch_norm_inference"

block trace_openxla_batch_norm_training:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 2], @[2], @[2]])
    let outs = batchNormTraining(inputs[0], inputs[1], inputs[2],
      0.0'f32, 1)
    doAssert outs.output.shape == @[2, 2]
    doAssert outs.batchMean.shape == @[2]
    doAssert outs.batchVar.shape == @[2]
    ctx.traceReturn([outs.output, outs.batchMean, outs.batchVar])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.batch_norm_training"

block trace_openxla_batch_norm_grad:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(
      @[dtFloat32, dtFloat32, dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 2], @[2], @[2], @[2], @[2, 2]])
    let outs = batchNormGrad(inputs[0], inputs[1], inputs[2],
      inputs[3], inputs[4], 0.0'f32, 1)
    doAssert outs.gradOperand.shape == @[2, 2]
    doAssert outs.gradScale.shape == @[2]
    doAssert outs.gradOffset.shape == @[2]
    ctx.traceReturn([outs.gradOperand, outs.gradScale, outs.gradOffset])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.batch_norm_grad"

block trace_openxla_dot:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 3], @[3, 4]])
    let r = dot(inputs[0], inputs[1])
    doAssert r.shape == @[2, 4]
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.dot"

block trace_openxla_cholesky_get_dimension_size_and_pad:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 2], @[]])
    let chol = cholesky(inputs[0])
    let dim = getDimensionSize(inputs[0], 0)
    let padded = pad(inputs[0], inputs[1], [1, 1], [1, 1], [0, 0])
    doAssert chol.shape == @[2, 2]
    doAssert dim.dtype == dtInt32
    doAssert dim.shape == @[]
    doAssert padded.shape == @[4, 4]
    ctx.traceReturn([chol, dim, padded])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.cholesky"
  assertContains text, "stablehlo.get_dimension_size"
  assertContains text, "stablehlo.pad"

block trace_openxla_broadcast_and_dynamic_slice_ops:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 2], @[1, 1]])
    let start0 = scalarI32(0'i32)
    let start1 = scalarI32(1'i32)
    let broad = broadcast(inputs[0], [3])
    let sliced = dynamicSlice(inputs[0], [start0, start1], [2, 1])
    let updated = dynamicUpdateSlice(inputs[0], inputs[1], [start0, start1])
    doAssert broad.shape == @[3, 2, 2]
    doAssert sliced.shape == @[2, 1]
    doAssert updated.shape == @[2, 2]
    ctx.traceReturn([broad, sliced, updated])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.broadcast"
  assertContains text, "stablehlo.dynamic_slice"
  assertContains text, "stablehlo.dynamic_update_slice"

block trace_openxla_iota:
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1]])
    let r = iota(dtInt32, [2, 3], 1)
    doAssert r.dtype == dtInt32
    doAssert r.shape == @[2, 3]
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.iota"

block trace_openxla_replica_and_partition_id:
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1]])
    let rid = replicaId()
    let pid = partitionId()
    doAssert rid.dtype == dtUint32
    doAssert pid.dtype == dtUint32
    ctx.traceReturn([rid, pid])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.replica_id"
  assertContains text, "stablehlo.partition_id"

block trace_openxla_dynamic_shape_ops:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 2]])
    let size = scalarI32(2'i32)
    let shape4 = constant(dtInt32, [1], i32Bytes([4'i32]))
    let low = constant(dtInt32, [2], i32Bytes([1'i32, 1'i32]))
    let high = constant(dtInt32, [2], i32Bytes([1'i32, 1'i32]))
    let interior = constant(dtInt32, [2], i32Bytes([0'i32, 0'i32]))
    let sized = setDimensionSize(inputs[0], size, 0)
    let reshaped = dynamicReshape(inputs[0], shape4, [4])
    let padded = dynamicPad(inputs[0], scalarF32(0.0'f32),
      low, high, interior, [4, 4])
    doAssert sized.shape == @[2, 2]
    doAssert reshaped.shape == @[4]
    doAssert padded.shape == @[4, 4]
    ctx.traceReturn([sized, reshaped, padded])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.set_dimension_size"
  assertContains text, "stablehlo.dynamic_reshape"
  assertContains text, "stablehlo.dynamic_pad"

block trace_openxla_dynamic_iota_and_real_dynamic_slice:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 2]])
    let shape = constant(dtInt32, [2], i32Bytes([2'i32, 3'i32]))
    let start = constant(dtInt32, [2], i32Bytes([0'i32, 1'i32]))
    let limit = constant(dtInt32, [2], i32Bytes([2'i32, 2'i32]))
    let strides = constant(dtInt32, [2], i32Bytes([1'i32, 1'i32]))
    let dynIota = dynamicIota(dtInt32, shape, [2, 3], 1)
    let rds = realDynamicSlice(inputs[0], start, limit, strides, [2, 1])
    doAssert dynIota.shape == @[2, 3]
    doAssert rds.shape == @[2, 1]
    ctx.traceReturn([dynIota, rds])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.dynamic_iota"
  assertContains text, "stablehlo.real_dynamic_slice"

block trace_is_finite_rejects_integer:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtInt32], @[@[4]])
    doAssertRaises(TensorError):
      discard isFinite(inputs[0])
    ctx.traceReturn([inputs[0]])

block trace_unary_math:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let e = exp(inputs[0])
    let l = log(e)
    let s = sqrt(l)
    let a = abs(s)
    let t = tanh(a)
    ctx.traceReturn([t])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["exponential", "log", "sqrt", "abs", "tanh"]:
    assertContains text, "stablehlo." & needle

block trace_openxla_unary_math:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let c = cbrt(inputs[0])
    let ce = ceil(c)
    let em = expm1(ce)
    let fl = floor(em)
    let lp = log1p(fl)
    let lo = logistic(lp)
    let ta = tan(lo)
    let si = sign(ta)
    let ra = roundNearestAfz(si)
    let re = roundNearestEven(ra)
    ctx.traceReturn([re])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["cbrt", "ceil", "exponential_minus_one", "floor",
                 "log_plus_one", "logistic", "tan", "sign",
                 "round_nearest_afz", "round_nearest_even"]:
    assertContains text, "stablehlo." & needle

block trace_reshape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let r = reshape(inputs[0], [6])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.reshape"
  assertContains text, "tensor<6xf32>"

block trace_transpose:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3, 4]])
    let r = transpose(inputs[0], [2, 0, 1])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.transpose"
  assertContains text, "dims = [2, 0, 1]"
  assertContains text, "tensor<4x2x3xf32>"

block trace_reverse:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3, 4]])
    let r = reverse(inputs[0], [0, 2])
    ctx.traceReturn([r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.reverse"
  assertContains text, "dims = [0, 2]"

block reshape_bad_count_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    doAssertRaises(TensorError):
      discard reshape(inputs[0], [5])
    ctx.traceReturn([inputs[0]])

block transpose_bad_perm_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    doAssertRaises(TensorError):
      discard transpose(inputs[0], [0, 0])
    ctx.traceReturn([inputs[0]])

block phase4_vjp_registry_is_complete:
  for op in ["add", "sub", "mul", "neg", "divide", "maximum", "minimum",
             "atan2", "power",
             "exp", "log", "sqrt", "abs", "tanh",
             "cbrt", "expm1", "log1p", "logistic", "tan",
             "optimizationBarrier",
             "reshape", "transpose", "reverse"]:
    doAssert hasVjp(op), "missing vjp registration for '" & op & "'"
  doAssert hasVjp("dot"), "missing vjp registration for 'dot'"
  for op in ["ceil", "floor", "remainder"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  for op in ["sign", "roundNearestAfz", "roundNearestEven"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  for op in ["bitwiseAnd", "bitwiseOr", "bitwiseXor", "bitwiseNot"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  for op in ["shiftLeft", "shiftRightArithmetic", "shiftRightLogical"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  for op in ["countLeadingZeros", "popcnt"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  for op in ["astype", "bitcastConvert", "isFinite", "complex", "real",
             "imag"]:
    doAssert hasNoGradient(op), "missing no-gradient policy for '" & op & "'"
  doAssert hasNoGradient("reducePrecision")
  doAssert hasNoGradient("batchNormInference")
