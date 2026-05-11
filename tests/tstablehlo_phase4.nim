## Phase 4 — verifier + textual emitter for the new ops.

import rew
import std/strutils

block verify_div_max_min:
  var b = initBuilder("m")
  let f32x2 = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main", [f32x2, f32x2], [f32x2])
  let d = b.divide(args[0], args[1])
  let m = b.maximum(d, args[0])
  let n = b.minimum(m, args[1])
  b.returnOp([n])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "stablehlo.divide" in text
  doAssert "stablehlo.maximum" in text
  doAssert "stablehlo.minimum" in text

block verify_openxla_binary_math:
  var b = initBuilder("ob")
  let f32x2 = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main", [f32x2, f32x2], [f32x2])
  let a = b.atan2(args[0], args[1])
  let p = b.power(a, args[0])
  let r = b.remainder(p, args[1])
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  for needle in ["atan2", "power", "remainder"]:
    doAssert "stablehlo." & needle in text, text

block verify_openxla_bitwise:
  var b = initBuilder("bits")
  let u32x4 = initTensorType(dtUint32, [4])
  let args = b.beginFunc("main", [u32x4, u32x4], [u32x4])
  let a = b.andOp(args[0], args[1])
  let o = b.orOp(a, args[0])
  let x = b.xorOp(o, args[1])
  let n = b.notOp(x)
  b.returnOp([n])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  for needle in ["and", "or", "xor", "not"]:
    doAssert "stablehlo." & needle in text, text

block verify_openxla_shifts:
  var b = initBuilder("shifts")
  let u32x4 = initTensorType(dtUint32, [4])
  let args = b.beginFunc("main", [u32x4, u32x4], [u32x4])
  let l = b.shiftLeft(args[0], args[1])
  let a = b.shiftRightArithmetic(l, args[1])
  let r = b.shiftRightLogical(a, args[1])
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  for needle in ["shift_left", "shift_right_arithmetic",
                 "shift_right_logical"]:
    doAssert "stablehlo." & needle in text, text

block verify_openxla_integer_bit_counts:
  var b = initBuilder("bitcounts")
  let u32x4 = initTensorType(dtUint32, [4])
  let args = b.beginFunc("main", [u32x4], [u32x4])
  let clz = b.countLeadingZeros(args[0])
  let pc = b.popcnt(clz)
  b.returnOp([pc])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  for needle in ["count_leading_zeros", "popcnt"]:
    doAssert "stablehlo." & needle in text, text

block verify_openxla_optimization_barrier:
  var b = initBuilder("barrier")
  let f32x4 = initTensorType(dtFloat32, [4])
  let args = b.beginFunc("main", [f32x4], [f32x4])
  let r = b.optimizationBarrier(args[0])
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "stablehlo.optimization_barrier" in text, text

block verify_openxla_type_changing_unary:
  var b = initBuilder("typed")
  let f32x4 = initTensorType(dtFloat32, [4])
  let i32x4 = initTensorType(dtInt32, [4])
  let boolx4 = initTensorType(dtBool, [4])
  let args = b.beginFunc("main", [f32x4], [boolx4, i32x4, i32x4])
  let finite = b.isFinite(args[0])
  let converted = b.convert(args[0], dtInt32)
  let bits = b.bitcastConvert(args[0], dtInt32, [4])
  b.returnOp([finite, converted, bits])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  for needle in ["is_finite", "convert", "bitcast_convert"]:
    doAssert "\"stablehlo." & needle & "\"" in text, text

block verify_openxla_reduce_precision:
  var b = initBuilder("reduced_precision")
  let f32x4 = initTensorType(dtFloat32, [4])
  let args = b.beginFunc("main", [f32x4], [f32x4])
  let r = b.reducePrecision(args[0], 8, 23)
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.reduce_precision\"" in text, text
  doAssert "exponent_bits = 8 : i32" in text, text
  doAssert "mantissa_bits = 23 : i32" in text, text

block verify_openxla_batch_norm_inference:
  var b = initBuilder("batch_norm")
  let f32x4 = initTensorType(dtFloat32, [4])
  let args = b.beginFunc("main", [f32x4, f32x4, f32x4, f32x4, f32x4], [f32x4])
  let r = b.batchNormInference(args[0], args[1], args[2],
    args[3], args[4], 0.0'f32, 0)
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.batch_norm_inference\"" in text, text
  doAssert "feature_index = 0 : i64" in text, text

block verify_openxla_batch_norm_training:
  var b = initBuilder("batch_norm_training")
  let operandTy = initTensorType(dtFloat32, [2, 2])
  let featureTy = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main", [operandTy, featureTy, featureTy],
    [operandTy, featureTy, featureTy])
  let outs = b.batchNormTraining(args[0], args[1], args[2], 0.0'f32, 1)
  b.returnOp(outs)
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.batch_norm_training\"" in text, text
  doAssert "feature_index = 1 : i64" in text, text
  doAssert "(tensor<2x2xf32>, tensor<2xf32>, tensor<2xf32>)" in text, text

block verify_openxla_batch_norm_grad:
  var b = initBuilder("batch_norm_grad")
  let operandTy = initTensorType(dtFloat32, [2, 2])
  let featureTy = initTensorType(dtFloat32, [2])
  let args = b.beginFunc("main",
    [operandTy, featureTy, featureTy, featureTy, operandTy],
    [operandTy, featureTy, featureTy])
  let outs = b.batchNormGrad(args[0], args[1], args[2], args[3],
    args[4], 0.0'f32, 1)
  b.returnOp(outs)
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.batch_norm_grad\"" in text, text
  doAssert "feature_index = 1 : i64" in text, text

block verify_openxla_dot:
  var b = initBuilder("dot")
  let lhsTy = initTensorType(dtFloat32, [2, 3])
  let rhsTy = initTensorType(dtFloat32, [3, 4])
  let outTy = initTensorType(dtFloat32, [2, 4])
  let args = b.beginFunc("main", [lhsTy, rhsTy], [outTy])
  let r = b.dot(args[0], args[1])
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.dot\"" in text, text
  doAssert "tensor<2x4xf32>" in text, text

block verify_openxla_cholesky_get_dimension_size_and_pad:
  var b = initBuilder("shape_linalg_misc")
  let matrixTy = initTensorType(dtFloat32, [2, 2])
  let scalarTy = initTensorType(dtFloat32, [])
  let i32ScalarTy = initTensorType(dtInt32, [])
  let args = b.beginFunc("main", [matrixTy, scalarTy],
    [matrixTy, i32ScalarTy, initTensorType(dtFloat32, [4, 4])])
  let chol = b.cholesky(args[0], lower = true)
  let dim = b.getDimensionSize(args[0], 0)
  let padded = b.pad(args[0], args[1], [1, 1], [1, 1], [0, 0])
  b.returnOp([chol, dim, padded])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.cholesky\"" in text, text
  doAssert "\"stablehlo.get_dimension_size\"" in text, text
  doAssert "\"stablehlo.pad\"" in text, text
  doAssert "tensor<4x4xf32>" in text, text

block verify_openxla_broadcast_and_dynamic_slice_ops:
  var b = initBuilder("dynamic_slice_ops")
  let matrixTy = initTensorType(dtFloat32, [2, 2])
  let updateTy = initTensorType(dtFloat32, [1, 1])
  let i32ScalarTy = initTensorType(dtInt32, [])
  let args = b.beginFunc("main",
    [matrixTy, updateTy, i32ScalarTy, i32ScalarTy],
    [initTensorType(dtFloat32, [3, 2, 2]),
     initTensorType(dtFloat32, [2, 1]),
     matrixTy])
  let broad = b.broadcast(args[0], [3])
  let sliced = b.dynamicSlice(args[0], [args[2], args[3]], [2, 1])
  let updated = b.dynamicUpdateSlice(args[0], args[1], [args[2], args[3]])
  b.returnOp([broad, sliced, updated])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.broadcast\"" in text, text
  doAssert "\"stablehlo.dynamic_slice\"" in text, text
  doAssert "\"stablehlo.dynamic_update_slice\"" in text, text
  doAssert "slice_sizes = array<i64: 2, 1>" in text, text

block verify_openxla_complex_real_imag:
  var b = initBuilder("complex_ops")
  let f32x2 = initTensorType(dtFloat32, [2])
  let c64x2 = initTensorType(dtComplex64, [2])
  let args = b.beginFunc("main", [f32x2, f32x2], [c64x2, f32x2, f32x2])
  let c = b.complexOp(args[0], args[1])
  let r = b.real(c)
  let i = b.imag(c)
  b.returnOp([c, r, i])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "stablehlo.complex" in text, text
  doAssert "\"stablehlo.real\"" in text, text
  doAssert "\"stablehlo.imag\"" in text, text
  doAssert "tensor<2xcomplex<f32>>" in text, text

block verify_openxla_iota:
  var b = initBuilder("iota")
  let outTy = initTensorType(dtInt32, [2, 3])
  discard b.beginFunc("main", [], [outTy])
  let r = b.iota(dtInt32, [2, 3], 1)
  b.returnOp([r])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.iota\"" in text, text
  doAssert "iota_dimension = 1 : i64" in text, text

block verify_openxla_replica_and_partition_id:
  var b = initBuilder("topology_ids")
  let scalarTy = initTensorType(dtUint32, [])
  discard b.beginFunc("main", [], [scalarTy, scalarTy])
  let rid = b.replicaId()
  let pid = b.partitionId()
  b.returnOp([rid, pid])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.replica_id\"" in text, text
  doAssert "\"stablehlo.partition_id\"" in text, text

block verify_openxla_dynamic_shape_ops:
  var b = initBuilder("dynamic_shape_ops")
  let matrixTy = initTensorType(dtFloat32, [2, 2])
  let scalarF32Ty = initTensorType(dtFloat32, [])
  let scalarI32Ty = initTensorType(dtInt32, [])
  let vec1I32Ty = initTensorType(dtInt32, [1])
  let vec2I32Ty = initTensorType(dtInt32, [2])
  let args = b.beginFunc("main",
    [matrixTy, scalarI32Ty, vec1I32Ty, scalarF32Ty, vec2I32Ty, vec2I32Ty,
     vec2I32Ty],
    [matrixTy, initTensorType(dtFloat32, [4]), initTensorType(dtFloat32, [4, 4])])
  let sized = b.setDimensionSize(args[0], args[1], 0)
  let reshaped = b.dynamicReshape(args[0], args[2], [4])
  let padded = b.dynamicPad(args[0], args[3], args[4], args[5], args[6],
    [4, 4])
  b.returnOp([sized, reshaped, padded])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.set_dimension_size\"" in text, text
  doAssert "\"stablehlo.dynamic_reshape\"" in text, text
  doAssert "\"stablehlo.dynamic_pad\"" in text, text

block verify_openxla_dynamic_iota_and_real_dynamic_slice:
  var b = initBuilder("dynamic_iota_slice")
  let matrixTy = initTensorType(dtFloat32, [2, 2])
  let vec2I32Ty = initTensorType(dtInt32, [2])
  let args = b.beginFunc("main",
    [vec2I32Ty, matrixTy, vec2I32Ty, vec2I32Ty, vec2I32Ty],
    [initTensorType(dtInt32, [2, 3]), initTensorType(dtFloat32, [2, 1])])
  let dynIota = b.dynamicIota(dtInt32, args[0], [2, 3], 1)
  let rds = b.realDynamicSlice(args[1], args[2], args[3], args[4], [2, 1])
  b.returnOp([dynIota, rds])
  b.endFunc()
  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "\"stablehlo.dynamic_iota\"" in text, text
  doAssert "\"stablehlo.real_dynamic_slice\"" in text, text

block verify_is_finite_requires_float:
  doAssertRaises(ShBuilderError):
    var b = initBuilder("bad_finite")
    let i32x4 = initTensorType(dtInt32, [4])
    let args = b.beginFunc("main", [i32x4], [])
    discard b.isFinite(args[0])

block verify_reduce_precision_requires_float:
  doAssertRaises(ShBuilderError):
    var b = initBuilder("bad_reduce_precision")
    let i32x4 = initTensorType(dtInt32, [4])
    let args = b.beginFunc("main", [i32x4], [])
    discard b.reducePrecision(args[0], 8, 23)

block verify_bitcast_convert_bit_count_mismatch_raises:
  doAssertRaises(ShBuilderError):
    var b = initBuilder("bad_bits")
    let f32x4 = initTensorType(dtFloat32, [4])
    let args = b.beginFunc("main", [f32x4], [])
    discard b.bitcastConvert(args[0], dtInt32, [3])

block verify_unary_math:
  var b = initBuilder("u")
  let f32 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32], [f32])
  let e = b.exponential(args[0])
  let l = b.log(e)
  let s = b.sqrt(l)
  let a = b.abs(s)
  let t = b.tanh(a)
  b.returnOp([t])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in ["exponential", "log", "sqrt", "abs", "tanh"]:
    doAssert "stablehlo." & needle in text, text

block verify_openxla_unary_math:
  var b = initBuilder("ou")
  let f32 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32], [f32])
  let c = b.cbrt(args[0])
  let ce = b.ceil(c)
  let em = b.exponentialMinusOne(ce)
  let fl = b.floor(em)
  let lp = b.logPlusOne(fl)
  let lo = b.logistic(lp)
  let ta = b.tan(lo)
  let si = b.sign(ta)
  let ra = b.roundNearestAfz(si)
  let re = b.roundNearestEven(ra)
  b.returnOp([re])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in ["cbrt", "ceil", "exponential_minus_one", "floor",
                 "log_plus_one", "logistic", "tan", "sign",
                 "round_nearest_afz", "round_nearest_even"]:
    doAssert "stablehlo." & needle in text, text

block verify_reshape:
  var b = initBuilder("r")
  let f32In = initTensorType(dtFloat32, [2, 3])
  let f32Out = initTensorType(dtFloat32, [6])
  let args = b.beginFunc("main", [f32In], [f32Out])
  let r = b.reshape(args[0], [6])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reshape %v1 : (tensor<2x3xf32>) -> tensor<6xf32>" in
    text, text

block verify_reshape_element_count_mismatch_raises:
  doAssertRaises(ShBuilderError):
    var b = initBuilder()
    let f32In = initTensorType(dtFloat32, [2, 3])
    let argIds = b.beginFunc("main", [f32In], [])
    discard b.reshape(argIds[0], [5])

block verify_transpose:
  var b = initBuilder("t")
  let f32In = initTensorType(dtFloat32, [2, 3, 4])
  let f32Out = initTensorType(dtFloat32, [4, 2, 3])
  let args = b.beginFunc("main", [f32In], [f32Out])
  let r = b.transpose(args[0], [2, 0, 1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.transpose %v1, dims = [2, 0, 1]" in text, text
  doAssert "(tensor<2x3x4xf32>) -> tensor<4x2x3xf32>" in text, text

block verify_reverse:
  var b = initBuilder("rev")
  let f32In = initTensorType(dtFloat32, [2, 3, 4])
  let args = b.beginFunc("main", [f32In], [f32In])
  let r = b.reverse(args[0], [0, 2])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reverse %v1, dims = [0, 2]" in text, text

block verify_transpose_bad_permutation:
  doAssertRaises(ShBuilderError):
    var b = initBuilder()
    let inTy = initTensorType(dtFloat32, [2, 3])
    let argIds = b.beginFunc("main", [inTy], [])
    discard b.transpose(argIds[0], [0, 0])
  doAssertRaises(ShBuilderError):
    var b = initBuilder()
    let inTy = initTensorType(dtFloat32, [2, 3])
    let argIds = b.beginFunc("main", [inTy], [])
    discard b.transpose(argIds[0], [0])
