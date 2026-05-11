## Element-wise unary math — `exp`, `log`, `sqrt`, `abs`, `tanh`,
## no-gradient rounding/sign ops, plus the StableHLO coverage unary family.
##
## Same dispatcher contract as the binary ops in `arith.nim`. Each op
## carries `{.rewOp.}` and has a corresponding `registerVjp` entry in
## `src/rew/autograd/registry.nim`.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker

template unaryOpImpl(opName, traceCall: untyped): untyped {.dirty.} =
  case currentMode()
  of dmTrace:
    requireTrace(a, opName)
    let ctx = currentTraceContext()
    let id = traceCall(ctx.builder, a.traceId)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp(opName, [a], result)
  of dmEager:
    requireEager(a, opName)
    let outs = dispatchEager(opName, [a])
    doAssert outs.len == 1, opName & ": eager backend returned wrong arity"
    result = outs[0]

proc exp*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise natural exponential. StableHLO calls this
  ## `stablehlo.exponential`.
  unaryOpImpl("exp", shops.exponential)

proc log*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise natural logarithm.
  unaryOpImpl("log", shops.log)

proc sqrt*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise square root.
  unaryOpImpl("sqrt", shops.sqrt)

proc abs*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise absolute value.
  unaryOpImpl("abs", shops.abs)

proc tanh*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise hyperbolic tangent.
  unaryOpImpl("tanh", shops.tanh)

proc cbrt*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise cube root.
  unaryOpImpl("cbrt", shops.cbrt)

proc ceil*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise ceiling.
  unaryOpImpl("ceil", shops.ceil)

proc expm1*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise `exp(x) - 1`, emitted as `stablehlo.exponential_minus_one`.
  unaryOpImpl("expm1", shops.exponentialMinusOne)

proc floor*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise floor.
  unaryOpImpl("floor", shops.floor)

proc log1p*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise `log(1 + x)`, emitted as `stablehlo.log_plus_one`.
  unaryOpImpl("log1p", shops.logPlusOne)

proc logistic*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise logistic sigmoid.
  unaryOpImpl("logistic", shops.logistic)

proc tan*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise tangent.
  unaryOpImpl("tan", shops.tan)

proc sign*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise sign: `-1`, `0`, or `1` according to each input value.
  unaryOpImpl("sign", shops.sign)

proc roundNearestAfz*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise round to nearest, with halfway cases away from zero.
  unaryOpImpl("roundNearestAfz", shops.roundNearestAfz)

proc roundNearestEven*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise round to nearest, with halfway cases to even.
  unaryOpImpl("roundNearestEven", shops.roundNearestEven)

proc bitwiseNot*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise bitwise/logical NOT.
  unaryOpImpl("bitwiseNot", shops.notOp)

proc countLeadingZeros*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise count of leading zero bits.
  unaryOpImpl("countLeadingZeros", shops.countLeadingZeros)

proc popcnt*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise population count, i.e. number of set bits.
  unaryOpImpl("popcnt", shops.popcnt)

proc optimizationBarrier*(a: Tensor): Tensor {.rewOp.} =
  ## Prevent compiler optimizations from moving values across this op.
  ## Numerically this is an identity.
  unaryOpImpl("optimizationBarrier", shops.optimizationBarrier)

proc stopGradient*(a: Tensor): Tensor {.rewOp.} =
  ## Identity forward, zero gradient backward. Useful for stopping
  ## gradient flow through a subgraph (e.g. target networks in RL,
  ## or GAN discriminator inputs).
  unaryOpImpl("stopGradient", shops.optimizationBarrier)

proc astype*(a: Tensor; dtype: DType): Tensor {.rewOp.} =
  ## Element-wise StableHLO conversion to `dtype`. Shape and device are
  ## preserved; no implicit promotion is performed by other ops.
  case currentMode()
  of dmTrace:
    requireTrace(a, "astype")
    let ctx = currentTraceContext()
    let id = shops.convert(ctx.builder, a.traceId, dtype)
    result = initTraceTensor(id, dtype, a.shape, a.device, a.sharding)
    recordTraceOp("astype", [a], result)
  of dmEager:
    requireEager(a, "astype")
    let outs = dispatchEager("astype", [a], [("dtype", dtype.name)])
    doAssert outs.len == 1, "astype: eager backend returned wrong arity"
    result = outs[0]

proc bitcastConvert*(a: Tensor; dtype: DType;
    outputShape: openArray[int]): Tensor {.rewOp.} =
  ## Reinterpret `a`'s bits as `dtype` with `outputShape`.
  ## The total input and output bit counts must match.
  var outElements = 1
  for d in outputShape:
    if d < 0:
      raise newException(TensorError,
        "bitcastConvert: output shape contains negative dimension " & $d)
    outElements *= d
  let inBits = a.numElements * a.dtype.bitWidth
  let outBits = outElements * dtype.bitWidth
  if inBits != outBits:
    raise newException(TensorError,
      "bitcastConvert: input bit count " & $inBits &
        " differs from output bit count " & $outBits)
  case currentMode()
  of dmTrace:
    requireTrace(a, "bitcastConvert")
    let ctx = currentTraceContext()
    let id = shops.bitcastConvert(ctx.builder, a.traceId, dtype, outputShape)
    result = initTraceTensor(id, dtype, outputShape, a.device, a.sharding)
    recordTraceOp("bitcastConvert", [a], result)
  of dmEager:
    requireEager(a, "bitcastConvert")
    let outs = dispatchEager("bitcastConvert", [a], [
      ("dtype", dtype.name),
      ("output_shape", $(@outputShape)),
    ])
    doAssert outs.len == 1,
      "bitcastConvert: eager backend returned wrong arity"
    result = outs[0]

proc bitcastConvert*(a: Tensor; dtype: DType): Tensor {.rewOp.} =
  ## Reinterpret `a`'s bits as `dtype` without changing shape.
  bitcastConvert(a, dtype, a.shape)

proc isFinite*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise IEEE-754 finiteness test. Returns a same-shape boolean
  ## tensor and accepts floating-point inputs.
  if not a.dtype.isFloat:
    raise newException(TensorError,
      "isFinite: operand must be floating point, got " & $a.dtype)
  case currentMode()
  of dmTrace:
    requireTrace(a, "isFinite")
    let ctx = currentTraceContext()
    let id = shops.isFinite(ctx.builder, a.traceId)
    result = initTraceTensor(id, dtBool, a.shape, a.device, a.sharding)
    recordTraceOp("isFinite", [a], result)
  of dmEager:
    requireEager(a, "isFinite")
    let outs = dispatchEager("isFinite", [a])
    doAssert outs.len == 1, "isFinite: eager backend returned wrong arity"
    result = outs[0]

proc real*(a: Tensor): Tensor {.rewOp.} =
  ## Extracts the real component. Floating-point inputs are returned as
  ## same-dtype tensors; complex inputs produce their component dtype.
  if not (a.dtype.isFloat or a.dtype.isComplex):
    raise newException(TensorError,
      "real: operand must be floating point or complex, got " & $a.dtype)
  let outDType = a.dtype.complexPartDType
  case currentMode()
  of dmTrace:
    requireTrace(a, "real")
    let ctx = currentTraceContext()
    let id = shops.real(ctx.builder, a.traceId)
    result = initTraceTensor(id, outDType, a.shape, a.device, a.sharding)
    recordTraceOp("real", [a], result)
  of dmEager:
    requireEager(a, "real")
    let outs = dispatchEager("real", [a])
    doAssert outs.len == 1, "real: eager backend returned wrong arity"
    result = outs[0]

proc imag*(a: Tensor): Tensor {.rewOp.} =
  ## Extracts the imaginary component. Floating-point inputs produce zeros
  ## with the same dtype; complex inputs produce their component dtype.
  if not (a.dtype.isFloat or a.dtype.isComplex):
    raise newException(TensorError,
      "imag: operand must be floating point or complex, got " & $a.dtype)
  let outDType = a.dtype.complexPartDType
  case currentMode()
  of dmTrace:
    requireTrace(a, "imag")
    let ctx = currentTraceContext()
    let id = shops.imag(ctx.builder, a.traceId)
    result = initTraceTensor(id, outDType, a.shape, a.device, a.sharding)
    recordTraceOp("imag", [a], result)
  of dmEager:
    requireEager(a, "imag")
    let outs = dispatchEager("imag", [a])
    doAssert outs.len == 1, "imag: eager backend returned wrong arity"
    result = outs[0]

proc reducePrecision*(a: Tensor; exponentBits, mantissaBits: int): Tensor
    {.rewOp.} =
  ## Round `a` to the precision described by `exponentBits` and
  ## `mantissaBits`, then convert back to the original dtype.
  if not a.dtype.isFloat:
    raise newException(TensorError,
      "reducePrecision: operand must be floating point, got " & $a.dtype)
  if exponentBits < 1:
    raise newException(TensorError,
      "reducePrecision: exponentBits must be >= 1")
  if mantissaBits < 0:
    raise newException(TensorError,
      "reducePrecision: mantissaBits must be >= 0")
  case currentMode()
  of dmTrace:
    requireTrace(a, "reducePrecision")
    let ctx = currentTraceContext()
    let id = shops.reducePrecision(ctx.builder, a.traceId,
      exponentBits, mantissaBits)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("reducePrecision", [a], result)
  of dmEager:
    requireEager(a, "reducePrecision")
    let outs = dispatchEager("reducePrecision", [a], [
      ("exponent_bits", $exponentBits),
      ("mantissa_bits", $mantissaBits),
    ])
    doAssert outs.len == 1,
      "reducePrecision: eager backend returned wrong arity"
    result = outs[0]

proc sine*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise sine.
  unaryOpImpl("sine", shops.sine)

proc cosine*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise cosine.
  unaryOpImpl("cosine", shops.cosine)

proc sin*(a: Tensor): Tensor =
  ## Element-wise sine. Alias for `sine`.
  sine(a)

proc cos*(a: Tensor): Tensor =
  ## Element-wise cosine. Alias for `cosine`.
  cosine(a)

proc rsqrt*(a: Tensor): Tensor {.rewOp.} =
  ## Element-wise reciprocal square root: `1 / sqrt(x)`.
  unaryOpImpl("rsqrt", shops.rsqrt)

proc clamp*(minVal, a, maxVal: Tensor): Tensor {.rewOp.} =
  ## Element-wise clamp: `max(minVal, min(a, maxVal))`.
  ## `minVal` and `maxVal` must have the same shape as `a` or be scalar.
  requireSameMode(minVal, a, "clamp")
  requireSameMode(a, maxVal, "clamp")
  requireSameDevice(minVal, a, "clamp")
  requireSameDevice(a, maxVal, "clamp")
  if minVal.dtype != a.dtype or maxVal.dtype != a.dtype:
    raise newException(TensorError,
      "clamp: dtype mismatch among operands")
  if minVal.shape.len != 0 and minVal.shape != a.shape:
    raise newException(TensorError,
      "clamp: minVal shape must be scalar or match operand shape")
  if maxVal.shape.len != 0 and maxVal.shape != a.shape:
    raise newException(TensorError,
      "clamp: maxVal shape must be scalar or match operand shape")
  case currentMode()
  of dmTrace:
    requireTrace(minVal, "clamp")
    requireTrace(a, "clamp")
    requireTrace(maxVal, "clamp")
    let ctx = currentTraceContext()
    let id = shops.clamp(ctx.builder, minVal.traceId, a.traceId, maxVal.traceId)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("clamp", [minVal, a, maxVal], result)
  of dmEager:
    requireEager(minVal, "clamp")
    requireEager(a, "clamp")
    requireEager(maxVal, "clamp")
    let outs = dispatchEager("clamp", [minVal, a, maxVal])
    doAssert outs.len == 1, "clamp: eager backend returned wrong arity"
    result = outs[0]

proc uniformQuantize*(operand: Tensor; resultDType: DType): Tensor
    {.rewOp.} =
  ## Quantizes `operand` to `resultDType`. Shape is preserved.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "uniform_quantize")
    let ctx = currentTraceContext()
    let resultType = initTensorType(resultDType, operand.shape)
    let id = shops.uniformQuantize(ctx.builder, operand.traceId, resultType)
    result = initTraceTensor(id, resultDType, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("uniformQuantize", [operand], result)
  of dmEager:
    raise newException(TensorError,
      "uniform_quantize: only supported in trace/jit mode")

proc uniformDequantize*(operand: Tensor; resultDType: DType): Tensor
    {.rewOp.} =
  ## Dequantizes `operand` to `resultDType`. Shape is preserved.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "uniform_dequantize")
    let ctx = currentTraceContext()
    let id = shops.uniformDequantize(ctx.builder, operand.traceId,
      resultDType)
    result = initTraceTensor(id, resultDType, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("uniformDequantize", [operand], result)
  of dmEager:
    raise newException(TensorError,
      "uniform_dequantize: only supported in trace/jit mode")
