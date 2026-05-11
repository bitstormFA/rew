## Activation functions.
##
## Each activation is composite — it expands to existing primitive
## ops (`maximum`, `mul`, etc.) so autodiff can differentiate it
## without a dedicated vjp rule. That keeps the registry small and
## makes the math auditable from one place.

import std/math
import ../tensor
import ../dtype
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/compare
import ../ops/shape
import ../ops/linalg
import ../ops/reduce
import ../ops/ternary
import ../ops/concat

proc relu*(x: Tensor): Tensor =
  ## Element-wise rectified linear unit: `max(x, 0)`.
  let zero = scalarF32(0'f32)
  var dims: seq[int] = @[]
  let zeroBroadcast = broadcastTo(zero, x.shape, dims)
  maximum(x, zeroBroadcast)

proc sigmoid*(x: Tensor): Tensor =
  ## Element-wise sigmoid: `1 / (1 + exp(-x))`.
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, x.shape, dims)
  let expNeg = exp(neg(x))
  divide(oneB, add(oneB, expNeg))

proc silu*(x: Tensor): Tensor =
  ## Element-wise SiLU (Swish): `x * sigmoid(x)`.
  mul(x, sigmoid(x))

proc gelu*(x: Tensor): Tensor =
  ## Element-wise GELU (approximate): `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
  let half = scalarF32(0.5'f32)
  let one = scalarF32(1'f32)
  let coeff = scalarF32(0.044715'f32)
  let sqrtTwoPi = scalarF32(sqrt(2.0'f32 / PI.float32))
  var dims: seq[int] = @[]
  let halfB = broadcastTo(half, x.shape, dims)
  let oneB = broadcastTo(one, x.shape, dims)
  let coeffB = broadcastTo(coeff, x.shape, dims)
  let sqrtB = broadcastTo(sqrtTwoPi, x.shape, dims)
  let xCubed = mul(x, mul(x, x))
  let inner = mul(sqrtB, add(x, mul(coeffB, xCubed)))
  let tanhInner = tanh(inner)
  mul(halfB, mul(x, add(oneB, tanhInner)))

proc leakyRelu*(x: Tensor; alpha: float32 = 0.01'f32): Tensor =
  ## Element-wise leaky ReLU: `max(x, alpha * x)`.
  let alphaScalar = scalarF32(alpha)
  var dims: seq[int] = @[]
  let alphaB = broadcastTo(alphaScalar, x.shape, dims)
  let scaled = mul(alphaB, x)
  maximum(x, scaled)

proc softmax*(x: Tensor; axis: int): Tensor =
  ## Numerically stable softmax along `axis`:
  ## `exp(x - max(x, axis)) / sum(exp(x - max(x, axis)), axis)`.
  let maxVal = reduceMax(x, [axis])
  # Broadcast max back to input shape for subtraction.
  var bdims: seq[int] = @[]
  for i in 0 ..< x.shape.len:
    if i != axis: bdims.add i
  let maxB = broadcastTo(maxVal, x.shape, bdims)
  let shifted = sub(x, maxB)
  let expShifted = exp(shifted)
  let sumExp = reduceSum(expShifted, [axis])
  let sumB = broadcastTo(sumExp, x.shape, bdims)
  divide(expShifted, sumB)

proc logSoftmax*(x: Tensor; axis: int): Tensor =
  ## Numerically stable log-softmax along `axis`:
  ## `x - max(x) - log(sum(exp(x - max(x)), axis))`.
  let maxVal = reduceMax(x, [axis])
  var bdims: seq[int] = @[]
  for i in 0 ..< x.shape.len:
    if i != axis: bdims.add i
  let maxB = broadcastTo(maxVal, x.shape, bdims)
  let shifted = sub(x, maxB)
  let expShifted = exp(shifted)
  let sumExp = reduceSum(expShifted, [axis])
  let logSumExp = log(sumExp)
  let logSumExpB = broadcastTo(logSumExp, x.shape, bdims)
  sub(shifted, logSumExpB)

# ---- additional activation composites -----------------------------------

proc scalarBroadcast(value: float32; shape: openArray[int]): Tensor =
  let s = scalarF32(value)
  if shape.len == 0: return s
  var dims: seq[int] = @[]
  broadcastTo(s, @shape, dims)

proc elu*(x: Tensor; alpha: float32 = 1'f32): Tensor =
  ## Element-wise ELU: `x` if `x > 0`, else `alpha * (exp(x) - 1)`.
  let zeroB = scalarBroadcast(0'f32, x.shape)
  let positive = compare(x, zeroB, "GT")
  let alphaB = scalarBroadcast(alpha, x.shape)
  let negPart = mul(alphaB, expm1(x))
  select(positive, x, negPart)

proc selu*(x: Tensor): Tensor =
  ## Element-wise SELU: scaled ELU with fixed alpha and lambda.
  let lambda = scalarBroadcast(1.0507009873554804934193349852946'f32, x.shape)
  let alpha = scalarBroadcast(1.6732632423543772848170429916717'f32, x.shape)
  let zeroB = scalarBroadcast(0'f32, x.shape)
  let positive = compare(x, zeroB, "GT")
  let negPart = mul(alpha, expm1(x))
  let eluOut = select(positive, x, negPart)
  mul(lambda, eluOut)

proc celu*(x: Tensor; alpha: float32 = 1'f32): Tensor =
  ## Element-wise CELU: `max(0, x) + min(0, alpha * (exp(x/alpha) - 1))`.
  let zeroB = scalarBroadcast(0'f32, x.shape)
  let alphaB = scalarBroadcast(alpha, x.shape)
  let scaled = divide(x, alphaB)
  let negPart = mul(alphaB, expm1(scaled))
  let maxPart = maximum(x, zeroB)
  let minPart = minimum(zeroB, negPart)
  add(maxPart, minPart)

proc softplus*(x: Tensor): Tensor =
  ## Element-wise softplus: `log(1 + exp(x))`.
  log1p(exp(x))

proc softsign*(x: Tensor): Tensor =
  ## Element-wise softsign: `x / (1 + |x|)`.
  let oneB = scalarBroadcast(1'f32, x.shape)
  divide(x, add(oneB, abs(x)))

proc mish*(x: Tensor): Tensor =
  ## Element-wise Mish: `x * tanh(softplus(x))`.
  mul(x, tanh(softplus(x)))

proc hardtanh*(x: Tensor; minVal = -1'f32; maxVal = 1'f32): Tensor =
  ## Element-wise hardtanh: clamps values to `[minVal, maxVal]`.
  let minB = scalarBroadcast(minVal, x.shape)
  let maxB = scalarBroadcast(maxVal, x.shape)
  clamp(minB, x, maxB)

proc hardsigmoid*(x: Tensor): Tensor =
  ## Element-wise hardsigmoid: `clamp(0, x/6 + 0.5, 1)`.
  let oneSixth = scalarBroadcast(1'f32 / 6'f32, x.shape)
  let half = scalarBroadcast(0.5'f32, x.shape)
  let zeroB = scalarBroadcast(0'f32, x.shape)
  let oneB = scalarBroadcast(1'f32, x.shape)
  let shifted = add(mul(x, oneSixth), half)
  clamp(zeroB, shifted, oneB)

proc hardswish*(x: Tensor): Tensor =
  ## Element-wise hardswish: `x * hardsigmoid(x)`.
  mul(x, hardsigmoid(x))

proc glu*(x: Tensor; dim = -1): Tensor =
  ## Gated Linear Unit. Splits `x` in half along `dim`, applies sigmoid
  ## to the first half, and multiplies element-wise by the second half.
  let pos = if dim < 0: x.shape.len + dim else: dim
  if pos < 0 or pos >= x.shape.len:
    raise newException(TensorError,
      "glu: dim " & $dim & " out of range for rank " & $x.shape.len)
  let halfSize = x.shape[pos] div 2
  if halfSize * 2 != x.shape[pos]:
    raise newException(TensorError,
      "glu: dim " & $pos & " size " & $x.shape[pos] & " is not even")
  let rank = x.shape.len
  var startsA = newSeq[int](rank)
  var limitsA = newSeq[int](rank)
  var strides = newSeq[int](rank)
  var startsB = newSeq[int](rank)
  var limitsB = newSeq[int](rank)
  for i in 0 ..< rank:
    startsA[i] = 0
    limitsA[i] = (if i == pos: halfSize else: x.shape[i])
    strides[i] = 1
    startsB[i] = (if i == pos: halfSize else: 0)
    limitsB[i] = x.shape[i]
  let a = slice(x, startsA, limitsA, strides)
  let b = slice(x, startsB, limitsB, strides)
  mul(sigmoid(a), b)

proc prelu*(x: Tensor; weight: Tensor): Tensor =
  ## Element-wise PReLU: `x` if `x > 0` else `weight * x`.
  ## `weight` broadcasts against `x`.
  let zeroB = scalarBroadcast(0'f32, x.shape)
  let positive = compare(x, zeroB, "GT")
  select(positive, x, mul(weight, x))

proc logSigmoid*(x: Tensor): Tensor =
  ## Element-wise log-sigmoid: `-softplus(-x)`.
  neg(softplus(neg(x)))

proc softmin*(x: Tensor; axis: int): Tensor =
  ## Numerically stable softmin along `axis`. Equivalent to
  ## `softmax(-x, axis)`.
  softmax(neg(x), axis)

proc oneHot*(indices: Tensor; numClasses: int;
    dtype: DType = dtFloat32): Tensor =
  ## One-hot encode `indices` into shape `indices.shape + [numClasses]`.
  ## `indices` must be an integer tensor with values in
  ## `[0, numClasses)`.
  if not (indices.dtype.isSignedInt or indices.dtype.isUnsignedInt):
    raise newException(TensorError,
      "oneHot: indices must be an integer tensor, got " & $indices.dtype)
  let rank = indices.shape.len
  let outRank = rank + 1
  # Build class-axis iota as [1, ..., 1, numClasses]
  var classShape = newSeq[int](outRank)
  for i in 0 ..< rank: classShape[i] = 1
  classShape[rank] = numClasses
  let classes = iota(dtInt32, classShape, rank, indices.device)
  let classesB = broadcastTo(classes,
    indices.shape & @[numClasses], @[rank])
  # Unsqueeze indices to [..., 1], broadcast to target shape
  let idx = unsqueeze(indices, rank)
  var idxBdims: seq[int] = @[]
  for i in 0 ..< rank: idxBdims.add i
  let idxB = broadcastTo(idx,
    indices.shape & @[numClasses], idxBdims)
  astype(compare(idxB, classesB, "EQ"), dtype)

proc relu6*(x: Tensor): Tensor =
  ## Clipped ReLU: `minimum(maximum(x, 0), 6)`.
  let zero = scalarF32(0'f32)
  var bdims: seq[int] = @[]
  let zeroB = broadcastTo(zero, x.shape, bdims)
  let six = scalarF32(6'f32)
  let sixB = broadcastTo(six, x.shape, bdims)
  minimum(maximum(x, zeroB), sixB)

proc standardize*(x: Tensor; axis: openArray[int];
    eps: float32 = 1e-5'f32): Tensor =
  ## Standardize `x` along `axis` to zero mean and unit variance:
  ## `(x - mean) / max(std, eps)`.
  let mean = reduceMean(x, axis)
  var bdims: seq[int] = @[]
  for i in 0 ..< x.shape.len:
    if i notin axis: bdims.add i
  let meanB = broadcastTo(mean, x.shape, bdims)
  let centered = sub(x, meanB)
  let variance = reduceMean(mul(centered, centered), axis)
  let epsScalar = broadcastTo(scalarF32(eps), variance.shape, @[])
  let std = sqrt(add(variance, epsScalar))
  let stdB = broadcastTo(std, x.shape, bdims)
  divide(centered, maximum(stdB,
    broadcastTo(scalarF32(eps), x.shape, @[])))

proc swiglu*(x: Tensor; beta: float32 = 1'f32): Tensor =
  ## Element-wise SwiGLU (Swish-Gated Linear Unit):
  ## `x * sigmoid(beta * x)`.
  ## This is the SiLU (Swish) activation — when used in a gated context,
  ## half of the input channels are activated by SwiGLU and the other
  ## half form the gate.
  if beta == 1'f32:
    silu(x)
  else:
    let betaScalar = scalarF32(beta)
    var dims: seq[int] = @[]
    let betaB = broadcastTo(betaScalar, x.shape, dims)
    mul(x, sigmoid(mul(betaB, x)))
