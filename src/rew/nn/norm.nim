## Normalization layers — `LayerNorm`.
##
## Pure value types following rew's functional nn invariant.
## `forward` normalizes over the last `normalizedDims` dimensions
## and applies a learnable affine transform (gamma * normalized + beta).

import ../tensor
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/reduce
import ../ops/linalg
import ../ops/shape
import ./init

type
  LayerNorm* = object
    ## Layer normalization over the last `normalizedShape.len` dims.
    ## `gamma` (scale) and `beta` (bias) have shape `normalizedShape`.
    gamma*: Tensor
    beta*: Tensor
    normalizedShape*: seq[int]
    eps*: float32

proc initLayerNorm*(normalizedShape: openArray[int];
    eps: float32 = 1e-5'f32): LayerNorm =
  ## Constructs a `LayerNorm`. `gamma` is initialized to ones, `beta` to
  ## zeros. Trace-mode only (uses `constantF32`).
  var count = 1
  for d in normalizedShape: count *= d
  let gammaData = onesF32(count)
  let betaData = zerosF32(count)
  LayerNorm(
    gamma: constantF32(@normalizedShape, gammaData),
    beta: constantF32(@normalizedShape, betaData),
    normalizedShape: @normalizedShape,
    eps: eps,
  )

proc forward*(layer: LayerNorm; x: Tensor): Tensor =
  ## Normalizes `x` over its last `normalizedShape.len` dimensions.
  ## `x` must have rank >= `normalizedShape.len` and its trailing dims
  ## must match `normalizedShape`.
  ##
  ## Computes: `gamma * (x - mean) / sqrt(variance + eps) + beta`.
  let normDims = layer.normalizedShape.len
  if x.shape.len < normDims:
    raise newException(TensorError,
      "LayerNorm.forward: input rank " & $x.shape.len &
        " is less than normalized dims " & $normDims)
  # The dims to reduce over are the last `normDims` axes.
  var reduceDims: seq[int] = @[]
  for i in (x.shape.len - normDims) ..< x.shape.len:
    reduceDims.add i
  # Compute mean.
  let mean = reduceMean(x, reduceDims)
  # Broadcast mean back to x shape.
  var bdims: seq[int] = @[]
  for i in 0 ..< x.shape.len:
    if i < x.shape.len - normDims:
      bdims.add i
  let meanB = broadcastTo(mean, x.shape, bdims)
  let centered = sub(x, meanB)
  # Compute variance = mean((x - mean)^2).
  let sq = mul(centered, centered)
  let variance = reduceMean(sq, reduceDims)
  # Add eps and compute rsqrt.
  let epsScalar = scalarF32(layer.eps)
  var epsDims: seq[int] = @[]
  let epsB = broadcastTo(epsScalar, variance.shape, epsDims)
  let varEps = add(variance, epsB)
  let invStd = rsqrt(varEps)
  # Broadcast invStd to x shape.
  let invStdB = broadcastTo(invStd, x.shape, bdims)
  let normalized = mul(centered, invStdB)
  # Apply affine: gamma * normalized + beta.
  # gamma and beta have shape normalizedShape, broadcast to x.
  var affDims: seq[int] = @[]
  for i in (x.shape.len - normDims) ..< x.shape.len:
    affDims.add i
  let gammaB = broadcastTo(layer.gamma, x.shape, affDims)
  let betaB = broadcastTo(layer.beta, x.shape, affDims)
  add(mul(gammaB, normalized), betaB)

# ---- InstanceNorm --------------------------------------------------------

type
  InstanceNorm* = object
    ## Instance normalization over spatial dims. For NHWC input, this
    ## normalizes over dims [1, 2] (H, W).
    gamma*: Tensor
    beta*: Tensor
    numFeatures*: int
    eps*: float32

proc initInstanceNorm*(numFeatures: int; eps: float32 = 1e-5'f32): InstanceNorm =
  ## Constructs an `InstanceNorm` with `numFeatures` channels.
  let dataOnes = onesF32(numFeatures)
  let dataZeros = zerosF32(numFeatures)
  InstanceNorm(
    gamma: constantF32([numFeatures], dataOnes),
    beta: constantF32([numFeatures], dataZeros),
    numFeatures: numFeatures,
    eps: eps,
  )

proc forward*(layer: InstanceNorm; x: Tensor): Tensor =
  ## Normalize `x` over spatial dims. Expects NHWC `[N, H, W, C]`.
  if x.shape.len != 4:
    raise newException(TensorError,
      "InstanceNorm.forward: expected NHWC rank-4, got " & $x.shape)
  if x.shape[3] != layer.numFeatures:
    raise newException(TensorError,
      "InstanceNorm.forward: channel dim " & $x.shape[3] &
        " != numFeatures " & $layer.numFeatures)
  let reduceDims = [1, 2]
  let mean = reduceMean(x, reduceDims)
  var bdims: seq[int] = @[0, 3]
  let meanB = broadcastTo(mean, x.shape, bdims)
  let centered = sub(x, meanB)
  let sq = mul(centered, centered)
  let variance = reduceMean(sq, reduceDims)
  let epsScalar = scalarF32(layer.eps)
  var epsDims: seq[int] = @[]
  let epsB = broadcastTo(epsScalar, variance.shape, epsDims)
  let varEps = add(variance, epsB)
  let invStd = rsqrt(varEps)
  let invStdB = broadcastTo(invStd, x.shape, bdims)
  let normalized = mul(centered, invStdB)
  var affDims: seq[int] = @[3]
  let gammaB = broadcastTo(layer.gamma, x.shape, affDims)
  let betaB = broadcastTo(layer.beta, x.shape, affDims)
  add(mul(gammaB, normalized), betaB)

# ---- GroupNorm -----------------------------------------------------------

type
  GroupNorm* = object
    ## Group normalization. Splits channels into groups, then normalizes
    ## each group independently.
    gamma*: Tensor
    beta*: Tensor
    numGroups*: int
    numChannels*: int
    eps*: float32

proc initGroupNorm*(numGroups, numChannels: int;
    eps: float32 = 1e-5'f32): GroupNorm =
  ## Constructs a `GroupNorm`. `numChannels` must be divisible by
  ## `numGroups`.
  if numChannels mod numGroups != 0:
    raise newException(TensorError,
      "GroupNorm: numChannels " & $numChannels &
        " must be divisible by numGroups " & $numGroups)
  let dataOnes = onesF32(numChannels)
  let dataZeros = zerosF32(numChannels)
  GroupNorm(
    gamma: constantF32([numChannels], dataOnes),
    beta: constantF32([numChannels], dataZeros),
    numGroups: numGroups,
    numChannels: numChannels,
    eps: eps,
  )

proc forward*(layer: GroupNorm; x: Tensor): Tensor =
  ## Normalize `x` by splitting channels into groups. Expects NHWC
  ## `[N, H, W, C]`.
  if x.shape.len != 4:
    raise newException(TensorError,
      "GroupNorm.forward: expected NHWC rank-4, got " & $x.shape)
  if x.shape[3] != layer.numChannels:
    raise newException(TensorError,
      "GroupNorm.forward: channel dim " & $x.shape[3] &
        " != numChannels " & $layer.numChannels)
  let channelsPerGroup = layer.numChannels div layer.numGroups
  # Reshape to [N, H, W, G, C/G]
  let reshaped = reshape(x,
    [x.shape[0], x.shape[1], x.shape[2],
     layer.numGroups, channelsPerGroup])
  # Normalize over [1, 2, 4] = H, W, C/G
  let reduceDims = [1, 2, 4]
  let mean = reduceMean(reshaped, reduceDims)
  var bdims: seq[int] = @[0, 3]
  let meanB = broadcastTo(mean, reshaped.shape, bdims)
  let centered = sub(reshaped, meanB)
  let sq = mul(centered, centered)
  let variance = reduceMean(sq, reduceDims)
  let epsScalar = scalarF32(layer.eps)
  var epsDims: seq[int] = @[]
  let epsB = broadcastTo(epsScalar, variance.shape, epsDims)
  let varEps = add(variance, epsB)
  let invStd = rsqrt(varEps)
  let invStdB = broadcastTo(invStd, reshaped.shape, bdims)
  let normalized = mul(centered, invStdB)
  # Reshape back to [N, H, W, C]
  let normFlat = reshape(normalized, x.shape)
  # Apply affine: gamma/beta have shape [C], broadcast over [N, H, W, C].
  var affDims: seq[int] = @[3]
  let gammaB = broadcastTo(layer.gamma, x.shape, affDims)
  let betaB = broadcastTo(layer.beta, x.shape, affDims)
  add(mul(gammaB, normFlat), betaB)

proc rmsNorm*(x: Tensor; gamma: Tensor;
    eps: float32 = 1e-5'f32): Tensor =
  ## Root-mean-square normalization over the last axis (typically
  ## the feature/embedding dimension). `gamma` has shape matching
  ## the last dim of `x`. No bias term.
  if x.shape.len == 0:
    raise newException(TensorError,
      "rmsNorm: input must have at least 1 dimension")
  let lastDim = x.shape.len - 1
  let sq = mul(x, x)
  let meanSq = reduceMean(sq, @[lastDim])
  let epsScalar = scalarF32(eps)
  var epsDims: seq[int] = @[]
  let epsB = broadcastTo(epsScalar, meanSq.shape, epsDims)
  let rms = sqrt(add(meanSq, epsB))
  var rmsDims: seq[int] = @[]
  for i in 0 ..< lastDim:
    rmsDims.add i
  let rmsB = broadcastTo(rms, x.shape, rmsDims)
  let normed = divide(x, rmsB)
  mul(broadcastTo(gamma, x.shape, @[lastDim]), normed)
