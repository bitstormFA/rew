## SpectralNorm — weight normalization via power iteration.
##
## Wraps a linear or conv layer, normalizing its weight by the spectral norm.
## During training, maintains a running estimate of the first singular vector(s)
## via power iteration.

import ../tensor
import ../ops/linalg
import ../ops/arith
import ../ops/reduce
import ../ops/unary
import ../ops/literal
import ../ops/shape

type
  SpectralNorm*[L] = object
    ## Spectral normalization wrapper. `inner` is the wrapped layer
    ## (e.g. `Linear` or `Conv2d`). `u` and `v` are the singular vector
    ## estimates. `nPowerIterations` controls the accuracy of the
    ## spectral norm estimate.
    inner*: L
    u*: Tensor
    v*: Tensor
    nPowerIterations*: int

proc initSpectralNorm*[L](inner: L; weight: Tensor;
    nPowerIterations = 1): SpectralNorm[L] =
  ## Constructs a spectral normalization wrapper. `weight` must be the
  ## weight tensor from the inner layer (used to initialize the singular
  ## vectors). `u` is initialized as a random unit vector with shape
  ## matching the first dimension of `weight`.
  if weight.shape.len < 2:
    raise newException(TensorError,
      "initSpectralNorm: weight must have at least 2 dims")
  let outDim = weight.shape[0]
  # Flatten all other dims for u: u has shape [outDim]
  let uData = newSeq[float32](outDim)
  # Initialize u[0] = 1.0 as a simple initial vector
  var uInit = uData
  if uInit.len > 0:
    uInit[0] = 1'f32
  # v shape is the flattened remaining dims
  var vSize = 1
  for i in 1 ..< weight.shape.len:
    vSize *= weight.shape[i]
  let vData = newSeq[float32](vSize)
  SpectralNorm[L](
    inner: inner,
    u: constantF32([outDim], uInit),
    v: constantF32([vSize], vData),
    nPowerIterations: nPowerIterations,
  )

proc spectralNormWeight(layer: SpectralNorm; weight: Tensor): Tensor {.used.} =
  ## Computes `weight / sigma(weight)` where `sigma` is the spectral norm.
  let outDim = weight.shape[0]
  var vSize = 1
  for i in 1 ..< weight.shape.len:
    vSize *= weight.shape[i]
  # Reshape weight to [outDim, vSize]
  let wMat = reshape(weight, [outDim, vSize])
  var u = layer.u
  var v = layer.v
  # Power iteration
  for _ in 0 ..< layer.nPowerIterations:
    # v = normalize(W^T @ u)
    let wTu = matmul(transpose(wMat, [1, 0]), unsqueeze(u, 1))
    let wTu1d = reshape(wTu, [vSize])
    var vUnscaled = wTu1d
    let vNorm = sqrt(reduceSum(mul(vUnscaled, vUnscaled), [0]))
    let vNormClip = maximum(vNorm,
      broadcastTo(scalarF32(1e-12'f32), vNorm.shape, @[]))
    v = divide(vUnscaled, vNormClip)
    # u = normalize(W @ v)
    let wv = matmul(wMat, unsqueeze(v, 1))
    let wv1d = reshape(wv, [outDim])
    var uUnscaled = wv1d
    let uNorm = sqrt(reduceSum(mul(uUnscaled, uUnscaled), [0]))
    let uNormClip = maximum(uNorm,
      broadcastTo(scalarF32(1e-12'f32), uNorm.shape, @[]))
    u = divide(uUnscaled, uNormClip)
  # sigma = u^T @ W @ v
  let wv = matmul(wMat, unsqueeze(v, 1))
  let sigma = reduceSum(mul(u, reshape(wv, [outDim])), [0])
  # Normalize
  let sigmaClip = maximum(sigma,
    broadcastTo(scalarF32(1e-12'f32), sigma.shape, @[]))
  var bdims: seq[int] = @[]
  for i in 0 ..< weight.shape.len:
    bdims.add i
  let sigmaB = broadcastTo(sigmaClip, weight.shape, bdims)
  divide(weight, sigmaB)
