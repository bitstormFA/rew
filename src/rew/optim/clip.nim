## Gradient clipping utilities.
##
## These operate on pytree-flattened gradient sequences and return
## clipped copies. They do not modify the original tensors.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/reduce
import ../ops/linalg

proc clipGradNorm*[P](grads: P; maxNorm: float32): P =
  ## Clips gradients by global L2 norm. If `||grads||_2 > maxNorm`,
  ## scales all gradient leaves by `maxNorm / ||grads||_2`.
  ## Returns the (potentially scaled) gradients with the same structure.
  let gl = treeFlatten(grads)
  # Compute global L2 norm: sqrt(sum of all squared elements).
  var normTerms: seq[Tensor] = @[]
  for g in gl:
    let sq = mul(g, g)
    var allDims = newSeq[int](g.shape.len)
    for i in 0 ..< g.shape.len: allDims[i] = i
    normTerms.add reduceSum(sq, allDims)
  # Sum all scalar norms.
  var totalSqNorm = normTerms[0]
  for i in 1 ..< normTerms.len:
    totalSqNorm = add(totalSqNorm, normTerms[i])
  let totalNorm = sqrt(totalSqNorm)
  # Scale factor: min(1, maxNorm / totalNorm) = maxNorm / max(totalNorm, maxNorm)
  let maxNormScalar = scalarF32(maxNorm)
  let clampedNorm = maximum(totalNorm, maxNormScalar)
  let scale = divide(maxNormScalar, clampedNorm)
  var clipped = newSeq[Tensor](gl.len)
  for i in 0 ..< gl.len:
    var bdims: seq[int] = @[]
    let scaleB = broadcastTo(scale, gl[i].shape, bdims)
    clipped[i] = mul(scaleB, gl[i])
  treeUnflatten(grads, clipped)

proc clipGradValue*[P](grads: P; clipValue: float32): P =
  ## Clips gradient values element-wise to `[-clipValue, +clipValue]`.
  let gl = treeFlatten(grads)
  let lo = scalarF32(-clipValue)
  let hi = scalarF32(clipValue)
  var clipped = newSeq[Tensor](gl.len)
  for i in 0 ..< gl.len:
    var bdims: seq[int] = @[]
    let loB = broadcastTo(lo, gl[i].shape, bdims)
    let hiB = broadcastTo(hi, gl[i].shape, bdims)
    # Clamp: max(lo, min(x, hi))
    clipped[i] = maximum(loB, minimum(gl[i], hiB))
  treeUnflatten(grads, clipped)
