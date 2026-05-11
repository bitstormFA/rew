## Local coordinate frame construction for point clouds.
##
## Builds orthonormal frames (e1, e2, e3) from relative position vectors.
## Used by architectures that rotate features into a local frame before
## message passing (e.g., TFN, Cormorant).
##
## All ops are composite and trace-compatible.

import std/sequtils
import ../../tensor
import ../../ops/arith
import ../../ops/unary
import ../../ops/reduce
import ../../ops/shape
import ../../ops/linalg
import ../../ops/literal

proc localFrame*(relPos: Tensor): (Tensor, Tensor, Tensor) =
  ## Compute a local orthonormal frame from relative position vectors.
  ##
  ## `relPos`: `[..., 3]` relative position vectors (r_j - r_i).
  ## Returns `(e1, e2, e3)` where:
  ##   e1 = relPos / ||relPos||      (radial direction)
  ##   e2 = orthogonal to e1         (first tangential direction)
  ##   e3 = e1 × e2                  (second tangential direction)
  ##
  ## Near-zero vectors fall back to an identity frame.
  if relPos.shape.len == 0 or relPos.shape[^1] != 3:
    raise newException(TensorError,
      "localFrame: last dim must be 3, got " & $relPos.shape)
  let rank = relPos.shape.len
  let r2 = reduceSum(mul(relPos, relPos), @[rank - 1])
  let r = sqrt(r2)
  let eps = 1e-8'f32
  let rSafe = maximum(r, broadcastTo(scalarF32(eps), r.shape, @[]))
  # e1 = relPos / ||relPos||.
  var ooR = broadcastTo(scalarF32(1'f32), rSafe.shape, @[])
  ooR = unsqueeze(divide(ooR, rSafe), rank - 1)
  let e1 = mul(relPos, broadcastTo(ooR, relPos.shape,
    (0 ..< rank - 1).toSeq & @[rank - 1]))
  # Reference vector not parallel to e1.
  var refShape: seq[int] = @[]
  for i in 0 ..< rank - 1: refShape.add 1
  let refVec = constantF32(@[1, 3], @[0'f32, 1'f32, 0'f32])
  let refBroad = broadcastTo(reshape(refVec, refShape & @[3]),
    relPos.shape, (rank - 1 ..< rank).toSeq)
  # e2 = refVec - (refVec·e1)*e1  (Gram-Schmidt).
  let dot = reduceSum(mul(refBroad, e1), @[rank - 1])
  let dotB = broadcastTo(unsqueeze(dot, rank - 1), relPos.shape,
    (0 ..< rank - 1).toSeq & @[rank - 1])
  let e2Unnormed = sub(refBroad, mul(dotB, e1))
  let e2R2 = reduceSum(mul(e2Unnormed, e2Unnormed), @[rank - 1])
  let e2Norm = sqrt(maximum(e2R2, broadcastTo(scalarF32(eps),
    e2R2.shape, @[])))
  var ooE2 = broadcastTo(scalarF32(1'f32), e2Norm.shape, @[])
  ooE2 = unsqueeze(divide(ooE2, e2Norm), rank - 1)
  let e2 = mul(e2Unnormed, broadcastTo(ooE2, relPos.shape,
    (0 ..< rank - 1).toSeq & @[rank - 1]))
  # e3 = e1 × e2.
  let e3 = cross(e1, e2, rank - 1)
  (e1, e2, e3)
