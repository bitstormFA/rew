## Scaled Dot-Product Attention — the core attention mechanism.
##
## `scale` = `1 / sqrt(headDim)` by default (standard scaled
## dot-product attention). Supports optional causal masking.

import std/math
import ../tensor
import ../dtype
import ../device
import ../ops/arith
import ../ops/linalg
import ../ops/compare
import ../ops/shape
import ../ops/literal
import ../ops/ternary
import ../nn/activation

proc makeCausalMask*(shape: openArray[int]): Tensor =
  ## Creates an additive mask where positions i <= j are 0 and
  ## positions i > j are -1e9 (causal masking).
  ## `shape` is `[seqLen, kvLen]`.
  if shape.len != 2:
    raise newException(TensorError,
      "makeCausalMask: shape must be rank-2 (seqLen, kvLen), got " &
        $shape)
  let device = defaultDevice()
  let rows = iota(dtInt32, shape, 0, device)
  let cols = iota(dtInt32, shape, 1, device)
  let upper = compare(rows, cols, "LE")
  let negInf = scalarF32(-1e9'f32)
  var zeroDims: seq[int] = @[]
  let negInfB = broadcastTo(negInf, shape, zeroDims)
  let zeroB = broadcastTo(scalarF32(0'f32), shape, zeroDims)
  select(upper, zeroB, negInfB)

proc scaledDotProductAttention*(q, k, v: Tensor;
    causal: bool = false;
    scale: float32 = 0'f32): Tensor =
  ## Computes `softmax(q @ k^T / sqrt(d_k)) @ v`.
  ##
  ## Shapes: `q` `[..., seqLen, headDim]`, `k` `[..., kvLen, headDim]`,
  ## `v` `[..., kvLen, headDim]`. Returns `[..., seqLen, headDim]`.
  ##
  ## `causal`: applies an upper-triangular mask so position `i` can
  ## only attend to positions `<= i` (auto-regressive decoding).
  ##
  ## `scale`: if 0 (default), uses `1 / sqrt(headDim)`.
  if q.shape.len < 2 or k.shape.len < 2 or v.shape.len < 2:
    raise newException(TensorError,
      "scaledDotProductAttention: operands must have rank >= 2")
  let rank = q.shape.len
  # Build batching dims: all leading dims except the last two.
  var batchDims: seq[int] = @[]
  for i in 0 ..< rank - 2:
    batchDims.add i
  let headDim = q.shape[^1].float32
  let sc = if scale != 0'f32: scale
           else: 1'f32 / math.sqrt(headDim)
  # Compute attention scores: q @ k^T using dotGeneral.
  # q: [..., seqLen, headDim], k: [..., kvLen, headDim]
  # Contract q's headDim (last) with k's headDim (last).
  var scores = dotGeneral(q, k, batchDims, batchDims,
    [rank - 1], [rank - 1])
  # Apply scale.
  let scScalar = scalarF32(sc)
  var zeroDims: seq[int] = @[]
  let scB = broadcastTo(scScalar, scores.shape, zeroDims)
  scores = mul(scores, scB)
  # Apply causal mask.
  if causal:
    let seqLen = scores.shape[^2]
    let kvLen = scores.shape[^1]
    let causalMaskInner = makeCausalMask([seqLen, kvLen])
    var causalBdims: seq[int] = @[scores.shape.len - 2,
      scores.shape.len - 1]
    let causalMaskB = broadcastTo(causalMaskInner, scores.shape, causalBdims)
    scores = add(scores, causalMaskB)
  # Softmax over the last dim.
  let attn = softmax(scores, scores.shape.len - 1)
  # Weighted sum of values: attn @ v using dotGeneral.
  dotGeneral(attn, v, batchDims, batchDims, [rank - 1], [rank - 2])
