## Multi-Head Attention — Vaswani et al. (2017) "Attention Is All You Need".
##
## Splits Q, K, V into `numHeads` heads, applies scaled dot-product
## attention, and concatenates the results.
##
## Pure value type following rew's functional nn invariant.

import std/math
import ../tensor
import ../rng
import ../ops/shape
import ./linear
import ./attention

type
  MultiHeadAttention* = object
    ## Multi-head scaled dot-product attention with linear projections.
    qProj*: Linear
    kProj*: Linear
    vProj*: Linear
    outProj*: Linear
    numHeads*: int
    headDim*: int
    dropout*: float32    ## Dropout probability on attention weights (0 = no dropout).

proc initMultiHeadAttention*(key: Key; embedDim, numHeads: int;
    dropout: float32 = 0'f32): MultiHeadAttention =
  ## Constructs a `MultiHeadAttention` with `embedDim` input/output.
  ## The `headDim` is `embedDim / numHeads`.
  if embedDim mod numHeads != 0:
    raise newException(TensorError,
      "initMultiHeadAttention: embedDim " & $embedDim &
        " must be divisible by numHeads " & $numHeads)
  let headDim = embedDim div numHeads
  let keys = split(key, 4)
  MultiHeadAttention(
    qProj: initLinear(keys[0], embedDim, embedDim),
    kProj: initLinear(keys[1], embedDim, embedDim),
    vProj: initLinear(keys[2], embedDim, embedDim),
    outProj: initLinear(keys[3], embedDim, embedDim),
    numHeads: numHeads,
    headDim: headDim,
    dropout: dropout,
  )

proc forward*(layer: MultiHeadAttention; q, k, v: Tensor;
    causal: bool = false): Tensor =
  ## Applies multi-head attention.
  ##
  ## `q`: `[batch, seqLen, embedDim]`
  ## `k`: `[batch, kvLen, embedDim]`
  ## `v`: `[batch, kvLen, embedDim]`
  ## Returns: `[batch, seqLen, embedDim]`
  let batch = q.shape[0]
  let seqLen = q.shape[1]
  let kvLen = k.shape[1]
  # Project Q, K, V. Flatten batch+seq for Linear (matmul requires rank-2).
  let qFlat = reshape(q, [batch * seqLen, q.shape[2]])
  let kFlat = reshape(k, [batch * kvLen, k.shape[2]])
  let vFlat = reshape(v, [batch * kvLen, v.shape[2]])
  let qpFlat = forward(layer.qProj, qFlat)
  let kpFlat = forward(layer.kProj, kFlat)
  let vpFlat = forward(layer.vProj, vFlat)
  let qp = reshape(qpFlat, [batch, seqLen, qpFlat.shape[1]])
  let kp = reshape(kpFlat, [batch, kvLen, kpFlat.shape[1]])
  let vp = reshape(vpFlat, [batch, kvLen, vpFlat.shape[1]])
  # Reshape to [batch, numHeads, seqLen, headDim].
  let qh = reshape(qp,
    [batch, seqLen, layer.numHeads, layer.headDim])
  let kh = reshape(kp,
    [batch, kvLen, layer.numHeads, layer.headDim])
  let vh = reshape(vp,
    [batch, kvLen, layer.numHeads, layer.headDim])
  # Transpose to [batch, numHeads, seqLen, headDim].
  let qr = transpose(qh, [0, 2, 1, 3])
  let kr = transpose(kh, [0, 2, 1, 3])
  let vr = transpose(vh, [0, 2, 1, 3])
  # Scaled dot-product attention per head.
  let attnOut = scaledDotProductAttention(qr, kr, vr,
    causal = causal)
  # Transpose back to [batch, seqLen, numHeads, headDim].
  let attnT = transpose(attnOut, [0, 2, 1, 3])
  # Reshape to [batch, seqLen, embedDim].
  let attnFlat = reshape(attnT, [batch, seqLen,
    layer.numHeads * layer.headDim])
  # Output projection (flatten batch+seq for Linear).
  let attnFlat2d = reshape(attnFlat, [batch * seqLen, attnFlat.shape[2]])
  let outFlat = forward(layer.outProj, attnFlat2d)
  reshape(outFlat, [batch, seqLen, outFlat.shape[1]])
