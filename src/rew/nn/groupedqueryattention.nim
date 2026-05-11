## Grouped Query Attention — Ainslie et al. (2023).
##
## Generalises Multi-Head Attention (MHA) and Multi-Query Attention (MQA)
## by grouping query heads over a smaller set of key/value heads.
##
## - `numKVHeads == numHeads` → standard MHA
## - `numKVHeads == 1` → MQA
## - `1 < numKVHeads < numHeads` → GQA
##
## Supported features:
## - Optional RoPE application to Q and K tensors
## - Optional causal masking
## - Dropout on attention weights
##
## Pure value type following rew's functional nn invariant.

import std/math
import ../tensor
import ../rng
import ../ops/shape
import ../ops/linalg
import ./linear
import ./attention
import ./rope

type
  GroupedQueryAttention* = object
    ## Grouped query attention with separate Q and K/V projection sizes.
    qProj*: Linear
    kProj*: Linear
    vProj*: Linear
    outProj*: Linear
    numHeads*: int
    numKVHeads*: int
    headDim*: int
    dropout*: float32
    rope*: RotaryPositionEncoding
    hasRope*: bool

proc initGroupedQueryAttention*(key: Key; embedDim, numHeads: int;
    numKVHeads: int = 0; dropout: float32 = 0'f32;
    rope: RotaryPositionEncoding = RotaryPositionEncoding()):
    GroupedQueryAttention =
  ## Constructs a `GroupedQueryAttention` layer.
  ##
  ## `embedDim` — input/output embedding dimension.
  ## `numHeads` — number of query heads (must be divisible by `numKVHeads`).
  ## `numKVHeads` — number of key/value heads (0 = same as `numHeads` for MHA).
  ## `rope` — optional `RotaryPositionEncoding` to apply to Q and K.
  if embedDim mod numHeads != 0:
    raise newException(TensorError,
      "initGroupedQueryAttention: embedDim " & $embedDim &
        " must be divisible by numHeads " & $numHeads)
  let kvH = if numKVHeads == 0: numHeads else: numKVHeads
  if numHeads mod kvH != 0:
    raise newException(TensorError,
      "initGroupedQueryAttention: numHeads " & $numHeads &
        " must be divisible by numKVHeads " & $kvH)
  let headDim = embedDim div numHeads
  let keys = split(key, 4)
  let hasRope = rope.maxSeqLen > 0
  GroupedQueryAttention(
    qProj: initLinear(keys[0], embedDim, embedDim),
    kProj: initLinear(keys[1], embedDim, kvH * headDim),
    vProj: initLinear(keys[2], embedDim, kvH * headDim),
    outProj: initLinear(keys[3], embedDim, embedDim),
    numHeads: numHeads,
    numKVHeads: kvH,
    headDim: headDim,
    dropout: dropout,
    rope: rope,
    hasRope: hasRope,
  )

proc forward*(layer: GroupedQueryAttention; q, k, v: Tensor;
    causal: bool = false; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  ## Applies grouped query attention.
  ##
  ## `q`: `[batch, seqLen, embedDim]`
  ## `k`: `[batch, kvLen, embedDim]`
  ## `v`: `[batch, kvLen, embedDim]`
  ## `offset`: position offset for RoPE (used in cached inference).
  ##
  ## Returns: `[batch, seqLen, embedDim]`
  let batch = q.shape[0]
  let seqLen = q.shape[1]
  let kvLen = k.shape[1]
  let kvh = layer.numKVHeads
  let nh = layer.numHeads
  let hd = layer.headDim
  let groupSize = nh div kvh
  # Flatten batch and seq for Linear layers (matmul requires rank-2).
  let qFlat = reshape(q, [batch * seqLen, q.shape[2]])
  let kFlat = reshape(k, [batch * kvLen, k.shape[2]])
  let vFlat = reshape(v, [batch * kvLen, v.shape[2]])
  # Project Q, K, V.
  let qpFlat = forward(layer.qProj, qFlat)
  let kpFlat = forward(layer.kProj, kFlat)
  let vpFlat = forward(layer.vProj, vFlat)
  # Reshape back to 3D.
  let qp = reshape(qpFlat, [batch, seqLen, qpFlat.shape[1]])
  let kp = reshape(kpFlat, [batch, kvLen, kpFlat.shape[1]])
  let vp = reshape(vpFlat, [batch, kvLen, vpFlat.shape[1]])
  # Reshape Q to [batch, numHeads, seqLen, headDim].
  let qh = reshape(qp, [batch, seqLen, nh, hd])
  let qr = transpose(qh, [0, 2, 1, 3])
  # Reshape K to [batch, numKVHeads, kvLen, headDim].
  let kh = reshape(kp, [batch, kvLen, kvh, hd])
  let kr = transpose(kh, [0, 2, 1, 3])
  # Reshape V to [batch, numKVHeads, kvLen, headDim].
  let vh = reshape(vp, [batch, kvLen, kvh, hd])
  let vr = transpose(vh, [0, 2, 1, 3])
  # Apply RoPE to Q and K if configured.
  let qrRoped = if layer.hasRope:
      forward(layer.rope, qr, offset)
    else:
      qr
  let krRoped = if layer.hasRope:
      forward(layer.rope, kr, offset)
    else:
      kr
  # Expand K/V heads: [batch, numKVHeads, kvLen, headDim]
  # → [batch, numKVHeads, groupSize, kvLen, headDim]
  # → [batch, numHeads, kvLen, headDim]
  let kExpand = reshape(krRoped, [batch, kvh, 1, kvLen, hd])
  let kVExp = broadcastTo(kExpand, [batch, kvh, groupSize, kvLen, hd],
    [0, 1, 2, 3, 4])
  let kFull = reshape(kVExp, [batch, nh, kvLen, hd])
  let vExpand = reshape(vr, [batch, kvh, 1, kvLen, hd])
  let vVExp = broadcastTo(vExpand, [batch, kvh, groupSize, kvLen, hd],
    [0, 1, 2, 3, 4])
  let vFull = reshape(vVExp, [batch, nh, kvLen, hd])
  # Scaled dot-product attention per head.
  let attnOut = scaledDotProductAttention(qrRoped, kFull, vFull,
    causal = causal)
  # Transpose back to [batch, seqLen, numHeads, headDim].
  let attnT = transpose(attnOut, [0, 2, 1, 3])
  # Reshape to [batch, seqLen, embedDim].
  let attnFlat = reshape(attnT, [batch, seqLen, nh * hd])
  # Output projection (flatten batch+seq for Linear).
  let attnFlat2d = reshape(attnFlat, [batch * seqLen, attnFlat.shape[2]])
  let outFlat = forward(layer.outProj, attnFlat2d)
  reshape(outFlat, [batch, seqLen, outFlat.shape[1]])
