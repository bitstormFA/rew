## Transformer Encoder and Decoder blocks.
##
## Standard Transformer architecture from "Attention Is All You Need".
## Both blocks follow the pre-LN pattern (LayerNorm before attention/FFN)
## as it is more stable for training.

import ../tensor
import ../rng
import ../ops/arith
import ./norm
import ./dropout
import ./multiheadattention
import ./feedforward

type
  TransformerEncoder* = object
    ## A single transformer encoder block:
    ##   x = x + MHA(LayerNorm(x))
    ##   x = x + FFN(LayerNorm(x))
    mha*: MultiHeadAttention
    ff*: FeedForward
    ln1*: LayerNorm
    ln2*: LayerNorm
    dropout1*: Dropout
    dropout2*: Dropout

proc initTransformerEncoder*(key: Key; embedDim, numHeads, hiddenDim: int;
    dropoutAttn: float32 = 0.1'f32; dropoutFfn: float32 = 0.1'f32;
    dropoutRes: float32 = 0.1'f32; eps: float32 = 1e-5'f32):
    TransformerEncoder =
  let keys = split(key, 2)
  TransformerEncoder(
    mha: initMultiHeadAttention(keys[0], embedDim, numHeads, dropoutAttn),
    ff: initFeedForward(keys[1], embedDim, hiddenDim, dropoutFfn),
    ln1: initLayerNorm([embedDim], eps),
    ln2: initLayerNorm([embedDim], eps),
    dropout1: initDropout(dropoutRes),
    dropout2: initDropout(dropoutRes),
  )

proc forward*(layer: TransformerEncoder; x: Tensor;
    key: Key = Key(); training: bool = true): Tensor =
  ## Applies one encoder block.
  let keys = split(key, 3)
  let k1 = if training: keys[0] else: key
  let k2 = if training: keys[1] else: key
  let k3 = if training: keys[2] else: key
  # Self-attention with residual.
  let norm1 = forward(layer.ln1, x)
  let attn = forward(layer.mha, norm1, norm1, norm1)
  let attnDrop = forward(layer.dropout1, attn, k1, training)
  var res = add(x, attnDrop)
  # FFN with residual.
  let norm2 = forward(layer.ln2, res)
  let ffn = forward(layer.ff, norm2, k2, training)
  let ffnDrop = forward(layer.dropout2, ffn, k3, training)
  add(res, ffnDrop)

type
  TransformerDecoder* = object
    ## A single transformer decoder block:
    ##   x = x + MHA_self(LayerNorm(x))      (causal self-attention)
    ##   x = x + MHA_cross(LayerNorm(x), enc) (cross-attention)
    ##   x = x + FFN(LayerNorm(x))
    selfMha*: MultiHeadAttention
    crossMha*: MultiHeadAttention
    ff*: FeedForward
    ln1*: LayerNorm
    ln2*: LayerNorm
    ln3*: LayerNorm
    dropout1*: Dropout
    dropout2*: Dropout
    dropout3*: Dropout

proc initTransformerDecoder*(key: Key; embedDim, numHeads, hiddenDim: int;
    dropoutAttn: float32 = 0.1'f32; dropoutFfn: float32 = 0.1'f32;
    dropoutRes: float32 = 0.1'f32; eps: float32 = 1e-5'f32):
    TransformerDecoder =
  let keys = split(key, 3)
  TransformerDecoder(
    selfMha: initMultiHeadAttention(keys[0], embedDim, numHeads, dropoutAttn),
    crossMha: initMultiHeadAttention(keys[1], embedDim, numHeads, dropoutAttn),
    ff: initFeedForward(keys[2], embedDim, hiddenDim, dropoutFfn),
    ln1: initLayerNorm([embedDim], eps),
    ln2: initLayerNorm([embedDim], eps),
    ln3: initLayerNorm([embedDim], eps),
    dropout1: initDropout(dropoutRes),
    dropout2: initDropout(dropoutRes),
    dropout3: initDropout(dropoutRes),
  )

proc forward*(layer: TransformerDecoder; x, encoderOut: Tensor;
    key: Key = Key(); training: bool = true): Tensor =
  ## Applies one decoder block. `x` is the decoder input,
  ## `encoderOut` is the encoder output used for cross-attention.
  let keys = split(key, 4)
  let k1 = if training: keys[0] else: key
  let k2 = if training: keys[1] else: key
  let k3 = if training: keys[2] else: key
  let k4 = if training: keys[3] else: key
  # Causal self-attention.
  let norm1 = forward(layer.ln1, x)
  let selfAttn = forward(layer.selfMha, norm1, norm1, norm1, causal = true)
  let selfDrop = forward(layer.dropout1, selfAttn, k1, training)
  var res = add(x, selfDrop)
  # Cross-attention.
  let norm2 = forward(layer.ln2, res)
  let crossAttn = forward(layer.crossMha, norm2, encoderOut, encoderOut)
  let crossDrop = forward(layer.dropout2, crossAttn, k2, training)
  res = add(res, crossDrop)
  # FFN.
  let norm3 = forward(layer.ln3, res)
  let ffn = forward(layer.ff, norm3, k3, training)
  let ffnDrop = forward(layer.dropout3, ffn, k4, training)
  add(res, ffnDrop)
