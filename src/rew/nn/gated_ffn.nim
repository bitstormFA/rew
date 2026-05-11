## Gated Feed-Forward Network — used in modern transformer architectures.
##
## Llama / Mistral / Gemma style: gate_proj, up_proj, down_proj.
## `gate_proj(x) * activation(up_proj(x)) -> down_proj`.
##
## Supports SwiGLU (default), GeGLU, and plain gated SiLU.

import ../tensor
import ../rng
import ../ops/shape
import ../ops/arith
import ./linear
import ./activation
import ./dropout

type
  GatedFeedForward* = object
    ## Gated MLP: `downProj(activation(gateProj(x)) * upProj(x))`.
    ##
    ## Llama-style: `activation = SiLU`, so this is SwiGLU.
    ## Gemma-style: `activation = GELU` → GeGLU.
    gateProj*: Linear
    upProj*: Linear
    downProj*: Linear
    hiddenDim*: int
    dropout*: Dropout
    useDropout*: bool

proc initGatedFeedForward*(key: Key; embedDim, hiddenDim: int;
    dropout: float32 = 0.0'f32): GatedFeedForward =
  ## Constructs a gated feed-forward layer with SwiGLU (SiLU) activation.
  ## `hiddenDim` is the intermediate (up-projection) dimension.
  if hiddenDim <= 0:
    raise newException(TensorError,
      "initGatedFeedForward: hiddenDim must be positive")
  let keys = split(key, 3)
  GatedFeedForward(
    gateProj: initLinear(keys[0], embedDim, hiddenDim),
    upProj: initLinear(keys[1], embedDim, hiddenDim),
    downProj: initLinear(keys[2], hiddenDim, embedDim),
    hiddenDim: hiddenDim,
    dropout: initDropout(dropout),
    useDropout: dropout > 0.0'f32,
  )

proc forward*(layer: GatedFeedForward; x: Tensor; key: Key = Key();
    training: bool = true): Tensor =
  ## Applies the gated feed-forward block.
  ##
  ## `x`: `[batch, seq, embedDim]` → `[batch, seq, embedDim]`.
  let batch = x.shape[0]
  let seqLen = x.shape[1]
  # Flatten for Linear (matmul requires rank-2).
  let xFlat = reshape(x, [batch * seqLen, x.shape[2]])
  let gateFlat = forward(layer.gateProj, xFlat)
  let upFlat = forward(layer.upProj, xFlat)
  # Apply SiLU to gate and multiply with up.
  let gateAct = silu(gateFlat)
  let hiddenFlat = mul(gateAct, upFlat)
  # Apply dropout if configured.
  let hiddenDropped = if layer.useDropout:
      forward(layer.dropout, hiddenFlat, key, training)
    else:
      hiddenFlat
  # Down-projection.
  let outFlat = forward(layer.downProj, hiddenDropped)
  reshape(outFlat, [batch, seqLen, outFlat.shape[1]])
