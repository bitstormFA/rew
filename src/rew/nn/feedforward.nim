## Feed-Forward Network — standard MLP block used in transformers.
##
## Two linear layers with a GELU (or other) activation in between,
## followed by dropout.

import ../tensor
import ../rng
import ../ops/shape
import ./linear
import ./dropout
import ./activation

type
  FeedForward* = object
    ## Two-layer MLP with an activation and dropout.
    ## `Linear(embedDim, hiddenDim) -> activation -> dropout -> Linear(hiddenDim, embedDim) -> dropout`.
    fc1*: Linear
    fc2*: Linear
    dropout1*: Dropout
    dropout2*: Dropout
    hiddenDim*: int

proc initFeedForward*(key: Key; embedDim, hiddenDim: int;
    dropout: float32 = 0.1'f32): FeedForward =
  if hiddenDim <= 0:
    raise newException(TensorError,
      "initFeedForward: hiddenDim must be positive")
  let keys = split(key, 2)
  FeedForward(
    fc1: initLinear(keys[0], embedDim, hiddenDim),
    fc2: initLinear(keys[1], hiddenDim, embedDim),
    dropout1: initDropout(dropout),
    dropout2: initDropout(dropout),
    hiddenDim: hiddenDim,
  )

proc forward*(layer: FeedForward; x: Tensor;
    key: Key = Key(); training: bool = true): Tensor =
  ## Applies the feed-forward block. Handles batched 3-D inputs:
  ## `[batch, seq, features]` → `[batch, seq, features]`.
  let keys = split(key, 2)
  let k1 = if training: keys[0] else: key
  let k2 = if training: keys[1] else: key
  # Flatten batch+seq for Linear layers.
  let isBatched = x.shape.len > 2
  var xFlat: Tensor
  var batch, seqLen: int
  if isBatched:
    batch = x.shape[0]
    seqLen = x.shape[x.shape.len - 2]
    xFlat = reshape(x, [batch * seqLen, x.shape[x.shape.len - 1]])
  else:
    xFlat = x
  let h = forward(layer.fc1, xFlat)
  let ha = gelu(h)
  let hd = forward(layer.dropout1, ha, k1, training)
  let y = forward(layer.fc2, hd)
  let yDrop = forward(layer.dropout2, y, k2, training)
  if isBatched:
    reshape(yDrop, [batch, seqLen, yDrop.shape[1]])
  else:
    yDrop
