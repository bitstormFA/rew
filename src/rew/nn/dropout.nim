## `Dropout` — regularization via random zeroing.
##
## Functional dropout: the forward pass takes an explicit `Key` for
## reproducibility (per rew invariant #8: no global RNG). During
## training, elements are zeroed with probability `p` and the remaining
## elements are scaled by `1/(1-p)` to maintain expected value.
##
## In inference mode (`training = false`), dropout is a no-op.
##
## Since rew v1 has no runtime random-uniform tensor op, dropout
## generates the mask host-side from the PRNG key and materializes
## it as a constant tensor. This is correct for trace mode (the mask
## becomes part of the graph) and matches JAX's approach.

import ../tensor
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ./init

type
  Dropout* = object
    ## Dropout layer with drop probability `p`.
    p*: float32

proc initDropout*(p: float32 = 0.5'f32): Dropout =
  ## Constructs a `Dropout` layer. `p` is the probability of zeroing
  ## each element (0 = no dropout, 1 = all zeros).
  if p < 0'f32 or p >= 1'f32:
    raise newException(TensorError,
      "initDropout: p must be in [0, 1), got " & $p)
  Dropout(p: p)

proc forward*(layer: Dropout; x: Tensor; key: Key;
    training: bool = true): Tensor =
  ## Applies dropout to `x`. When `training` is false, returns `x`
  ## unchanged. When training, generates a binary mask from `key` and
  ## applies element-wise: `x * mask / (1 - p)`.
  if not training or layer.p == 0'f32:
    return x
  let count = x.numElements
  let samples = uniformF32(key, count, 0'f32, 1'f32)
  # Build mask: 1.0 where sample >= p, 0.0 where sample < p.
  var maskData = newSeq[float32](count)
  let scale = 1'f32 / (1'f32 - layer.p)
  for i in 0 ..< count:
    if samples[i] >= layer.p:
      maskData[i] = scale
    else:
      maskData[i] = 0'f32
  let mask = constantF32(x.shape, maskData)
  mul(x, mask)

# ---- Dropout2d ---------------------------------------------------------------

type
  Dropout2d* = object
    ## Channel-wise 2D dropout. Drops entire channels with probability `p`
    ## for NHWC inputs. The trailing axis is treated as the channel dimension.
    p*: float32

proc initDropout2d*(p: float32 = 0.5'f32): Dropout2d =
  ## Constructs a Dropout2d layer.
  if p < 0'f32 or p >= 1'f32:
    raise newException(TensorError,
      "initDropout2d: p must be in [0, 1), got " & $p)
  Dropout2d(p: p)

proc forward*(layer: Dropout2d; x: Tensor; key: Key;
    training: bool = true): Tensor =
  ## Applies channel-wise dropout. During training, entire channels along
  ## the last axis are dropped. Surviving channels are scaled by `1/(1-p)`.
  if not training or layer.p == 0'f32:
    return x
  let nc = x.shape[x.shape.len - 1]
  let samples = uniformF32(key, nc, 0'f32, 1'f32)
  var maskData = newSeq[float32](nc)
  let scale = 1'f32 / (1'f32 - layer.p)
  for i in 0 ..< nc:
    if samples[i] >= layer.p:
      maskData[i] = scale
    else:
      maskData[i] = 0'f32
  let mask = constantF32([nc], maskData)
  # Broadcast mask: expand dims to match x rank, broadcast to x shape
  var bcDims: seq[int] = @[]
  for i in 0 ..< x.shape.len - 1:
    bcDims.add i
  let maskB = broadcastTo(mask, x.shape, bcDims)
  mul(x, maskB)
