## Positional Encodings for transformer models.
##
## Provides sinusoidal (fixed) positional encodings and a learned
## positional embedding layer.

import std/math
import ../tensor
import ../dtype
import ../device
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/shape
import ../ops/concat
import ../ops/unary
import ./init

proc sinusoidalPositionEncoding*(seqLen, embedDim: int): Tensor =
  ## Creates a sinusoidal positional encoding of shape
  ## `[1, seqLen, embedDim]` that can be added to token embeddings.
  ## Formula from Vaswani et al. (2017).
  if embedDim mod 2 != 0:
    raise newException(TensorError,
      "sinusoidalPositionEncoding: embedDim must be even, got " &
        $embedDim)
  let dev = defaultDevice()
  let positions = iota(dtFloat32, [seqLen, 1], 0, dev)
  let half = embedDim div 2
  var divTermData = newSeq[float32](half)
  for i in 0 ..< half:
    let exponent = 2'f32 * float32(i) / float32(embedDim)
    divTermData[i] = 1'f32 / pow(10000'f32.float64,
      exponent.float64).float32
  let divTerm = constantF32([1, half], divTermData)
  let angles = mul(positions, divTerm)
  let sinPart = sine(angles)
  let cosPart = cosine(angles)
  let pe = concat([sinPart, cosPart], 1)
  unsqueeze(pe, 0)

type
  LearnedPositionEncoding* = object
    ## Learnable position embedding table.
    weight*: Tensor
    maxLen*: int
    embedDim*: int

proc initLearnedPositionEncoding*(key: Key; maxLen, embedDim: int):
    LearnedPositionEncoding =
  ## Creates a learned positional embedding of shape `[maxLen, embedDim]`.
  let data = normalF32(key, maxLen * embedDim, 0'f32,
    1'f32 / sqrt(float32(embedDim)))
  LearnedPositionEncoding(
    weight: constantF32([maxLen, embedDim], data),
    maxLen: maxLen,
    embedDim: embedDim,
  )

proc forward*(layer: LearnedPositionEncoding; seqLen: int): Tensor =
  ## Returns position embeddings for a sequence of length `seqLen`.
  ## Shape: `[1, seqLen, embedDim]`.
  if seqLen > layer.maxLen:
    raise newException(TensorError,
      "LearnedPositionEncoding: seqLen " & $seqLen &
        " exceeds maxLen " & $layer.maxLen)
  let starts = [0, 0]
  let limits = [seqLen, layer.embedDim]
  let strides = [1, 1]
  let sliced = slice(layer.weight, starts, limits, strides)
  unsqueeze(sliced, 0)
