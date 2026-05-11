## Sinusoidal time-step embedding for diffusion models.
##
## Maps scalar timesteps to high-dimensional embeddings via sinusoidal
## encoding followed by a 2-layer MLP. Based on the Transformer
## sinusoidal position encoding (Vaswani et al., 2017) adapted for
## continuous time inputs.
##
## Pure value type following rew's functional nn invariant.

import std/math
import ../../tensor
import ../../rng
import ../../ops/literal
import ../../ops/arith
import ../../ops/unary
import ../../ops/shape
import ../../ops/linalg
import ../../ops/concat
import ../linear
import ../activation

type
  TimeEmbedding* = object
    ## Sinusoidal time embedding with MLP projection.
    ##
    ## Encodes scalar timesteps t ∈ [0, 1] into an `embedDim`-dimensional
    ## vector via:
    ##   emb = [sin(t * w_k), cos(t * w_k)] for k in 0..halfEmbed//2-1
    ##   out = SiLU(Linear(emb)) → Linear(⋅)
    linear1*: Linear
    linear2*: Linear
    embedDim*: int        ## Output embedding dimension
    halfDim*: int         ## Half the sinusoidal encoding dimension

proc initTimeEmbedding*(key: Key; embedDim, timeEmbedDim: int): TimeEmbedding =
  ## Constructs a `TimeEmbedding`.
  ##
  ## `embedDim` is the output dimension of the time embedding vector.
  ## `timeEmbedDim` is the intermediate sinusoidal encoding dimension
  ## (typically `embedDim * 4`).
  if embedDim <= 0 or timeEmbedDim <= 0:
    raise newException(TensorError,
      "initTimeEmbedding: dimensions must be positive")
  let halfDim = timeEmbedDim div 2
  if halfDim * 2 != timeEmbedDim:
    raise newException(TensorError,
      "initTimeEmbedding: timeEmbedDim must be even, got " & $timeEmbedDim)
  let keys = split(key, 2)
  TimeEmbedding(
    linear1: initLinear(keys[0], timeEmbedDim, embedDim * 4),
    linear2: initLinear(keys[1], embedDim * 4, embedDim),
    embedDim: embedDim,
    halfDim: halfDim,
  )

proc forward*(layer: TimeEmbedding; t: Tensor): Tensor =
  ## Encodes scalar timesteps into embedding vectors.
  ##
  ## `t`: `[batch]` float32 timesteps in [0, 1].
  ## Returns: `[batch, embedDim]` float32.
  if t.shape.len != 1:
    raise newException(TensorError,
      "TimeEmbedding.forward: t must be rank-1 [batch], got " & $t.shape)
  let batch = t.shape[0]
  # Compute log-spaced frequency bands.
  let maxPeriod = 10000'f32
  var freqs = newSeq[float32](layer.halfDim)
  for i in 0 ..< layer.halfDim:
    freqs[i] = pow(maxPeriod.float64,
      (-2'f32 * float32(i) / float32(layer.halfDim)).float64).float32
  let freqTensor = constantF32(@[1, layer.halfDim], freqs)
  # t: [batch] → [batch, 1], multiply with freqs to get [batch, halfDim].
  let tUns = unsqueeze(t, 1)
  let tB = broadcastTo(tUns, @[batch, layer.halfDim], @[0, 1])
  let angles = mul(tB, freqTensor)
  let sinPart = sine(angles)
  let cosPart = cosine(angles)
  # Concatenate sin and cos: [batch, timeEmbedDim].
  let emb = concat(@[sinPart, cosPart], 1)
  # MLP projection with SiLU activation.
  let h = forward(layer.linear1, emb)
  let ha = silu(h)
  forward(layer.linear2, ha)
