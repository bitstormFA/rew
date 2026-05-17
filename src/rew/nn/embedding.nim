## `Embedding` — lookup table layer.
##
## Maps integer indices to dense vectors. The weight matrix has shape
## `[vocabSize, embedDim]`. Forward slices rows corresponding to the
## input indices.
##
## Since rew v1 does not have a gather/scatter primitive, we implement
## embedding lookup via one-hot encoding + matmul for trace-mode
## differentiability.

import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/linalg
import ../ops/shape
import ./init

type
  Embedding* = object
    ## Embedding lookup table. `weight` is `[vocabSize, embedDim]`.
    weight*: Param[Tensor]
    vocabSize*: int
    embedDim*: int

proc initEmbedding*(key: Key; vocabSize, embedDim: int): Embedding =
  ## Constructs an `Embedding` with normal initialization (std=1.0).
  ## Trace-mode only (uses `constantF32`).
  if vocabSize <= 0 or embedDim <= 0:
    raise newException(TensorError,
      "initEmbedding: vocabSize and embedDim must be positive")
  let data = normalF32(key, vocabSize * embedDim, 0'f32, 1'f32)
  Embedding(
    weight: param(constantF32([vocabSize, embedDim], data)),
    vocabSize: vocabSize,
    embedDim: embedDim,
  )

proc forward*(layer: Embedding; indices: Tensor): Tensor =
  ## Looks up embeddings for `indices`.
  ##
  ## `indices` shape: `[..., vocabSize]` — one-hot encoded tensor where
  ## the last dim equals `vocabSize`. Leading dims can be batch/seq.
  ## Result shape: `[..., embedDim]`.
  ##
  ## If `indices` is `[seqLen]` (1-D integer tokens), caller should
  ## convert to one-hot first.
  if indices.shape.len < 2 or indices.shape[^1] != layer.vocabSize:
    raise newException(TensorError,
      "Embedding.forward: indices must be [..., vocabSize] one-hot, got " &
        $indices.shape)
  # Flatten leading dims to [N, vocabSize] for matmul.
  let lastDim = indices.shape.len - 1
  let leadingSize = block:
    var n = 1
    for i in 0 ..< lastDim: n *= indices.shape[i]
    n
  let indices2d = reshape(indices, [leadingSize, layer.vocabSize])
  let emb2d = matmul(indices2d, layer.weight)
  # Reshape back: [..., embedDim].
  var outShape = indices.shape[0 ..< lastDim] & @[layer.embedDim]
  reshape(emb2d, outShape)
