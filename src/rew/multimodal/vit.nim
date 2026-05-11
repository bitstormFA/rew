## Vision Transformer (ViT).
##
## Splits NHWC images into patches, projects patches into an embedding space,
## prepends a learnable class token, adds learnable positional embeddings, and
## processes the sequence through a pre-LN transformer encoder stack.

import ../rng
import ../tensor
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ../ops/shape
import ../ops/concat
import ../nn/[init, linear, norm, transformer]

type
  ViTConfig* = object
    imageSize*: int
    patchSize*: int
    numChannels*: int
    hiddenSize*: int
    numLayers*: int
    numHeads*: int
    mlpHiddenSize*: int
    numClasses*: int
    dropout*: float32
    layerNormEps*: float32

  ViT* = object
    patchProj*: Linear
    classToken*: Tensor
    positionEmbedding*: Tensor
    layers*: seq[TransformerEncoder]
    norm*: LayerNorm
    head*: Linear
    config*: ViTConfig

proc initViTConfig*(imageSize = 224; patchSize = 16; numChannels = 3;
    hiddenSize = 768; numLayers = 12; numHeads = 12;
    mlpHiddenSize = 0; numClasses = 1000; dropout = 0.0'f32;
    layerNormEps = 1e-6'f32): ViTConfig =
  let actualMlpHidden =
    if mlpHiddenSize == 0:
      hiddenSize * 4
    else:
      mlpHiddenSize
  ViTConfig(
    imageSize: imageSize, patchSize: patchSize, numChannels: numChannels,
    hiddenSize: hiddenSize, numLayers: numLayers, numHeads: numHeads,
    mlpHiddenSize: actualMlpHidden, numClasses: numClasses,
    dropout: dropout, layerNormEps: layerNormEps,
  )

func numPatches*(config: ViTConfig): int =
  ## Number of image patches produced by `config`.
  let patchesPerSide = config.imageSize div config.patchSize
  patchesPerSide * patchesPerSide

func patchDim*(config: ViTConfig): int =
  ## Flattened feature dimension of one image patch.
  config.patchSize * config.patchSize * config.numChannels

proc validate(config: ViTConfig) =
  if config.imageSize <= 0 or config.patchSize <= 0:
    raise newException(TensorError,
      "initViT: imageSize and patchSize must be positive")
  if config.imageSize mod config.patchSize != 0:
    raise newException(TensorError,
      "initViT: imageSize " & $config.imageSize &
        " must be divisible by patchSize " & $config.patchSize)
  if config.numChannels <= 0:
    raise newException(TensorError,
      "initViT: numChannels must be positive")
  if config.hiddenSize <= 0 or config.mlpHiddenSize <= 0:
    raise newException(TensorError,
      "initViT: hiddenSize and mlpHiddenSize must be positive")
  if config.numLayers <= 0:
    raise newException(TensorError,
      "initViT: numLayers must be positive")
  if config.numHeads <= 0 or config.hiddenSize mod config.numHeads != 0:
    raise newException(TensorError,
      "initViT: hiddenSize " & $config.hiddenSize &
        " must be divisible by numHeads " & $config.numHeads)
  if config.numClasses <= 0:
    raise newException(TensorError,
      "initViT: numClasses must be positive")
  if config.dropout < 0'f32 or config.dropout >= 1'f32:
    raise newException(TensorError,
      "initViT: dropout must be in [0, 1)")

proc initViT*(key: Key; config: ViTConfig): ViT =
  ## Constructs a ViT encoder and classification head.
  config.validate()
  let keys = split(key, config.numLayers + 4)
  var layers = newSeq[TransformerEncoder](config.numLayers)
  for i in 0 ..< config.numLayers:
    layers[i] = initTransformerEncoder(keys[i + 3],
      config.hiddenSize, config.numHeads, config.mlpHiddenSize,
      dropoutAttn = config.dropout, dropoutFfn = config.dropout,
      dropoutRes = config.dropout, eps = config.layerNormEps)
  let tokenData = uniformF32(keys[1], config.hiddenSize,
    -0.02'f32, 0.02'f32)
  let posCount = (config.numPatches + 1) * config.hiddenSize
  let posData = uniformF32(keys[2], posCount, -0.02'f32, 0.02'f32)
  ViT(
    patchProj: initLinear(keys[0], config.patchDim, config.hiddenSize),
    classToken: constantF32([1, 1, config.hiddenSize], tokenData),
    positionEmbedding: constantF32(
      [1, config.numPatches + 1, config.hiddenSize], posData),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.layerNormEps),
    head: initLinear(keys[config.numLayers + 3],
      config.hiddenSize, config.numClasses),
    config: config,
  )

proc patchify*(vit: ViT; images: Tensor): Tensor =
  ## Converts NHWC images `[batch, height, width, channels]` into flattened
  ## patches `[batch, numPatches, patchDim]`.
  if images.shape.len != 4:
    raise newException(TensorError,
      "ViT.patchify: expected NHWC rank-4 images, got " & $images.shape)
  if images.shape[1] != vit.config.imageSize or
      images.shape[2] != vit.config.imageSize or
      images.shape[3] != vit.config.numChannels:
    raise newException(TensorError,
      "ViT.patchify: expected [batch, " & $vit.config.imageSize & ", " &
        $vit.config.imageSize & ", " & $vit.config.numChannels &
        "], got " & $images.shape)
  let batch = images.shape[0]
  let patchesPerSide = vit.config.imageSize div vit.config.patchSize
  let p = vit.config.patchSize
  let c = vit.config.numChannels
  let grid = reshape(images, [batch, patchesPerSide, p, patchesPerSide, p, c])
  let ordered = transpose(grid, [0, 1, 3, 2, 4, 5])
  reshape(ordered, [batch, vit.config.numPatches, vit.config.patchDim])

proc forwardFeatures*(vit: ViT; patches: Tensor; key: Key = Key();
    training: bool = true): Tensor =
  ## Encodes flattened patches and returns the final class-token features
  ## with shape `[batch, hiddenSize]`.
  if patches.shape.len != 3:
    raise newException(TensorError,
      "ViT.forwardFeatures: expected rank-3 patches, got " & $patches.shape)
  if patches.shape[1] != vit.config.numPatches or
      patches.shape[2] != vit.config.patchDim:
    raise newException(TensorError,
      "ViT.forwardFeatures: expected patch shape [batch, " &
        $vit.config.numPatches & ", " & $vit.config.patchDim &
        "], got " & $patches.shape)
  let batch = patches.shape[0]
  let numPatches = patches.shape[1]
  let xFlat = reshape(patches, [batch * numPatches, patches.shape[2]])
  let embedded = vit.patchProj.forward(xFlat)
  var tokens = reshape(embedded, [batch, numPatches, vit.config.hiddenSize])
  let cls = broadcastTo(vit.classToken, [batch, 1, vit.config.hiddenSize],
    [0, 1, 2])
  tokens = concat([cls, tokens], 1)
  let pos = broadcastTo(vit.positionEmbedding, tokens.shape, [0, 1, 2])
  var h = add(tokens, pos)
  let keys = split(key, vit.layers.len)
  for i, layer in vit.layers:
    let layerKey =
      if training:
        keys[i]
      else:
        key
    h = layer.forward(h, layerKey, training)
  h = vit.norm.forward(h)
  let clsOut = slice(h, [0, 0, 0], [batch, 1, vit.config.hiddenSize],
    [1, 1, 1])
  reshape(clsOut, [batch, vit.config.hiddenSize])

proc forwardFromPatches*(vit: ViT; patches: Tensor; key: Key = Key();
    training: bool = true): Tensor =
  ## Classifies already-flattened patches `[batch, numPatches, patchDim]`.
  vit.head.forward(vit.forwardFeatures(patches, key, training))

proc forward*(vit: ViT; images: Tensor; key: Key = Key();
    training: bool = true): Tensor =
  ## Classifies NHWC images. Rank-3 inputs are accepted as pre-flattened
  ## patches for compatibility with older examples.
  let patches =
    if images.shape.len == 4:
      vit.patchify(images)
    elif images.shape.len == 3:
      images
    else:
      raise newException(TensorError,
        "ViT.forward: expected NHWC images or rank-3 patches, got " &
          $images.shape)
  vit.forwardFromPatches(patches, key, training)
