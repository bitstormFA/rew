## CLIP — Contrastive Language-Image Pre-training.
##
## Vision + text dual encoder projection heads with normalized shared
## embedding space and symmetric contrastive logits.

import std/math
import ../rng
import ../tensor
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/reduce
import ../ops/linalg
import ../ops/shape
import ../nn/linear

type
  ClipConfig* = object
    visionHiddenSize*: int
    textHiddenSize*: int
    projectionDim*: int
    logitScale*: float32
    eps*: float32

  ClipModel* = object
    visionProj*: Linear
    textProj*: Linear
    logitScale*: float32
    config*: ClipConfig

  ClipOutput* = object
    ## Normalized embeddings plus the symmetric image/text contrastive logits.
    imageEmbeds*: Tensor
    textEmbeds*: Tensor
    logitsPerImage*: Tensor
    logitsPerText*: Tensor

proc initClipConfig*(visionHiddenSize = 768; textHiddenSize = 512;
    projectionDim = 512; logitScale = ln(1.0'f32 / 0.07'f32);
    eps = 1e-6'f32): ClipConfig =
  ClipConfig(
    visionHiddenSize: visionHiddenSize, textHiddenSize: textHiddenSize,
    projectionDim: projectionDim, logitScale: logitScale, eps: eps,
  )

proc validate(config: ClipConfig) =
  if config.visionHiddenSize <= 0 or config.textHiddenSize <= 0 or
      config.projectionDim <= 0:
    raise newException(TensorError,
      "initClipModel: hidden sizes and projectionDim must be positive")
  if config.eps <= 0'f32:
    raise newException(TensorError,
      "initClipModel: eps must be positive")

proc initClipModel*(key: Key; config: ClipConfig): ClipModel =
  ## Constructs projection heads for an image encoder and a text encoder.
  config.validate()
  let keys = split(key, 2)
  ClipModel(
    visionProj: initLinear(keys[0], config.visionHiddenSize,
      config.projectionDim),
    textProj: initLinear(keys[1], config.textHiddenSize,
      config.projectionDim),
    logitScale: config.logitScale,
    config: config,
  )

proc l2Normalize*(x: Tensor; axis: int = -1; eps: float32 = 1e-6'f32):
    Tensor =
  ## Normalizes `x` to unit L2 norm along `axis`.
  let pos =
    if axis < 0:
      x.shape.len + axis
    else:
      axis
  if pos < 0 or pos >= x.shape.len:
    raise newException(TensorError,
      "l2Normalize: axis " & $axis & " out of range for rank " &
        $x.shape.len)
  let sq = mul(x, x)
  let normSq = reduceSum(sq, [pos])
  let epsScalar = scalarF32(eps, x.device)
  var epsDims: seq[int] = @[]
  let epsB = broadcastTo(epsScalar, normSq.shape, epsDims)
  let denom = sqrt(maximum(normSq, epsB))
  var bdims: seq[int] = @[]
  for i in 0 ..< x.shape.len:
    if i != pos:
      bdims.add i
  divide(x, broadcastTo(denom, x.shape, bdims))

proc encodeImage*(model: ClipModel; visionFeatures: Tensor): Tensor =
  ## Projects image-encoder features and L2-normalizes the projection.
  if visionFeatures.shape.len != 2 or
      visionFeatures.shape[1] != model.config.visionHiddenSize:
    raise newException(TensorError,
      "ClipModel.encodeImage: expected [batch, " &
        $model.config.visionHiddenSize & "], got " & $visionFeatures.shape)
  l2Normalize(model.visionProj.forward(visionFeatures),
    eps = model.config.eps)

proc encodeText*(model: ClipModel; textFeatures: Tensor): Tensor =
  ## Projects text-encoder features and L2-normalizes the projection.
  if textFeatures.shape.len != 2 or
      textFeatures.shape[1] != model.config.textHiddenSize:
    raise newException(TensorError,
      "ClipModel.encodeText: expected [batch, " &
        $model.config.textHiddenSize & "], got " & $textFeatures.shape)
  l2Normalize(model.textProj.forward(textFeatures),
    eps = model.config.eps)

proc contrastiveLogits*(model: ClipModel; imageEmbeds, textEmbeds: Tensor):
    tuple[logitsPerImage: Tensor; logitsPerText: Tensor] =
  ## Computes scaled cosine-similarity logits for image-to-text and
  ## text-to-image retrieval.
  if imageEmbeds.shape.len != 2 or textEmbeds.shape.len != 2:
    raise newException(TensorError,
      "ClipModel.contrastiveLogits: embeddings must be rank-2")
  if imageEmbeds.shape[1] != textEmbeds.shape[1]:
    raise newException(TensorError,
      "ClipModel.contrastiveLogits: projection dims differ (" &
        $imageEmbeds.shape & " vs " & $textEmbeds.shape & ")")
  let sim = matmul(imageEmbeds, transpose(textEmbeds, [1, 0]))
  let scale = scalarF32(math.exp(model.logitScale), sim.device)
  var dims: seq[int] = @[]
  let scaled = mul(sim, broadcastTo(scale, sim.shape, dims))
  (logitsPerImage: scaled, logitsPerText: transpose(scaled, [1, 0]))

proc forward*(model: ClipModel; visionFeatures, textFeatures: Tensor):
    tuple[imageEmbeds: Tensor; textEmbeds: Tensor] =
  ## Returns normalized image and text embeddings in the shared CLIP space.
  let imageEmbeds = model.encodeImage(visionFeatures)
  let textEmbeds = model.encodeText(textFeatures)
  (imageEmbeds: imageEmbeds, textEmbeds: textEmbeds)

proc forwardContrastive*(model: ClipModel; visionFeatures,
    textFeatures: Tensor): ClipOutput =
  ## Encodes both towers and returns symmetric contrastive logits.
  let encoded = model.forward(visionFeatures, textFeatures)
  let logits = model.contrastiveLogits(encoded.imageEmbeds,
    encoded.textEmbeds)
  ClipOutput(
    imageEmbeds: encoded.imageEmbeds,
    textEmbeds: encoded.textEmbeds,
    logitsPerImage: logits.logitsPerImage,
    logitsPerText: logits.logitsPerText,
  )
