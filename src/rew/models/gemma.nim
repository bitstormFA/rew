## Gemma 2 model — Google's lightweight decoder LM.
##
## Alternating local/global attention, GeGLU activation, soft-capping.

import ../rng
import ../tensor
import ../ops/arith
import ../ops/shape
import ../ops/unary
import ../ops/literal
import ../ops/linalg
import ../nn/[linear, norm, embedding, rope, groupedqueryattention, gated_ffn]

type
  GemmaConfig* = object
    vocabSize*: int
    hiddenSize*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    headDim*: int
    maxPositionEmbeddings*: int
    ropeTheta*: float64
    rmsNormEps*: float32
    attnLogitSoftcapping*: float32
    finalLogitSoftcapping*: float32
    dropout*: float32

  GemmaDecoderLayer* = object
    selfAttn*: GroupedQueryAttention
    mlp*: GatedFeedForward
    inputLayernorm*: LayerNorm
    preFfnLayernorm*: LayerNorm
    postFfnLayernorm*: LayerNorm
    isGlobalAttn*: bool

  GemmaForCausalLM* = object
    embedTokens*: Embedding
    layers*: seq[GemmaDecoderLayer]
    norm*: LayerNorm
    lmHead*: Linear
    config*: GemmaConfig

proc softCap*(x: Tensor; cap: float32): Tensor =
  ## Soft-capping: `cap * tanh(x / cap)`.
  let capS = scalarF32(cap)
  var dims: seq[int] = @[]
  let capB = broadcastTo(capS, x.shape, dims)
  mul(capB, tanh(divide(x, capB)))

proc initGemmaConfig*(vocabSize = 256000; hiddenSize = 2304;
    intermediateSize = 9216; numHiddenLayers = 26;
    numAttentionHeads = 8; numKeyValueHeads = 4;
    maxPositionEmbeddings = 8192; ropeTheta = 10000.0;
    rmsNormEps = 1e-6'f32; attnLogitSoftcapping = 50.0'f32;
    finalLogitSoftcapping = 30.0'f32; dropout = 0.0'f32): GemmaConfig =
  GemmaConfig(
    vocabSize: vocabSize, hiddenSize: hiddenSize,
    intermediateSize: intermediateSize, numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: numKeyValueHeads,
    headDim: hiddenSize div numAttentionHeads,
    maxPositionEmbeddings: maxPositionEmbeddings, ropeTheta: ropeTheta,
    rmsNormEps: rmsNormEps, attnLogitSoftcapping: attnLogitSoftcapping,
    finalLogitSoftcapping: finalLogitSoftcapping, dropout: dropout,
  )

proc initGemmaForCausalLM*(key: Key; config: GemmaConfig): GemmaForCausalLM =
  let keys = split(key, config.numHiddenLayers + 3)
  let rope = initRotaryPositionEncoding(config.headDim,
    config.maxPositionEmbeddings, config.ropeTheta)
  var layers = newSeq[GemmaDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lkeys = split(keys[i + 1], 2)
    let attn = initGroupedQueryAttention(lkeys[0],
      config.hiddenSize, config.numAttentionHeads,
      config.numKeyValueHeads, config.dropout, rope)
    let mlp = initGatedFeedForward(lkeys[1],
      config.hiddenSize, config.intermediateSize, config.dropout)
    layers[i] = GemmaDecoderLayer(
      selfAttn: attn, mlp: mlp,
      inputLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      preFfnLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      postFfnLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      isGlobalAttn: (i mod 2) == 0,  # alternating pattern
    )
  GemmaForCausalLM(
    embedTokens: initEmbedding(keys[config.numHiddenLayers + 1],
      config.vocabSize, config.hiddenSize),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
    lmHead: initLinear(keys[config.numHiddenLayers + 2],
      config.hiddenSize, config.vocabSize),
    config: config,
  )

proc forward*(layer: GemmaDecoderLayer; x: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  let mlpKey = if training: split(key, 1)[0] else: key
  # Pre-attention RMSNorm.
  let normed = forward(layer.inputLayernorm, x)
  let attn = layer.selfAttn.forward(normed, normed, normed,
    causal = causal, offset = offset)
  var res = add(x, attn)
  # Pre-FFN RMSNorm + FFN + Post-FFN RMSNorm.
  let preFfn = forward(layer.preFfnLayernorm, res)
  let mlpOut = layer.mlp.forward(preFfn, mlpKey, training)
  let postFfn = forward(layer.postFfnLayernorm, mlpOut)
  add(res, postFfn)

proc forward*(model: GemmaForCausalLM; inputIds: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  var h = model.embedTokens.forward(inputIds)
  let keys = split(key, model.config.numHiddenLayers)
  for i, layer in model.layers:
    let k = if training: keys[i] else: key
    h = layer.forward(h, causal = causal, offset = offset, key = k,
      training = training)
  h = forward(model.norm, h)
  let batch = h.shape[0]
  let seqLen = h.shape[1]
  let hFlat = reshape(h, [batch * seqLen, h.shape[2]])
  var logits = model.lmHead.forward(hFlat)
  if model.config.finalLogitSoftcapping > 0'f32:
    logits = softCap(logits, model.config.finalLogitSoftcapping)
  reshape(logits, [batch, seqLen, logits.shape[1]])
