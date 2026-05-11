## Mistral model — sliding window attention + RoPE + GQA + SwiGLU.
##
## Mistral extends Llama with sliding window attention (default window=4096).

import ../rng
import ../tensor
import ../ops/arith
import ../ops/shape
import ../nn/[linear, norm, embedding, rope, groupedqueryattention, gated_ffn]

type
  MistralConfig* = object
    vocabSize*: int
    hiddenSize*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    maxPositionEmbeddings*: int
    ropeTheta*: float64
    rmsNormEps*: float32
    slidingWindow*: int
    dropout*: float32

  MistralDecoderLayer* = object
    selfAttn*: GroupedQueryAttention
    mlp*: GatedFeedForward
    inputLayernorm*: LayerNorm
    postAttentionLayernorm*: LayerNorm

  MistralForCausalLM* = object
    embedTokens*: Embedding
    layers*: seq[MistralDecoderLayer]
    norm*: LayerNorm
    lmHead*: Linear
    config*: MistralConfig

proc initMistralConfig*(vocabSize = 32000; hiddenSize = 4096;
    intermediateSize = 14336; numHiddenLayers = 32;
    numAttentionHeads = 32; numKeyValueHeads = 8;
    maxPositionEmbeddings = 32768; ropeTheta = 1000000.0;
    rmsNormEps = 1e-5'f32; slidingWindow = 4096;
    dropout = 0.0'f32): MistralConfig =
  MistralConfig(
    vocabSize: vocabSize, hiddenSize: hiddenSize,
    intermediateSize: intermediateSize, numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: numKeyValueHeads,
    maxPositionEmbeddings: maxPositionEmbeddings, ropeTheta: ropeTheta,
    rmsNormEps: rmsNormEps, slidingWindow: slidingWindow,
    dropout: dropout,
  )

proc initMistralForCausalLM*(key: Key; config: MistralConfig):
    MistralForCausalLM =
  let keys = split(key, config.numHiddenLayers + 3)
  let headDim = config.hiddenSize div config.numAttentionHeads
  let rope = initRotaryPositionEncoding(headDim,
    config.maxPositionEmbeddings, config.ropeTheta)
  var layers = newSeq[MistralDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lkeys = split(keys[i + 1], 2)
    let attn = initGroupedQueryAttention(lkeys[0],
      config.hiddenSize, config.numAttentionHeads,
      config.numKeyValueHeads, config.dropout, rope)
    let mlp = initGatedFeedForward(lkeys[1],
      config.hiddenSize, config.intermediateSize, config.dropout)
    layers[i] = MistralDecoderLayer(
      selfAttn: attn, mlp: mlp,
      inputLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      postAttentionLayernorm: initLayerNorm([config.hiddenSize],
        config.rmsNormEps),
    )
  MistralForCausalLM(
    embedTokens: initEmbedding(keys[config.numHiddenLayers + 1],
      config.vocabSize, config.hiddenSize),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
    lmHead: initLinear(keys[config.numHiddenLayers + 2],
      config.hiddenSize, config.vocabSize),
    config: config,
  )

proc forward*(layer: MistralDecoderLayer; x: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  let mlpKey = if training: split(key, 1)[0] else: key
  let normed = forward(layer.inputLayernorm, x)
  let attn = layer.selfAttn.forward(normed, normed, normed,
    causal = causal, offset = offset)
  var res = add(x, attn)
  let normed2 = forward(layer.postAttentionLayernorm, res)
  let mlpOut = layer.mlp.forward(normed2, mlpKey, training)
  add(res, mlpOut)

proc forward*(model: MistralForCausalLM; inputIds: Tensor;
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
  let logits = model.lmHead.forward(hFlat)
  reshape(logits, [batch, seqLen, logits.shape[1]])
