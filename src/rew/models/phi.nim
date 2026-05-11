## Phi model — Microsoft's small language model.
##
## Partial RoPE, block-sparse attention, GELU-based FFN.

import ../rng
import ../tensor
import ../ops/arith
import ../ops/shape
import ../nn/[linear, norm, embedding, rope, groupedqueryattention, gated_ffn]

type
  PhiConfig* = object
    vocabSize*: int
    hiddenSize*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    maxPositionEmbeddings*: int
    ropeTheta*: float64
    rmsNormEps*: float32
    partialRopeFactor*: float32
    dropout*: float32

  PhiDecoderLayer* = object
    selfAttn*: GroupedQueryAttention
    mlp*: GatedFeedForward
    inputLayernorm*: LayerNorm
    postAttentionLayernorm*: LayerNorm

  PhiForCausalLM* = object
    embedTokens*: Embedding
    layers*: seq[PhiDecoderLayer]
    norm*: LayerNorm
    lmHead*: Linear
    config*: PhiConfig

proc initPhiConfig*(vocabSize = 51200; hiddenSize = 2048;
    intermediateSize = 5632; numHiddenLayers = 24;
    numAttentionHeads = 32; numKeyValueHeads = 8;
    maxPositionEmbeddings = 2048; ropeTheta = 10000.0;
    rmsNormEps = 1e-5'f32; partialRopeFactor = 0.5'f32;
    dropout = 0.0'f32): PhiConfig =
  PhiConfig(
    vocabSize: vocabSize, hiddenSize: hiddenSize,
    intermediateSize: intermediateSize, numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: numKeyValueHeads,
    maxPositionEmbeddings: maxPositionEmbeddings, ropeTheta: ropeTheta,
    rmsNormEps: rmsNormEps, partialRopeFactor: partialRopeFactor,
    dropout: dropout,
  )

proc initPhiForCausalLM*(key: Key; config: PhiConfig): PhiForCausalLM =
  let keys = split(key, config.numHiddenLayers + 3)
  let headDim = config.hiddenSize div config.numAttentionHeads
  let rope = initRotaryPositionEncoding(headDim,
    config.maxPositionEmbeddings, config.ropeTheta)
  var layers = newSeq[PhiDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lkeys = split(keys[i + 1], 2)
    let attn = initGroupedQueryAttention(lkeys[0],
      config.hiddenSize, config.numAttentionHeads,
      config.numKeyValueHeads, config.dropout, rope)
    let mlp = initGatedFeedForward(lkeys[1],
      config.hiddenSize, config.intermediateSize, config.dropout)
    layers[i] = PhiDecoderLayer(
      selfAttn: attn, mlp: mlp,
      inputLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      postAttentionLayernorm: initLayerNorm([config.hiddenSize],
        config.rmsNormEps),
    )
  PhiForCausalLM(
    embedTokens: initEmbedding(keys[config.numHiddenLayers + 1],
      config.vocabSize, config.hiddenSize),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
    lmHead: initLinear(keys[config.numHiddenLayers + 2],
      config.hiddenSize, config.vocabSize),
    config: config,
  )

proc forward*(layer: PhiDecoderLayer; x: Tensor;
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

proc forward*(model: PhiForCausalLM; inputIds: Tensor;
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
