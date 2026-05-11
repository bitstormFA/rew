## Llama model: decoder-only transformer with RMSNorm, RoPE, GQA, and SwiGLU.
##
## Based on the LLaMA architecture (Touvron et al., 2023) with grouped
## query attention and rotary position embeddings.

import ../rng
import ../tensor
import ../ops/arith
import ../ops/shape
import ../nn/[linear, norm, embedding, rope, groupedqueryattention,
  gated_ffn]

type
  LlamaConfig* = object
    ## Configuration for a Llama model.
    vocabSize*: int
    hiddenSize*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    maxPositionEmbeddings*: int
    ropeTheta*: float64
    rmsNormEps*: float32
    dropout*: float32

  LlamaDecoderLayer* = object
    ## One Llama decoder block: RMSNorm → GQA with RoPE → residual →
    ## RMSNorm → GatedFFN (SwiGLU) → residual.
    selfAttn*: GroupedQueryAttention
    mlp*: GatedFeedForward
    inputLayernorm*: LayerNorm
    postAttentionLayernorm*: LayerNorm

  LlamaModel* = object
    ## Llama transformer backbone (no LM head).
    embedTokens*: Embedding
    layers*: seq[LlamaDecoderLayer]
    norm*: LayerNorm
    config*: LlamaConfig

  LlamaForCausalLM* = object
    ## Llama model with causal LM head.
    model*: LlamaModel
    lmHead*: Linear

proc initLlamaConfig*(vocabSize = 32000; hiddenSize = 4096;
    intermediateSize = 11008; numHiddenLayers = 32;
    numAttentionHeads = 32; numKeyValueHeads = 0;
    maxPositionEmbeddings = 2048; ropeTheta = 10000.0;
    rmsNormEps = 1e-5'f32; dropout = 0.0'f32): LlamaConfig =
  let kvHeads = if numKeyValueHeads == 0: numAttentionHeads
                else: numKeyValueHeads
  LlamaConfig(
    vocabSize: vocabSize,
    hiddenSize: hiddenSize,
    intermediateSize: intermediateSize,
    numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: kvHeads,
    maxPositionEmbeddings: maxPositionEmbeddings,
    ropeTheta: ropeTheta,
    rmsNormEps: rmsNormEps,
    dropout: dropout,
  )

proc initLlamaModel*(key: Key; config: LlamaConfig): LlamaModel =
  ## Constructs a Llama decoder-only transformer model.
  let keys = split(key, config.numHiddenLayers + 2)
  let headDim = config.hiddenSize div config.numAttentionHeads
  let rope = initRotaryPositionEncoding(headDim,
    config.maxPositionEmbeddings, config.ropeTheta)
  var layers = newSeq[LlamaDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lkeys = split(keys[i + 1], 2)
    let attn = initGroupedQueryAttention(lkeys[0],
      config.hiddenSize, config.numAttentionHeads,
      config.numKeyValueHeads, config.dropout, rope)
    let mlp = initGatedFeedForward(lkeys[1],
      config.hiddenSize, config.intermediateSize, config.dropout)
    layers[i] = LlamaDecoderLayer(
      selfAttn: attn,
      mlp: mlp,
      inputLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      postAttentionLayernorm: initLayerNorm([config.hiddenSize],
        config.rmsNormEps),
    )
  LlamaModel(
    embedTokens: initEmbedding(keys[config.numHiddenLayers + 1],
      config.vocabSize, config.hiddenSize),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
    config: config,
  )

proc initLlamaForCausalLM*(key: Key; config: LlamaConfig): LlamaForCausalLM =
  ## Constructs a Llama model with causal LM head.
  let keys = split(key, 2)
  LlamaForCausalLM(
    model: initLlamaModel(keys[0], config),
    lmHead: initLinear(keys[1], config.hiddenSize, config.vocabSize),
  )

# ---- Forward methods --------------------------------------------------------

proc forward*(layer: LlamaDecoderLayer; x: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  ## Applies one Llama decoder block.
  let mlpKey = if training: split(key, 1)[0] else: key
  let normed = forward(layer.inputLayernorm, x)
  let attn = layer.selfAttn.forward(normed, normed, normed,
    causal = causal, offset = offset)
  var res = add(x, attn)
  let normed2 = forward(layer.postAttentionLayernorm, res)
  let mlp = layer.mlp.forward(normed2, mlpKey, training)
  add(res, mlp)

proc forward*(model: LlamaModel; inputIds: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  ## Forward pass through the Llama backbone.
  var h = model.embedTokens.forward(inputIds)
  let keys = split(key, model.config.numHiddenLayers)
  for i, layer in model.layers:
    let k = if training: keys[i] else: key
    h = layer.forward(h, causal = causal, offset = offset, key = k,
      training = training)
  forward(model.norm, h)

proc forward*(model: LlamaForCausalLM; inputIds: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  ## Forward pass through Llama with LM head.
  let h = model.model.forward(inputIds, causal, offset, key, training)
  let batch = h.shape[0]
  let seqLen = h.shape[1]
  let hFlat = reshape(h, [batch * seqLen, h.shape[2]])
  let logits = model.lmHead.forward(hFlat)
  reshape(logits, [batch, seqLen, logits.shape[1]])
