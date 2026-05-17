## Gemma 4 text-only decoder support.
##
## This module intentionally models only `model.language_model.*` weights.
## Multimodal towers are outside the first QLoRA path.

import std/json
import ../tensor
import ../pytree
import ../rng
import ../ops/[arith, linalg, literal, shape, unary]
import ../nn/[activation, attention, embedding, init, linear, norm]

type
  Gemma4LayerType* = enum
    ## Gemma 4 alternates sliding-window and full-attention layers.
    g4SlidingAttention
    g4FullAttention

  Gemma4TextConfig* = object
    ## Text decoder configuration parsed from Gemma 4 `text_config`.
    vocabSize*: int
    hiddenSize*: int
    hiddenSizePerLayerInput*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    headDim*: int
    globalHeadDim*: int
    slidingWindow*: int
    maxPositionEmbeddings*: int
    rmsNormEps*: float32
    finalLogitSoftcapping*: float32
    tieWordEmbeddings*: bool
    layerTypes*: seq[Gemma4LayerType]

  Gemma4RmsNorm* = object
    ## RMSNorm layer with a single scale vector.
    weight*: Param[Tensor]
    eps*: float32

  Gemma4TextAttention* = object
    ## Gemma 4 grouped-query attention projection block.
    qProj*: Linear
    kProj*: Linear
    vProj*: Linear
    oProj*: Linear
    numHeads*: int
    numKeyValueHeads*: int
    headDim*: int
    layerType*: Gemma4LayerType
    slidingWindow*: int

  Gemma4Mlp* = object
    ## GELU-tanh gated MLP used by Gemma 4 text layers.
    gateProj*: Linear
    upProj*: Linear
    downProj*: Linear

  Gemma4TextLayer* = object
    ## One text decoder layer.
    inputLayernorm*: Gemma4RmsNorm
    postAttentionLayernorm*: Gemma4RmsNorm
    preFeedforwardLayernorm*: Gemma4RmsNorm
    postFeedforwardLayernorm*: Gemma4RmsNorm
    selfAttn*: Gemma4TextAttention
    mlp*: Gemma4Mlp

  Gemma4TextForCausalLM* = object
    ## Text-only Gemma 4 causal LM.
    config*: Gemma4TextConfig
    embedTokens*: Embedding
    layers*: seq[Gemma4TextLayer]
    norm*: Gemma4RmsNorm
    lmHead*: Linear

proc initGemma4TextConfig*(
    vocabSize = 262144;
    hiddenSize = 2560;
    hiddenSizePerLayerInput = 256;
    intermediateSize = 10240;
    numHiddenLayers = 42;
    numAttentionHeads = 8;
    numKeyValueHeads = 2;
    headDim = 256;
    globalHeadDim = 512;
    slidingWindow = 512;
    maxPositionEmbeddings = 131072;
    rmsNormEps = 1e-6'f32;
    finalLogitSoftcapping = 30'f32;
    tieWordEmbeddings = true;
    layerTypes: seq[Gemma4LayerType] = @[]): Gemma4TextConfig =
  ## Builds a Gemma 4 text config, defaulting to `google/gemma-4-E4B-it`.
  result = Gemma4TextConfig(
    vocabSize: vocabSize,
    hiddenSize: hiddenSize,
    hiddenSizePerLayerInput: hiddenSizePerLayerInput,
    intermediateSize: intermediateSize,
    numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: numKeyValueHeads,
    headDim: headDim,
    globalHeadDim: globalHeadDim,
    slidingWindow: slidingWindow,
    maxPositionEmbeddings: maxPositionEmbeddings,
    rmsNormEps: rmsNormEps,
    finalLogitSoftcapping: finalLogitSoftcapping,
    tieWordEmbeddings: tieWordEmbeddings,
    layerTypes: layerTypes,
  )
  if result.layerTypes.len == 0:
    for i in 0 ..< numHiddenLayers:
      if (i + 1) mod 6 == 0:
        result.layerTypes.add g4FullAttention
      else:
        result.layerTypes.add g4SlidingAttention

proc intField(node: JsonNode; key: string; default: int): int =
  if node.hasKey(key): node[key].getInt() else: default

proc f32Field(node: JsonNode; key: string; default: float32): float32 =
  if node.hasKey(key): node[key].getFloat().float32 else: default

proc boolField(node: JsonNode; key: string; default: bool): bool =
  if node.hasKey(key): node[key].getBool() else: default

proc parseLayerType(value: string): Gemma4LayerType =
  case value
  of "full_attention", "full": g4FullAttention
  else: g4SlidingAttention

proc parseGemma4TextConfig*(node: JsonNode): Gemma4TextConfig =
  ## Parses a Gemma 4 config JSON object or nested `text_config` object.
  let textNode =
    if node.hasKey("text_config") and node["text_config"].kind == JObject:
      node["text_config"]
    else:
      node
  var layerTypes: seq[Gemma4LayerType] = @[]
  if textNode.hasKey("layer_types") and textNode["layer_types"].kind == JArray:
    for item in textNode["layer_types"]:
      layerTypes.add parseLayerType(item.getStr())
  initGemma4TextConfig(
    vocabSize = intField(textNode, "vocab_size", 262144),
    hiddenSize = intField(textNode, "hidden_size", 2560),
    hiddenSizePerLayerInput = intField(textNode,
      "hidden_size_per_layer_input", 256),
    intermediateSize = intField(textNode, "intermediate_size", 10240),
    numHiddenLayers = intField(textNode, "num_hidden_layers", 42),
    numAttentionHeads = intField(textNode, "num_attention_heads", 8),
    numKeyValueHeads = intField(textNode, "num_key_value_heads", 2),
    headDim = intField(textNode, "head_dim", 256),
    globalHeadDim = intField(textNode, "global_head_dim", 512),
    slidingWindow = intField(textNode, "sliding_window", 512),
    maxPositionEmbeddings = intField(textNode,
      "max_position_embeddings", 131072),
    rmsNormEps = f32Field(textNode, "rms_norm_eps", 1e-6'f32),
    finalLogitSoftcapping = f32Field(textNode,
      "final_logit_softcapping", 30'f32),
    tieWordEmbeddings = boolField(textNode, "tie_word_embeddings", true),
    layerTypes = layerTypes,
  )

proc loadGemma4TextConfig*(path: string): Gemma4TextConfig =
  ## Loads and parses a Gemma 4 `config.json` file.
  parseGemma4TextConfig(parseFile(path))

proc initGemma4RmsNorm*(hiddenSize: int; eps: float32): Gemma4RmsNorm =
  ## Constructs an RMSNorm scale vector initialized to ones.
  Gemma4RmsNorm(
    weight: param(constantF32([hiddenSize], onesF32(hiddenSize))),
    eps: eps,
  )

proc forward*(layer: Gemma4RmsNorm; x: Tensor): Tensor =
  ## Applies RMSNorm over the last axis.
  rmsNorm(x, layer.weight, layer.eps)

proc initGemma4TextAttention*(key: Key; cfg: Gemma4TextConfig;
    layerType: Gemma4LayerType): Gemma4TextAttention =
  ## Constructs Gemma 4 text attention projections.
  let keys = split(key, 4)
  let qDim = cfg.numAttentionHeads * cfg.headDim
  let kvDim = cfg.numKeyValueHeads * cfg.headDim
  Gemma4TextAttention(
    qProj: initLinear(keys[0], cfg.hiddenSize, qDim),
    kProj: initLinear(keys[1], cfg.hiddenSize, kvDim),
    vProj: initLinear(keys[2], cfg.hiddenSize, kvDim),
    oProj: initLinear(keys[3], qDim, cfg.hiddenSize),
    numHeads: cfg.numAttentionHeads,
    numKeyValueHeads: cfg.numKeyValueHeads,
    headDim: cfg.headDim,
    layerType: layerType,
    slidingWindow: cfg.slidingWindow,
  )

proc repeatKvHeads(x: Tensor; numHeads, numKeyValueHeads: int): Tensor =
  if numHeads == numKeyValueHeads:
    return x
  if (numHeads mod numKeyValueHeads) != 0:
    raise newException(TensorError,
      "repeatKvHeads: numHeads must be divisible by numKeyValueHeads")
  let groups = numHeads div numKeyValueHeads
  let b = x.shape[0]
  let kv = x.shape[1]
  let t = x.shape[2]
  let d = x.shape[3]
  let expanded = reshape(x, [b, kv, 1, t, d])
  let broadcasted = broadcastTo(expanded, [b, kv, groups, t, d],
    [0, 1, 2, 3, 4])
  reshape(broadcasted, [b, numHeads, t, d])

proc forward*(layer: Gemma4TextAttention; x: Tensor): Tensor =
  ## Runs grouped-query causal attention. Sliding-window masking is recorded
  ## in config but currently lowered as ordinary causal attention.
  if x.shape.len != 3:
    raise newException(TensorError,
      "Gemma4TextAttention.forward: expected [batch, seq, hidden]")
  let b = x.shape[0]
  let t = x.shape[1]
  let h = x.shape[2]
  let xFlat = reshape(x, [b * t, h])
  let qFlat = layer.qProj.forward(xFlat)
  let kFlat = layer.kProj.forward(xFlat)
  let vFlat = layer.vProj.forward(xFlat)
  let q = transpose(reshape(qFlat, [b, t, layer.numHeads, layer.headDim]),
    [0, 2, 1, 3])
  let k = transpose(reshape(kFlat,
    [b, t, layer.numKeyValueHeads, layer.headDim]), [0, 2, 1, 3])
  let v = transpose(reshape(vFlat,
    [b, t, layer.numKeyValueHeads, layer.headDim]), [0, 2, 1, 3])
  let kr = repeatKvHeads(k, layer.numHeads, layer.numKeyValueHeads)
  let vr = repeatKvHeads(v, layer.numHeads, layer.numKeyValueHeads)
  let attn = scaledDotProductAttention(q, kr, vr, causal = true)
  let attnFlat = reshape(transpose(attn, [0, 2, 1, 3]),
    [b * t, layer.numHeads * layer.headDim])
  reshape(layer.oProj.forward(attnFlat), [b, t, h])

proc initGemma4Mlp*(key: Key; cfg: Gemma4TextConfig): Gemma4Mlp =
  ## Constructs the Gemma 4 GELU-gated MLP.
  let keys = split(key, 3)
  Gemma4Mlp(
    gateProj: initLinear(keys[0], cfg.hiddenSize, cfg.intermediateSize),
    upProj: initLinear(keys[1], cfg.hiddenSize, cfg.intermediateSize),
    downProj: initLinear(keys[2], cfg.intermediateSize, cfg.hiddenSize),
  )

proc forward*(layer: Gemma4Mlp; x: Tensor): Tensor =
  ## Applies `down(gelu_tanh(gate(x)) * up(x))`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "Gemma4Mlp.forward: expected [batch, seq, hidden]")
  let b = x.shape[0]
  let t = x.shape[1]
  let h = x.shape[2]
  let flat = reshape(x, [b * t, h])
  let gated = mul(gelu(layer.gateProj.forward(flat)),
    layer.upProj.forward(flat))
  reshape(layer.downProj.forward(gated), [b, t, h])

proc initGemma4TextLayer*(key: Key; cfg: Gemma4TextConfig;
    layerType: Gemma4LayerType): Gemma4TextLayer =
  ## Constructs one Gemma 4 text decoder layer.
  let keys = split(key, 2)
  Gemma4TextLayer(
    inputLayernorm: initGemma4RmsNorm(cfg.hiddenSize, cfg.rmsNormEps),
    postAttentionLayernorm: initGemma4RmsNorm(cfg.hiddenSize, cfg.rmsNormEps),
    preFeedforwardLayernorm: initGemma4RmsNorm(cfg.hiddenSize, cfg.rmsNormEps),
    postFeedforwardLayernorm: initGemma4RmsNorm(cfg.hiddenSize,
      cfg.rmsNormEps),
    selfAttn: initGemma4TextAttention(keys[0], cfg, layerType),
    mlp: initGemma4Mlp(keys[1], cfg),
  )

proc forward*(layer: Gemma4TextLayer; x: Tensor): Tensor =
  ## Applies attention and MLP residual blocks.
  let attnOut = layer.postAttentionLayernorm.forward(
    layer.selfAttn.forward(layer.inputLayernorm.forward(x)))
  let h = add(x, attnOut)
  let mlpOut = layer.postFeedforwardLayernorm.forward(
    layer.mlp.forward(layer.preFeedforwardLayernorm.forward(h)))
  add(h, mlpOut)

proc initGemma4TextForCausalLM*(key: Key;
    cfg: Gemma4TextConfig): Gemma4TextForCausalLM =
  ## Constructs a randomly initialized text-only Gemma 4 causal LM.
  let keys = split(key, cfg.numHiddenLayers + 3)
  result.config = cfg
  result.embedTokens = initEmbedding(keys[0], cfg.vocabSize, cfg.hiddenSize)
  for i in 0 ..< cfg.numHiddenLayers:
    let layerType =
      if i < cfg.layerTypes.len: cfg.layerTypes[i] else: g4SlidingAttention
    result.layers.add initGemma4TextLayer(keys[i + 1], cfg, layerType)
  result.norm = initGemma4RmsNorm(cfg.hiddenSize, cfg.rmsNormEps)
  result.lmHead = initLinear(keys[^1], cfg.hiddenSize, cfg.vocabSize)

proc softcap(logits: Tensor; cap: float32): Tensor =
  if cap <= 0'f32:
    return logits
  let capScalar = scalarF32(cap)
  let capB = broadcastTo(capScalar, logits.shape, @[])
  mul(capB, tanh(divide(logits, capB)))

proc forward*(model: Gemma4TextForCausalLM; inputOneHot: Tensor): Tensor =
  ## Runs a causal-LM forward pass from one-hot token ids to logits.
  var hidden = model.embedTokens.forward(inputOneHot)
  for layer in model.layers:
    hidden = layer.forward(hidden)
  let normed = model.norm.forward(hidden)
  let b = normed.shape[0]
  let t = normed.shape[1]
  let h = normed.shape[2]
  let logits = reshape(model.lmHead.forward(reshape(normed, [b * t, h])),
    [b, t, model.config.vocabSize])
  softcap(logits, model.config.finalLogitSoftcapping)

proc gemma4QloraTargetModules*(): seq[string] =
  ## Returns default Gemma 4 text QLoRA target module suffixes.
  @[
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "per_layer_input_gate", "per_layer_projection",
  ]

proc gemma4TextWeightNames*(cfg: Gemma4TextConfig): seq[string] =
  ## Returns expected `model.language_model.*` weight names for text layers.
  result.add "model.language_model.embed_tokens.weight"
  result.add "model.language_model.embed_tokens_per_layer.weight"
  result.add "model.language_model.norm.weight"
  result.add "model.language_model.per_layer_model_projection.weight"
  result.add "model.language_model.per_layer_projection_norm.weight"
  for i in 0 ..< cfg.numHiddenLayers:
    let prefix = "model.language_model.layers." & $i & "."
    for name in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "per_layer_input_gate.weight",
        "per_layer_projection.weight",
        "post_per_layer_input_norm.weight",
        "layer_scalar",
      ]:
      result.add prefix & name
