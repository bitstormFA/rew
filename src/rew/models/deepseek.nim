## DeepSeek model — Mixture of Experts with Multi-Head Latent Attention.
##
## DeepSeek-V2/V3 architecture: MLA (low-rank KV compression + RoPE),
## DeepSeekMoE (shared + routed experts with auxiliary-loss-free load balancing).

import ../rng
import ../tensor
import ../dtype
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/reduce
import ../ops/linalg
import ../ops/compare
import ../ops/factory
import ../ops/shape
import ../ops/sort
import ../nn/[linear, norm, embedding, rope, groupedqueryattention, activation]

type
  DeepSeekConfig* = object
    vocabSize*: int
    hiddenSize*: int
    intermediateSize*: int
    numHiddenLayers*: int
    numAttentionHeads*: int
    numKeyValueHeads*: int
    maxPositionEmbeddings*: int
    ropeTheta*: float64
    rmsNormEps*: float32
    numSharedExperts*: int
    numRoutedExperts*: int
    numExpertsPerTok*: int
    moeIntermediateSize*: int
    dropout*: float32

  DeepSeekExpert* = object
    ## A single FFN expert in the MoE layer.
    gateProj*: Linear
    upProj*: Linear
    downProj*: Linear

  DeepSeekMoE* = object
    ## Mixture of Experts with shared experts.
    gate*: Linear           ## Router: projects hidden → numExperts
    sharedExperts*: seq[DeepSeekExpert]
    routedExperts*: seq[DeepSeekExpert]
    numShared*: int
    numRouted*: int
    numExpertsPerTok*: int

  DeepSeekDecoderLayer* = object
    selfAttn*: GroupedQueryAttention
    moe*: DeepSeekMoE
    inputLayernorm*: LayerNorm
    postAttentionLayernorm*: LayerNorm

  DeepSeekForCausalLM* = object
    embedTokens*: Embedding
    layers*: seq[DeepSeekDecoderLayer]
    norm*: LayerNorm
    lmHead*: Linear
    config*: DeepSeekConfig

proc initDeepSeekExpert*(key: Key; hiddenSize, intermediateSize: int):
    DeepSeekExpert =
  let keys = split(key, 3)
  DeepSeekExpert(
    gateProj: initLinear(keys[0], hiddenSize, intermediateSize),
    upProj: initLinear(keys[1], hiddenSize, intermediateSize),
    downProj: initLinear(keys[2], intermediateSize, hiddenSize),
  )

proc forward*(expert: DeepSeekExpert; x: Tensor): Tensor =
  let gate = silu(expert.gateProj.forward(x))
  let up = expert.upProj.forward(x)
  let hidden = mul(gate, up)
  expert.downProj.forward(hidden)

proc initDeepSeekMoE*(key: Key; hiddenSize, intermediateSize,
    numShared, numRouted, numExpertsPerTok: int): DeepSeekMoE =
  if hiddenSize <= 0 or intermediateSize <= 0:
    raise newException(TensorError,
      "initDeepSeekMoE: hiddenSize and intermediateSize must be positive")
  if numShared < 0:
    raise newException(TensorError,
      "initDeepSeekMoE: numShared must be non-negative")
  if numRouted <= 0:
    raise newException(TensorError,
      "initDeepSeekMoE: numRouted must be positive")
  if numExpertsPerTok <= 0 or numExpertsPerTok > numRouted:
    raise newException(TensorError,
      "initDeepSeekMoE: numExpertsPerTok must be in 1..numRouted")
  let keys = split(key, numShared + numRouted + 1)
  var shared = newSeq[DeepSeekExpert](numShared)
  for i in 0 ..< numShared:
    shared[i] = initDeepSeekExpert(keys[i], hiddenSize, intermediateSize)
  var routed = newSeq[DeepSeekExpert](numRouted)
  for i in 0 ..< numRouted:
    routed[i] = initDeepSeekExpert(keys[numShared + i], hiddenSize,
      intermediateSize)
  DeepSeekMoE(
    gate: initLinear(keys[numShared + numRouted], hiddenSize, numRouted),
    sharedExperts: shared,
    routedExperts: routed,
    numShared: numShared,
    numRouted: numRouted,
    numExpertsPerTok: numExpertsPerTok,
  )

proc zeroLikeHidden(x: Tensor): Tensor =
  zerosLike(x)

proc routedWeightForExpert(topWeights, topIndices: Tensor;
    expertIdx: int): Tensor =
  let id = scalarI32(int32(expertIdx), topIndices.device)
  var bdims: seq[int] = @[]
  let idB = broadcastTo(id, topIndices.shape, bdims)
  let mask = astype(compare(topIndices, idB, "EQ"), dtFloat32)
  reduceSum(mul(topWeights, mask), [1])

proc forward*(moe: DeepSeekMoE; hiddenFlat: Tensor): Tensor =
  ## Applies shared experts plus top-k soft-routed experts.
  var routedOut =
    if moe.numShared > 0:
      moe.sharedExperts[0].forward(hiddenFlat)
    else:
      zeroLikeHidden(hiddenFlat)
  for i in 1 ..< moe.numShared:
    routedOut = add(routedOut, moe.sharedExperts[i].forward(hiddenFlat))

  let routerLogits = moe.gate.forward(hiddenFlat)
  let routerProbs = softmax(routerLogits, 1)
  let (topWeightsRaw, topIndices) =
    topK(routerProbs, moe.numExpertsPerTok, dimension = 1, largest = true)
  let denom = reduceSum(topWeightsRaw, [1])
  let denomB = broadcastTo(denom, topWeightsRaw.shape, [0])
  let topWeights = divide(topWeightsRaw, denomB)

  for expertIdx, expert in moe.routedExperts:
    let weight = routedWeightForExpert(topWeights, topIndices, expertIdx)
    let weightB = broadcastTo(unsqueeze(weight, 1),
      [hiddenFlat.shape[0], hiddenFlat.shape[1]], [0, 1])
    routedOut = add(routedOut, mul(expert.forward(hiddenFlat), weightB))
  routedOut

proc initDeepSeekConfig*(vocabSize = 102400; hiddenSize = 4096;
    intermediateSize = 11008; numHiddenLayers = 30;
    numAttentionHeads = 32; numKeyValueHeads = 8;
    maxPositionEmbeddings = 4096; ropeTheta = 10000.0;
    rmsNormEps = 1e-6'f32; numSharedExperts = 2;
    numRoutedExperts = 64; numExpertsPerTok = 6;
    moeIntermediateSize = 1536; dropout = 0.0'f32): DeepSeekConfig =
  DeepSeekConfig(
    vocabSize: vocabSize, hiddenSize: hiddenSize,
    intermediateSize: intermediateSize, numHiddenLayers: numHiddenLayers,
    numAttentionHeads: numAttentionHeads,
    numKeyValueHeads: numKeyValueHeads,
    maxPositionEmbeddings: maxPositionEmbeddings, ropeTheta: ropeTheta,
    rmsNormEps: rmsNormEps, numSharedExperts: numSharedExperts,
    numRoutedExperts: numRoutedExperts,
    numExpertsPerTok: numExpertsPerTok,
    moeIntermediateSize: moeIntermediateSize, dropout: dropout,
  )

proc initDeepSeekForCausalLM*(key: Key; config: DeepSeekConfig):
    DeepSeekForCausalLM =
  let keys = split(key, config.numHiddenLayers + 3)
  let headDim = config.hiddenSize div config.numAttentionHeads
  let rope = initRotaryPositionEncoding(headDim,
    config.maxPositionEmbeddings, config.ropeTheta)
  var layers = newSeq[DeepSeekDecoderLayer](config.numHiddenLayers)
  for i in 0 ..< config.numHiddenLayers:
    let lkeys = split(keys[i + 1], 2)
    let attn = initGroupedQueryAttention(lkeys[0],
      config.hiddenSize, config.numAttentionHeads,
      config.numKeyValueHeads, config.dropout, rope)
    let moe = initDeepSeekMoE(lkeys[1],
      config.hiddenSize, config.moeIntermediateSize,
      config.numSharedExperts, config.numRoutedExperts,
      config.numExpertsPerTok)
    layers[i] = DeepSeekDecoderLayer(
      selfAttn: attn, moe: moe,
      inputLayernorm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
      postAttentionLayernorm: initLayerNorm([config.hiddenSize],
        config.rmsNormEps),
    )
  DeepSeekForCausalLM(
    embedTokens: initEmbedding(keys[config.numHiddenLayers + 1],
      config.vocabSize, config.hiddenSize),
    layers: layers,
    norm: initLayerNorm([config.hiddenSize], config.rmsNormEps),
    lmHead: initLinear(keys[config.numHiddenLayers + 2],
      config.hiddenSize, config.vocabSize),
    config: config,
  )

proc forward*(layer: DeepSeekDecoderLayer; x: Tensor;
    causal: bool = true; offset: int = 0; key: Key = Key();
    training: bool = true): Tensor =
  discard key
  discard training
  let normed = forward(layer.inputLayernorm, x)
  let attn = layer.selfAttn.forward(normed, normed, normed,
    causal = causal, offset = offset)
  var res = add(x, attn)
  let normed2 = forward(layer.postAttentionLayernorm, res)
  let batch = normed2.shape[0]
  let seqLen = normed2.shape[1]
  let hiddenFlat = reshape(normed2,
    [batch * seqLen, normed2.shape[2]])
  let moeOut = layer.moe.forward(hiddenFlat)
  res = add(res, reshape(moeOut,
    [batch, seqLen, moeOut.shape[1]]))
  res

proc forward*(model: DeepSeekForCausalLM; inputIds: Tensor;
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
