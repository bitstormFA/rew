## Text generation — sampling strategies, beam search, KV cache integration.
##
## Provides `generate` for autoregressive text generation from a causal LM.

import std/[algorithm, math]
import ../tensor
import ../dtype
import ../eager
import ../rng
import ../nn/init

type
  GenerationConfig* = object
    ## Parameters controlling text generation.
    maxNewTokens*: int
    temperature*: float32
    topK*: int
    topP*: float32
    minP*: float32
    repetitionPenalty*: float32
    stopTokenIds*: seq[int]
    doSample*: bool

  GenerationResult* = object
    ## Result of a generation call.
    tokenIds*: seq[int]
    logProbs*: seq[float32]

proc initGenerationConfig*(maxNewTokens = 50; temperature = 1.0'f32;
    topK = 50; topP = 1.0'f32; minP = 0.0'f32;
    repetitionPenalty = 1.0'f32; doSample = true): GenerationConfig =
  if maxNewTokens < 0:
    raise newException(ValueError,
      "initGenerationConfig: maxNewTokens must be non-negative")
  if temperature < 0'f32:
    raise newException(ValueError,
      "initGenerationConfig: temperature must be non-negative")
  if topK < 0:
    raise newException(ValueError,
      "initGenerationConfig: topK must be non-negative")
  if topP <= 0'f32 or topP > 1'f32:
    raise newException(ValueError,
      "initGenerationConfig: topP must be in (0, 1]")
  if minP < 0'f32 or minP > 1'f32:
    raise newException(ValueError,
      "initGenerationConfig: minP must be in [0, 1]")
  if repetitionPenalty <= 0'f32:
    raise newException(ValueError,
      "initGenerationConfig: repetitionPenalty must be positive")
  GenerationConfig(
    maxNewTokens: maxNewTokens,
    temperature: temperature,
    topK: topK,
    topP: topP,
    minP: minP,
    repetitionPenalty: repetitionPenalty,
    doSample: doSample,
  )

# ---- Paged KV Cache ---------------------------------------------------------

type
  PagedKvCache* = object
    ## Paged attention KV cache (vLLM-style).
    ##
    ## Divides KV cache into fixed-size blocks (pages). Each request
    ## maintains a page table mapping logical position → physical block.
    numLayers*: int
    numHeads*: int
    headDim*: int
    blockSize*: int
    numBlocks*: int
    keyCache*: seq[Tensor]  ## per physical block, per layer: [blockSize, numHeads, headDim]
    valueCache*: seq[Tensor]
    freeBlocks*: seq[int]

proc initPagedKvCache*(numLayers, numHeads, headDim, blockSize,
    numBlocks: int): PagedKvCache =
  ## Constructs an empty paged KV cache.
  result = PagedKvCache(
    numLayers: numLayers,
    numHeads: numHeads,
    headDim: headDim,
    blockSize: blockSize,
    numBlocks: numBlocks,
  )
  for i in 0 ..< numBlocks:
    result.freeBlocks.add i

# ---- Sampling helpers ------------------------------------------------------

proc sampleTopK(logits: seq[float32]; topK: int; key: Key): int =
  ## Samples from softmax(logits) keeping only top-K entries.
  let vocabSize = logits.len
  var sorted = newSeq[(float32, int)](vocabSize)
  for i, val in logits:
    sorted[i] = (val, i)
  sorted.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
  let k = min(topK, vocabSize)
  var probs = newSeq[float32](k)
  var sumExp = 0.0'f32
  for i in 0 ..< k:
    probs[i] = exp(sorted[i][0])
    sumExp += probs[i]
  for i in 0 ..< k:
    probs[i] /= sumExp
  # Sample from multinomial.
  let u = uniformF32(key, 1, 0.0'f32, 1.0'f32)[0]
  var cum = 0.0'f32
  for i in 0 ..< k:
    cum += probs[i]
    if u <= cum:
      return sorted[i][1]
  sorted[k - 1][1]

proc sampleTopP(logits: seq[float32]; topP: float32; key: Key): int =
  ## Nucleus sampling: samples from smallest set with cum prob >= topP.
  let vocabSize = logits.len
  var sorted = newSeq[(float32, int)](vocabSize)
  for i, val in logits:
    sorted[i] = (val, i)
  sorted.sort(proc(a, b: (float32, int)): int = cmp(b[0], a[0]))
  let maxLogit = sorted[0][0]
  var probs = newSeq[float32](vocabSize)
  var sumExp = 0.0'f32
  for i in 0 ..< vocabSize:
    probs[i] = exp(sorted[i][0] - maxLogit)
    sumExp += probs[i]
  for i in 0 ..< vocabSize:
    probs[i] /= sumExp
  var cum = 0.0'f32
  var cutoff = vocabSize
  for i in 0 ..< vocabSize:
    cum += probs[i]
    if cum >= topP:
      cutoff = i + 1
      break
  # Renormalize truncated set.
  var renorm = 0.0'f32
  for i in 0 ..< cutoff:
    renorm += probs[i]
  for i in 0 ..< cutoff:
    probs[i] /= renorm
  let u = uniformF32(key, 1, 0.0'f32, 1.0'f32)[0]
  cum = 0.0'f32
  for i in 0 ..< cutoff:
    cum += probs[i]
    if u <= cum:
      return sorted[i][1]
  sorted[cutoff - 1][1]

proc sampleGreedy(logits: seq[float32]): int =
  if logits.len == 0:
    raise newException(TensorError, "sampleGreedy: logits must not be empty")
  var bestIdx = 0
  for i in 0 ..< logits.len:
    if logits[i] > logits[bestIdx]:
      bestIdx = i
  bestIdx

proc inputIdsTensor(ids: openArray[int]): Tensor =
  if ids.len == 0:
    raise newException(TensorError, "generate: tokenIds must not be empty")
  var data = newSeq[int32](ids.len)
  for i, id in ids:
    if id < 0 or id > int(high(int32)):
      raise newException(TensorError,
        "generate: token id out of int32 range at index " & $i)
    data[i] = int32(id)
  fromHost(data, [1, data.len])

proc lastLogits(t: Tensor): seq[float32] =
  if t.dtype != dtFloat32:
    raise newException(TensorError,
      "generate: forwardFn must return float32 logits, got " & $t.dtype)
  if t.shape.len == 0:
    raise newException(TensorError,
      "generate: forwardFn returned scalar logits")
  let vocabSize = t.shape[^1]
  if vocabSize <= 0:
    raise newException(TensorError,
      "generate: vocabulary dimension must be positive")
  let host = t.toHost(float32)
  if host.len < vocabSize:
    raise newException(TensorError,
      "generate: logits payload is smaller than the vocabulary dimension")
  result = newSeq[float32](vocabSize)
  let start = host.len - vocabSize
  for i in 0 ..< vocabSize:
    result[i] = host[start + i]

proc applySamplingConfig(logits: seq[float32]; seen: openArray[int];
    config: GenerationConfig): seq[float32] =
  result = logits
  if result.len == 0:
    raise newException(TensorError, "generate: logits must not be empty")

  if config.repetitionPenalty != 1.0'f32:
    if config.repetitionPenalty <= 0.0'f32:
      raise newException(ValueError,
        "generate: repetitionPenalty must be positive")
    for token in seen:
      if token >= 0 and token < result.len:
        if result[token] < 0'f32:
          result[token] *= config.repetitionPenalty
        else:
          result[token] /= config.repetitionPenalty

  let temp = max(config.temperature, 1e-6'f32)
  for i in 0 ..< result.len:
    result[i] = result[i] / temp

  if config.minP > 0'f32:
    let maxLogit = result.max
    let threshold = maxLogit + ln(config.minP)
    for i in 0 ..< result.len:
      if result[i] < threshold:
        result[i] = -Inf.float32

proc logProbOf(logits: openArray[float32]; token: int): float32 =
  if token < 0 or token >= logits.len:
    return -Inf.float32
  let maxLogit = logits.max
  var sumExp = 0.0'f32
  for val in logits:
    sumExp += exp(val - maxLogit)
  logits[token] - maxLogit - ln(sumExp)

# ---- Generation -------------------------------------------------------------

proc generate*(forwardFn: proc(inputIds: Tensor; offset: int;
    kvCache: Tensor = Tensor()): Tensor;
    tokenIds: seq[int]; config: GenerationConfig;
    eosTokenId: int = 2; key: Key = initKey(0)): GenerationResult =
  ## Autoregressive text generation.
  ##
  ## `forwardFn(inputIds, offset)` should return logits tensor
  ## `[1, 1, vocabSize]` for the new token.
  ##
  ## Host-side only (no trace/jit requirement). Callers should wrap
  ## their model's forward in a jit-compiled function.
  var generated: seq[int] = @[]
  var logProbs: seq[float32] = @[]
  var k = key
  var offset = 0
  for step in 0 ..< config.maxNewTokens:
    let inputTok = if step == 0: tokenIds
                   else: @[generated[^1]]
    let logitsTensor = forwardFn(inputIdsTensor(inputTok), offset)
    let logits = applySamplingConfig(
      lastLogits(logitsTensor), tokenIds & generated, config)
    let nextToken: int =
      if not config.doSample or config.temperature <= 0.01'f32:
        sampleGreedy(logits)
      elif config.topK > 0:
        sampleTopK(logits, config.topK, k)
      else:
        sampleTopP(logits, config.topP, k)
    generated.add nextToken
    logProbs.add logProbOf(logits, nextToken)
    offset += inputTok.len
    k = foldIn(k, uint64(step))
    if nextToken in config.stopTokenIds or nextToken == eosTokenId:
      break
  GenerationResult(tokenIds: tokenIds & generated, logProbs: logProbs)
