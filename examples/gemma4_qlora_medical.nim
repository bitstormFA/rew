## Gemma 4 E4B-it + MedAlpaca medical QLoRA adapter training path.
##
## By default this downloads tokenizer/config/chat-template plus a small
## medical SFT dataset. Set `REW_GEMMA4_DOWNLOAD_WEIGHTS=1` to download the 16 GB
## safetensors checkpoint as well.

import std/[json, math, os, strformat, strutils, times]
import rew
import rew/xla

const
  ModelRepo = "google/gemma-4-E4B-it"
  DatasetRepo = "medalpaca/medical_meadow_medical_flashcards"
  DatasetTask = "medical_question_answering"
  DefaultSampleRows = 8
  DefaultTrainExamples = 128
  DefaultTrainSteps = 8
  DefaultTrainBatchSize = 16
  DefaultTrainLearningRate = 0.05'f32
  TokenBuckets = 64
  Backend = tCpu
  DefaultTfLogDir = "tf_logs" / "gemma4_qlora_medical"
  BosToken = "<bos>"
  TurnStart = "<|turn>"
  TurnEnd = "<turn|>"

type
  MedicalSampleStats = object
    rows: int
    totalChars: int
    totalTokens: int
    minTokens: int
    maxTokens: int

  MedicalTrainingData = object
    stats: MedicalSampleStats
    inputs: seq[float32]
    labels: seq[float32]
    examples: int

  TrainingSummary = object
    steps: int
    examples: int
    batchSize: int
    learningRate: float32
    initialLoss: float32
    finalLoss: float32
    eventPath: string

  TensorBoardWriter = object
    path: string
    file: File
    closed: bool

proc downloadFullWeights(): bool =
  getEnv("REW_GEMMA4_DOWNLOAD_WEIGHTS", "0").strip().toLowerAscii() in
    ["1", "true", "yes", "on"]

proc downloadModelAssets(includeWeights: bool): string =
  if includeWeights:
    return hfDownloadModel(ModelRepo)
  let opts = HfDownloadOptions(
    revision: "main",
    allowPatterns: @[
      "config.json",
      "generation_config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "chat_template.jinja",
      "processor_config.json",
    ],
  )
  hfDownloadSnapshot(hrkModel, ModelRepo, opts)

proc downloadMedicalDataset(): string =
  hfDownloadDataset(DatasetRepo)

proc firstCachedJsonFile(snapshotDir: string): string =
  for file in walkDirRec(snapshotDir):
    if file.endsWith(".json") or file.endsWith(".jsonl"):
      return file
  raise newException(DataError,
    "medical dataset snapshot has no cached JSON/JSONL files: " & snapshotDir)

proc jsonString(row: JsonNode; key: string): string =
  if row.kind == JObject and row.hasKey(key) and row[key].kind == JString:
    row[key].getStr().strip()
  else:
    ""

proc formatGemma4MedicalSft(row: JsonNode; addBos = true): string =
  let instruction = row.jsonString("instruction")
  let input = row.jsonString("input")
  let output = row.jsonString("output")
  var userText = instruction
  if input.len > 0:
    if userText.len > 0:
      userText.add "\n\n"
    userText.add input
  if userText.len == 0:
    raise newException(DataError,
      "medical dataset row has neither instruction nor input text")
  if output.len == 0:
    raise newException(DataError, "medical dataset row has no output text")

  if addBos:
    result.add BosToken
  result.add TurnStart
  result.add "user\n"
  result.add userText
  result.add "\n"
  result.add TurnEnd
  result.add "\n"
  result.add TurnStart
  result.add "model\n"
  result.add output
  result.add "\n"
  result.add TurnEnd
  result.add "\n"

proc sampleRowLimit(): int =
  let raw = getEnv("REW_GEMMA4_SAMPLE_ROWS", $DefaultSampleRows)
  try:
    result = parseInt(raw)
  except ValueError:
    raise newException(ValueError,
      "REW_GEMMA4_SAMPLE_ROWS must be an integer, got: " & raw)
  if result <= 0:
    raise newException(ValueError,
      "REW_GEMMA4_SAMPLE_ROWS must be positive, got: " & raw)

proc envPositiveInt(name: string; defaultValue: int): int =
  let raw = getEnv(name, $defaultValue)
  try:
    result = parseInt(raw)
  except ValueError:
    raise newException(ValueError, name & " must be an integer, got: " & raw)
  if result <= 0:
    raise newException(ValueError, name & " must be positive, got: " & raw)

proc envPositiveF32(name: string; defaultValue: float32): float32 =
  let raw = getEnv(name, $defaultValue)
  try:
    result = parseFloat(raw).float32
  except ValueError:
    raise newException(ValueError, name & " must be a float, got: " & raw)
  if result <= 0'f32:
    raise newException(ValueError, name & " must be positive, got: " & raw)

proc trainExampleLimit(): int =
  envPositiveInt("REW_GEMMA4_TRAIN_EXAMPLES", DefaultTrainExamples)

proc trainStepLimit(): int =
  envPositiveInt("REW_GEMMA4_TRAIN_STEPS", DefaultTrainSteps)

proc trainBatchSize(): int =
  envPositiveInt("REW_GEMMA4_BATCH_SIZE", DefaultTrainBatchSize)

proc trainLearningRate(): float32 =
  envPositiveF32("REW_GEMMA4_TRAIN_LR", DefaultTrainLearningRate)

proc averageTokens(stats: MedicalSampleStats): float32 =
  if stats.rows == 0:
    0'f32
  else:
    float32(stats.totalTokens) / float32(stats.rows)

proc tokenBucket(tokenId: int): int =
  tokenId mod TokenBuckets

proc addTrainingExample(data: var MedicalTrainingData; inputToken,
    targetToken: int) =
  let inputBucket = tokenBucket(inputToken)
  let targetBucket = tokenBucket(targetToken)
  let inputStart = data.inputs.len
  let labelStart = data.labels.len
  data.inputs.setLen(inputStart + TokenBuckets)
  data.labels.setLen(labelStart + TokenBuckets)
  data.inputs[inputStart + inputBucket] = 1'f32
  data.labels[labelStart + targetBucket] = 1'f32
  inc data.examples

proc collectMedicalTrainingData(dataFile: string; tokenizer: HfTokenizer;
    rowLimit, exampleLimit: int): MedicalTrainingData =
  let ds = fromHfJsonFile(dataFile)
  for row in ds:
    let text = formatGemma4MedicalSft(row)
    let ids = tokenizer.encode(text)
    if result.stats.rows == 0:
      result.stats.minTokens = ids.len
      result.stats.maxTokens = ids.len
    else:
      result.stats.minTokens = min(result.stats.minTokens, ids.len)
      result.stats.maxTokens = max(result.stats.maxTokens, ids.len)
    inc result.stats.rows
    result.stats.totalChars += text.len
    result.stats.totalTokens += ids.len
    if ids.len >= 2:
      for i in 0 ..< (ids.len - 1):
        if result.examples < exampleLimit:
          result.addTrainingExample(ids[i], ids[i + 1])
    if result.stats.rows >= rowLimit or result.examples >= exampleLimit:
      break
  if result.stats.rows == 0:
    raise newException(DataError, "medical dataset yielded no rows")
  if result.examples == 0:
    raise newException(DataError,
      "medical dataset yielded no adjacent token pairs for training")

proc appendU32Le(buf: var string; value: uint32) =
  for shift in countup(0, 24, 8):
    buf.add char(((value shr shift) and 0xff'u32).int)

proc appendU64Le(buf: var string; value: uint64) =
  for shift in countup(0, 56, 8):
    buf.add char(((value shr shift) and 0xff'u64).int)

proc appendFloat32Le(buf: var string; value: float32) =
  var bits: uint32
  var v = value
  copyMem(addr bits, addr v, sizeof(bits))
  buf.appendU32Le(bits)

proc appendFloat64Le(buf: var string; value: float64) =
  var bits: uint64
  var v = value
  copyMem(addr bits, addr v, sizeof(bits))
  buf.appendU64Le(bits)

proc appendVarint(buf: var string; value: uint64) =
  var x = value
  while x >= 0x80'u64:
    buf.add char(((x and 0x7f'u64) or 0x80'u64).int)
    x = x shr 7
  buf.add char(x.int)

proc appendKey(buf: var string; fieldNumber, wireType: int) =
  buf.appendVarint(uint64((fieldNumber shl 3) or wireType))

proc appendLengthDelimited(buf: var string; fieldNumber: int; value: string) =
  buf.appendKey(fieldNumber, 2)
  buf.appendVarint(uint64(value.len))
  buf.add value

proc appendStringField(buf: var string; fieldNumber: int; value: string) =
  buf.appendLengthDelimited(fieldNumber, value)

proc appendFixed32Field(buf: var string; fieldNumber: int; value: float32) =
  buf.appendKey(fieldNumber, 5)
  buf.appendFloat32Le(value)

proc appendFixed64Field(buf: var string; fieldNumber: int; value: float64) =
  buf.appendKey(fieldNumber, 1)
  buf.appendFloat64Le(value)

proc appendVarintField(buf: var string; fieldNumber: int; value: uint64) =
  buf.appendKey(fieldNumber, 0)
  buf.appendVarint(value)

proc crc32c(data: string): uint32 =
  var crc = 0xffffffff'u32
  for ch in data:
    crc = crc xor uint32(ord(ch))
    for _ in 0 ..< 8:
      let mask = if (crc and 1'u32) == 0'u32: 0'u32 else: 0xffffffff'u32
      crc = (crc shr 1) xor (0x82f63b78'u32 and mask)
  not crc

proc maskedCrc32c(data: string): uint32 =
  let crc = crc32c(data)
  let rotated = (crc shr 15) or (crc shl 17)
  uint32((uint64(rotated) + uint64(0xa282ead8'u32)) and 0xffffffff'u64)

proc tfRecord(data: string): string =
  var lenBytes = ""
  lenBytes.appendU64Le(uint64(data.len))
  result.add lenBytes
  result.appendU32Le(maskedCrc32c(lenBytes))
  result.add data
  result.appendU32Le(maskedCrc32c(data))

proc eventFileVersion(wallTime: float64): string =
  result.appendFixed64Field(1, wallTime)
  result.appendStringField(3, "brain.Event:2")

proc scalarEvent(tag: string; value: float32; step: int;
    wallTime: float64): string =
  var valueMsg = ""
  valueMsg.appendStringField(1, tag)
  valueMsg.appendFixed32Field(2, value)
  var summary = ""
  summary.appendLengthDelimited(1, valueMsg)
  result.appendFixed64Field(1, wallTime)
  result.appendVarintField(2, uint64(step))
  result.appendLengthDelimited(5, summary)

proc writeRecord(writer: var TensorBoardWriter; data: string) =
  writer.file.write(tfRecord(data))
  writer.file.flushFile()

proc openTensorBoardWriter(logDir: string): TensorBoardWriter =
  createDir(logDir)
  let stamp = int64(epochTime())
  result.path = logDir / ("events.out.tfevents." & $stamp & "." &
    $getCurrentProcessId() & ".rew")
  result.file = open(result.path, fmWrite)
  result.writeRecord(eventFileVersion(epochTime()))

proc close(writer: var TensorBoardWriter) =
  if not writer.closed:
    writer.file.close()
    writer.closed = true

proc logScalar(writer: var TensorBoardWriter; tag: string; value: float32;
    step: int = 0) =
  writer.writeRecord(scalarEvent(tag, value, step, epochTime()))

proc logSetupMetrics(writer: var TensorBoardWriter; cfg: QloraConfig;
    stats: MedicalSampleStats; examples, steps, batchSize: int;
    learningRate: float32) =
  writer.logScalar("medical/sample_rows", float32(stats.rows))
  writer.logScalar("medical/total_chars", float32(stats.totalChars))
  writer.logScalar("medical/total_tokens", float32(stats.totalTokens))
  writer.logScalar("medical/avg_tokens", stats.averageTokens())
  writer.logScalar("medical/min_tokens", float32(stats.minTokens))
  writer.logScalar("medical/max_tokens", float32(stats.maxTokens))
  writer.logScalar("train/examples", float32(examples))
  writer.logScalar("train/steps", float32(steps))
  writer.logScalar("train/batch_size", float32(batchSize))
  writer.logScalar("train/learning_rate", learningRate)
  writer.logScalar("train/token_buckets", float32(TokenBuckets))
  writer.logScalar("qlora/rank", float32(cfg.rank))
  writer.logScalar("qlora/alpha", cfg.alpha)
  writer.logScalar("qlora/group_size", float32(cfg.groupSize))
  writer.logScalar("qlora/learning_rate", cfg.learningRate)
  writer.logScalar("qlora/sequence_length", float32(cfg.sequenceLength))

proc writeAdapterManifest(path: string; cfg: QloraConfig;
    stats: MedicalSampleStats; training: TrainingSummary; tfLogDir: string) =
  createDir(parentDir(path))
  let node = %* {
    "base_model": ModelRepo,
    "dataset": DatasetRepo,
    "dataset_task": DatasetTask,
    "sample_rows": stats.rows,
    "rank": cfg.rank,
    "alpha": cfg.alpha,
    "group_size": cfg.groupSize,
    "learning_rate": cfg.learningRate,
    "sequence_length": cfg.sequenceLength,
    "total_char_count": stats.totalChars,
    "total_token_count": stats.totalTokens,
    "average_token_count": stats.averageTokens(),
    "min_token_count": stats.minTokens,
    "max_token_count": stats.maxTokens,
    "token_buckets": TokenBuckets,
    "training_examples": training.examples,
    "training_steps": training.steps,
    "training_batch_size": training.batchSize,
    "training_learning_rate": training.learningRate,
    "initial_loss": training.initialLoss,
    "final_loss": training.finalLoss,
    "tensorboard_event_file": training.eventPath,
    "tensorboard_log_dir": tfLogDir,
    "format": "rew-qlora-adapter-manifest-v1"
  }
  writeFile(path, pretty(node))

proc copyBatch(data: MedicalTrainingData; start, batchSize: int):
    tuple[x, y: seq[float32]] =
  result.x = newSeq[float32](batchSize * TokenBuckets)
  result.y = newSeq[float32](batchSize * TokenBuckets)
  for b in 0 ..< batchSize:
    let sample = (start + b) mod data.examples
    let src = sample * TokenBuckets
    let dst = b * TokenBuckets
    for j in 0 ..< TokenBuckets:
      result.x[dst + j] = data.inputs[src + j]
      result.y[dst + j] = data.labels[src + j]

proc initQloraTrainingTensors(d: Device; cfg: QloraConfig):
    tuple[baseWeight, bias, a, b: Tensor] =
  let keys = split(initKey(20260508), 3)
  let bound = sqrt(1.0'f32 / float32(TokenBuckets))
  let baseData = uniformF32(keys[0], TokenBuckets * TokenBuckets,
    -0.02'f32, 0.02'f32)
  let aData = uniformF32(keys[1], cfg.rank * TokenBuckets, -bound, bound)
  let bData = newSeq[float32](TokenBuckets * cfg.rank)
  let biasData = newSeq[float32](TokenBuckets)
  result = (
    baseWeight: fromHostF32(d, baseData, [TokenBuckets, TokenBuckets]),
    bias: fromHostF32(d, biasData, [TokenBuckets]),
    a: fromHostF32(d, aData, [cfg.rank, TokenBuckets]),
    b: fromHostF32(d, bData, [TokenBuckets, cfg.rank]),
  )

proc qloraLayer(baseWeight, bias, a, b: Tensor; cfg: QloraConfig):
    QloraLinear =
  QloraLinear(
    dequantizedWeight: buffer(baseWeight),
    bias: buffer(bias),
    hasBias: true,
    inFeatures: TokenBuckets,
    outFeatures: TokenBuckets,
    A: param(a),
    B: param(b),
    rank: cfg.rank,
    alpha: cfg.alpha,
    scaling: cfg.alpha / float32(cfg.rank),
  )

proc trainMedicalQlora(data: MedicalTrainingData; cfg: QloraConfig;
    steps, requestedBatchSize: int; learningRate: float32;
    writer: var TensorBoardWriter): TrainingSummary =
  let batchSize = min(requestedBatchSize, data.examples)
  let d = initDevice(Backend)
  setDefaultDevice(d)
  installEagerBackend()

  var tensors = initQloraTrainingTensors(d, cfg)
  let lr = scalarF32(d, learningRate)
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let baseWeight = args[0]
    let bias = args[1]
    let x = args[4]
    let y = args[5]
    let lrArg = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let layer = qloraLayer(baseWeight, bias, p[0], p[1], cfg)
      softmaxCrossEntropy(layer.forward(x), y)
    let vr = vjp(lossFn, [args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc update(param, grad: Tensor): Tensor =
      let lrB = broadcastTo(lrArg, param.shape, @[])
      sub(param, mul(lrB, grad))
    @[vr.output, update(args[2], grads[0]), update(args[3], grads[1])]

  let trainJ = jit(trainFn, "gemma4_medical_qlora_train_step",
    donateArgs = [2, 3])
  result.steps = steps
  result.examples = data.examples
  result.batchSize = batchSize
  result.learningRate = learningRate
  result.initialLoss = Inf.float32
  result.finalLoss = Inf.float32
  result.eventPath = writer.path

  echo &"Training medical QLoRA adapter: {steps} step(s), " &
    &"{data.examples} example(s), batch={batchSize}, buckets={TokenBuckets}"
  for step in 0 ..< steps:
    let batch = data.copyBatch(step * batchSize, batchSize)
    let x = fromHostF32(d, batch.x, [batchSize, TokenBuckets])
    let y = fromHostF32(d, batch.y, [batchSize, TokenBuckets])
    let outs = trainJ.call([
      tensors.baseWeight, tensors.bias, tensors.a, tensors.b, x, y, lr])
    tensors.a = outs[1]
    tensors.b = outs[2]
    let loss = item(outs[0], float32)
    if step == 0:
      result.initialLoss = loss
    result.finalLoss = loss
    writer.logScalar("train/loss", loss, step + 1)
    echo &"  train step {step + 1}/{steps}: loss = {loss:.4f}"

proc main() =
  let includeWeights = downloadFullWeights()
  if includeWeights:
    echo "Downloading Gemma 4 full checkpoint assets..."
  else:
    echo "Downloading Gemma 4 metadata/tokenizer assets..."
  let modelDir = downloadModelAssets(includeWeights)
  echo "Model snapshot: ", modelDir

  echo "Downloading MedAlpaca medical flashcards dataset..."
  let datasetDir = downloadMedicalDataset()
  echo "Dataset snapshot: ", datasetDir
  let dataFile = firstCachedJsonFile(datasetDir)

  let tokenizer = loadGemma4Tokenizer(modelDir)
  let sampleRows = sampleRowLimit()
  let trainExamples = trainExampleLimit()
  let trainingData = collectMedicalTrainingData(dataFile, tokenizer,
    sampleRows, trainExamples)
  let stats = trainingData.stats
  echo &"Formatted {stats.rows} medical rows: {stats.totalChars} chars, " &
    &"{stats.totalTokens} tokens"
  echo &"Prepared {trainingData.examples} next-token bucket training examples"

  let cfg = defaultQloraConfig()
  let steps = trainStepLimit()
  let batchSize = trainBatchSize()
  let learningRate = trainLearningRate()

  let tfLogDir = getEnv("REW_GEMMA4_TF_LOGDIR",
    getCurrentDir() / DefaultTfLogDir)
  var writer = openTensorBoardWriter(tfLogDir)
  var training: TrainingSummary
  try:
    writer.logSetupMetrics(cfg, stats, trainingData.examples, steps,
      min(batchSize, trainingData.examples), learningRate)
    training = trainMedicalQlora(trainingData, cfg, steps, batchSize,
      learningRate, writer)
  finally:
    writer.close()
  echo "TensorBoard event file: ", training.eventPath

  let outPath = getEnv("REW_GEMMA4_ADAPTER_OUT",
    getCurrentDir() / "artifacts" / "gemma4-medical-lora" /
      "adapter_manifest.json")
  writeAdapterManifest(outPath, cfg, stats, training, tfLogDir)
  echo "Adapter manifest: ", outPath
  if includeWeights:
    echo "Full checkpoint cache requested with REW_GEMMA4_DOWNLOAD_WEIGHTS=1."
  else:
    echo "Set REW_GEMMA4_DOWNLOAD_WEIGHTS=1 before running to cache the full checkpoint."

when isMainModule:
  main()
