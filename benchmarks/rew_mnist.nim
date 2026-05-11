## Rechenwerk MNIST example benchmarks.
##
## Emits one JSON object per benchmark row. The companion
## `benchmarks/run.py` script compiles this file, runs the PyTorch
## equivalent, and prints a Markdown summary table.

import std/[json, math, monotimes, os, strutils, times]
import rew
import rew/pjrt/loader

const
  Framework = "Rechenwerk"
  DefaultBackend = "cpu"
  DefaultWarmup = 5
  DefaultIterations = 20
  DefaultBatchSize = 32
  ImageSide = 28
  ImagePixels = ImageSide * ImageSide
  NumClasses = 10
  MlpHiddenDim = 64
  CnnConv1Channels = 8
  CnnConv2Channels = 16
  CnnHiddenDim = 32
  CnnFeatureSide = ImageSide div 4
  CnnFeatureCount = CnnConv2Channels * CnnFeatureSide * CnnFeatureSide
  VitPatchSize = 7
  VitNumPatches = (ImageSide div VitPatchSize) * (ImageSide div VitPatchSize)
  VitPatchDim = VitPatchSize * VitPatchSize
  VitEmbedDim = 16
  VitNumHeads = 1
  VitHeadDim = VitEmbedDim div VitNumHeads
  VitTokenCount = VitNumPatches + 1
  VitMlpDim = 32
  VitParamCount = 18
  VpPatchW = 0
  VpPatchB = 1
  VpClassToken = 2
  VpPosition = 3
  VpQueryW = 4
  VpQueryB = 5
  VpKeyW = 6
  VpKeyB = 7
  VpValueW = 8
  VpValueB = 9
  VpProjW = 10
  VpProjB = 11
  VpMlp1W = 12
  VpMlp1B = 13
  VpMlp2W = 14
  VpMlp2B = 15
  VpHeadW = 16
  VpHeadB = 17
  VitDonateArgs = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16, 17]

type
  BenchOptions = object
    backend: string
    warmup: int
    iterations: int
    batchSize: int

  BenchMeasurement = object
    firstStepMs: float64
    meanStepMs: float64
    samplesPerSecond: float64
    lastLoss: float64

proc usage() =
  echo """
Usage: rew_mnist [--backend=cpu] [--warmup=5] [--iterations=20] [--batch-size=32]

Runs the Rechenwerk MNIST MLP and CNN training-step benchmarks and emits
JSON lines for benchmarks/run.py.
"""

proc parseCount(name, value: string; allowZero: bool = false): int =
  try:
    result = parseInt(value)
  except ValueError:
    quit "invalid " & name & ": " & value, 2
  if result < 0 or (result == 0 and not allowZero):
    quit name & " must be positive, got " & value, 2

proc parseOptions(): BenchOptions =
  result = BenchOptions(
    backend: DefaultBackend,
    warmup: DefaultWarmup,
    iterations: DefaultIterations,
    batchSize: DefaultBatchSize,
  )
  for arg in commandLineParams():
    if arg == "--help" or arg == "-h":
      usage()
      quit 0
    elif arg.startsWith("--backend="):
      result.backend = arg["--backend=".len .. ^1]
    elif arg.startsWith("--warmup="):
      result.warmup =
        parseCount("warmup", arg["--warmup=".len .. ^1], allowZero = true)
    elif arg.startsWith("--iterations="):
      result.iterations =
        parseCount("iterations", arg["--iterations=".len .. ^1])
    elif arg.startsWith("--batch-size="):
      result.batchSize =
        parseCount("batch-size", arg["--batch-size=".len .. ^1])
    else:
      quit "unknown argument: " & arg, 2
  if result.backend.len == 0:
    quit "backend must not be empty", 2

proc baseRow(example, status: string; opts: BenchOptions): JsonNode =
  result = newJObject()
  result["example"] = %example
  result["framework"] = %Framework
  result["device"] = %opts.backend
  result["status"] = %status
  result["batch_size"] = %opts.batchSize
  result["warmup"] = %opts.warmup
  result["iterations"] = %opts.iterations

proc emitSkipped(example: string; opts: BenchOptions; reason: string) =
  let row = baseRow(example, "skipped", opts)
  row["reason"] = %reason
  row["first_step_ms"] = newJNull()
  row["mean_step_ms"] = newJNull()
  row["samples_per_s"] = newJNull()
  row["last_loss"] = newJNull()
  echo $row

proc emitFailed(example: string; opts: BenchOptions; reason: string) =
  let row = baseRow(example, "failed", opts)
  row["reason"] = %reason
  row["first_step_ms"] = newJNull()
  row["mean_step_ms"] = newJNull()
  row["samples_per_s"] = newJNull()
  row["last_loss"] = newJNull()
  echo $row

proc emitOk(example: string; opts: BenchOptions; m: BenchMeasurement) =
  let row = baseRow(example, "ok", opts)
  row["reason"] = %""
  row["first_step_ms"] = %m.firstStepMs
  row["mean_step_ms"] = %m.meanStepMs
  row["samples_per_s"] = %m.samplesPerSecond
  row["last_loss"] = %m.lastLoss
  echo $row

proc elapsedMs(start: MonoTime): float64 =
  float64((getMonoTime() - start).inNanoseconds) / 1_000_000.0

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape:
    n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

proc initLinearEager(d: Device; key: Key; inFeat, outFeat: int): Linear =
  let keys = split(key, 2)
  let bound = sqrt(1.0'f32 / float32(inFeat))
  var w = uniformF32(keys[0], inFeat * outFeat, -bound, bound)
  var b = newSeq[float32](outFeat)
  Linear(
    weight: f32ToDevice(d, w, [inFeat, outFeat]),
    bias: f32ToDevice(d, b, [outFeat]),
  )

proc initConv2dEager(d: Device; key: Key; inChannels, outChannels,
    kernelSize: int): Conv2d =
  let keys = split(key, 2)
  let fanIn = inChannels * kernelSize * kernelSize
  let bound = sqrt(1.0'f32 / float32(fanIn))
  var w = uniformF32(keys[0],
    outChannels * inChannels * kernelSize * kernelSize, -bound, bound)
  var b = newSeq[float32](outChannels)
  Conv2d(
    weight: f32ToDevice(d, w,
      [outChannels, inChannels, kernelSize, kernelSize]),
    bias: f32ToDevice(d, b, [outChannels]),
    stride: [1, 1],
    padding: [[1, 1], [1, 1]],
    dilation: [1, 1],
  )

proc initParamEager(d: Device; key: Key; shape: openArray[int];
    bound: float32): Tensor =
  var count = 1
  for dim in shape:
    count *= dim
  var data = uniformF32(key, count, -bound, bound)
  f32ToDevice(d, data, shape)

proc labelsOneHot(batchSize: int): seq[float32] =
  result = newSeq[float32](batchSize * NumClasses)
  for i in 0 ..< batchSize:
    result[i * NumClasses + (i mod NumClasses)] = 1.0'f32

proc forwardMlp(l1, l2: Linear; x: Tensor): Tensor =
  let h = relu(forward(l1, x))
  forward(l2, h)

proc forwardCnn(c1, c2: Conv2d; l1, l2: Linear; x: Tensor): Tensor =
  let h1 = relu(forward(c1, x))
  let p1 = maxPool2d(h1, [2, 2], [2, 2])
  let h2 = relu(forward(c2, p1))
  let p2 = maxPool2d(h2, [2, 2], [2, 2])
  let flat = flatten(p2)
  let h3 = relu(forward(l1, flat))
  forward(l2, h3)

proc dense2(x, weight, bias: Tensor): Tensor =
  let y = matmul(x, weight)
  let biasB = broadcastTo(bias, y.shape, [1])
  add(y, biasB)

proc dense3(x, weight, bias: Tensor): Tensor =
  let batch = x.shape[0]
  let tokens = x.shape[1]
  let outFeat = weight.shape[1]
  let flat = reshape(x, [batch * tokens, x.shape[2]])
  let y = dense2(flat, weight, bias)
  reshape(y, [batch, tokens, outFeat])

proc forwardVit(p: openArray[Tensor]; x: Tensor): Tensor =
  let batch = x.shape[0]
  let patches = reshape(x, [batch * VitNumPatches, VitPatchDim])
  let patchFlat = dense2(patches, p[VpPatchW], p[VpPatchB])
  let patchTokens = reshape(patchFlat,
    [batch, VitNumPatches, VitEmbedDim])
  let tokens0 = add(concat([p[VpClassToken], patchTokens], 1),
    p[VpPosition])

  let q0 = dense3(tokens0, p[VpQueryW], p[VpQueryB])
  let k0 = dense3(tokens0, p[VpKeyW], p[VpKeyB])
  let v0 = dense3(tokens0, p[VpValueW], p[VpValueB])
  let q = transpose(reshape(q0,
    [batch, VitTokenCount, VitNumHeads, VitHeadDim]), [0, 2, 1, 3])
  let k = transpose(reshape(k0,
    [batch, VitTokenCount, VitNumHeads, VitHeadDim]), [0, 2, 1, 3])
  let v = transpose(reshape(v0,
    [batch, VitTokenCount, VitNumHeads, VitHeadDim]), [0, 2, 1, 3])
  let rawScores = dotGeneral(q, k, [0, 1], [0, 1], [3], [3])
  let scale = scalarF32(1.0'f32 / sqrt(float32(VitHeadDim)))
  var scalarDims: seq[int] = @[]
  let scaleB = broadcastTo(scale, rawScores.shape, scalarDims)
  let weights = softmax(mul(rawScores, scaleB), 3)
  let context = dotGeneral(weights, v, [0, 1], [0, 1], [3], [2])
  let merged = reshape(transpose(context, [0, 2, 1, 3]),
    [batch, VitTokenCount, VitEmbedDim])
  let tokens1 = add(tokens0, dense3(merged, p[VpProjW], p[VpProjB]))

  let hidden = gelu(dense3(tokens1, p[VpMlp1W], p[VpMlp1B]))
  let tokens2 = add(tokens1, dense3(hidden, p[VpMlp2W], p[VpMlp2B]))
  let classOut = reshape(slice(tokens2, [0, 0, 0],
    [batch, 1, VitEmbedDim]), [batch, VitEmbedDim])
  dense2(classOut, p[VpHeadW], p[VpHeadB])

proc finishMeasurement(batchSize, iterations: int; firstMs, totalMs: float64;
    lastLoss: float32): BenchMeasurement =
  let meanMs = totalMs / float64(iterations)
  let samplesPerSecond =
    if meanMs > 0.0: float64(batchSize) * 1000.0 / meanMs else: 0.0
  BenchMeasurement(
    firstStepMs: firstMs,
    meanStepMs: meanMs,
    samplesPerSecond: samplesPerSecond,
    lastLoss: float64(lastLoss),
  )

proc benchMlp(d: Device; opts: BenchOptions): BenchMeasurement =
  let key = initKey(0xC0FFEE'u64)
  let keys = split(key, 4)
  var pixels = uniformF32(keys[0], opts.batchSize * ImagePixels, 0.0'f32,
    1.0'f32)
  var labels = labelsOneHot(opts.batchSize)
  let x = f32ToDevice(d, pixels, [opts.batchSize, ImagePixels])
  let y = f32ToDevice(d, labels, [opts.batchSize, NumClasses])

  var layer1 = initLinearEager(d, keys[1], ImagePixels, MlpHiddenDim)
  var layer2 = initLinearEager(d, keys[2], MlpHiddenDim, NumClasses)
  var lrHost = @[0.01'f32]
  let lr = f32ToDevice(d, lrHost, [])

  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]
    let ya = args[5]
    let lrArg = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: p[0], bias: p[1])
      let l2 = Linear(weight: p[2], bias: p[3])
      softmaxCrossEntropy(forwardMlp(l1, l2, xa), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1.0'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lrArg, p.shape, bdims)
      sub(p, mul(lrB, g))
    @[vr.output,
      upd(args[0], grads[0]),
      upd(args[1], grads[1]),
      upd(args[2], grads[2]),
      upd(args[3], grads[3])]

  let trainJ = jit(trainFn, "bench_mnist_mlp_train_step",
    donateArgs = [0, 1, 2, 3])

  proc runStep(): float32 =
    let outs = trainJ.call([
      layer1.weight, layer1.bias,
      layer2.weight, layer2.bias,
      x, y, lr])
    layer1 = Linear(weight: outs[1], bias: outs[2])
    layer2 = Linear(weight: outs[3], bias: outs[4])
    readBackF32(outs[0])[0]

  var lastLoss: float32
  let firstStart = getMonoTime()
  lastLoss = runStep()
  let firstMs = elapsedMs(firstStart)
  for _ in 0 ..< opts.warmup:
    lastLoss = runStep()
  var totalMs = 0.0
  for _ in 0 ..< opts.iterations:
    let start = getMonoTime()
    lastLoss = runStep()
    totalMs += elapsedMs(start)
  finishMeasurement(opts.batchSize, opts.iterations, firstMs, totalMs, lastLoss)

proc benchCnn(d: Device; opts: BenchOptions): BenchMeasurement =
  let key = initKey(0xCAFE'u64)
  let keys = split(key, 6)
  var pixels = uniformF32(keys[0],
    opts.batchSize * ImageSide * ImageSide, 0.0'f32, 1.0'f32)
  var labels = labelsOneHot(opts.batchSize)
  let x = f32ToDevice(d, pixels, [opts.batchSize, ImageSide, ImageSide, 1])
  let y = f32ToDevice(d, labels, [opts.batchSize, NumClasses])

  var conv1 = initConv2dEager(d, keys[1], 1, CnnConv1Channels, 3)
  var conv2 = initConv2dEager(d, keys[2], CnnConv1Channels,
    CnnConv2Channels, 3)
  var fc1 = initLinearEager(d, keys[3], CnnFeatureCount, CnnHiddenDim)
  var fc2 = initLinearEager(d, keys[4], CnnHiddenDim, NumClasses)
  var lrHost = @[0.05'f32]
  let lr = f32ToDevice(d, lrHost, [])

  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[8]
    let ya = args[9]
    let lrArg = args[10]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let c1 = Conv2d(weight: p[0], bias: p[1],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let c2 = Conv2d(weight: p[2], bias: p[3],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let l1 = Linear(weight: p[4], bias: p[5])
      let l2 = Linear(weight: p[6], bias: p[7])
      softmaxCrossEntropy(forwardCnn(c1, c2, l1, l2, xa), ya)
    let vr = vjp(lossFn, [
      args[0], args[1], args[2], args[3],
      args[4], args[5], args[6], args[7]])
    let grads = vr.pullback(scalarF32(1.0'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lrArg, p.shape, bdims)
      sub(p, mul(lrB, g))
    var outs: seq[Tensor] = @[vr.output]
    for i in 0 .. 7:
      outs.add upd(args[i], grads[i])
    outs

  let trainJ = jit(trainFn, "bench_mnist_cnn_train_step",
    donateArgs = [0, 1, 2, 3, 4, 5, 6, 7])

  proc runStep(): float32 =
    let outs = trainJ.call([
      conv1.weight, conv1.bias,
      conv2.weight, conv2.bias,
      fc1.weight, fc1.bias,
      fc2.weight, fc2.bias,
      x, y, lr])
    conv1 = Conv2d(weight: outs[1], bias: outs[2],
      stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
    conv2 = Conv2d(weight: outs[3], bias: outs[4],
      stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
    fc1 = Linear(weight: outs[5], bias: outs[6])
    fc2 = Linear(weight: outs[7], bias: outs[8])
    readBackF32(outs[0])[0]

  var lastLoss: float32
  let firstStart = getMonoTime()
  lastLoss = runStep()
  let firstMs = elapsedMs(firstStart)
  for _ in 0 ..< opts.warmup:
    lastLoss = runStep()
  var totalMs = 0.0
  for _ in 0 ..< opts.iterations:
    let start = getMonoTime()
    lastLoss = runStep()
    totalMs += elapsedMs(start)
  finishMeasurement(opts.batchSize, opts.iterations, firstMs, totalMs, lastLoss)

proc benchVit(d: Device; opts: BenchOptions): BenchMeasurement =
  let key = initKey(0xB17'u64)
  let keys = split(key, 12)
  var pixels = uniformF32(keys[0],
    opts.batchSize * ImagePixels, 0.0'f32, 1.0'f32)
  var labels = labelsOneHot(opts.batchSize)
  let x = f32ToDevice(d, pixels, [opts.batchSize, ImagePixels])
  let y = f32ToDevice(d, labels, [opts.batchSize, NumClasses])

  let patch = initLinearEager(d, keys[1], VitPatchDim, VitEmbedDim)
  let q = initLinearEager(d, keys[2], VitEmbedDim, VitEmbedDim)
  let k = initLinearEager(d, keys[3], VitEmbedDim, VitEmbedDim)
  let v = initLinearEager(d, keys[4], VitEmbedDim, VitEmbedDim)
  let proj = initLinearEager(d, keys[5], VitEmbedDim, VitEmbedDim)
  let mlp1 = initLinearEager(d, keys[6], VitEmbedDim, VitMlpDim)
  let mlp2 = initLinearEager(d, keys[7], VitMlpDim, VitEmbedDim)
  let head = initLinearEager(d, keys[8], VitEmbedDim, NumClasses)
  var params = @[
    patch.weight, patch.bias,
    initParamEager(d, keys[9], [opts.batchSize, 1, VitEmbedDim], 0.02'f32),
    initParamEager(d, keys[10],
      [opts.batchSize, VitTokenCount, VitEmbedDim], 0.02'f32),
    q.weight, q.bias,
    k.weight, k.bias,
    v.weight, v.bias,
    proj.weight, proj.bias,
    mlp1.weight, mlp1.bias,
    mlp2.weight, mlp2.bias,
    head.weight, head.bias]
  var lrHost = @[0.03'f32]
  let lr = f32ToDevice(d, lrHost, [])

  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[VitParamCount]
    let ya = args[VitParamCount + 1]
    let lrArg = args[VitParamCount + 2]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      softmaxCrossEntropy(forwardVit(p, xa), ya)
    let vr = vjp(lossFn, args.toOpenArray(0, VitParamCount - 1))
    let grads = vr.pullback(scalarF32(1.0'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lrArg, p.shape, bdims)
      sub(p, mul(lrB, g))
    var outs: seq[Tensor] = @[vr.output]
    for i in 0 ..< VitParamCount:
      outs.add upd(args[i], grads[i])
    outs

  let trainJ = jit(trainFn, "bench_mnist_vit_train_step",
    donateArgs = VitDonateArgs)

  proc runStep(): float32 =
    var args = params
    args.add x
    args.add y
    args.add lr
    let outs = trainJ.call(args)
    for i in 0 ..< VitParamCount:
      params[i] = outs[i + 1]
    readBackF32(outs[0])[0]

  var lastLoss: float32
  let firstStart = getMonoTime()
  lastLoss = runStep()
  let firstMs = elapsedMs(firstStart)
  for _ in 0 ..< opts.warmup:
    lastLoss = runStep()
  var totalMs = 0.0
  for _ in 0 ..< opts.iterations:
    let start = getMonoTime()
    lastLoss = runStep()
    totalMs += elapsedMs(start)
  finishMeasurement(opts.batchSize, opts.iterations, firstMs, totalMs, lastLoss)

proc runOne(example: string; d: Device; opts: BenchOptions;
    fn: proc(d: Device; opts: BenchOptions): BenchMeasurement) =
  try:
    emitOk(example, opts, fn(d, opts))
  except CatchableError as e:
    emitFailed(example, opts, e.msg)
  except Defect as e:
    emitFailed(example, opts, e.msg)

proc run() =
  let opts = parseOptions()
  try:
    discard loadPlugin(parseTarget(opts.backend))
  except PjrtError as e:
    emitSkipped("mnist_mlp", opts, e.msg)
    emitSkipped("mnist_cnn", opts, e.msg)
    emitSkipped("mnist_vit", opts, e.msg)
    return
  except TargetError as e:
    emitSkipped("mnist_mlp", opts, e.msg)
    emitSkipped("mnist_cnn", opts, e.msg)
    emitSkipped("mnist_vit", opts, e.msg)
    return

  let d = initDevice(parseTarget(opts.backend), 0)
  setDefaultDevice(d)
  installEagerBackend()
  runOne("mnist_mlp", d, opts, benchMlp)
  runOne("mnist_cnn", d, opts, benchCnn)
  runOne("mnist_vit", d, opts, benchVit)

when isMainModule:
  run()
