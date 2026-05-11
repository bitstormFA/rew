## MNIST CNN training example (data pipeline API).
##
## Demonstrates the full Phase 11 stack — `Conv2d`, `maxPool2d`,
## and `Linear` traced and trained as a single `jit`-compiled SGD
## step — with data fed through the `rew/data` pipeline:
## `fromNpy` -> `map` (NHWC) -> `shuffle` -> `batch` -> `collate` ->
## `toTensors`.
##
## Architecture (NHWC):
##  - `Conv2d(1 -> 8, 3x3, padding=1)` -> ReLU -> `maxPool2d(2x2, stride=2)`
##  - `Conv2d(8 -> 16, 3x3, padding=1)` -> ReLU -> `maxPool2d(2x2, stride=2)`
##  - `flatten` -> `Linear(16*7*7 -> 32)` -> ReLU -> `Linear(32 -> 10)`
##
## ## Running
##
## With real data, point `REW_MNIST_DIR` at a directory containing
## `train_images.npy` (uint8, shape `[N, 784]` or `[N, 28, 28]`) and
## `train_labels.npy` (uint8 or int64, shape `[N]`):
##
## ```
## REW_MNIST_DIR=$HOME/datasets/mnist nim c -r examples/mnist_cnn.nim
## ```
##
## Without `REW_MNIST_DIR`, a small synthetic dataset is generated.
## Skips with a one-line diagnostic when no PJRT plugin is installed.

import std/[math, os, strformat]
import rew
import rew/pjrt/loader

const
  Backend = tCpu
  ImageSide = 28
  ImagePixels = ImageSide * ImageSide
  NumClasses = 10
  Conv1Channels = 8
  Conv2Channels = 16
  HiddenDim = 32
  PoolStride = 2
  ## Spatial size after two 2x2/stride-2 pools: 28 -> 14 -> 7.
  FeatureSide = ImageSide div (PoolStride * PoolStride)
  FeatureCount = Conv2Channels * FeatureSide * FeatureSide
  BatchSize = 32
  TrainEpochs = 1
  TrainSteps = 5
  LearningRate = 0.05'f32
  SyntheticSamples = 100
  ShuffleBuffer = 10000

# ---- host tensor helpers ----------------------------------------------

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

# ---- eager init helpers ------------------------------------------------

proc initLinearEager(d: Device; key: Key; inFeat, outFeat: int): Linear =
  let keys = split(key, 2)
  let bound = sqrt(1.0f32 / float32(inFeat))
  var w = uniformF32(keys[0], inFeat * outFeat, -bound, bound)
  var b = newSeq[float32](outFeat)
  Linear(
    weight: f32ToDevice(d, w, [inFeat, outFeat]),
    bias:   f32ToDevice(d, b, [outFeat]),
  )

proc initConv2dEager(d: Device; key: Key;
    inChannels, outChannels, kernelSize: int): Conv2d =
  let keys = split(key, 2)
  let fanIn = inChannels * kernelSize * kernelSize
  let bound = sqrt(1.0f32 / float32(fanIn))
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

# ---- dataset shaping ---------------------------------------------------

proc toNHWC(s: Sample): Sample =
  ## Reshapes each MNIST sample (delivered by `fromNpy` as `[784]` or
  ## `[28, 28]`) to NHWC `[28, 28, 1]`. After `collate`, batches become
  ## `[batchSize, 28, 28, 1]` — the layout the CNN expects.
  Sample(
    data: s.data,
    dataShape: @[ImageSide, ImageSide, 1],
    label: s.label,
  )

proc syntheticDataset(): Dataset[Sample] =
  ## Generates a small uniform-random dataset for the no-files path.
  ## Samples are already shaped as `[28, 28, 1]` so the same pipeline
  ## works without an extra reshape.
  let key = initKey(0xBEEF'u64)
  var samples = newSeq[Sample](SyntheticSamples)
  for i in 0 ..< SyntheticSamples:
    let k = foldIn(key, uint64(i))
    var pixels = uniformF32(k, ImagePixels, 0.0'f32, 1.0'f32)
    samples[i] = Sample(
      data: pixels,
      dataShape: @[ImageSide, ImageSide, 1],
      label: i mod NumClasses,
    )
  fromSeq(samples)

# ---- model -------------------------------------------------------------

proc forwardCnn(c1, c2: Conv2d; l1, l2: Linear; x: Tensor): Tensor =
  ## Two conv blocks + two-layer MLP head.
  let h1 = relu(forward(c1, x))
  let p1 = maxPool2d(h1, [2, 2], [2, 2])
  let h2 = relu(forward(c2, p1))
  let p2 = maxPool2d(h2, [2, 2], [2, 2])
  let flat = flatten(p2)
  let h3 = relu(forward(l1, flat))
  forward(l2, h3)

proc accuracy(logits, oneHot: Tensor): float32 =
  let lh = readBackF32(logits)
  let yh = readBackF32(oneHot)
  var correct = 0
  let n = logits.shape[0]
  for i in 0 ..< n:
    var bestLogit = -Inf.float32
    var bestL = 0
    var bestY = 0
    var bestYV = -1.0'f32
    for c in 0 ..< NumClasses:
      let lv = lh[i * NumClasses + c]
      if lv > bestLogit:
        bestLogit = lv; bestL = c
      let yv = yh[i * NumClasses + c]
      if yv > bestYV:
        bestYV = yv; bestY = c
    if bestL == bestY: inc correct
  float32(correct) / float32(n)

# ---- main --------------------------------------------------------------

proc run() =
  echo "── PJRT plugin: ", Backend
  try:
    discard loadPlugin(Backend)
  except PjrtError as e:
    echo "  (skip) ", e.msg
    return

  let d = initDevice(Backend)
  setDefaultDevice(d)
  installEagerBackend()
  echo &"  using device: {d}"

  let key = initKey(0xCAFE'u64)
  let keys = split(key, 5)

  # ---- build data pipeline -------------------------------------------
  let dir = getEnv("REW_MNIST_DIR")
  var pipeline: Dataset[seq[Sample]]
  if dir.len > 0:
    let imgPath = dir / "train_images.npy"
    let lblPath = dir / "train_labels.npy"
    if not fileExists(imgPath) or not fileExists(lblPath):
      echo "  (skip) REW_MNIST_DIR=", dir,
        " missing train_images.npy / train_labels.npy"
      return
    echo "  loading MNIST from ", dir
    pipeline = fromNpy(imgPath, lblPath)
      .map(toNHWC)
      .shuffle(keys[0], bufferSize = ShuffleBuffer)
      .batch(BatchSize)
  else:
    echo "  REW_MNIST_DIR unset — using synthetic dataset"
    pipeline = syntheticDataset()
      .shuffle(keys[0], bufferSize = SyntheticSamples)
      .batch(BatchSize)

  # ---- init model ----------------------------------------------------
  var conv1 = initConv2dEager(d, keys[1], 1, Conv1Channels, 3)
  var conv2 = initConv2dEager(d, keys[2], Conv1Channels, Conv2Channels, 3)
  var fc1   = initLinearEager(d, keys[3], FeatureCount, HiddenDim)
  var fc2   = initLinearEager(d, keys[4], HiddenDim, NumClasses)

  # ---- jit-compiled training step ------------------------------------
  #
  # Param order in the jit input vector:
  #   0..1  conv1.weight, conv1.bias
  #   2..3  conv2.weight, conv2.bias
  #   4..5  fc1.weight, fc1.bias
  #   6..7  fc2.weight, fc2.bias
  #   8     x   9 y   10 lr
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[8]
    let ya = args[9]
    let lr = args[10]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let c1 = Conv2d(weight: p[0], bias: p[1],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let c2 = Conv2d(weight: p[2], bias: p[3],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let l1 = Linear(weight: p[4], bias: p[5])
      let l2 = Linear(weight: p[6], bias: p[7])
      softmaxCrossEntropy(forwardCnn(c1, c2, l1, l2, xa), ya)
    let vr = vjp(lossFn,
      [args[0], args[1], args[2], args[3],
       args[4], args[5], args[6], args[7]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lr, p.shape, bdims)
      sub(p, mul(lrB, g))
    var outs: seq[Tensor] = @[vr.output]
    for i in 0 .. 7:
      outs.add upd(args[i], grads[i])
    outs

  let trainJ = jit(trainFn, "mnist_cnn_train_step",
    donateArgs = [0, 1, 2, 3, 4, 5, 6, 7])

  var lrHost = @[LearningRate]
  let lr = f32ToDevice(d, lrHost, [])

  # ---- training loop using the data pipeline -------------------------
  for epoch in 1 .. TrainEpochs:
    echo &"\n  epoch {epoch}/{TrainEpochs}"
    let iter = pipeline.source()
    var step = 0
    while true:
      let batchSamples = iter()
      if finished(iter): break
      let b = collate(batchSamples)
      let (x, y) = toTensors(d, b, NumClasses)
      let outs = trainJ.call([
        conv1.weight, conv1.bias,
        conv2.weight, conv2.bias,
        fc1.weight,   fc1.bias,
        fc2.weight,   fc2.bias,
        x, y, lr])
      conv1 = Conv2d(weight: outs[1], bias: outs[2],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      conv2 = Conv2d(weight: outs[3], bias: outs[4],
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      fc1   = Linear(weight: outs[5], bias: outs[6])
      fc2   = Linear(weight: outs[7], bias: outs[8])
      let loss = readBackF32(outs[0])[0]
      let acc = accuracy(forwardCnn(conv1, conv2, fc1, fc2, x), y)
      inc step
      if step <= TrainSteps or step mod 100 == 0:
        echo &"    step {step}: loss = {loss:.4f}, accuracy = {acc * 100:.1f}% (batch={b.batchSize})"
      if step >= TrainSteps and dir.len == 0:
        break

when isMainModule:
  run()
