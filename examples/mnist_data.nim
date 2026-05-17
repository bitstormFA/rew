## MNIST training with the Dataset Pipeline API.
##
## Demonstrates the full data loading pipeline: `.npy` source through
## shuffle, batch, collate, and device transfer, feeding a two-layer
## MLP trained via `jit(vjp)`.
##
## ## Running
##
## Point `REW_MNIST_DIR` at a directory containing `train_images.npy`
## (uint8, shape `[N, 784]` or `[N, 28, 28]`) and `train_labels.npy`
## (uint8 or int64, shape `[N]`):
##
## ```
## REW_MNIST_DIR=$HOME/datasets/mnist nim c -r examples/mnist_data.nim
## ```
##
## Without `REW_MNIST_DIR`, a small synthetic dataset is generated.
## Skips with a one-line diagnostic when no PJRT plugin is installed.

import std/[math, os, strformat]
import rew
import rew/xla
import rew/pjrt/loader

const
  Backend = tCpu
  ImagePixels = 784
  HiddenDim = 64
  NumClasses = 10
  BatchSize = 32
  TrainEpochs = 3
  TrainSteps = 5
  LearningRate = 0.05'f32

# ---- host tensor helpers ---------------------------------------------------

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

# ---- eager init helpers ----------------------------------------------------

proc initLinearEager(d: Device; key: Key; inFeat, outFeat: int): Linear =
  let keys = split(key, 2)
  let bound = sqrt(1.0f32 / float32(inFeat))
  var w = uniformF32(keys[0], inFeat * outFeat, -bound, bound)
  var b = newSeq[float32](outFeat)
  Linear(
    weight: param(f32ToDevice(d, w, [inFeat, outFeat])),
    bias: param(f32ToDevice(d, b, [outFeat])),
  )

# ---- synthetic dataset -----------------------------------------------------

proc syntheticDataset(): Dataset[Sample] =
  ## Creates a small synthetic dataset of 100 samples for testing when
  ## no MNIST files are available.
  let key = initKey(0xBEEF'u64)
  var samples = newSeq[Sample](100)
  for i in 0 ..< 100:
    let k = foldIn(key, uint64(i))
    var pixels = uniformF32(k, ImagePixels, 0.0'f32, 1.0'f32)
    samples[i] = Sample(
      data: pixels,
      dataShape: @[ImagePixels],
      label: i mod NumClasses,
    )
  fromSeq(samples)

# ---- model -----------------------------------------------------------------

proc forwardMlp(l1, l2: Linear; x: Tensor): Tensor =
  let h = relu(forward(l1, x))
  forward(l2, h)

proc accuracy(logits, oneHot: Tensor): float32 =
  let lh = readBackF32(logits)
  let yh = readBackF32(oneHot)
  var correct = 0
  let n = logits.shape[0]
  for i in 0 ..< n:
    var bestL = 0; var bestLogit = -Inf.float32
    var bestY = 0; var bestYV = -1.0'f32
    for c in 0 ..< NumClasses:
      let lv = lh[i * NumClasses + c]
      if lv > bestLogit: bestLogit = lv; bestL = c
      let yv = yh[i * NumClasses + c]
      if yv > bestYV: bestYV = yv; bestY = c
    if bestL == bestY: inc correct
  float32(correct) / float32(n)

# ---- main ------------------------------------------------------------------

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

  let key = initKey(42'u64)
  let keys = split(key, 4)

  # ---- build dataset pipeline ---------------------------------------------
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
      .shuffle(keys[0], bufferSize = 10000)
      .batch(BatchSize)
      .prefetch(2)
  else:
    echo "  REW_MNIST_DIR unset — using synthetic dataset"
    pipeline = syntheticDataset()
      .shuffle(keys[0], bufferSize = 50)
      .batch(BatchSize)
      .prefetch(1)

  # ---- init model ----------------------------------------------------------
  var fc1 = initLinearEager(d, keys[1], ImagePixels, HiddenDim)
  var fc2 = initLinearEager(d, keys[2], HiddenDim, NumClasses)

  # ---- jit-compiled training step ------------------------------------------
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]
    let ya = args[5]
    let lr = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: param(p[0]), bias: param(p[1]))
      let l2 = Linear(weight: param(p[2]), bias: param(p[3]))
      softmaxCrossEntropy(forwardMlp(l1, l2, xa), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lr, p.shape, bdims)
      sub(p, mul(lrB, g))
    var outs: seq[Tensor] = @[vr.output]
    for i in 0 .. 3:
      outs.add upd(args[i], grads[i])
    outs

  let trainJ = jit(trainFn, "mnist_data_train_step",
    donateArgs = [0, 1, 2, 3])

  var lrHost = @[LearningRate]
  let lr = f32ToDevice(d, lrHost, [])

  # ---- training loop using the dataset pipeline ----------------------------
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
        fc1.weight.value, fc1.bias.value,
        fc2.weight.value, fc2.bias.value,
        x, y, lr])
      fc1 = Linear(weight: param(outs[1]), bias: param(outs[2]))
      fc2 = Linear(weight: param(outs[3]), bias: param(outs[4]))
      let loss = readBackF32(outs[0])[0]
      let acc = accuracy(forwardMlp(fc1, fc2, x), y)
      inc step
      if step <= TrainSteps or step mod 100 == 0:
        echo &"    step {step}: loss = {loss:.4f}, accuracy = {acc * 100:.1f}% (batch={b.batchSize})"
      if step >= TrainSteps and dir.len == 0:
        break

when isMainModule:
  run()
