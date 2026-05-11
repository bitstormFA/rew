## MNIST MLP training with the Trainer API (Tier 2).
##
## The Trainer owns the loop — you define `trainingStep` and
## `configureOptimizers` on your Task type, add callbacks, and call
## `trainer.fit(task, data)`.
##
## Compares with `mnist_workbench.nim` where you own the loop.
##
## ## Running
##
## Point `REW_MNIST_DIR` at a directory containing `train_images.npy`
## and `train_labels.npy`, or the example runs with synthetic data:
##
## ```
## REW_MNIST_DIR=$HOME/datasets/mnist nim c -r examples/mnist_trainer.nim
## ```

import std/[math, os, strformat, options]
import rew
import rew/train
import rew/pjrt/loader

const
  Backend = tCpu
  ImagePixels = 784
  HiddenDim = 64
  NumClasses = 10
  BatchSize = 32
  LearningRate = 0.05'f32

# ---- Synthetic data helpers -----------------------------------------------

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

proc makeBatch(key: Key): Batch =
  let keys = split(key, 2)
  var pixels = uniformF32(keys[0], BatchSize * ImagePixels, 0.0'f32, 1.0'f32)
  var labels = newSeq[int](BatchSize)
  for i in 0 ..< BatchSize:
    labels[i] = i mod NumClasses
  Batch(data: pixels, dataShape: @[BatchSize, ImagePixels],
        labels: labels, batchSize: BatchSize)

# ---- Task definition -------------------------------------------------------

type
  MnistTask = object
    fc1: Linear
    fc2: Linear
    opt: OptimizerConfig
    key: Key
    lr: Tensor
    trainJ: JitFunction
    stepCount: int

proc initMnistTask(d: Device; key: Key; lr: Tensor; trainJ: JitFunction): MnistTask =
  let keys = split(key, 2)
  let bound = sqrt(1.0'f32 / float32(ImagePixels))
  var w1 = uniformF32(keys[0], ImagePixels * HiddenDim, -bound, bound)
  var b1 = newSeq[float32](HiddenDim)
  var w2 = uniformF32(keys[1], HiddenDim * NumClasses, -bound, bound)
  var b2 = newSeq[float32](NumClasses)
  MnistTask(
    fc1: Linear(weight: fromHostF32(d, w1, [ImagePixels, HiddenDim]),
                bias:   fromHostF32(d, b1, [HiddenDim])),
    fc2: Linear(weight: fromHostF32(d, w2, [HiddenDim, NumClasses]),
                bias:   fromHostF32(d, b2, [NumClasses])),
    key: key, lr: lr, trainJ: trainJ,
    opt: initOptimizerConfig(OptimizerKind(kind: otSgd,
        sgd: initSgd(lr))),
  )

# ---- Required hooks --------------------------------------------------------

proc trainingStep(t: var MnistTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  let (bx, by) = toTensors(t.lr.device, batch, NumClasses)
  let outs = t.trainJ.call([
      t.fc1.weight, t.fc1.bias, t.fc2.weight, t.fc2.bias,
      bx, by, t.lr])
  t.fc1 = Linear(weight: outs[1], bias: outs[2])
  t.fc2 = Linear(weight: outs[3], bias: outs[4])
  let lossVal = readBackF32(outs[0])[0]
  ctx.log("train/loss", lossVal, onStep = true, onEpoch = true, progBar = true)
  t.stepCount += 1
  outs[0]

proc configureOptimizers(t: MnistTask): OptimizerConfig =
  t.opt

# ---- Optional hooks --------------------------------------------------------

proc onFitStart(t: var MnistTask; ctx: var TrainContext) =
  echo "Training started"

proc onTrainEpochStart(t: var MnistTask; ctx: var TrainContext) =
  echo &"\n── Epoch {ctx.epoch} ──"

proc onTrainEpochEnd(t: var MnistTask; ctx: var TrainContext) =
  for m in ctx.epochMetrics:
    if m.name == "train/loss":
      echo &"  avg loss = {m.value:.4f}"

# ---- Main -----------------------------------------------------------------

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

  # ---- Build jit-compiled training step ----
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]; let ya = args[5]; let lr = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: p[0], bias: p[1])
      let l2 = Linear(weight: p[2], bias: p[3])
      softmaxCrossEntropy(forward(l2, relu(forward(l1, xa))), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      sub(p, mul(broadcastTo(lr, p.shape, bdims), g))
    @[vr.output, upd(args[0], grads[0]), upd(args[1], grads[1]),
      upd(args[2], grads[2]), upd(args[3], grads[3])]

  let trainJ = jit(trainFn, "mnist_trainer_step", donateArgs = [0, 1, 2, 3])

  # ---- Create task and data ----
  var task = initMnistTask(d, initKey(42),
      scalarF32(d, LearningRate), trainJ)

  var batches = newSeq[Batch](25)  # 5 epochs * 5 batches
  var key = initKey(123)
  for i in 0 ..< batches.len:
    key = foldIn(key, uint64(i))
    batches[i] = makeBatch(key)

  let data = initDataPipe(fromSeq(batches))

  # ---- Create Trainer with callbacks ----
  var trainer = initTrainer(maxEpochs = 5, accelerator = akCpu,
      automaticOptimization = false)
  trainer.logEvery = 1
  trainer.callbacks = @[
    initEarlyStopping(monitor = "train/loss", patience = 10,
        mode = cmMin).toCallback(),
    initProgressBar(refreshRate = 1).toCallback(),
  ]

  # ---- Train ----
  trainer.fit(task, data)

  echo &"\nTraining done. {task.stepCount} steps total."

when isMainModule:
  run()
