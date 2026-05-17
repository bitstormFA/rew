## MNIST-shaped MLP training with the typed Trainer API.
##
## The Trainer owns the loop over `TrainState`, `DataSplits`, and a typed loss.
## Compare with `mnist_runtime.nim`, where user code owns the loop.
##
## ## Running
##
## ```
## nim c -r examples/mnist_trainer.nim
## ```

import std/[math, strformat]
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

type
  MnistMlp = object
    fc1: Linear
    fc2: Linear

  MnistBatch = object
    x: Tensor
    y: Tensor

proc initMnistMlp(d: Device; key: Key): MnistMlp =
  let keys = split(key, 2)
  let bound = sqrt(1.0'f32 / float32(ImagePixels))
  var w1 = uniformF32(keys[0], ImagePixels * HiddenDim, -bound, bound)
  var b1 = newSeq[float32](HiddenDim)
  var w2 = uniformF32(keys[1], HiddenDim * NumClasses, -bound, bound)
  var b2 = newSeq[float32](NumClasses)
  MnistMlp(
    fc1: Linear(
      weight: param(fromHostF32(d, w1, [ImagePixels, HiddenDim])),
      bias: param(fromHostF32(d, b1, [HiddenDim]))),
    fc2: Linear(
      weight: param(fromHostF32(d, w2, [HiddenDim, NumClasses])),
      bias: param(fromHostF32(d, b2, [NumClasses]))),
  )

proc forward(model: MnistMlp; x: Tensor): Tensor =
  forward(model.fc2, relu(forward(model.fc1, x)))

proc makeBatch(d: Device; key: Key): MnistBatch =
  let keys = split(key, 2)
  var pixels = uniformF32(keys[0], BatchSize * ImagePixels, 0.0'f32, 1.0'f32)
  var labels = newSeq[float32](BatchSize * NumClasses)
  for i in 0 ..< BatchSize:
    labels[i * NumClasses + (i mod NumClasses)] = 1'f32
  MnistBatch(
    x: fromHostF32(d, pixels, [BatchSize, ImagePixels]),
    y: fromHostF32(d, labels, [BatchSize, NumClasses]),
  )

proc loss(model: MnistMlp; batch: MnistBatch; ctx: CallCtx): Tensor =
  discard ctx
  softmaxCrossEntropy(forward(model, batch.x), batch.y)

proc run() =
  echo "PJRT plugin: ", Backend
  try:
    discard loadPlugin(Backend)
  except PjrtError as e:
    echo "  (skip) ", e.msg
    return

  let d = initDevice(Backend)
  setDefaultDevice(d)
  installEagerBackend()
  echo &"using device: {d}"

  var batches: seq[MnistBatch]
  var key = initKey(123)
  for i in 0 ..< 25:
    key = foldIn(key, uint64(i))
    batches.add makeBatch(d, key)

  let lr = scalarF32(d, LearningRate)
  var state = initTrainState(initMnistMlp(d, initKey(42)), sgd(lr))
  let data = initDataSplits(fromSeq(batches))

  var trainer = initTrainer(maxEpochs = 5, accelerator = akCpu)
  trainer.donateParams = true
  trainer.callbacks = @[
    initEarlyStopping(monitor = "loss", patience = 10, mode = cmMin).toCallback(),
  ]

  trainer.fit(state, data, loss)
  echo &"Training done. {state.step} steps total."

when isMainModule:
  run()
