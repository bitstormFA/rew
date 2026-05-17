## Coherent high-level training API example.
##
## This uses MNIST-shaped synthetic data and shows the same model trained
## through three surfaces:
## - a manual loop with a typed compiled step,
## - `Trainer.fit(state, data, loss)`,
## - `Trainer.fit(state, data, customStep)`.

import std/[strformat]
import rew

const
  Backend = tCpu
  ImagePixels = 784
  NumClasses = 10
  BatchSize = 4

type
  MnistTiny = object
    head: Linear

  MnistBatch = object
    x: Tensor
    y: Tensor

proc initMnistTiny(key: Key): MnistTiny =
  MnistTiny(head: initLinear(key, ImagePixels, NumClasses))

proc syntheticBatch(d: Device): MnistBatch =
  var pixels = newSeq[float32](BatchSize * ImagePixels)
  var labels = newSeq[float32](BatchSize * NumClasses)
  for i in 0 ..< BatchSize:
    pixels[i * ImagePixels + (i mod ImagePixels)] = 1'f32
    labels[i * NumClasses + (i mod NumClasses)] = 1'f32
  MnistBatch(
    x: fromHostF32(d, pixels, [BatchSize, ImagePixels]),
    y: fromHostF32(d, labels, [BatchSize, NumClasses]),
  )

proc loss(model: MnistTiny; batch: MnistBatch; ctx: CallCtx): Tensor =
  discard ctx
  softmaxCrossEntropy(forward(model.head, batch.x), batch.y)

proc run() =
  let d = initDevice(Backend)
  setDefaultDevice(d)
  installEagerBackend()
  try:
    discard scalarF32(d, 0'f32)
  except EagerError as e:
    echo "  (skip) ", e.msg
    return

  let runtime = initRuntime(akCpu)
  let batch = syntheticBatch(d)
  let lr = scalarF32(d, 0.05'f32)
  let data = initDataSplits(fromSeq(@[batch, batch]))

  echo "manual typed step"
  var manualState = initTrainState(initMnistTiny(initKey(1)), sgd(lr))
  var manualStep = compileTrainStep(loss, manualState, runtime,
    donate = paramsOf(manualState.model))
  for _ in 0 ..< 2:
    manualState = manualStep.call(manualState, batch).state
  echo &"  steps={manualState.step}"

  echo "trainer loss mode"
  var trainerState = initTrainState(initMnistTiny(initKey(2)), sgd(lr))
  var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
  trainer.fit(trainerState, data, loss)
  echo &"  steps={trainerState.step}"

  echo "trainer custom step mode"
  var customState = initTrainState(initMnistTiny(initKey(3)), sgd(lr))
  var compiled = compileTrainStep(loss, customState, runtime)
  let customStep: TrainStepFn[MnistTiny, MnistBatch] =
    proc(state: TrainState[MnistTiny]; batch: MnistBatch;
        ctx: CallCtx): StepResult[TrainState[MnistTiny]] =
      discard ctx
      compiled.call(state, batch)
  trainer.fit(customState, data, customStep)
  echo &"  steps={customState.step}"

when isMainModule:
  run()
