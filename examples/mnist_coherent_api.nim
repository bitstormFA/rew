## Coherent high-level MNIST-style training example.
##
## This file intentionally stays on `import rew`: raw compiler handles live in
## `rew/xla`, while normal users build typed value programs.

import rew

type
  MnistBatch = object
    x: Tensor
    y: Tensor

  MnistMlp = object
    l1: Linear
    l2: Linear

proc initMnistMlp(key: Key): MnistMlp =
  let keys = split(key, 2)
  MnistMlp(
    l1: initLinear(keys[0], 784, 512),
    l2: initLinear(keys[1], 512, 10),
  )

proc forward(model: MnistMlp; x: Tensor; ctx: CallCtx): Tensor =
  discard ctx
  model.l2.forward(relu(model.l1.forward(x)))

proc loss(model: MnistMlp; batch: MnistBatch; ctx: CallCtx): Tensor =
  softmaxCrossEntropy(forward(model, batch.x, ctx), batch.y)

proc trainOneBatch(batch: MnistBatch; runtime: Runtime): TrainState[
    MnistMlp, AdamW, AdamState] =
  var state = initTrainState(initMnistMlp(initKey(42)),
    adamw(scalarF32(runtime.device, 3e-4'f32)))
  var step = compileTrainStep(loss, state, runtime,
    donate = paramsOf(state.model))
  step(state, batch).state
