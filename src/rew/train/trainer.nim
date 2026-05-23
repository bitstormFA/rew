## Trainer — typed training loop orchestration.
##
## The Trainer is the high-level loop over `TrainState`, `DataSplits`, typed
## losses, typed custom steps, metrics, validation, and callbacks. Raw `jit`
## remains available through `rew/xla`; the user-facing Trainer surface stays
## on typed state and batch values.

import std/options
import ../distributed
import ../pytree
import ../tensor
import ./runtime
import ./context
import ./callback
import ./datasplits
import ./state

type
  TrainerError* = object of CatchableError
    ## Raised when a Trainer configuration is invalid.

  Trainer* = object
    maxEpochs*: int
    maxSteps*: Option[int]
    accelerator*: Accelerator
    devices*: int
    precision*: Precision
    logEvery*: int
    valInterval*: Option[int]
    donateParams*: bool
    strategy*: ParallelPolicy
    callbacks*: seq[Callback]

proc initTrainer*(maxEpochs: int = 10; accelerator: Accelerator = akAuto;
    devices: int = 1; precision: Precision = prFloat32;
    strategy: ParallelPolicy = autoParallel()): Trainer =
  ## Creates a typed Trainer configuration.
  if maxEpochs <= 0:
    raise newException(TrainerError,
      "initTrainer: maxEpochs must be positive")
  Trainer(
    maxEpochs: maxEpochs,
    accelerator: accelerator,
    devices: devices,
    precision: precision,
    logEvery: 50,
    valInterval: none[int](),
    strategy: strategy,
  )

proc checkpointWriter[S](runtime: Runtime; state: ptr S): CheckpointWriter =
  proc(path: string; saveCtx: TrainContext) {.closure.} =
    discard saveCtx
    runtime.save(path, state[])

template fireCtxCallbacks(cbs: openArray[Callback]; hookName: untyped;
    ctx: var TrainContext) =
  for cb in cbs:
    if cb.hookName.isSome:
      cb.hookName.get()(ctx)

template fireEpochEndCallbacks(cbs: openArray[Callback];
    ctx: var TrainContext; saveCheckpoint: CheckpointWriter) =
  for cb in cbs:
    if cb.onTrainEpochEnd.isSome:
      cb.onTrainEpochEnd.get()(ctx, saveCheckpoint)

template fireBatchStartCallbacks(cbs: openArray[Callback]; batchIdx: int;
    ctx: var TrainContext) =
  for cb in cbs:
    if cb.onTrainBatchStart.isSome:
      cb.onTrainBatchStart.get()(batchIdx, ctx)

template fireBatchEndCallbacks(cbs: openArray[Callback]; batchIdx: int;
    loss: Tensor; ctx: var TrainContext) =
  for cb in cbs:
    if cb.onTrainBatchEnd.isSome:
      cb.onTrainBatchEnd.get()(batchIdx, loss, ctx)

proc logStepResult[S](ctx: var TrainContext; result: StepResult[S]) =
  ctx.log("loss", result.loss, onStep = true, onEpoch = true, progBar = true)
  for metric in result.metrics:
    if metric.name != "loss":
      ctx.log(metric.name, metric.value,
        onStep = true, onEpoch = true, progBar = metric.progBar)

proc runValidation[M, B, O, S](trainer: var Trainer; state: TrainState[M, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]; runtime: Runtime;
    ctx: var TrainContext) =
  if data.val.isNone:
    return

  let previousMode = ctx.mode
  ctx.mode = tmValidate
  fireCtxCallbacks(trainer.callbacks, onValidationStart, ctx)

  let iter = data.val.get().source()
  var batchIdx = 0
  while true:
    let batch = iter()
    if finished(iter):
      break
    let callCtx = initCallCtx(runtime, tmValidate)
    let value = loss(state.model, batch, callCtx)
    ctx.log("val/loss", value, onStep = false, onEpoch = true,
      progBar = true)
    batchIdx += 1

  fireCtxCallbacks(trainer.callbacks, onValidationEnd, ctx)
  ctx.mode = previousMode

proc shouldStopAtMaxSteps(trainer: Trainer; ctx: TrainContext): bool =
  trainer.maxSteps.isSome and ctx.globalStep >= trainer.maxSteps.get()

proc finishEpoch[M, B, O, S](trainer: var Trainer; state: TrainState[M, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]; runtime: Runtime;
    ctx: var TrainContext; saveCheckpoint: CheckpointWriter) =
  if trainer.valInterval.isNone:
    runValidation(trainer, state, data, loss, runtime, ctx)
  ctx.reduceEpochMetrics()
  fireEpochEndCallbacks(trainer.callbacks, ctx, saveCheckpoint)

proc fit*[M, B, O, S](trainer: var Trainer; state: var TrainState[M, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]) =
  ## Fits a `TrainState` from a typed loss function.
  ##
  ## Users provide `loss(model, batch, ctx)`. The Trainer owns iteration,
  ## optimizer state, step counting, compiled-step caching, validation,
  ## metrics, and callbacks.
  let runtime = initRuntime(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  var ctx = initTrainContext(tmFit)
  var step = compileTrainStep(loss, state, runtime,
    donate = if trainer.donateParams: paramsOf(state.model) else: PathSet())
  let saveCheckpoint = checkpointWriter(runtime, addr(state))

  fireCtxCallbacks(trainer.callbacks, onFitStart, ctx)
  fireCtxCallbacks(trainer.callbacks, onTrainStart, ctx)

  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    ctx.mode = tmFit
    fireCtxCallbacks(trainer.callbacks, onTrainEpochStart, ctx)

    let iter = data.train.source()
    var batchIdx = 0
    while true:
      if shouldStopAtMaxSteps(trainer, ctx):
        break
      let batch = iter()
      if finished(iter):
        break

      fireBatchStartCallbacks(trainer.callbacks, batchIdx, ctx)
      let result = step(state, batch)
      state = result.state
      ctx.logStepResult(result)
      fireBatchEndCallbacks(trainer.callbacks, batchIdx, result.loss, ctx)

      ctx.globalStep += 1
      batchIdx += 1

      if trainer.valInterval.isSome and
          ctx.globalStep mod trainer.valInterval.get() == 0:
        runValidation(trainer, state, data, loss, runtime, ctx)

    finishEpoch(trainer, state, data, loss, runtime, ctx, saveCheckpoint)
    if ctx.shouldStop or shouldStopAtMaxSteps(trainer, ctx):
      break

  fireCtxCallbacks(trainer.callbacks, onTrainEnd, ctx)
  fireCtxCallbacks(trainer.callbacks, onFitEnd, ctx)

proc fit*[M, B, O, S](trainer: var Trainer; state: var TrainState[M, O, S];
    data: DataSplits[B]; trainStep: TrainStepFn[M, B, O, S]) =
  ## Fits a `TrainState` from a user-owned typed custom step.
  let runtime = initRuntime(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  var ctx = initTrainContext(tmFit)
  let saveCheckpoint = checkpointWriter(runtime, addr(state))

  fireCtxCallbacks(trainer.callbacks, onFitStart, ctx)
  fireCtxCallbacks(trainer.callbacks, onTrainStart, ctx)

  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    ctx.mode = tmFit
    fireCtxCallbacks(trainer.callbacks, onTrainEpochStart, ctx)

    let iter = data.train.source()
    var batchIdx = 0
    while true:
      if shouldStopAtMaxSteps(trainer, ctx):
        break
      let batch = iter()
      if finished(iter):
        break

      fireBatchStartCallbacks(trainer.callbacks, batchIdx, ctx)
      let callCtx = initCallCtx(runtime, tmFit)
      let result = trainStep(state, batch, callCtx)
      state = result.state
      ctx.logStepResult(result)
      fireBatchEndCallbacks(trainer.callbacks, batchIdx, result.loss, ctx)

      ctx.globalStep += 1
      batchIdx += 1

    ctx.reduceEpochMetrics()
    fireEpochEndCallbacks(trainer.callbacks, ctx, saveCheckpoint)
    if ctx.shouldStop or shouldStopAtMaxSteps(trainer, ctx):
      break

  fireCtxCallbacks(trainer.callbacks, onTrainEnd, ctx)
  fireCtxCallbacks(trainer.callbacks, onFitEnd, ctx)

proc validate*[M, B, O, S](trainer: var Trainer; state: TrainState[M, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]): seq[MetricEntry] =
  ## Runs one typed validation pass and returns epoch metrics.
  let runtime = initRuntime(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  var ctx = initTrainContext(tmValidate)
  runValidation(trainer, state, data, loss, runtime, ctx)
  ctx.reduceEpochMetrics()
  ctx.epochMetrics
