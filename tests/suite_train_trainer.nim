## Phase 8 — Typed Trainer: TrainState/DataSplits fit and validation.

type
  TinyModel = object
    w: Param[Tensor]

  TinyBatch = object
    x: Tensor

proc initTinyModel(d: Device): TinyModel =
  TinyModel(w: param(scalarF32(d, 0'f32)))

proc tinyLoss(model: TinyModel; batch: TinyBatch; ctx: CallCtx): Tensor =
  discard ctx
  let diff = sub(model.w, batch.x)
  mul(diff, diff)

proc makeTinyData(d: Device; n: int): DataSplits[TinyBatch] =
  var batches: seq[TinyBatch]
  for i in 0 ..< n:
    batches.add TinyBatch(x: scalarF32(d, float32(i + 1)))
  initDataSplits(fromSeq(batches), val = some(fromSeq(batches)))

block trainer_init_defaults:
  let trainer = initTrainer()
  doAssert trainer.maxEpochs == 10
  doAssert trainer.accelerator == akAuto
  doAssert trainer.devices == 1
  doAssert trainer.maxSteps.isNone
  doAssert not trainer.donateParams
  doAssert trainer.callbacks.len == 0

block trainer_init_custom:
  let trainer = initTrainer(maxEpochs = 5, accelerator = akCpu)
  doAssert trainer.maxEpochs == 5
  doAssert trainer.accelerator == akCpu

block trainer_fields_mutable:
  var trainer = initTrainer()
  trainer.maxEpochs = 20
  trainer.maxSteps = some(100)
  trainer.logEvery = 10
  trainer.donateParams = true
  doAssert trainer.maxEpochs == 20
  doAssert trainer.maxSteps.get() == 100
  doAssert trainer.logEvery == 10
  doAssert trainer.donateParams

var trainerPluginAvailable = false
try:
  discard loadPlugin(tCpu)
  trainerPluginAvailable = true
except PjrtError:
  discard

if not trainerPluginAvailable:
  echo "  (skip) no CPU plugin — skipping typed Trainer device tests"
else:
  let d = initDevice(tCpu)
  setDefaultDevice(d)
  installEagerBackend()

  block trainer_fit_loss_mode:
    var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    let data = makeTinyData(d, 3)
    var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu)
    trainer.fit(state, data, tinyLoss)
    doAssert state.step == 6

  block trainer_fit_respects_max_steps:
    var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    let data = makeTinyData(d, 10)
    var trainer = initTrainer(maxEpochs = 5, accelerator = akCpu)
    trainer.maxSteps = some(4)
    trainer.fit(state, data, tinyLoss)
    doAssert state.step == 4

  block trainer_fit_custom_step:
    var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    var compiled = compileTrainStep(tinyLoss, state, initRuntime(akCpu))
    let step: TrainStepFn[TinyModel, TinyBatch, Sgd, EmptyOptState] =
      proc(state: TrainState[TinyModel, Sgd, EmptyOptState]; batch: TinyBatch;
          ctx: CallCtx): StepResult[TrainState[TinyModel, Sgd, EmptyOptState]] =
        discard ctx
        compiled.call(state, batch)
    let data = makeTinyData(d, 2)
    var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu)
    trainer.fit(state, data, step)
    doAssert state.step == 4

  block trainer_validate_loss_mode:
    let state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    let data = makeTinyData(d, 2)
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    let metrics = trainer.validate(state, data, tinyLoss)
    var found = false
    for metric in metrics:
      if metric.name == "val/loss":
        found = true
    doAssert found

  block trainer_callbacks_lifecycle:
    var fitStarted = false
    var fitEnded = false
    var epochs = 0
    let cb = makeCallback("typed"):
      c.onFitStart = some(proc(ctx: var TrainContext) {.closure.} =
        discard ctx
        fitStarted = true)
      c.onTrainEpochEnd = some(proc(ctx: var TrainContext;
          saveCheckpoint: CheckpointWriter) {.closure.} =
        discard ctx
        epochs += 1)
      c.onFitEnd = some(proc(ctx: var TrainContext) {.closure.} =
        discard ctx
        fitEnded = true)
    var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    let data = makeTinyData(d, 1)
    var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu)
    trainer.callbacks = @[cb]
    trainer.fit(state, data, tinyLoss)
    doAssert fitStarted
    doAssert fitEnded
    doAssert epochs == 2
