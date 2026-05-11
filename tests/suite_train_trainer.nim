## Phase 8 — Full Trainer: fit/validate/test, smoke test with tiny model.

# ---- Plugin check ----
var trainerPluginAvailable = false
try:
  discard loadPlugin(tCpu)
  trainerPluginAvailable = true
except PjrtError:
  discard

block trainer_init_defaults:
  let trainer = initTrainer()
  doAssert trainer.maxEpochs == 10
  doAssert trainer.accelerator == akAuto
  doAssert trainer.devices == 1
  doAssert not trainer.automaticOptimization
  doAssert trainer.maxSteps.isNone
  doAssert trainer.callbacks.len == 0

block trainer_init_custom:
  let trainer = initTrainer(maxEpochs = 5, accelerator = akCpu,
      automaticOptimization = false)
  doAssert trainer.maxEpochs == 5
  doAssert not trainer.automaticOptimization

block trainer_fields_mutable:
  var trainer = initTrainer()
  trainer.maxEpochs = 20
  trainer.maxSteps = some(100)
  trainer.logEvery = 10
  doAssert trainer.maxEpochs == 20
  doAssert trainer.maxSteps.get() == 100
  doAssert trainer.logEvery == 10

# ---- Task types defined at module level (needed by checkRequiredHooks) ----

type
  SimpleTask = object
    steps: int
    model: int

proc trainingStep(t: var SimpleTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.steps += 1
  ctx.log("loss", 0.5'f32, onStep = true, onEpoch = true)
  scalarF32(cpu(0), 0.5'f32)

proc configureOptimizers(t: SimpleTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

block trainer_automatic_mode_rejected:
  var task = SimpleTask(steps: 0, model: 0)
  let batches = @[Batch(data: @[], dataShape: @[0],
      labels: @[], batchSize: 1)]
  let data = initDataPipe(fromSeq(batches))
  var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu,
      automaticOptimization = true)
  var raised = false
  try:
    trainer.fit(task, data)
  except TrainerError:
    raised = true
  doAssert raised

type
  HookTask = object
    log: seq[string]
    steps: int

proc trainingStep(t: var HookTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.steps += 1
  scalarF32(cpu(0), 0.5'f32)

proc configureOptimizers(t: HookTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

proc onFitStart(t: var HookTask; ctx: var TrainContext) =
  t.log.add "fitStart"

proc onFitEnd(t: var HookTask; ctx: var TrainContext) =
  t.log.add "fitEnd"

type
  ValTask = object
    valRan: bool

proc validationStep(t: var ValTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.valRan = true
  scalarF32(cpu(0), 0.0'f32)

type
  StepTask = object
    steps: int

proc trainingStep(t: var StepTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.steps += 1
  scalarF32(cpu(0), 0.0'f32)

proc configureOptimizers(t: StepTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

type
  EstopTask = object
    steps: int

proc trainingStep(t: var EstopTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.steps += 1
  ctx.log("loss", 1.0'f32, onStep = true, onEpoch = true)
  scalarF32(cpu(0), 1.0'f32)

proc configureOptimizers(t: EstopTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

type
  TestTask = object
    testBatches: seq[int]

proc testStep(t: var TestTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.testBatches.add batchIdx
  scalarF32(cpu(0), 0.0'f32)

type
  PredictTask = object
    predictBatches: seq[int]

proc predictStep(t: var PredictTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.predictBatches.add batchIdx
  scalarF32(cpu(0), float32(batchIdx))

type
  CheckpointTask = object
    savedPaths: seq[string]

proc trainingStep(t: var CheckpointTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  ctx.log("loss", 0.5'f32, onStep = true, onEpoch = true)
  scalarF32(cpu(0), 0.5'f32)

proc configureOptimizers(t: CheckpointTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

proc onSaveCheckpoint(t: var CheckpointTask; path: string;
    ctx: var TrainContext) =
  t.savedPaths.add path

type
  AutoParams = object
    w: Tensor

  AutoTask = object
    params: AutoParams
    opt: OptimizerConfig
    afterBackward: int
    optSteps: int
    losses: seq[float32]

proc configureOptimizers(t: AutoTask): OptimizerConfig =
  t.opt

proc parameters(t: AutoTask): AutoParams =
  t.params

proc setParameters(t: var AutoTask; params: AutoParams) =
  t.params = params

proc trainingInputs(t: var AutoTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): seq[Tensor] =
  @[scalarF32(cpu(0), batch.data[0])]

proc trainingLoss(t: var AutoTask; params: AutoParams;
    inputs: openArray[Tensor]; batchIdx: int): Tensor =
  let diff = sub(params.w, inputs[0])
  mul(diff, diff)

proc onAfterBackward(t: var AutoTask; grads: AutoParams;
    ctx: var TrainContext) =
  t.afterBackward += 1

proc onAfterOptimizerStep(t: var AutoTask; optIdx: int;
    ctx: var TrainContext) =
  t.optSteps += 1

proc onTrainBatchEnd(t: var AutoTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext; loss: Tensor) =
  t.losses.add loss.item(float32)

# ---- Device-dependent tests ----
if not trainerPluginAvailable:
  echo "  (skip) no CPU plugin — skipping device tests"
else:
  let d = initDevice(tCpu)
  setDefaultDevice(d)
  installEagerBackend()

  block trainer_fit_minimal:
    var task = SimpleTask(steps: 0, model: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 5:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu)
    trainer.logEvery = 1
    trainer.fit(task, data)
    doAssert task.steps == 10

  block trainer_fit_with_hooks:
    var task = HookTask(log: @[], steps: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 2:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    trainer.logEvery = 1
    trainer.fit(task, data)
    doAssert "fitStart" in task.log
    doAssert "fitEnd" in task.log

  block trainer_validate_standalone:
    var task = ValTask(valRan: false)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 2:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let trainFn = fromSeq(batches)
    let data = initDataPipe(trainFn, val = some(trainFn))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    discard trainer.validate(task, data)
    doAssert task.valRan

  block trainer_max_steps:
    var task = StepTask(steps: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 20:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 5, accelerator = akCpu)
    trainer.logEvery = 1
    trainer.maxSteps = some(3)
    trainer.fit(task, data)
    doAssert task.steps == 3

  block trainer_early_stopping:
    var task = EstopTask(steps: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 3:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 10, accelerator = akCpu)
    trainer.logEvery = 1
    trainer.callbacks = @[initEarlyStopping(
        monitor = "loss", patience = 1, mode = cmMin).toCallback()]
    trainer.fit(task, data)
    doAssert task.steps < 30

  block trainer_test_uses_dataset_source:
    var task = TestTask(testBatches: @[])
    let trainBatches = @[Batch(data: @[], dataShape: @[0],
        labels: @[], batchSize: 1)]
    var testBatches: seq[Batch] = @[]
    for i in 0 ..< 2:
      testBatches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(trainBatches),
      test = some(fromSeq(testBatches)))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    discard trainer.test(task, data)
    doAssert task.testBatches == @[0, 1]

  block trainer_predict_increments_batch_idx:
    var task = PredictTask(predictBatches: @[])
    let trainBatches = @[Batch(data: @[], dataShape: @[0],
        labels: @[], batchSize: 1)]
    var predictBatches: seq[Batch] = @[]
    for i in 0 ..< 3:
      predictBatches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(trainBatches),
      predict = some(fromSeq(predictBatches)))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    let preds = trainer.predict(task, data)
    doAssert task.predictBatches == @[0, 1, 2]
    doAssert preds.len == 3

  block trainer_checkpoint_calls_task_hook:
    var task = CheckpointTask(savedPaths: @[])
    let batches = @[Batch(data: @[], dataShape: @[0],
        labels: @[], batchSize: 1)]
    let data = initDataPipe(fromSeq(batches))
    let dir = getTempDir() / "rew_trainer_checkpoint_test"
    if dirExists(dir):
      removeDir(dir)
    defer:
      if dirExists(dir):
        removeDir(dir)
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)
    trainer.callbacks = @[initCheckpoint(monitor = "loss", dirPath = dir,
        saveLast = false, saveTopK = 1, filename = "ckpt-{epoch}").toCallback()]
    trainer.fit(task, data)
    doAssert task.savedPaths.len == 1
    doAssert dirExists(task.savedPaths[0])

  block trainer_automatic_sgd_updates_parameters:
    let lr = scalarF32(cpu(0), 0.1'f32)
    var task = AutoTask(
      params: AutoParams(w: scalarF32(cpu(0), 0.0'f32)),
      opt: initOptimizerConfig(OptimizerKind(kind: otSgd,
        sgd: initSgd(lr))),
      losses: @[],
    )
    let batches = @[Batch(data: @[1.0'f32], dataShape: @[1],
        labels: @[], batchSize: 1)]
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu,
      automaticOptimization = true)
    trainer.fit(task, data)
    doAssert abs(task.params.w.item(float32) - 0.2'f32) < 1e-3'f32
    doAssert task.afterBackward == 1
    doAssert task.optSteps == 1
    doAssert task.losses.len == 1

  block trainer_automatic_accumulates_and_clips:
    let lr = scalarF32(cpu(0), 0.1'f32)
    var task = AutoTask(
      params: AutoParams(w: scalarF32(cpu(0), 0.0'f32)),
      opt: initOptimizerConfig(OptimizerKind(kind: otSgd,
        sgd: initSgd(lr))),
      losses: @[],
    )
    let batches = @[
      Batch(data: @[1.0'f32], dataShape: @[1], labels: @[], batchSize: 1),
      Batch(data: @[1.0'f32], dataShape: @[1], labels: @[], batchSize: 1),
    ]
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu,
      automaticOptimization = true)
    trainer.accumulateGradBatches = 2
    trainer.gradientClipVal = some(0.5'f32)
    trainer.fit(task, data)
    doAssert abs(task.params.w.item(float32) - 0.05'f32) < 1e-3'f32
    doAssert task.afterBackward == 2
    doAssert task.optSteps == 1

  block trainer_automatic_applies_step_scheduler:
    let lr = scalarF32(cpu(0), 0.1'f32)
    let sched = SchedulerConfig(
      scheduler: SchedulerKind(kind: stExponentialLR,
        exponential: initExponentialLR(gamma = 0.5'f32)),
      interval: siStep,
      frequency: 1,
    )
    var task = AutoTask(
      params: AutoParams(w: scalarF32(cpu(0), 0.0'f32)),
      opt: initOptimizerConfig(OptimizerKind(kind: otSgd,
        sgd: initSgd(lr)), scheduler = some(sched)),
      losses: @[],
    )
    let batches = @[
      Batch(data: @[1.0'f32], dataShape: @[1], labels: @[], batchSize: 1),
      Batch(data: @[1.0'f32], dataShape: @[1], labels: @[], batchSize: 1),
      Batch(data: @[1.0'f32], dataShape: @[1], labels: @[], batchSize: 1),
    ]
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu,
      automaticOptimization = true)
    trainer.fit(task, data)
    doAssert abs(task.params.w.item(float32) - 0.424'f32) < 1e-3'f32
    doAssert task.optSteps == 3
