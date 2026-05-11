## Phase 8 — Manual optimization mode.

var manualPluginAvailable = false
try:
  discard loadPlugin(tCpu)
  manualPluginAvailable = true
except PjrtError:
  discard

block manual_mode_flag:
  let trainer = initTrainer(automaticOptimization = false)
  doAssert not trainer.automaticOptimization

block manual_mode_default_is_manual:
  let trainer = initTrainer()
  doAssert not trainer.automaticOptimization

# ---- Task types at module level ----

type
  ManualTask = object
    steps: int
    gSteps: int
    dSteps: int

proc trainingStep(t: var ManualTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.gSteps += 1
  t.dSteps += 1
  t.steps += 1
  ctx.log("g_loss", 0.5'f32, onStep = true)
  ctx.log("d_loss", 0.3'f32, onStep = true)
  scalarF32(cpu(0), 0.5'f32)

proc configureOptimizers(t: ManualTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

type
  ManualCbTask = object
    steps: int

proc trainingStep(t: var ManualCbTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  t.steps += 1
  scalarF32(cpu(0), 0.5'f32)

proc configureOptimizers(t: ManualCbTask): OptimizerConfig =
  initOptimizerConfig(OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.01'f32))))

# ---- Device-dependent tests ----
if not manualPluginAvailable:
  echo "  (skip) no CPU plugin — skipping device tests"
else:
  let d = initDevice(tCpu)
  setDefaultDevice(d)
  installEagerBackend()

  block manual_mode_fit:
    var task = ManualTask(steps: 0, gSteps: 0, dSteps: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 3:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu,
        automaticOptimization = false)
    trainer.logEvery = 1
    trainer.fit(task, data)
    doAssert task.steps == 6

  block manual_mode_with_callbacks:
    var task = ManualCbTask(steps: 0)
    var batches: seq[Batch] = @[]
    for i in 0 ..< 3:
      batches.add Batch(data: @[], dataShape: @[0],
          labels: @[], batchSize: 1)
    let data = initDataPipe(fromSeq(batches))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu,
        automaticOptimization = false)
    trainer.logEvery = 1
    trainer.callbacks = @[initProgressBar(refreshRate = 1).toCallback()]
    trainer.fit(task, data)
    doAssert task.steps == 3
