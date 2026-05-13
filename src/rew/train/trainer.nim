## Trainer — training loop orchestration.
##
## The Trainer owns the iteration, hook firing, logging, validation
## scheduling, and callback dispatch. The user's Task type provides
## `configureOptimizers` plus either a manual `trainingStep` or the
## automatic-optimization parameter/loss hooks; all other hooks are
## optional and checked with `when compiles`.
##
## The Trainer supports manual optimization, where the task owns the
## parameter update inside `trainingStep`, and automatic optimization,
## where the task exposes parameters plus a tensor-only loss function and
## the Trainer owns gradient computation and optimizer state.
##
## ## Modes
##
## - **Manual optimization** (`trainer.automaticOptimization = false`):
##   The user handles backward, optimizer steps, and schedulers inside
##   `trainingStep`.
## - **Automatic optimization** (`trainer.automaticOptimization = true`):
##   The task must additionally define `parameters`, `setParameters`,
##   `trainingInputs`, and `trainingLoss`. The Trainer computes gradients,
##   applies clipping/accumulation, runs the configured optimizer, and writes
##   updated parameters back to the task.

import std/options
import ../tensor
import ../distributed
import ../pytree
import ../autograd/transform
import ../transform/jit
import ../ops/arith
import ../ops/linalg
import ../ops/literal as literals
import ../optim/clip
import ../optim/scheduler
from ../eager import item, scalarF32
import ./workbench
import ./context
import ./hooks
import ./optimizer
import ./callback
import ./callbacks/checkpoint
import ./datapipe
import ./state

type
  TrainerError* = object of CatchableError
    ## Raised when a Trainer configuration or task hook contract is invalid.

  Trainer* = object
    maxEpochs*: int
    maxSteps*: Option[int]
    accelerator*: Accelerator
    devices*: int
    precision*: Precision
    logEvery*: int
    valInterval*: Option[int]
    gradientClipNorm*: Option[float32]
    gradientClipVal*: Option[float32]
    accumulateGradBatches*: int
    automaticOptimization*: bool
    jit*: bool
    donateParams*: bool
    strategy*: ParallelPolicy
    callbacks*: seq[Callback]

  TrainerInternal = ref object
    wb: Workbench
    trainerAccess: TrainerAccess

var autoJitCounter {.threadvar.}: int

proc initTrainer*(maxEpochs: int = 10; accelerator: Accelerator = akAuto;
    devices: int = 1; precision: Precision = prFloat32;
    automaticOptimization: bool = false;
    strategy: ParallelPolicy = autoParallel()): Trainer =
  ## Creates a Trainer with the given configuration.
  ## Manual optimization is the default. Set `automaticOptimization = true`
  ## for Trainer-owned gradients and optimizer steps.
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
    automaticOptimization: automaticOptimization,
    accumulateGradBatches: 1,
    strategy: strategy,
  )

# ---- Internal helpers --------------------------------------------------------

proc setupInternal(trainer: var Trainer): TrainerInternal =
  let wb = initWorkbench(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  if (trainer.devices > 1 or wb.processCount > 1) and not trainer.jit:
    raise newException(TrainerError,
      "distributed Trainer requires trainer.jit = true")
  TrainerInternal(wb: wb)

template fireTaskHook(task: var typed; hookName: untyped;
    args: varargs[untyped]) =
  when compiles(task.hookName(args)):
    task.hookName(args)

template fireCbHook(cbs: openArray[Callback]; trainerPtr, taskPtr: pointer;
    hookName: untyped; args: varargs[untyped]) =
  for cb in cbs:
    if cb.hookName.isSome:
      cb.hookName.get()(trainerPtr, taskPtr, args)

template fireAllHooks(task: var typed; cbs: openArray[Callback];
    trainerPtr, taskPtr: pointer; hookName: untyped;
    args: varargs[untyped]) =
  fireTaskHook(task, hookName, args)
  fireCbHook(cbs, trainerPtr, taskPtr, hookName, args)

proc initTrainerAccess[T](callbacks: seq[Callback]; intern: TrainerInternal;
    task: var T; taskPtr: pointer): TrainerAccess =
  let taskRef = addr(task)
  TrainerAccess(
    saveCheckpoint: proc(path: string; saveCtx: TrainContext) {.closure.} =
      var hookCtx = saveCtx
      when compiles(taskRef[].onSaveCheckpoint(path, hookCtx)):
        taskRef[].onSaveCheckpoint(path, hookCtx)
      let trainerPtr = cast[pointer](addr(intern.trainerAccess))
      for cb in callbacks:
        if cb.onSaveCheckpoint.isSome:
          cb.onSaveCheckpoint.get()(trainerPtr, taskPtr, path, hookCtx)
  )

# ---- Forward declarations ----------------------------------------------------

proc runValidation*[T, D](trainer: var Trainer; task: var T; data: D;
    ctx: var TrainContext)

proc tensorSlice(args: openArray[Tensor]; start, count: int): seq[Tensor] =
  if start < 0 or count < 0 or start + count > args.len:
    raise newException(TrainerError,
      "Trainer automatic optimization internal slice out of bounds")
  result = newSeq[Tensor](count)
  for i in 0 ..< count:
    result[i] = args[start + i]

proc concatTensors(parts: openArray[seq[Tensor]]): seq[Tensor] =
  for part in parts:
    for t in part:
      result.add t

proc nextAutoJitName(prefix: string): string =
  autoJitCounter += 1
  prefix & "_" & $autoJitCounter

proc addTreeLeaves[P](a, b: P): P =
  let al = treeFlatten(a)
  let bl = treeFlatten(b)
  if al.len != bl.len:
    raise newException(TrainerError,
      "automatic optimization: gradient leaf-count mismatch")
  var leavesOut = newSeq[Tensor](al.len)
  for i in 0 ..< al.len:
    leavesOut[i] = add(al[i], bl[i])
  treeUnflatten(a, leavesOut)

proc scaleTreeLeaves[P](tree: P; scale: Tensor): P =
  let leaves = treeFlatten(tree)
  var leavesOut = newSeq[Tensor](leaves.len)
  for i, leaf in leaves:
    var bdims: seq[int] = @[]
    let scaleB = broadcastTo(scale, leaf.shape, bdims)
    leavesOut[i] = mul(scaleB, leaf)
  treeUnflatten(tree, leavesOut)

proc optimizerFlatten(opt: OptimizerKind): seq[Tensor] =
  case opt.kind
  of otSgd:
    treeFlatten(opt.sgd)
  of otAdam:
    treeFlatten(opt.adam)
  of otAdamW:
    treeFlatten(opt.adamw)
  of otMomentumSgd:
    treeFlatten(opt.momentum)

proc optimizerLeafCount(opt: OptimizerKind): int =
  optimizerFlatten(opt).len

proc optimizerUnflatten(opt: OptimizerKind;
    leaves: seq[Tensor]): OptimizerKind =
  case opt.kind
  of otSgd:
    OptimizerKind(kind: otSgd, sgd: treeUnflatten(opt.sgd, leaves))
  of otAdam:
    OptimizerKind(kind: otAdam, adam: treeUnflatten(opt.adam, leaves))
  of otAdamW:
    OptimizerKind(kind: otAdamW, adamw: treeUnflatten(opt.adamw, leaves))
  of otMomentumSgd:
    OptimizerKind(kind: otMomentumSgd,
      momentum: treeUnflatten(opt.momentum, leaves))

proc optimizerStateFlatten(state: OptimizerState): seq[Tensor] =
  case state.kind
  of otSgd:
    @[]
  of otAdam, otAdamW:
    treeFlatten(state.adamState)
  of otMomentumSgd:
    treeFlatten(state.momentumState)

proc optimizerStateLeafCount(state: OptimizerState): int =
  optimizerStateFlatten(state).len

proc optimizerStateUnflatten(state: OptimizerState;
    leaves: seq[Tensor]): OptimizerState =
  case state.kind
  of otSgd:
    if leaves.len != 0:
      raise newException(TrainerError,
        "automatic optimization: SGD state has no tensor leaves")
    OptimizerState(kind: otSgd)
  of otAdam:
    OptimizerState(kind: otAdam,
      adamState: treeUnflatten(state.adamState, leaves))
  of otAdamW:
    OptimizerState(kind: otAdamW,
      adamState: treeUnflatten(state.adamState, leaves))
  of otMomentumSgd:
    OptimizerState(kind: otMomentumSgd,
      momentumState: treeUnflatten(state.momentumState, leaves))

proc optimizerLr(opt: OptimizerKind): float32 =
  case opt.kind
  of otSgd:
    opt.sgd.lr.item(float32)
  of otAdam:
    opt.adam.lr.item(float32)
  of otAdamW:
    opt.adamw.lr.item(float32)
  of otMomentumSgd:
    opt.momentum.lr.item(float32)

proc setOptimizerLr(opt: var OptimizerKind; lr: float32) =
  case opt.kind
  of otSgd:
    opt.sgd.lr = scalarF32(opt.sgd.lr.device, lr)
  of otAdam:
    opt.adam.lr = scalarF32(opt.adam.lr.device, lr)
  of otAdamW:
    opt.adamw.lr = scalarF32(opt.adamw.lr.device, lr)
  of otMomentumSgd:
    opt.momentum.lr = scalarF32(opt.momentum.lr.device, lr)

func metricValue(ctx: TrainContext; name: string): Option[float32] =
  for m in ctx.epochMetrics:
    if m.name == name:
      return some(m.value)
  for i in countdown(ctx.metrics.high, 0):
    if ctx.metrics[i].name == name:
      return some(ctx.metrics[i].value)
  none[float32]()

proc applyScheduler(optConfig: var OptimizerConfig; ctx: TrainContext;
    interval: SchedInterval; unit: int; baseLr: float32;
    plateauState: var Option[PlateauState]) =
  if optConfig.scheduler.isNone:
    return
  let sc = optConfig.scheduler.get()
  if sc.interval != interval:
    return
  if sc.frequency <= 0:
    raise newException(TrainerError,
      "automatic optimization: scheduler frequency must be positive")
  if (unit + 1) mod sc.frequency != 0:
    return

  var currentLr = optimizerLr(optConfig.optimizer)
  let newLr =
    case sc.scheduler.kind
    of stReduceOnPlateau:
      if sc.monitor.isNone:
        raise newException(TrainerError,
          "ReduceOnPlateau scheduler requires a monitor metric")
      let observed = ctx.metricValue(sc.monitor.get())
      if observed.isNone:
        return
      if plateauState.isNone:
        plateauState = some(initPlateauState(currentLr, modeIsMin = true))
      var state = plateauState.get()
      let stepped = step(sc.scheduler.reduce, currentLr, observed.get(),
        state, modeIsMin = true)
      plateauState = some(state)
      stepped
    else:
      step(sc.scheduler, currentLr, unit, baseLr)
  if newLr != currentLr:
    optConfig.optimizer.setOptimizerLr(newLr)

proc advanceOptimizerStateHost(opt: OptimizerKind; state: var OptimizerState) =
  case opt.kind
  of otSgd:
    discard
  of otAdam, otAdamW:
    let beta1 =
      case opt.kind
      of otAdam: opt.adam.beta1
      of otAdamW: opt.adamw.beta1
      else:
        raise newException(TrainerError,
          "automatic optimization: invalid Adam optimizer state")
    let beta2 =
      case opt.kind
      of otAdam: opt.adam.beta2
      of otAdamW: opt.adamw.beta2
      else:
        raise newException(TrainerError,
          "automatic optimization: invalid Adam optimizer state")
    let prevB1 = if state.adamState.t == 0:
        1'f32
      else:
        state.adamState.beta1Power
    let prevB2 = if state.adamState.t == 0:
        1'f32
      else:
        state.adamState.beta2Power
    state.adamState.t += 1
    state.adamState.beta1Power = prevB1 * beta1
    state.adamState.beta2Power = prevB2 * beta2
  of otMomentumSgd:
    discard

template hasAutomaticHooks(task: var typed; batch: typed;
    ctx: var TrainContext): bool =
  compiles(block:
    var autoParams = task.parameters()
    let autoInputs = task.trainingInputs(batch, 0, ctx)
    discard task.trainingLoss(autoParams, autoInputs, 0)
    task.setParameters(autoParams)
  )

proc missingAutomaticHooksError() {.noreturn.} =
  raise newException(TrainerError,
    "automatic optimization requires task hooks: " &
      "parameters(task): P, setParameters(task, params), " &
      "trainingInputs(task, batch, batchIdx, ctx): seq[Tensor], and " &
      "trainingLoss(task, params, inputs, batchIdx): Tensor")

proc automaticLossAndGrads[T, P](task: var T; params: P;
    inputs: seq[Tensor]; batchIdx: int): (Tensor, P) =
  let paramCount = treeLeafCount(params)
  let paramsProto = params
  let inputCount = inputs.len
  var taskSnapshot = task
  let stepFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let traceParamLeaves = tensorSlice(args, 0, paramCount)
    let traceInputs = tensorSlice(args, paramCount, inputCount)
    let traceParams = treeUnflatten(paramsProto, traceParamLeaves)
    let lossFn = proc(flatParams: openArray[Tensor]): Tensor =
      let p = treeUnflatten(paramsProto,
        tensorSlice(flatParams, 0, paramCount))
      taskSnapshot.trainingLoss(p, traceInputs, batchIdx)
    let vg = valueAndGrad(lossFn, treeFlatten(traceParams))
    result = @[vg.value]
    for gradLeaf in vg.grads:
      result.add gradLeaf

  let j = jit(stepFn, nextAutoJitName("rew_trainer_auto_grad"))
  let outs = j.call(concatTensors([treeFlatten(params), inputs]))
  if outs.len != paramCount + 1:
    raise newException(TrainerError,
      "automatic optimization: gradient step returned wrong arity")
  let loss = outs[0]
  let grads = treeUnflatten(params, tensorSlice(outs, 1, paramCount))
  (loss, grads)

proc automaticOptimizerStep[P](trainer: Trainer; params: P; grads: P;
    opt: OptimizerKind; state: var OptimizerState; gradBatches: int): P =
  if gradBatches <= 0:
    raise newException(TrainerError,
      "automatic optimization: gradBatches must be positive")
  let paramCount = treeLeafCount(params)
  let optCount = optimizerLeafCount(opt)
  let stateCount = optimizerStateLeafCount(state)
  let gradCount = treeLeafCount(grads)
  let paramsProto = params
  let optProto = opt
  let stateProto = state
  let gradsProto = grads
  let trainerCopy = trainer
  var donateArgs: seq[int] = @[]
  if trainer.donateParams:
    for i in 0 ..< paramCount:
      donateArgs.add i

  let stepFn = proc(args: openArray[Tensor]): seq[Tensor] =
    var offset = 0
    let traceParams = treeUnflatten(paramsProto,
      tensorSlice(args, offset, paramCount))
    offset += paramCount
    let traceOpt = optimizerUnflatten(optProto,
      tensorSlice(args, offset, optCount))
    offset += optCount
    let traceState = optimizerStateUnflatten(stateProto,
      tensorSlice(args, offset, stateCount))
    offset += stateCount
    var traceGrads = treeUnflatten(gradsProto,
      tensorSlice(args, offset, gradCount))

    if gradBatches > 1:
      let scale = literals.scalarF32(1'f32 / float32(gradBatches))
      traceGrads = scaleTreeLeaves(traceGrads, scale)
    if trainerCopy.gradientClipNorm.isSome:
      traceGrads = clipGradNorm(traceGrads,
        trainerCopy.gradientClipNorm.get())
    if trainerCopy.gradientClipVal.isSome:
      traceGrads = clipGradValue(traceGrads,
        trainerCopy.gradientClipVal.get())

    let (updatedParams, updatedState) =
      traceOpt.step(traceParams, traceGrads, traceState)
    result = concatTensors([treeFlatten(updatedParams),
      optimizerStateFlatten(updatedState)])

  let j = jit(stepFn, nextAutoJitName("rew_trainer_auto_update"),
    donateArgs = donateArgs)
  let outs = j.call(concatTensors([treeFlatten(params), optimizerFlatten(opt),
    optimizerStateFlatten(state), treeFlatten(grads)]))
  let expected = paramCount + stateCount
  if outs.len != expected:
    raise newException(TrainerError,
      "automatic optimization: optimizer step returned wrong arity")
  result = treeUnflatten(params, tensorSlice(outs, 0, paramCount))
  state = optimizerStateUnflatten(state,
    tensorSlice(outs, paramCount, stateCount))
  advanceOptimizerStateHost(opt, state)

proc shouldStepOptimizer(trainer: Trainer; optConfig: OptimizerConfig;
    pendingGradBatches: int): bool =
  let stepEvery = max(1, trainer.accumulateGradBatches) *
    max(1, optConfig.frequency)
  pendingGradBatches >= stepEvery

proc runAutomaticFit[T, D](trainer: var Trainer; task: var T; data: D;
    ctx: var TrainContext; trainerPtr, taskPtr: pointer) =
  var optConfig = task.configureOptimizers()
  var params = task.parameters()
  var optState = initOptimizerState(optConfig.optimizer, params)
  let baseLr = optimizerLr(optConfig.optimizer)
  var plateauState = none[PlateauState]()
  var gradAccum = params
  var hasGradAccum = false
  var pendingGradBatches = 0
  var stoppedByMaxSteps = false

  template flushOptimizer() =
    if hasGradAccum:
      fireTaskHook(task, onBeforeOptimizerStep, 0, ctx)
      for cb in trainer.callbacks:
        if cb.onBeforeOptimizerStep.isSome:
          cb.onBeforeOptimizerStep.get()(trainerPtr, taskPtr, 0, ctx)
      params = automaticOptimizerStep(trainer, params, gradAccum,
        optConfig.optimizer, optState, pendingGradBatches)
      task.setParameters(params)
      fireTaskHook(task, onAfterOptimizerStep, 0, ctx)
      for cb in trainer.callbacks:
        if cb.onAfterOptimizerStep.isSome:
          cb.onAfterOptimizerStep.get()(trainerPtr, taskPtr, 0, ctx)
      applyScheduler(optConfig, ctx, siStep, ctx.globalStep, baseLr,
        plateauState)
      hasGradAccum = false
      pendingGradBatches = 0

  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    ctx.mode = tmFit

    fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
        onTrainEpochStart, ctx)

    let trainIter = data.train.source()
    var batchIdx = 0
    while true:
      let batch = trainIter()
      if finished(trainIter): break

      if trainer.maxSteps.isSome and ctx.globalStep >= trainer.maxSteps.get():
        stoppedByMaxSteps = true
        break

      when hasAutomaticHooks(task, batch, ctx):
        fireTaskHook(task, onTrainBatchStart, batch, batchIdx, ctx)
        for cb in trainer.callbacks:
          if cb.onTrainBatchStart.isSome:
            cb.onTrainBatchStart.get()(trainerPtr, taskPtr, batch, batchIdx,
              ctx)

        let inputs = task.trainingInputs(batch, batchIdx, ctx)
        let (loss, grads) =
          automaticLossAndGrads(task, params, inputs, batchIdx)

        fireTaskHook(task, onBeforeBackward, loss, ctx)
        for cb in trainer.callbacks:
          if cb.onBeforeBackward.isSome:
            cb.onBeforeBackward.get()(trainerPtr, taskPtr, loss, ctx)

        if hasGradAccum:
          gradAccum = addTreeLeaves(gradAccum, grads)
        else:
          gradAccum = grads
          hasGradAccum = true
        pendingGradBatches += 1

        fireTaskHook(task, onAfterBackward, grads, ctx)
        for cb in trainer.callbacks:
          if cb.onAfterBackward.isSome:
            cb.onAfterBackward.get()(trainerPtr, taskPtr, ctx)

        if shouldStepOptimizer(trainer, optConfig, pendingGradBatches):
          flushOptimizer()

        fireTaskHook(task, onTrainBatchEnd, batch, batchIdx, ctx, loss)
        for cb in trainer.callbacks:
          if cb.onTrainBatchEnd.isSome:
            cb.onTrainBatchEnd.get()(trainerPtr, taskPtr, batch, batchIdx,
              loss, ctx)

        ctx.globalStep += 1
        batchIdx += 1

        if trainer.valInterval.isSome and
            ctx.globalStep mod trainer.valInterval.get() == 0:
          let hasVal = when compiles(data.val.isSome):
              data.val.isSome
            else:
              false
          if hasVal:
            runValidation(trainer, task, data, ctx)
      else:
        missingAutomaticHooksError()

    flushOptimizer()

    let hasVal = when compiles(data.val.isSome): data.val.isSome else: false
    if trainer.valInterval.isNone and hasVal:
      runValidation(trainer, task, data, ctx)

    ctx.reduceEpochMetrics()

    applyScheduler(optConfig, ctx, siEpoch, epoch, baseLr, plateauState)

    fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
        onTrainEpochEnd, ctx)

    if ctx.shouldStop or stoppedByMaxSteps:
      break

proc runManualFit[T, D](trainer: var Trainer; task: var T; data: D;
    ctx: var TrainContext; trainerPtr, taskPtr: pointer) =
  var stoppedByMaxSteps = false
  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    ctx.mode = tmFit

    fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
        onTrainEpochStart, ctx)

    let trainIter = data.train.source()
    var batchIdx = 0
    while true:
      let batch = trainIter()
      if finished(trainIter): break

      if trainer.maxSteps.isSome and ctx.globalStep >= trainer.maxSteps.get():
        stoppedByMaxSteps = true
        break

      fireTaskHook(task, onTrainBatchStart, batch, batchIdx, ctx)
      for cb in trainer.callbacks:
        if cb.onTrainBatchStart.isSome:
          cb.onTrainBatchStart.get()(trainerPtr, taskPtr, batch, batchIdx,
            ctx)

      let loss = task.trainingStep(batch, batchIdx, ctx)

      fireTaskHook(task, onTrainBatchEnd, batch, batchIdx, ctx, loss)
      for cb in trainer.callbacks:
        if cb.onTrainBatchEnd.isSome:
          cb.onTrainBatchEnd.get()(trainerPtr, taskPtr, batch, batchIdx,
            loss, ctx)

      ctx.globalStep += 1
      batchIdx += 1

      if trainer.valInterval.isSome and
          ctx.globalStep mod trainer.valInterval.get() == 0:
        let hasVal = when compiles(data.val.isSome): data.val.isSome else: false
        if hasVal:
          runValidation(trainer, task, data, ctx)

    let hasVal = when compiles(data.val.isSome): data.val.isSome else: false
    if trainer.valInterval.isNone and hasVal:
      runValidation(trainer, task, data, ctx)

    ctx.reduceEpochMetrics()

    fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
        onTrainEpochEnd, ctx)

    if ctx.shouldStop or stoppedByMaxSteps:
      break

# ---- fit --------------------------------------------------------------------

proc fit*[T, D](trainer: var Trainer; task: var T; data: D) =
  ## Runs the full training loop.
  ##
  ## `T` is the user's Task type. `configureOptimizers` is required.
  ## Manual optimization requires `trainingStep`; automatic optimization
  ## requires `parameters`, `setParameters`, `trainingInputs`, and
  ## `trainingLoss`. All other hooks are optional.
  ##
  ## `D` must provide a `train` field of type `Dataset[Batch]` and
  ## optionally `val`, `test`, `predict` fields.
  checkConfigureOptimizers(T)

  let intern = setupInternal(trainer)
  var ctx = initTrainContext(tmFit)

  let taskPtr = cast[pointer](addr(task))
  intern.trainerAccess = initTrainerAccess(trainer.callbacks, intern,
    task, taskPtr)
  let trainerPtr = cast[pointer](addr(intern.trainerAccess))

  # ---- Resolve hooks at compile time ----
  fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr, onFitStart, ctx)
  fireTaskHook(task, onTrainStart, ctx)
  for cb in trainer.callbacks:
    if cb.onTrainStart.isSome:
      cb.onTrainStart.get()(trainerPtr, taskPtr, ctx)

  template finishTraining() =
    fireTaskHook(task, onTrainEnd, ctx)
    for cb in trainer.callbacks:
      if cb.onTrainEnd.isSome:
        cb.onTrainEnd.get()(trainerPtr, taskPtr, ctx)
    fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr, onFitEnd, ctx)

  if trainer.automaticOptimization:
    when compiles(block:
      var autoParams = task.parameters()
      task.setParameters(autoParams)
    ):
      runAutomaticFit(trainer, task, data, ctx, trainerPtr, taskPtr)
      finishTraining()
      return
    else:
      missingAutomaticHooksError()

  when compiles(runManualFit(trainer, task, data, ctx, trainerPtr, taskPtr)):
    runManualFit(trainer, task, data, ctx, trainerPtr, taskPtr)
    finishTraining()
  else:
    raise newException(TrainerError,
      "manual optimization requires task.trainingStep(batch, batchIdx, ctx)")

# ---- typed TrainState fit ----------------------------------------------------

proc logStepResult[S](ctx: var TrainContext; result: StepResult[S]) =
  ctx.log("loss", result.loss, onStep = true, onEpoch = true, progBar = true)
  for metric in result.metrics:
    if metric.name != "loss":
      ctx.log(metric.name, metric.value,
        onStep = true, onEpoch = true, progBar = metric.progBar)

proc fit*[M, B](trainer: var Trainer; state: var TrainState[M];
    data: DataSplits[B]; loss: LossFn[M, B]) =
  ## Fits a `TrainState` from a typed loss function.
  ##
  ## This is the progressive-disclosure default: users provide
  ## `loss(model, batch, ctx)` and the Trainer owns gradient computation,
  ## optimizer state, step counting, callbacks, and compiled-step caching.
  let runtime = initRuntime(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  var ctx = initTrainContext(tmFit)
  var step = compileTrainStep(loss, state, runtime,
    donate = if trainer.donateParams: paramsOf(state.model) else: @[])

  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    let iter = data.train.source()
    var batchIdx = 0
    while true:
      let batch = iter()
      if finished(iter): break
      if trainer.maxSteps.isSome and ctx.globalStep >= trainer.maxSteps.get():
        return
      let result = step(state, batch)
      state = result.state
      ctx.logStepResult(result)
      ctx.globalStep += 1
      batchIdx += 1
    ctx.reduceEpochMetrics()
    if ctx.shouldStop:
      break

proc fit*[M, B](trainer: var Trainer; state: var TrainState[M];
    data: DataSplits[B]; trainStep: TrainStepFn[M, B]) =
  ## Fits a `TrainState` from a user-owned custom step.
  let runtime = initRuntime(trainer.accelerator, trainer.devices,
    trainer.precision, trainer.strategy)
  var ctx = initTrainContext(tmFit)
  for epoch in 0 ..< trainer.maxEpochs:
    ctx.epoch = epoch
    let iter = data.train.source()
    while true:
      let batch = iter()
      if finished(iter): break
      if trainer.maxSteps.isSome and ctx.globalStep >= trainer.maxSteps.get():
        return
      let callCtx = initCallCtx(runtime, epoch, ctx.globalStep, tmFit,
        state.key)
      let result = trainStep(state, batch, callCtx)
      state = result.state
      ctx.logStepResult(result)
      ctx.globalStep += 1
    ctx.reduceEpochMetrics()
    if ctx.shouldStop:
      break

# ---- validate ----------------------------------------------------------------

proc runValidation*[T, D](trainer: var Trainer; task: var T; data: D;
    ctx: var TrainContext) =
  ## Runs one validation pass. Called internally by `fit` or standalone via
  ## `validate`.
  let prevMode = ctx.mode
  ctx.mode = tmValidate

  let trainerPtr = cast[pointer](nil)
  let taskPtr = cast[pointer](addr(task))

  fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
      onValidationStart, ctx)

  let hasVal = when compiles(data.val.isSome): data.val.isSome else: false
  if not hasVal:
    ctx.mode = prevMode
    return

  let valIter = data.val.get().source()

  var batchIdx = 0
  while true:
    let batch = valIter()
    if finished(valIter): break

    fireTaskHook(task, onValidationBatchStart, batch, batchIdx, ctx)
    for cb in trainer.callbacks:
      if cb.onValidationBatchStart.isSome:
        cb.onValidationBatchStart.get()(trainerPtr, taskPtr, batch, batchIdx,
            ctx)

    # validationStep (optional)
    when compiles(task.validationStep(batch, batchIdx, ctx)):
      let valLoss = task.validationStep(batch, batchIdx, ctx)
      discard valLoss

    fireTaskHook(task, onValidationBatchEnd, batch, batchIdx, ctx)
    for cb in trainer.callbacks:
      if cb.onValidationBatchEnd.isSome:
        cb.onValidationBatchEnd.get()(trainerPtr, taskPtr, batch, batchIdx, ctx)

    batchIdx += 1

  fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
      onValidationEnd, ctx)

  ctx.mode = prevMode

# ---- validate / test / predict public API -----------------------------------

proc validate*[T, D](trainer: var Trainer; task: var T;
    data: D): seq[MetricEntry] =
  ## Runs a standalone validation pass. Returns epoch-level metrics.
  var ctx = initTrainContext(tmValidate)
  runValidation(trainer, task, data, ctx)
  ctx.reduceEpochMetrics()
  ctx.epochMetrics

proc test*[T, D](trainer: var Trainer; task: var T;
    data: D): seq[MetricEntry] =
  ## Runs a standalone test pass. Iterates `data.test` and calls
  ## `testStep` if defined.
  var ctx = initTrainContext(tmTest)

  let trainerPtr = cast[pointer](nil)
  let taskPtr = cast[pointer](addr(task))

  fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
      onTestStart, ctx)

  let hasTest = when compiles(data.test.isSome): data.test.isSome else: false
  if hasTest:
    let testIter = data.test.get().source()
    var batchIdx = 0
    while true:
      let batch = testIter()
      if finished(testIter): break

      fireTaskHook(task, onTestBatchStart, batch, batchIdx, ctx)
      for cb in trainer.callbacks:
        if cb.onTestBatchStart.isSome:
          cb.onTestBatchStart.get()(trainerPtr, taskPtr, batch, batchIdx, ctx)

      when compiles(task.testStep(batch, batchIdx, ctx)):
        let testLoss = task.testStep(batch, batchIdx, ctx)
        discard testLoss

      fireTaskHook(task, onTestBatchEnd, batch, batchIdx, ctx)
      for cb in trainer.callbacks:
        if cb.onTestBatchEnd.isSome:
          cb.onTestBatchEnd.get()(trainerPtr, taskPtr, batch, batchIdx, ctx)

      batchIdx += 1

  fireAllHooks(task, trainer.callbacks, trainerPtr, taskPtr,
      onTestEnd, ctx)

  ctx.reduceEpochMetrics()
  ctx.epochMetrics

proc predict*[T, D](trainer: var Trainer; task: var T;
    data: D): seq[Tensor] =
  ## Runs a prediction pass. Iterates `data.predict` and calls
  ## `predictStep` if defined. Returns collected predictions.
  var ctx = initTrainContext(tmPredict)

  let hasPredict = when compiles(data.predict.isSome):
      data.predict.isSome else: false
  if hasPredict:
    let predictIter = data.predict.get().source()
    var batchIdx = 0
    while true:
      let batch = predictIter()
      if finished(predictIter): break

      when compiles(task.predictStep(batch, batchIdx, ctx)):
        let pred = task.predictStep(batch, batchIdx, ctx)
        result.add pred
      batchIdx += 1
