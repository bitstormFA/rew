## Hook system — canonical signatures, compile-time validation, and
## convenience templates for Task hooks.
##
## Hooks are regular procs defined on the user's Task type. The Trainer
## checks for their existence at compile time with `when compiles` and
## provides clear error messages for missing required hooks.
##
## ## Required hook
##
##   task.configureOptimizers(): OptimizerConfig
##
## ## Manual optimization hook
##
##   task.trainingStep(batch: Batch, batchIdx: int, ctx: var TrainContext): Tensor
##
## ## Automatic optimization hooks
##
##   task.parameters(): P
##   task.setParameters(params: P)
##   task.trainingInputs(batch: Batch, batchIdx: int, ctx: var TrainContext): seq[Tensor]
##   task.trainingLoss(params: P, inputs: openArray[Tensor], batchIdx: int): Tensor
##
## ## Optional step hooks (checked with `when compiles`)
##
##   task.validationStep(batch: Batch, batchIdx: int, ctx: var TrainContext): Tensor
##   task.testStep(batch: Batch, batchIdx: int, ctx: var TrainContext): Tensor
##   task.predictStep(batch: Batch, batchIdx: int, ctx: var TrainContext): Tensor
##
## ## Optional lifecycle hooks
##
##   task.onFitStart(ctx: var TrainContext)
##   task.onFitEnd(ctx: var TrainContext)
##   task.onTrainStart(ctx: var TrainContext)
##   task.onTrainEnd(ctx: var TrainContext)
##   task.onTrainEpochStart(ctx: var TrainContext)
##   task.onTrainEpochEnd(ctx: var TrainContext)
##   task.onTrainBatchStart(batch: Batch, batchIdx: int, ctx: var TrainContext)
##   task.onTrainBatchEnd(batch: Batch, batchIdx: int, ctx: var TrainContext, loss: Tensor)
##   task.onValidationStart(ctx: var TrainContext)
##   task.onValidationEnd(ctx: var TrainContext)
##   task.onValidationBatchStart(batch: Batch, batchIdx: int, ctx: var TrainContext)
##   task.onValidationBatchEnd(batch: Batch, batchIdx: int, ctx: var TrainContext)
##   task.onTestStart(ctx: var TrainContext)
##   task.onTestEnd(ctx: var TrainContext)
##   task.onTestBatchStart(batch: Batch, batchIdx: int, ctx: var TrainContext)
##   task.onTestBatchEnd(batch: Batch, batchIdx: int, ctx: var TrainContext)
##   task.onBeforeBackward(loss: Tensor, ctx: var TrainContext)
##   task.onAfterBackward(grads: auto, ctx: var TrainContext)
##   task.onBeforeOptimizerStep(optIdx: int, ctx: var TrainContext)
##   task.onAfterOptimizerStep(optIdx: int, ctx: var TrainContext)
##   task.onSaveCheckpoint(path: string, ctx: var TrainContext)
##   task.onLoadCheckpoint(path: string, ctx: var TrainContext)
##
## ## Execution order
##
##   onFitStart
##     onTrainStart
##       for each epoch:
##         onTrainEpochStart
##           for each batch:
##             onTrainBatchStart
##             trainingStep           ← manual mode
##             trainingLoss           ← automatic mode forward
##               (Trainer backward + optimizer)
##             onTrainBatchEnd
##         onTrainEpochEnd
##         (validation if scheduled):
##           onValidationStart
##             for each val batch:
##               onValidationBatchStart
##               validationStep       ← optional
##               onValidationBatchEnd
##           onValidationEnd
##     onTrainEnd
##   onFitEnd
##
## Task hooks fire before Callback hooks at the same hook point.

import ../data/sample
import ./context

# ---- Required hook validation -----------------------------------------------

template checkTrainingStep*(T: typed) =
  ## Compile-time check: `T` must define `trainingStep`.
  var t = default(T)
  var ctx = default(TrainContext)
  when not compiles(t.trainingStep(default(Batch), 0, ctx)):
    {.error: "Task type " & $T & " must define:\n  " &
      "proc trainingStep(t: var " & $T &
      ", batch: Batch, batchIdx: int, ctx: var TrainContext): Tensor".}

template checkConfigureOptimizers*(T: typed) =
  ## Compile-time check: `T` must define `configureOptimizers`.
  let t = default(T)
  when not compiles(t.configureOptimizers()):
    {.error: "Task type " & $T & " must define:\n  " &
      "proc configureOptimizers(t: " & $T & "): OptimizerConfig".}

template checkRequiredHooks*(T: typed) =
  ## Compile-time check for the legacy manual Trainer hook set.
  checkTrainingStep(T)
  checkConfigureOptimizers(T)

# ---- Optional hook invocation -----------------------------------------------

template callHookIfDefined*(callExpr: untyped) =
  ## Calls `callExpr` if it compiles (i.e. the hook is defined on the task).
  ## No-op otherwise. Designed for void-returning lifecycle hooks.
  ##
  ## For Tensor-returning optional hooks (validationStep, etc.), use an
  ## explicit `when compiles(task.hook(...))` guard so the return value
  ## can be captured.
  ##
  ## Usage:
  ##   callHookIfDefined(task.onFitStart(ctx))
  ##   callHookIfDefined(task.onTrainEpochEnd(ctx))
  when compiles(callExpr):
    callExpr
