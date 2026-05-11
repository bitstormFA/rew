## Callback — a closed set of hook proc slots for pluggable training behaviours.
##
## Each hook slot is an `Option[proc(...)]` with `pointer` parameters for the
## trainer and task. The `pointer` approach avoids generic entanglement —
## callbacks receive opaque references and cast internally. This trades
## some type safety for faster compiles and simpler callback storage.
##
## Built-in callbacks (Checkpoint, EarlyStopping, Progress, LogMonitor)
## live in `rew/train/callbacks/` and use `toCallback` to produce a
## `Callback` value.

import std/options
import ../data/sample
import ../tensor
import ./context

type
  Callback* = object
    name*: string
    onFitStart*: Option[proc(trainer, task: pointer; ctx: var TrainContext)
        {.closure.}]
    onFitEnd*: Option[proc(trainer, task: pointer; ctx: var TrainContext)
        {.closure.}]
    onTrainStart*: Option[proc(trainer, task: pointer; ctx: var TrainContext)
        {.closure.}]
    onTrainEnd*: Option[proc(trainer, task: pointer; ctx: var TrainContext)
        {.closure.}]
    onTrainEpochStart*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onTrainEpochEnd*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onTrainBatchStart*: Option[proc(trainer, task: pointer; batch: Batch;
        batchIdx: int; ctx: var TrainContext) {.closure.}]
    onTrainBatchEnd*: Option[proc(trainer, task: pointer; batch: Batch;
        batchIdx: int; loss: Tensor; ctx: var TrainContext)
        {.closure.}]
    onValidationStart*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onValidationEnd*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onValidationBatchStart*: Option[proc(trainer, task: pointer;
        batch: Batch; batchIdx: int; ctx: var TrainContext)
        {.closure.}]
    onValidationBatchEnd*: Option[proc(trainer, task: pointer;
        batch: Batch; batchIdx: int; ctx: var TrainContext)
        {.closure.}]
    onTestStart*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onTestEnd*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onTestBatchStart*: Option[proc(trainer, task: pointer; batch: Batch;
        batchIdx: int; ctx: var TrainContext) {.closure.}]
    onTestBatchEnd*: Option[proc(trainer, task: pointer; batch: Batch;
        batchIdx: int; ctx: var TrainContext) {.closure.}]
    onBeforeBackward*: Option[proc(trainer, task: pointer; loss: Tensor;
        ctx: var TrainContext) {.closure.}]
    onAfterBackward*: Option[proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.}]
    onBeforeOptimizerStep*: Option[proc(trainer, task: pointer;
        optIdx: int; ctx: var TrainContext) {.closure.}]
    onAfterOptimizerStep*: Option[proc(trainer, task: pointer;
        optIdx: int; ctx: var TrainContext) {.closure.}]
    onSaveCheckpoint*: Option[proc(trainer, task: pointer; path: string;
        ctx: var TrainContext) {.closure.}]
    onLoadCheckpoint*: Option[proc(trainer, task: pointer; path: string;
        ctx: var TrainContext) {.closure.}]

func initCallback*(name: string): Callback =
  ## Creates a Callback with no hooks set. Use field assignment or
  ## `makeCallback` to add hook procs.
  Callback(name: name)

template makeCallback*(name: string; body: untyped): Callback =
  ## Creates a Callback with hooks set inside `body`.
  ##
  ## `c` is a `var Callback` visible inside `body`.
  ##
  ## Usage:
  ##   let cb = makeCallback("my_cb"):
  ##     c.onTrainEpochEnd = some(proc(trainer, task: pointer,
  ##         ctx: var TrainContext) =
  ##       echo "epoch ", ctx.epoch, " done")
  block:
    var c {.inject.} = initCallback(name)
    body
    c

# ---- Callback invocation helpers --------------------------------------------

template fireCallback*(cb: Callback; hook: untyped; args: varargs[untyped]) =
  ## Calls `cb.hook(args)` if the hook slot is set, otherwise no-op.
  ##
  ## Usage:
  ##   cb.fireCallback(onFitStart, trainerPtr, taskPtr, ctx)
  if cb.hook.isSome:
    cb.hook.get()(args)

template fireCallbacks*(cbs: openArray[Callback]; hook: untyped;
    args: varargs[untyped]) =
  ## Calls `hook` on every callback in `cbs` where the slot is set.
  for cb in cbs:
    if cb.hook.isSome:
      cb.hook.get()(args)
