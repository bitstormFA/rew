## Callback — context-centric hook slots for typed training loops.
##
## Callbacks observe `TrainContext` and scalar step results. They do not own
## model state and do not depend on a concrete batch type, so they work with
## arbitrary `DataSplits[B]`.

import std/options
import ../tensor
import ./context

type
  CheckpointWriter* = proc(path: string; ctx: TrainContext) {.closure.}

  Callback* = object
    name*: string
    onFitStart*: Option[proc(ctx: var TrainContext) {.closure.}]
    onFitEnd*: Option[proc(ctx: var TrainContext) {.closure.}]
    onTrainStart*: Option[proc(ctx: var TrainContext) {.closure.}]
    onTrainEnd*: Option[proc(ctx: var TrainContext) {.closure.}]
    onTrainEpochStart*: Option[proc(ctx: var TrainContext) {.closure.}]
    onTrainEpochEnd*: Option[proc(ctx: var TrainContext;
      saveCheckpoint: CheckpointWriter) {.closure.}]
    onTrainBatchStart*: Option[proc(batchIdx: int; ctx: var TrainContext)
      {.closure.}]
    onTrainBatchEnd*: Option[proc(batchIdx: int; loss: Tensor;
      ctx: var TrainContext) {.closure.}]
    onValidationStart*: Option[proc(ctx: var TrainContext) {.closure.}]
    onValidationEnd*: Option[proc(ctx: var TrainContext) {.closure.}]

func initCallback*(name: string): Callback =
  ## Creates a Callback with no hooks set. Use field assignment or
  ## `makeCallback` to add hook procs.
  Callback(name: name)

template makeCallback*(name: string; body: untyped): Callback =
  ## Creates a Callback with hooks set inside `body`.
  ##
  ## `c` is a `var Callback` visible inside `body`.
  block:
    var c {.inject.} = initCallback(name)
    body
    c

template fireCallback*(cb: Callback; hook: untyped; args: varargs[untyped]) =
  ## Calls `cb.hook(args)` if the hook slot is set, otherwise no-op.
  if cb.hook.isSome:
    cb.hook.get()(args)

template fireCallbacks*(cbs: openArray[Callback]; hook: untyped;
    args: varargs[untyped]) =
  ## Calls `hook` on every callback in `cbs` where the slot is set.
  for cb in cbs:
    if cb.hook.isSome:
      cb.hook.get()(args)
