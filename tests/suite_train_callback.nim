## Phase 8 — Callback hooks: type, init, fireCallback, fireCallbacks.

block init_callback:
  let cb = initCallback("test")
  doAssert cb.name == "test"
  doAssert cb.onFitStart.isNone
  doAssert cb.onFitEnd.isNone
  doAssert cb.onTrainEpochEnd.isNone

block make_callback_template:
  let cb = makeCallback("my_cb"):
    c.onFitStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx)
  doAssert cb.name == "my_cb"
  doAssert cb.onFitStart.isSome
  doAssert cb.onFitEnd.isNone

block fire_callback_calls_hook:
  var called = false
  let cb = makeCallback("test"):
    c.onFitStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      called = true)
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onFitStart, ctx)
  doAssert called

block fire_callback_unset_noop:
  let cb = initCallback("empty")
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onFitStart, ctx)  # must not crash

block fire_callbacks_multiple:
  var count = 0
  let cb1 = makeCallback("A"):
    c.onTrainEpochStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      count += 1)
  let cb2 = makeCallback("B"):
    c.onTrainEpochStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      count += 10)
  var ctx = initTrainContext(tmFit)
  let cbs = @[cb1, cb2]
  cbs.fireCallbacks(onTrainEpochStart, ctx)
  doAssert count == 11

block fire_callbacks_mixed_set:
  var count = 0
  let cb1 = makeCallback("A"):
    c.onTrainEpochStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      count += 1)
  let cb2 = initCallback("B")  # no hooks set
  var ctx = initTrainContext(tmFit)
  let cbs = @[cb1, cb2]
  cbs.fireCallbacks(onTrainEpochStart, ctx)
  doAssert count == 1  # only cb1 should fire

block callback_multiple_hooks:
  var started = false
  var ended = false
  let cb = makeCallback("lifecycle"):
    c.onTrainStart = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      started = true)
    c.onTrainEnd = some(proc(ctx: var TrainContext) {.closure.} =
      discard ctx
      ended = true)
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onTrainStart, ctx)
  doAssert started
  doAssert not ended
  cb.fireCallback(onTrainEnd, ctx)
  doAssert ended
