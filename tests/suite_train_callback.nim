## Phase 8 — Callback hooks: type, init, fireCallback, fireCallbacks.

block init_callback:
  let cb = initCallback("test")
  doAssert cb.name == "test"
  doAssert cb.onFitStart.isNone
  doAssert cb.onFitEnd.isNone
  doAssert cb.onTrainEpochEnd.isNone

block make_callback_template:
  let cb = makeCallback("my_cb"):
    c.onFitStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = discard)
  doAssert cb.name == "my_cb"
  doAssert cb.onFitStart.isSome
  doAssert cb.onFitEnd.isNone

block fire_callback_calls_hook:
  var called = false
  let cb = makeCallback("test"):
    c.onFitStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} =
      called = true)
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onFitStart, nil, nil, ctx)
  doAssert called

block fire_callback_unset_noop:
  let cb = initCallback("empty")
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onFitStart, nil, nil, ctx)  # must not crash

block fire_callbacks_multiple:
  var count = 0
  let cb1 = makeCallback("A"):
    c.onTrainEpochStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = count += 1)
  let cb2 = makeCallback("B"):
    c.onTrainEpochStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = count += 10)
  var ctx = initTrainContext(tmFit)
  let cbs = @[cb1, cb2]
  cbs.fireCallbacks(onTrainEpochStart, nil, nil, ctx)
  doAssert count == 11

block fire_callbacks_mixed_set:
  var count = 0
  let cb1 = makeCallback("A"):
    c.onTrainEpochStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = count += 1)
  let cb2 = initCallback("B")  # no hooks set
  var ctx = initTrainContext(tmFit)
  let cbs = @[cb1, cb2]
  cbs.fireCallbacks(onTrainEpochStart, nil, nil, ctx)
  doAssert count == 1  # only cb1 should fire

block callback_multiple_hooks:
  var started = false
  var ended = false
  let cb = makeCallback("lifecycle"):
    c.onTrainStart = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = started = true)
    c.onTrainEnd = some(proc(trainer, task: pointer;
        ctx: var TrainContext) {.closure.} = ended = true)
  var ctx = initTrainContext(tmFit)
  cb.fireCallback(onTrainStart, nil, nil, ctx)
  doAssert started
  doAssert not ended
  cb.fireCallback(onTrainEnd, nil, nil, ctx)
  doAssert ended
