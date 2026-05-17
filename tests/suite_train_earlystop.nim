## Phase 8 — EarlyStopping: triggers ctx.shouldStop.

block earlystop_init:
  let es = initEarlyStopping(monitor = "val/loss", patience = 5,
      mode = cmMin)
  doAssert es.monitor == "val/loss"
  doAssert es.patience == 5
  doAssert es.mode == cmMin

block earlystop_to_callback:
  let es = initEarlyStopping(monitor = "val/loss", patience = 3)
  let cb = es.toCallback()
  doAssert cb.name == "EarlyStopping"
  doAssert cb.onTrainEpochEnd.isSome()

block earlystop_no_stop_when_improving:
  let es = initEarlyStopping(monitor = "val/loss", patience = 3, mode = cmMin)
  let cb = es.toCallback()
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} = discard
  var ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "val/loss", value: 0.5'f32)
  cb.onTrainEpochEnd.get()(ctx, writer)
  doAssert not ctx.shouldStop

block earlystop_stops_when_patience_exhausted:
  let es = initEarlyStopping(monitor = "val/loss", patience = 2,
      minDelta = 0.0'f32, mode = cmMin)
  let cb = es.toCallback()
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} = discard
  # Epoch 0: loss=0.5 (baseline)
  var ctx0 = initTrainContext(tmFit)
  ctx0.epochMetrics.add MetricEntry(name: "val/loss", value: 0.5'f32)
  cb.onTrainEpochEnd.get()(ctx0, writer)
  doAssert not ctx0.shouldStop
  # Epoch 1: loss=0.6 (worse)
  var ctx1 = initTrainContext(tmFit)
  ctx1.epochMetrics.add MetricEntry(name: "val/loss", value: 0.6'f32)
  cb.onTrainEpochEnd.get()(ctx1, writer)
  doAssert not ctx1.shouldStop
  # Epoch 2: loss=0.55 (worse, patience=2 exhausted)
  var ctx2 = initTrainContext(tmFit)
  ctx2.epochMetrics.add MetricEntry(name: "val/loss", value: 0.55'f32)
  cb.onTrainEpochEnd.get()(ctx2, writer)
  doAssert ctx2.shouldStop

block earlystop_resets_on_improvement:
  let es = initEarlyStopping(monitor = "val/loss", patience = 3, mode = cmMin)
  let cb = es.toCallback()
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} = discard
  # E0: 0.5 (baseline)
  var ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "val/loss", value: 0.5'f32)
  cb.onTrainEpochEnd.get()(ctx, writer)
  # E1: 0.6 (worse)
  ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "val/loss", value: 0.6'f32)
  cb.onTrainEpochEnd.get()(ctx, writer)
  # E2: 0.4 (improved! resets counter)
  ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "val/loss", value: 0.4'f32)
  cb.onTrainEpochEnd.get()(ctx, writer)
  doAssert not ctx.shouldStop

block earlystop_max_mode:
  let es = initEarlyStopping(monitor = "val/acc", patience = 2, mode = cmMax)
  let cb = es.toCallback()
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} = discard
  # baseline: 0.8
  var ctx0 = initTrainContext(tmFit)
  ctx0.epochMetrics.add MetricEntry(name: "val/acc", value: 0.8'f32)
  cb.onTrainEpochEnd.get()(ctx0, writer)
  # worse: 0.75
  var ctx1 = initTrainContext(tmFit)
  ctx1.epochMetrics.add MetricEntry(name: "val/acc", value: 0.75'f32)
  cb.onTrainEpochEnd.get()(ctx1, writer)
  # worse: 0.7 (patience=2 exhausted)
  var ctx2 = initTrainContext(tmFit)
  ctx2.epochMetrics.add MetricEntry(name: "val/acc", value: 0.7'f32)
  cb.onTrainEpochEnd.get()(ctx2, writer)
  doAssert ctx2.shouldStop

block earlystop_no_metric_found:
  let es = initEarlyStopping(monitor = "nonexistent", patience = 1)
  let cb = es.toCallback()
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} = discard
  var ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "other", value: 0.5'f32)
  cb.onTrainEpochEnd.get()(ctx, writer)
  doAssert not ctx.shouldStop  # metric not found, no action
