## Phase 8 — TrainContext: logging, metric reduction, epoch aggregation.

block init_context_defaults:
  let ctx = initTrainContext(tmFit)
  doAssert ctx.epoch == 0
  doAssert ctx.globalStep == 0
  doAssert ctx.mode == tmFit
  doAssert ctx.shouldStop == false
  doAssert ctx.metrics.len == 0
  doAssert ctx.epochMetrics.len == 0

block log_float32_metric:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onStep = true, onEpoch = true)
  doAssert ctx.metrics.len == 1
  doAssert ctx.metrics[0].name == "loss"
  doAssert ctx.metrics[0].value == 0.5'f32
  doAssert ctx.metrics[0].onStep
  doAssert ctx.metrics[0].onEpoch

block log_multiple_metrics:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onStep = true)
  ctx.log("acc", 0.8'f32, onStep = true)
  ctx.log("loss", 0.3'f32, onStep = true)
  doAssert ctx.metrics.len == 3

block log_dict:
  var ctx = initTrainContext(tmFit)
  ctx.logDict([("loss", 0.5'f32), ("acc", 0.8'f32)], onStep = true)
  doAssert ctx.metrics.len == 2

block step_metrics_iterator:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onStep = true, onEpoch = true)
  ctx.log("acc", 0.8'f32, onStep = true, onEpoch = false)
  var count = 0
  for name, val in ctx.stepMetrics():
    count += 1
    doAssert name in ["loss", "acc"]
  doAssert count == 2

block clear_step_metrics:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onStep = true)
  ctx.clearStepMetrics()
  doAssert ctx.metrics.len == 0

block reduce_epoch_mean:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onStep = true, onEpoch = true, reduceFn = rdMean)
  ctx.log("loss", 0.3'f32, onStep = true, onEpoch = true, reduceFn = rdMean)
  ctx.reduceEpochMetrics()
  doAssert ctx.epochMetrics.len == 1
  doAssert ctx.epochMetrics[0].name == "loss"
  doAssert abs(ctx.epochMetrics[0].value - 0.4'f32) < 1e-6'f32

block reduce_epoch_sum:
  var ctx = initTrainContext(tmFit)
  ctx.log("total", 1.0'f32, onStep = true, onEpoch = true, reduceFn = rdSum)
  ctx.log("total", 2.0'f32, onStep = true, onEpoch = true, reduceFn = rdSum)
  ctx.log("total", 3.0'f32, onStep = true, onEpoch = true, reduceFn = rdSum)
  ctx.reduceEpochMetrics()
  doAssert abs(ctx.epochMetrics[0].value - 6.0'f32) < 1e-6'f32

block reduce_epoch_max:
  var ctx = initTrainContext(tmFit)
  ctx.log("max_val", 0.3'f32, onStep = true, onEpoch = true, reduceFn = rdMax)
  ctx.log("max_val", 0.7'f32, onStep = true, onEpoch = true, reduceFn = rdMax)
  ctx.log("max_val", 0.5'f32, onStep = true, onEpoch = true, reduceFn = rdMax)
  ctx.reduceEpochMetrics()
  doAssert abs(ctx.epochMetrics[0].value - 0.7'f32) < 1e-6'f32

block reduce_epoch_min:
  var ctx = initTrainContext(tmFit)
  ctx.log("min_val", 0.3'f32, onStep = true, onEpoch = true, reduceFn = rdMin)
  ctx.log("min_val", 0.7'f32, onStep = true, onEpoch = true, reduceFn = rdMin)
  ctx.log("min_val", 0.5'f32, onStep = true, onEpoch = true, reduceFn = rdMin)
  ctx.reduceEpochMetrics()
  doAssert abs(ctx.epochMetrics[0].value - 0.3'f32) < 1e-6'f32

block reduce_multiple_metrics:
  var ctx = initTrainContext(tmFit)
  ctx.log("loss", 0.5'f32, onEpoch = true, reduceFn = rdMean)
  ctx.log("loss", 0.3'f32, onEpoch = true, reduceFn = rdMean)
  ctx.log("acc", 0.9'f32, onEpoch = true, reduceFn = rdMax)
  ctx.log("acc", 0.7'f32, onEpoch = true, reduceFn = rdMax)
  ctx.reduceEpochMetrics()
  doAssert ctx.epochMetrics.len == 2
  for m in ctx.epochMetrics:
    if m.name == "loss":
      doAssert abs(m.value - 0.4'f32) < 1e-6'f32
    elif m.name == "acc":
      doAssert abs(m.value - 0.9'f32) < 1e-6'f32

block train_mode_values:
  doAssert tmFit != tmValidate
  doAssert tmTest != tmPredict

block reduce_fn_values:
  doAssert rdMean != rdSum
  doAssert rdMax != rdMin

block should_stop_flag:
  var ctx = initTrainContext(tmFit)
  doAssert not ctx.shouldStop
  ctx.shouldStop = true
  doAssert ctx.shouldStop
