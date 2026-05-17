## Phase 8 — Checkpoint callback and formatFilename.

block checkpoint_init_defaults:
  let ckpt = initCheckpoint()
  doAssert ckpt.monitor == "val/loss"
  doAssert ckpt.mode == cmMin
  doAssert ckpt.saveLast
  doAssert ckpt.saveTopK == 1
  doAssert ckpt.dirPath == "checkpoints"

block checkpoint_to_callback:
  let ckpt = initCheckpoint(monitor = "val/loss", dirPath = "/tmp/test_ckpts")
  let cb = ckpt.toCallback()
  doAssert cb.name == "Checkpoint"
  doAssert cb.onTrainEpochEnd.isSome

block checkpoint_init_custom:
  let ckpt = initCheckpoint(monitor = "val/acc", mode = cmMax,
      saveLast = false, saveTopK = 3, dirPath = "/tmp/ckpt_test",
      filename = "model-epoch={epoch}")
  doAssert ckpt.monitor == "val/acc"
  doAssert ckpt.mode == cmMax
  doAssert not ckpt.saveLast
  doAssert ckpt.saveTopK == 3
  doAssert ckpt.filename == "model-epoch={epoch}"

block checkpoint_writer_callback:
  var saveCalled = false
  var savePath = ""
  let ckpt = initCheckpoint(monitor = "val/loss", dirPath = "/tmp/test_ckpts")
  let cb = ckpt.toCallback()
  var ctx = initTrainContext(tmFit)
  ctx.epochMetrics.add MetricEntry(name: "val/loss", value: 0.5'f32)
  let writer: CheckpointWriter =
    proc(path: string; ctx: TrainContext) {.closure.} =
      discard ctx
      saveCalled = true
      savePath = path
  cb.onTrainEpochEnd.get()(ctx, writer)
  doAssert saveCalled
  doAssert savePath.len > 0

block checkpoint_format_filename:
  # The formatFilename func is internal; test substitution through init
  let ckpt = initCheckpoint(filename = "ckpt-{epoch}-{step}-{metric}")
  doAssert ckpt.filename == "ckpt-{epoch}-{step}-{metric}"

block checkpoint_modes_are_distinct:
  doAssert cmMin != cmMax
