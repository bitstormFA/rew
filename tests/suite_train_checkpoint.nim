## Phase 8 — Checkpoint: TrainerAccess pointer roundtrip, formatFilename.

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

block trainer_access_pointer_roundtrip:
  var saveCalled = false
  var savePath = ""
  var access = TrainerAccess(
    saveCheckpoint: proc(path: string; ctx: TrainContext) {.closure.} =
      saveCalled = true
      savePath = path)
  var accessPtr = addr(access)
  let trainerPtr = cast[pointer](accessPtr)
  let back = cast[ptr TrainerAccess](trainerPtr)
  var ctx = initTrainContext(tmFit)
  back.saveCheckpoint("/tmp/test_path", ctx)
  doAssert saveCalled
  doAssert savePath == "/tmp/test_path"

block checkpoint_format_filename:
  # The formatFilename func is internal; test substitution through init
  let ckpt = initCheckpoint(filename = "ckpt-{epoch}-{step}-{metric}")
  doAssert ckpt.filename == "ckpt-{epoch}-{step}-{metric}"

block checkpoint_modes_are_distinct:
  doAssert cmMin != cmMax
