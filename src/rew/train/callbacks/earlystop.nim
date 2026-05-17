## EarlyStopping — stops training when a monitored metric stops improving.

import std/options
import ../callback
import ../context

type
  CheckpointMode* = enum
    cmMin  ## Lower metric values are better (e.g., loss)
    cmMax  ## Higher metric values are better (e.g., accuracy)

  EarlyStopping* = object
    monitor*: string
    patience*: int
    minDelta*: float32
    mode*: CheckpointMode

  EarlyStoppingState = ref object
    bestScore*: float32
    waitCount*: int

func initEarlyStopping*(monitor: string = "val/loss"; patience: int = 3;
    minDelta: float32 = 0.0'f32; mode: CheckpointMode = cmMin): EarlyStopping =
  ## Creates an EarlyStopping callback. Training stops when `monitor` fails
  ## to improve for `patience` epochs (by at least `minDelta`).
  if patience <= 0:
    raise newException(ValueError,
      "initEarlyStopping: patience must be positive")
  EarlyStopping(monitor: monitor, patience: patience, minDelta: minDelta, mode: mode)

func toCallback*(es: EarlyStopping): Callback =
  ## Converts the EarlyStopping config into a `Callback`.
  let state = EarlyStoppingState(
    bestScore: case es.mode
      of cmMin: float32.high
      of cmMax: float32.low,
    waitCount: 0,
  )
  result = initCallback("EarlyStopping")
  result.onTrainEpochEnd = some(proc(ctx: var TrainContext;
      saveCheckpoint: CheckpointWriter) {.closure.} =
    var currentScore: float32
    var found = false
    for m in ctx.epochMetrics:
      if m.name == es.monitor:
        currentScore = m.value
        found = true
        break
    if not found:
      return
    let improved = case es.mode
      of cmMin: currentScore < state.bestScore - es.minDelta
      of cmMax: currentScore > state.bestScore + es.minDelta
    if improved:
      state.bestScore = currentScore
      state.waitCount = 0
    else:
      state.waitCount += 1
      if state.waitCount >= es.patience:
        ctx.shouldStop = true
  )
