## Progress — simple terminal progress display for training.

import std/[options, strutils]
import ../callback
import ../context
import ../../data/sample
import ../../tensor

type
  ProgressBar* = object
    refreshRate*: int

  ProgressState = ref object
    epochStartStep*: int

func initProgressBar*(refreshRate: int = 1): ProgressBar =
  ## Creates a ProgressBar callback that prints a progress line every
  ## `refreshRate` steps.
  if refreshRate <= 0:
    raise newException(ValueError,
      "initProgressBar: refreshRate must be positive")
  ProgressBar(refreshRate: refreshRate)

func formatMetrics(ctx: TrainContext): string =
  var parts: seq[string] = @[]
  for m in ctx.metrics:
    if m.progBar and m.onStep:
      parts.add m.name & "=" & formatFloat(float(m.value), ffDecimal, 4)
  result = parts.join(" ")

func toCallback*(pb: ProgressBar): Callback =
  ## Converts the ProgressBar config into a `Callback`.
  let state = ProgressState(epochStartStep: 0)
  result = initCallback("ProgressBar")
  result.onTrainEpochStart = some(proc(trainer, task: pointer;
      ctx: var TrainContext) {.closure.} =
    state.epochStartStep = ctx.globalStep
  )
  result.onTrainBatchEnd = some(proc(trainer, task: pointer; batch: Batch;
      batchIdx: int; loss: Tensor; ctx: var TrainContext)
      {.closure.} =
    if ctx.globalStep mod pb.refreshRate != 0:
      return
    let stepInEpoch = ctx.globalStep - state.epochStartStep
    let metricsStr = formatMetrics(ctx)
    let line = "Epoch " & $ctx.epoch & " | step " & $stepInEpoch &
      " | global " & $ctx.globalStep & " | " & metricsStr
    try:
      stdout.write("\r" & line & "\r")
      stdout.flushFile()
    except IOError:
      discard
  )
  result.onTrainEpochEnd = some(proc(trainer, task: pointer;
      ctx: var TrainContext) {.closure.} =
    # Emit epoch summary
    var parts: seq[string] = @[]
    for m in ctx.epochMetrics:
      if m.progBar:
        parts.add m.name & "=" & formatFloat(float(m.value), ffDecimal, 4)
    if parts.len > 0:
      try:
        stdout.write("\rEpoch " & $ctx.epoch & " done | " &
          parts.join(" ") & "\n")
        stdout.flushFile()
      except IOError:
        discard
  )
