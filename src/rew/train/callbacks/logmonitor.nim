## LogMonitor — prints logged metrics to stdout at configurable intervals.

import std/[options, strutils]
import ../callback
import ../context
import ../../data/sample
import ../../tensor

type
  LogMonitor* = object
    logEvery*: int

func initLogMonitor*(logEvery: int = 50): LogMonitor =
  ## Creates a LogMonitor callback that prints step-level metrics every
  ## `logEvery` steps.
  if logEvery <= 0:
    raise newException(ValueError,
      "initLogMonitor: logEvery must be positive")
  LogMonitor(logEvery: logEvery)

func toCallback*(lm: LogMonitor): Callback =
  ## Converts the LogMonitor config into a `Callback`.
  result = initCallback("LogMonitor")
  result.onTrainBatchEnd = some(proc(trainer, task: pointer; batch: Batch;
      batchIdx: int; loss: Tensor; ctx: var TrainContext)
      {.closure.} =
    if ctx.globalStep mod lm.logEvery != 0:
      return
    var parts: seq[string] = @[]
    parts.add "step=" & $ctx.globalStep
    for m in ctx.metrics:
      if m.onStep:
        parts.add m.name & "=" & formatFloat(float(m.value), ffDecimal, 4)
    try:
      stdout.write(parts.join(" ") & "\n")
      stdout.flushFile()
    except IOError:
      discard
  )
