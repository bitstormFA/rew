## TrainContext — training state and metric logging.
##
## The `TrainContext` is passed to every user hook. It tracks the current
## epoch, global step, and accumulated metrics. The `log` procs mirror
## Lightning's `self.log` — they store scalar values with reduction hints
## so the Trainer can emit step-level and epoch-level summaries.

import ../tensor
import ../eager

type
  TrainMode* = enum
    tmFit
    tmValidate
    tmTest
    tmPredict

  ReduceFn* = enum
    rdMean
    rdSum
    rdMax
    rdMin

  LogOpts* = object
    onStep*: bool
    onEpoch*: bool
    progBar*: bool
    reduceFn*: ReduceFn

  MetricEntry* = object
    name*: string
    value*: float32
    onStep*: bool
    onEpoch*: bool
    reduceFn*: ReduceFn
    progBar*: bool
    stepValues*: seq[float32]

  TrainContext* = object
    epoch*: int
    globalStep*: int
    mode*: TrainMode
    metrics*: seq[MetricEntry]
    epochMetrics*: seq[MetricEntry]
    shouldStop*: bool

proc initTrainContext*(mode: TrainMode = tmFit): TrainContext =
  ## Creates a fresh TrainContext for the given mode.
  TrainContext(mode: mode)

# ---- log helpers ------------------------------------------------------------

proc toScalarF32(t: Tensor): float32 =
  ## Extracts a float32 scalar from an eager tensor.
  if not t.isEager:
    raise newException(TensorError,
      "cannot extract value from a trace tensor — call `log` with a " &
        "float32 value, or extract after trace execution")
  if t.numElements != 1:
    raise newException(TensorError,
      "log: expected a scalar tensor, got shape " & $t.shape)
  transferToHost(t.device, t.buffer, addr result, sizeof(float32))

proc log*(ctx: var TrainContext; name: string; value: float32;
    onStep: bool = false; onEpoch: bool = true; progBar: bool = false;
    reduceFn: ReduceFn = rdMean) =
  ## Logs a scalar metric value.
  ##
  ## **onStep** — emit this value immediately (per-step logging).
  ## **onEpoch** — accumulate this value for epoch-level reduction.
  ## **progBar** — include this metric in progress bar displays.
  ## **reduceFn** — how to aggregate step values at epoch end
  ##   (`rdMean` = average, `rdSum` = total, `rdMax`/`rdMin` = extremum).
  ctx.metrics.add MetricEntry(
    name: name,
    value: value,
    onStep: onStep,
    onEpoch: onEpoch,
    reduceFn: reduceFn,
    progBar: progBar,
    stepValues: if onEpoch: @[value] else: @[],
  )

proc log*(ctx: var TrainContext; name: string; value: Tensor;
    onStep: bool = false; onEpoch: bool = true; progBar: bool = false;
    reduceFn: ReduceFn = rdMean) =
  ## Logs a scalar tensor as a metric. The tensor is immediately transferred
  ## to host and stored as float32.
  ctx.log(name, toScalarF32(value), onStep, onEpoch, progBar, reduceFn)

proc logDict*(ctx: var TrainContext; metrics: openArray[(string, float32)];
    onStep: bool = false; onEpoch: bool = true; progBar: bool = false;
    reduceFn: ReduceFn = rdMean) =
  ## Logs multiple metrics at once. Convenience for logging several values
  ## from a single step.
  for (name, val) in metrics:
    ctx.log(name, val, onStep, onEpoch, progBar, reduceFn)

# ---- epoch reduction --------------------------------------------------------

proc reduceEpochMetrics*(ctx: var TrainContext) =
  ## Aggregates per-step metric values into epoch-level summaries.
  ## Called by the Trainer at the end of each epoch.
  ##
  ## For each unique metric name tracked with `onEpoch = true`, applies
  ## the associated `reduceFn` across all logged step values and stores
  ## the result in `epochMetrics`.
  var byName: seq[(string, ReduceFn, seq[float32], bool)]
  for m in ctx.metrics:
    if not m.onEpoch:
      continue
    var found = false
    for item in byName.mitems:
      if item[0] == m.name:
        item[2].add m.value
        found = true
        break
    if not found:
      byName.add (m.name, m.reduceFn, @[m.value], m.progBar)
  ctx.epochMetrics.setLen(0)
  for (name, rf, values, progBar) in byName:
    var agg: float32
    case rf
    of rdMean:
      var sum: float32 = 0'f32
      for v in values: sum += v
      agg = sum / float32(values.len)
    of rdSum:
      var sum: float32 = 0'f32
      for v in values: sum += v
      agg = sum
    of rdMax:
      agg = float32.low
      for v in values:
        if v > agg: agg = v
    of rdMin:
      agg = float32.high
      for v in values:
        if v < agg: agg = v
    ctx.epochMetrics.add MetricEntry(
      name: name,
      value: agg,
      onStep: false,
      onEpoch: true,
      reduceFn: rf,
      progBar: progBar,
    )

# ---- per-step extraction ----------------------------------------------------

iterator stepMetrics*(ctx: TrainContext): (string, float32) =
  ## Yields `(name, value)` for metrics logged with `onStep = true` in the
  ## current step. Used by progress bars and step-level loggers.
  for m in ctx.metrics:
    if m.onStep:
      yield (m.name, m.value)

proc clearStepMetrics*(ctx: var TrainContext) =
  ## Clears per-step metrics. Called by the Trainer after emitting step logs.
  ctx.metrics.setLen(0)
