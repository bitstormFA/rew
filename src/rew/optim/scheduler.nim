## Learning rate schedulers — pure math, no tensor ops.
##
## Schedulers are plain value types. Their `step` procs return a new
## float32 learning rate; the caller wraps the result in a tensor on
## the appropriate device. This keeps schedulers independent of the
## trace/eager dispatch system.

import std/[math, options]

type
  SchedulerType* = enum
    stStepLR
    stCosineAnnealing
    stReduceOnPlateau
    stExponentialLR
    stLinearWarmup
    stCosineWarmRestarts
    stPolynomialLR
    stLambdaLR
    stConstantLR

  SchedInterval* = enum
    siEpoch
    siStep

  StepLR* = object
    ## Decays the LR by `gamma` every `stepSize` epochs.
    stepSize*: int
    gamma*: float32

  CosineAnnealingLR* = object
    ## Anneals the LR following a cosine curve.
    ## `lr(epoch) = etaMin + 0.5 * (baseLr - etaMin) * (1 + cos(pi * epoch / tMax))`
    tMax*: int
    etaMin*: float32

  ReduceOnPlateau* = object
    ## Reduces the LR when a monitored metric stops improving.
    factor*: float32
    patience*: int
    minLr*: float32

  ExponentialLR* = object
    ## Multiplies the LR by `gamma` every epoch.
    ## `lr(epoch) = baseLr * gamma^epoch`
    gamma*: float32

  LinearWarmupLR* = object
    ## Linearly increases LR from 0 to `targetLr` over `warmupSteps`.
    warmupSteps*: int
    targetLr*: float32

  CosineAnnealingWarmRestartsLR* = object
    ## Cosine annealing with warm restarts (Loshchilov & Hutter, 2016).
    ## `t0` is the initial period; each subsequent period is multiplied
    ## by `tMult`.
    t0*: int
    tMult*: int
    etaMin*: float32

  PolynomialLR* = object
    ## Polynomial decay: `lr = (baseLr - endLr) * (1 - step/maxSteps)^power + endLr`.
    maxSteps*: int
    power*: float32
    endLr*: float32

  LambdaLR* = object
    ## Placeholder: the actual lambda proc is stored by the caller.
    ## The `step` proc is not used through `SchedulerKind`; callers
    ## invoke `step(s: LambdaLR, baseLr, step: int, fn: proc(int): float32)`
    ## directly.

  ConstantLR* = object
    ## Keeps the LR constant. Useful for warmup chains or no-decay phases.

  SchedulerKind* = object
    ## Discriminated union of scheduler configs.
    case kind*: SchedulerType
    of stStepLR: stepLr*: StepLR
    of stCosineAnnealing: cosine*: CosineAnnealingLR
    of stReduceOnPlateau: reduce*: ReduceOnPlateau
    of stExponentialLR: exponential*: ExponentialLR
    of stLinearWarmup: warmup*: LinearWarmupLR
    of stCosineWarmRestarts: cosineWarm*: CosineAnnealingWarmRestartsLR
    of stPolynomialLR: polynomial*: PolynomialLR
    of stLambdaLR: lambdaLr*: LambdaLR
    of stConstantLR: constantLr*: ConstantLR

  SchedulerConfig* = object
    ## A scheduler with its scheduling interval and optional metric monitor.
    scheduler*: SchedulerKind
    interval*: SchedInterval
    monitor*: Option[string]
    frequency*: int

  PlateauState* = object
    ## Runtime state for ReduceOnPlateau: tracks the best metric value and
    ## the number of epochs without improvement.
    best*: float32
    counter*: int
    bestLr*: float32

# ---- StepLR ----------------------------------------------------------------

func initStepLR*(stepSize: int; gamma: float32 = 0.1'f32): StepLR =
  ## Creates a StepLR scheduler that decays the LR by `gamma` every
  ## `stepSize` epochs.
  if stepSize <= 0:
    raise newException(ValueError, "initStepLR: stepSize must be positive")
  StepLR(stepSize: stepSize, gamma: gamma)

func step*(s: StepLR; currentLr: float32; epoch: int): float32 =
  ## Returns the new LR after applying the step decay.
  ## Decay is applied when `(epoch + 1) mod stepSize == 0`.
  if epoch > 0 and (epoch + 1) mod s.stepSize == 0:
    currentLr * s.gamma
  else:
    currentLr

# ---- CosineAnnealingLR -----------------------------------------------------

func initCosineAnnealingLR*(tMax: int; etaMin: float32 = 0.0'f32): CosineAnnealingLR =
  ## Creates a CosineAnnealingLR scheduler. `tMax` is the period (in epochs)
  ## of one full cosine cycle.
  if tMax <= 0:
    raise newException(ValueError,
      "initCosineAnnealingLR: tMax must be positive")
  CosineAnnealingLR(tMax: tMax, etaMin: etaMin)

func step*(s: CosineAnnealingLR; baseLr: float32; epoch: int): float32 =
  ## Returns the LR at `epoch` following the cosine schedule.
  ## `baseLr` is the initial (maximum) learning rate.
  let t = float64(epoch mod s.tMax) / float64(s.tMax)
  let cosVal = cos(PI * t)
  s.etaMin + 0.5'f32 * (baseLr - s.etaMin) * float32(1'f64 + cosVal)

# ---- ReduceOnPlateau -------------------------------------------------------

func initReduceOnPlateau*(factor: float32 = 0.1'f32; patience: int = 10;
    minLr: float32 = 0.0'f32): ReduceOnPlateau =
  ## Creates a ReduceOnPlateau scheduler. When the monitored metric stops
  ## improving for `patience` epochs, the LR is multiplied by `factor`.
  if factor <= 0'f32 or factor >= 1'f32:
    raise newException(ValueError,
      "initReduceOnPlateau: factor must be in (0, 1)")
  if patience <= 0:
    raise newException(ValueError,
      "initReduceOnPlateau: patience must be positive")
  ReduceOnPlateau(factor: factor, patience: patience, minLr: minLr)

func initPlateauState*(initialLr: float32; modeIsMin: bool = true): PlateauState =
  ## Initializes the runtime state for ReduceOnPlateau.
  ## `modeIsMin = true` means lower metric values are better.
  PlateauState(
    best: if modeIsMin: Inf else: -Inf,
    counter: 0,
    bestLr: initialLr,
  )

func step*(s: ReduceOnPlateau; currentLr: float32; metric: float32;
    state: var PlateauState; modeIsMin: bool = true): float32 =
  ## Returns the (possibly reduced) LR based on whether `metric` improved.
  ## `modeIsMin = true` means lower metric values are better.
  ## The caller must persist `state` across calls.
  let improved = if modeIsMin: metric < state.best
                 else: metric > state.best
  if improved:
    state.best = metric
    state.counter = 0
    state.bestLr = currentLr
    currentLr
  else:
    state.counter += 1
    if state.counter >= s.patience:
      let newLr = max(currentLr * s.factor, s.minLr)
      state.counter = 0
      state.bestLr = newLr
      newLr
    else:
      currentLr

# ---- ExponentialLR ---------------------------------------------------------

func initExponentialLR*(gamma: float32 = 0.99'f32): ExponentialLR =
  if gamma <= 0'f32 or gamma > 1'f32:
    raise newException(ValueError,
      "initExponentialLR: gamma must be in (0, 1]")
  ExponentialLR(gamma: gamma)

func step*(s: ExponentialLR; baseLr: float32; epoch: int): float32 =
  ## Returns LR = baseLr * gamma^epoch.
  baseLr * pow(s.gamma.float64, float64(epoch)).float32

# ---- LinearWarmupLR ---------------------------------------------------------

func initLinearWarmupLR*(warmupSteps: int; targetLr: float32 = 1e-3'f32):
    LinearWarmupLR =
  if warmupSteps <= 0:
    raise newException(ValueError,
      "initLinearWarmupLR: warmupSteps must be positive")
  LinearWarmupLR(warmupSteps: warmupSteps, targetLr: targetLr)

func step*(s: LinearWarmupLR; step: int): float32 =
  ## Returns LR linearly ramped from 0 to targetLr over warmupSteps.
  if step >= s.warmupSteps:
    s.targetLr
  else:
    s.targetLr * float32(step + 1) / float32(s.warmupSteps)

# ---- CosineAnnealingWarmRestartsLR ------------------------------------------

func initCosineAnnealingWarmRestartsLR*(t0: int;
    tMult: int = 1; etaMin: float32 = 0.0'f32): CosineAnnealingWarmRestartsLR =
  if t0 <= 0:
    raise newException(ValueError,
      "initCosineAnnealingWarmRestartsLR: t0 must be positive")
  if tMult < 1:
    raise newException(ValueError,
      "initCosineAnnealingWarmRestartsLR: tMult must be >= 1")
  CosineAnnealingWarmRestartsLR(t0: t0, tMult: tMult, etaMin: etaMin)

func step*(s: CosineAnnealingWarmRestartsLR; baseLr: float32; epoch: int):
    float32 =
  ## Returns the LR at `epoch` following the cosine-with-warm-restarts schedule.
  var tCur = 0
  var period = s.t0
  var nextRestart = s.t0
  while nextRestart <= epoch:
    tCur = nextRestart
    period = period * s.tMult
    nextRestart = tCur + period
    if s.tMult == 1:
      break
  let tSinceRestart = epoch - tCur
  let cosVal = cos(PI * float64(tSinceRestart) / float64(period))
  s.etaMin + 0.5'f32 * (baseLr - s.etaMin) * float32(1'f64 + cosVal)

# ---- PolynomialLR -----------------------------------------------------------

func initPolynomialLR*(maxSteps: int; power: float32 = 1.0'f32;
    endLr: float32 = 0.0'f32): PolynomialLR =
  if maxSteps <= 0:
    raise newException(ValueError,
      "initPolynomialLR: maxSteps must be positive")
  PolynomialLR(maxSteps: maxSteps, power: power, endLr: endLr)

func step*(s: PolynomialLR; baseLr: float32; step: int): float32 =
  ## Returns polynomial-decay LR.
  if step >= s.maxSteps:
    s.endLr
  else:
    let factor = 1'f32 - float32(step) / float32(s.maxSteps)
    let decayed = float32(pow(factor.float64, s.power.float64))
    s.endLr + (baseLr - s.endLr) * decayed

# ---- LambdaLR ---------------------------------------------------------------

func initLambdaLR*(): LambdaLR =
  LambdaLR()

proc step*(s: LambdaLR; baseLr: float32; step: int;
    lrLambda: proc(step: int): float32): float32 =
  ## Returns `baseLr * lrLambda(step)`.
  baseLr * lrLambda(step)

# ---- ConstantLR -------------------------------------------------------------

func initConstantLR*(): ConstantLR =
  ConstantLR()

func step*(s: ConstantLR; currentLr: float32): float32 =
  ## Returns the current LR unchanged.
  currentLr

# ---- SchedulerKind helpers -------------------------------------------------

func step*(sk: SchedulerKind; currentLr: float32; epoch: int;
    baseLr: float32 = 0'f32): float32 =
  ## Dispatches to the correct `step` based on the scheduler kind.
  case sk.kind
  of stStepLR: step(sk.stepLr, currentLr, epoch)
  of stCosineAnnealing: step(sk.cosine, baseLr, epoch)
  of stReduceOnPlateau: currentLr  # needs metric; caller must handle separately
  of stExponentialLR: step(sk.exponential, baseLr, epoch)
  of stLinearWarmup: step(sk.warmup, epoch)
  of stCosineWarmRestarts: step(sk.cosineWarm, baseLr, epoch)
  of stPolynomialLR: step(sk.polynomial, baseLr, epoch)
  of stLambdaLR: currentLr           # needs lambda; caller must handle
  of stConstantLR: step(sk.constantLr, currentLr)
