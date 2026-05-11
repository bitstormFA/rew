## Optimizer integration — typed wrappers so the Trainer can manage
## functional optimizer state generically.
##
## The Trainer owns a `seq[OptimizerState]` and threads it through the
## loop. Users return an `OptimizerConfig` from `configureOptimizers()`;
## the Trainer calls `initOptimizerState` at the start and `step` after
## each backward pass.

import std/options
import ../optim/[sgd, adam, adamw, momentum]
import ../optim/scheduler

type
  OptimizerType* = enum
    otSgd
    otAdam
    otAdamW
    otMomentumSgd

  OptimizerKind* = object
    ## Discriminated union over the concrete optimizer types.
    case kind*: OptimizerType
    of otSgd: sgd*: Sgd
    of otAdam: adam*: Adam
    of otAdamW: adamw*: AdamW
    of otMomentumSgd: momentum*: MomentumSgd

  OptimizerState* = object
    ## Discriminated union over the concrete optimizer state types.
    ## `otSgd` has no state; the other variants carry their respective state.
    case kind*: OptimizerType
    of otAdam, otAdamW:
      adamState*: AdamState
    of otMomentumSgd:
      momentumState*: MomentumState
    of otSgd:
      discard

  OptimizerConfig* = object
    ## Returned by the user's `configureOptimizers` hook.
    optimizer*: OptimizerKind
    scheduler*: Option[SchedulerConfig]
    frequency*: int

func initOptimizerConfig*(optimizer: OptimizerKind;
    scheduler: Option[SchedulerConfig] = none[SchedulerConfig]();
    frequency: int = 1): OptimizerConfig =
  ## Creates an OptimizerConfig. `frequency` controls gradient accumulation
  ## (optimizer steps every `frequency` batches).
  if frequency <= 0:
    raise newException(ValueError,
      "initOptimizerConfig: frequency must be positive")
  OptimizerConfig(optimizer: optimizer, scheduler: scheduler, frequency: frequency)

# ---- state initialization ---------------------------------------------------

proc initOptimizerState*[P](opt: OptimizerKind; params: P): OptimizerState =
  ## Allocates zero-initialized optimizer state matching `params`.
  ## Must be called inside a `withTrace` block (state tensors are created
  ## via `scalarF32`/`constantF32`).
  case opt.kind
  of otSgd:
    OptimizerState(kind: otSgd)
  of otAdam:
    OptimizerState(kind: otAdam, adamState: initAdamState(params))
  of otAdamW:
    OptimizerState(kind: otAdamW, adamState: initAdamState(params))
  of otMomentumSgd:
    OptimizerState(kind: otMomentumSgd, momentumState: initMomentumState(params))

# ---- optimizer step dispatch ------------------------------------------------

proc step*[P](opt: OptimizerKind; params: P; grads: P;
    state: OptimizerState): (P, OptimizerState) =
  ## Performs one optimizer step. Dispatches to the concrete optimizer's
  ## `step` based on `opt.kind`. Returns `(newParams, newState)`.
  ## Must be called inside a `withTrace` block.
  case opt.kind
  of otSgd:
    (opt.sgd.step(params, grads), OptimizerState(kind: otSgd))
  of otAdam:
    let (newParams, newAdamState) = opt.adam.step(params, grads, state.adamState)
    (newParams, OptimizerState(kind: otAdam, adamState: newAdamState))
  of otAdamW:
    let (newParams, newAdamState) = opt.adamw.step(params, grads, state.adamState)
    (newParams, OptimizerState(kind: otAdamW, adamState: newAdamState))
  of otMomentumSgd:
    let (newParams, newMomState) = opt.momentum.step(params, grads, state.momentumState)
    (newParams, OptimizerState(kind: otMomentumSgd, momentumState: newMomState))
