## Lookahead meta-optimizer — wraps another optimizer with a "lookahead"
## update that moves slow weights toward fast weights.
##
## Implements Zhang et al. (2019). At every `k` steps, the slow weights
## are updated toward the fast weights via:
##   slow = slow + alpha * (fast - slow)
## then fast weights are reset to equal slow.
##
## Functional: `step` returns `(newParams, newState)`. No mutation.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/linalg

type
  Lookahead*[O, S] = object
    inner*: O
    alpha*: float32   ## Interpolation factor (default 0.5)
    k*: int           ## Number of inner steps per lookahead update
    stepCount*: int   ## Current step counter

  LookaheadState*[S] = object
    slowWeights*: seq[Tensor]  ## Slow weight copies
    innerState*: S              ## State of the inner optimizer

proc initLookahead*[O, S, P](inner: O; alpha: float32 = 0.5'f32;
    k: int = 5; params: P; makeInnerState: proc(p: P): S):
    Lookahead[O, S] =
  if alpha <= 0'f32 or alpha > 1'f32:
    raise newException(TensorError,
      "initLookahead: alpha must be in (0, 1], got " & $alpha)
  if k <= 0:
    raise newException(TensorError,
      "initLookahead: k must be positive, got " & $k)
  Lookahead[O, S](inner: inner, alpha: alpha, k: k, stepCount: 0)

proc initLookaheadState*[O, S, P](opt: Lookahead[O, S]; params: P;
    makeInnerState: proc(p: P): S): LookaheadState[S] =
  let pl = treeFlatten(params)
  var slow = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    slow[i] = pl[i]
  LookaheadState[S](slowWeights: slow, innerState: makeInnerState(params))

proc step*[P, O, S](opt: var Lookahead[O, S]; params: P; grads: P;
    state: var LookaheadState[S];
    innerStep: proc(opt: var O; params: P; grads: P;
                    state: var S): (P, S)):
    (P, LookaheadState[S]) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "Lookahead.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  # Run inner optimizer step
  var innerOpt = opt.inner
  var (newFast, newInnerState) = innerStep(innerOpt, params, grads,
    state.innerState)
  opt.inner = innerOpt
  opt.stepCount += 1
  var fastL = treeFlatten(newFast)
  # Every k steps: update slow weights, reset fast = slow
  if opt.stepCount mod opt.k == 0:
    let alphaScalar = scalarF32(opt.alpha)
    let oneMinusAlpha = scalarF32(1'f32 - opt.alpha)
    var newSlow = newSeq[Tensor](fastL.len)
    var updatedFast = newSeq[Tensor](fastL.len)
    for i in 0 ..< fastL.len:
      var bdims: seq[int] = @[]
      let aB = broadcastTo(alphaScalar, fastL[i].shape, bdims)
      let oneMinusAB = broadcastTo(oneMinusAlpha, fastL[i].shape, bdims)
      newSlow[i] = add(mul(oneMinusAB, state.slowWeights[i]),
                        mul(aB, fastL[i]))
      updatedFast[i] = newSlow[i]
      # Reset fast to match slow
      fastL[i] = newSlow[i]
    let resultParams = treeUnflatten(params, updatedFast)
    let newState = LookaheadState[S](
      slowWeights: newSlow,
      innerState: newInnerState,
    )
    (resultParams, newState)
  else:
    let resultParams = treeUnflatten(params, fastL)
    let newState = LookaheadState[S](
      slowWeights: state.slowWeights,
      innerState: newInnerState,
    )
    (resultParams, newState)
