## Adadelta optimizer — adaptive learning rate without a global LR.
##
## Implements Zeiler (2012): accumulates squared gradients and squared
## updates with exponential moving averages. No global learning rate needed.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  AdadeltaState* = object
    gAvg*: seq[Tensor]   ## EMA of squared gradients
    dxAvg*: seq[Tensor]  ## EMA of squared updates

  Adadelta* = object
    rho*: float32        ## Decay rate (default 0.9)
    eps*: float32

proc initAdadelta*(rho: float32 = 0.9'f32;
    eps: float32 = 1e-6'f32): Adadelta =
  Adadelta(rho: rho, eps: eps)

proc initAdadeltaState*[P](params: P): AdadeltaState =
  let pl = treeFlatten(params)
  var gAvg = newSeq[Tensor](pl.len)
  var dxAvg = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    gAvg[i] = constantF32(pl[i].shape, zData)
    dxAvg[i] = constantF32(pl[i].shape, zData)
  AdadeltaState(gAvg: gAvg, dxAvg: dxAvg)

proc step*[P](opt: Adadelta; params: P; grads: P;
    state: AdadeltaState): (P, AdadeltaState) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len or state.gAvg.len != pl.len:
    raise newException(PytreeError, "Adadelta.step: size mismatch")
  let rhoB = scalarF32(opt.rho)
  let oneMinusRhoB = scalarF32(1'f32 - opt.rho)
  let epsB = scalarF32(opt.eps)
  var newGAvg = newSeq[Tensor](pl.len)
  var newDxAvg = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    # gAvg = rho * gAvg + (1 - rho) * g^2
    let rhoBB = broadcastTo(rhoB, pl[i].shape, @[])
    let oneMinusRhoBB = broadcastTo(oneMinusRhoB, pl[i].shape, @[])
    newGAvg[i] = add(
      mul(rhoBB, state.gAvg[i]),
      mul(oneMinusRhoBB, mul(gl[i], gl[i])))
    # delta = -sqrt(dxAvg + eps) / sqrt(gAvg + eps) * g
    let epsBB = broadcastTo(epsB, pl[i].shape, @[])
    let num = sqrt(add(state.dxAvg[i], epsBB))
    let den = sqrt(add(newGAvg[i], epsBB))
    let delta = neg(mul(divide(num, den), gl[i]))
    # dxAvg = rho * dxAvg + (1 - rho) * delta^2
    newDxAvg[i] = add(
      mul(rhoBB, state.dxAvg[i]),
      mul(oneMinusRhoBB, mul(delta, delta)))
    # p = p + delta
    updated[i] = add(pl[i], delta)
  let newParams = treeUnflatten(params, updated)
  let newState = AdadeltaState(gAvg: newGAvg, dxAvg: newDxAvg)
  (newParams, newState)
