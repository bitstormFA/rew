## Adamax optimizer — Adam variant with infinity-norm.
##
## Replaces the L2 norm in Adam with the infinity norm (max of weighted
## past gradients). Sometimes more stable for embeddings.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  AdamaxState* = object
    m*: seq[Tensor]   ## First moment
    u*: seq[Tensor]   ## Infinity norm (exponentially weighted)
    t*: int

  Adamax* = object
    lr*: Tensor
    beta1*: float32
    beta2*: float32
    eps*: float32

proc initAdamax*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32): Adamax =
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initAdamax: lr must be a 0-d scalar tensor")
  Adamax(lr: lr, beta1: beta1, beta2: beta2, eps: eps)

proc initAdamaxState*[P](params: P): AdamaxState =
  let pl = treeFlatten(params)
  var m = newSeq[Tensor](pl.len)
  var u = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    m[i] = constantF32(pl[i].shape, zData)
    u[i] = constantF32(pl[i].shape, zData)
  AdamaxState(m: m, u: u, t: 0)

proc step*[P](opt: Adamax; params: P; grads: P;
    state: AdamaxState): (P, AdamaxState) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len or state.m.len != pl.len:
    raise newException(PytreeError, "Adamax.step: size mismatch")
  let t = state.t + 1
  let b1 = scalarF32(opt.beta1)
  let b2 = scalarF32(opt.beta2)
  let oneMinusB1 = scalarF32(1'f32 - opt.beta1)
  var beta1Power = 1'f32
  for _ in 0 ..< t:
    beta1Power *= opt.beta1
  let corr = scalarF32(1'f32 / (1'f32 - beta1Power))
  var newM = newSeq[Tensor](pl.len)
  var newU = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    # m = beta1 * m + (1 - beta1) * g
    let b1B = broadcastTo(b1, pl[i].shape, @[])
    let oneMinusB1B = broadcastTo(oneMinusB1, pl[i].shape, @[])
    newM[i] = add(mul(b1B, state.m[i]), mul(oneMinusB1B, gl[i]))
    # u = max(beta2 * u, |g|)
    let b2B = broadcastTo(b2, pl[i].shape, @[])
    let weightedOld = mul(b2B, state.u[i])
    newU[i] = maximum(weightedOld, abs(gl[i]))
    # p = p - lr / (1-beta1^t) * m / (u + eps)
    let corrB = broadcastTo(corr, pl[i].shape, @[])
    let lrB = broadcastTo(opt.lr, pl[i].shape, @[])
    let epsB = broadcastTo(scalarF32(opt.eps), pl[i].shape, @[])
    let mHat = mul(corrB, newM[i])
    let denom = add(newU[i], epsB)
    updated[i] = sub(pl[i], mul(lrB, divide(mHat, denom)))
  let newParams = treeUnflatten(params, updated)
  let newState = AdamaxState(m: newM, u: newU, t: t)
  (newParams, newState)
