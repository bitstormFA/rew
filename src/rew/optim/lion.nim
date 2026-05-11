## Lion optimizer — EvoLved Sign Momentum.
##
## Chen et al. (2023): "Symbolic Discovery of Optimization Algorithms".
## Uses sign-based updates with decoupled weight decay.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  LionState* = object
    m*: seq[Tensor]  ## Exponential moving average of gradients
    t*: int

  Lion* = object
    lr*: Tensor
    beta1*: float32
    beta2*: float32
    weightDecay*: float32

proc initLion*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.99'f32; weightDecay: float32 = 0'f32): Lion =
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initLion: lr must be a 0-d scalar tensor")
  Lion(lr: lr, beta1: beta1, beta2: beta2, weightDecay: weightDecay)

proc initLionState*[P](params: P): LionState =
  let pl = treeFlatten(params)
  var m = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    m[i] = constantF32(pl[i].shape, zData)
  LionState(m: m, t: 0)

proc step*[P](opt: Lion; params: P; grads: P;
    state: LionState): (P, LionState) =
  ## Performs one Lion update step.
  ##
  ## Algorithm:
  ##   c = beta1 * m + (1 - beta1) * g
  ##   m = beta2 * m + (1 - beta2) * g
  ##   update = sign(c)
  ##   p = p - lr * (update + weightDecay * p)
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len or state.m.len != pl.len:
    raise newException(PytreeError, "Lion.step: size mismatch")
  let b1 = scalarF32(opt.beta1)
  let b2 = scalarF32(opt.beta2)
  let oneMinusB1 = scalarF32(1'f32 - opt.beta1)
  let oneMinusB2 = scalarF32(1'f32 - opt.beta2)
  let wd = scalarF32(opt.weightDecay)
  var newM = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    # c = beta1 * m + (1 - beta1) * g
    let b1B = broadcastTo(b1, pl[i].shape, @[])
    let oneMinusB1B = broadcastTo(oneMinusB1, pl[i].shape, @[])
    let c = add(mul(b1B, state.m[i]), mul(oneMinusB1B, gl[i]))
    # m = beta2 * m + (1 - beta2) * g
    let b2B = broadcastTo(b2, pl[i].shape, @[])
    let oneMinusB2B = broadcastTo(oneMinusB2, pl[i].shape, @[])
    newM[i] = add(mul(b2B, state.m[i]), mul(oneMinusB2B, gl[i]))
    # update = lr * sign(c)
    let lrB = broadcastTo(opt.lr, pl[i].shape, @[])
    let update = mul(lrB, sign(c))
    # Weight decay: p = p - lr * wd * p
    var paramUpdate = update
    if opt.weightDecay > 0'f32:
      let wdB = broadcastTo(wd, pl[i].shape, @[])
      paramUpdate = add(update, mul(mul(lrB, wdB), pl[i]))
    updated[i] = sub(pl[i], paramUpdate)
  let newParams = treeUnflatten(params, updated)
  let newState = LionState(m: newM, t: state.t + 1)
  (newParams, newState)
