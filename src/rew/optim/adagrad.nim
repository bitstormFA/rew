## Adagrad optimizer — adaptive subgradient method.
##
## Accumulates the sum of squared gradients and adapts the learning rate
## per-parameter: `p -= lr * g / sqrt(G + eps)`.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  AdagradState* = object
    gSum*: seq[Tensor]  ## Accumulated squared gradients

  Adagrad* = object
    lr*: Tensor
    eps*: float32
    lrDecay*: float32   ## Learning rate decay factor per step

proc initAdagrad*(lr: Tensor; eps: float32 = 1e-10'f32;
    lrDecay: float32 = 0'f32): Adagrad =
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initAdagrad: lr must be a 0-d scalar tensor")
  Adagrad(lr: lr, eps: eps, lrDecay: lrDecay)

proc initAdagradState*[P](params: P): AdagradState =
  let pl = treeFlatten(params)
  var gSum = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    gSum[i] = constantF32(pl[i].shape, zData)
  AdagradState(gSum: gSum)

proc step*[P](opt: Adagrad; params: P; grads: P;
    state: AdagradState): (P, AdagradState) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len or state.gSum.len != pl.len:
    raise newException(PytreeError, "Adagrad.step: size mismatch")
  let lrDecayFactor = 1'f32 / (1'f32 + opt.lrDecay * float32(pl.len))
  let lrScale = scalarF32(lrDecayFactor)
  var lrB = broadcastTo(mul(opt.lr, lrScale), pl[0].shape, @[])
  var newGSum = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    # G_t = G_{t-1} + g^2
    newGSum[i] = add(state.gSum[i], mul(gl[i], gl[i]))
    # p = p - lr * g / sqrt(G_t + eps)
    let epsScalar = broadcastTo(scalarF32(opt.eps), pl[i].shape, @[])
    let denom = add(sqrt(newGSum[i]), epsScalar)
    if i > 0:
      lrB = broadcastTo(mul(opt.lr, lrScale), pl[i].shape, @[])
    updated[i] = sub(pl[i], mul(lrB, divide(gl[i], denom)))
  let newParams = treeUnflatten(params, updated)
  let newState = AdagradState(gSum: newGSum)
  (newParams, newState)
