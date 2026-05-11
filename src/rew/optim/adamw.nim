## AdamW optimizer — Adam with decoupled weight decay.
##
## Implements Loshchilov & Hutter (2017). Weight decay is applied
## directly to the parameters rather than through the gradient,
## which prevents the regularization from being affected by the
## adaptive learning rate.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg
import ./adam

type
  AdamW* = object
    ## AdamW optimizer: Adam + decoupled weight decay.
    lr*: Tensor         ## 0-d scalar learning rate
    beta1*: float32     ## First moment decay (default 0.9)
    beta2*: float32     ## Second moment decay (default 0.999)
    eps*: float32       ## Numerical stability (default 1e-8)
    weightDecay*: float32  ## Weight decay coefficient (default 0.01)

proc initAdamW*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32;
    weightDecay: float32 = 0.01'f32): AdamW =
  ## Constructs an AdamW optimizer. `lr` must be a 0-d scalar tensor.
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initAdamW: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  AdamW(lr: lr, beta1: beta1, beta2: beta2, eps: eps,
        weightDecay: weightDecay)

proc step*[P](opt: AdamW; params: P; grads: P;
    state: AdamState): (P, AdamState) =
  ## Performs one AdamW update step. Returns `(newParams, newState)`.
  ## Applies weight decay before the Adam update.
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "AdamW.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  if state.m.len != pl.len:
    raise newException(PytreeError,
      "AdamW.step: state size mismatch")
  let t = state.t + 1
  let b1 = scalarF32(opt.beta1)
  let b2 = scalarF32(opt.beta2)
  let oneMinusB1 = scalarF32(1'f32 - opt.beta1)
  let oneMinusB2 = scalarF32(1'f32 - opt.beta2)
  let prevB1Power = if state.t == 0: 1'f32 else: state.beta1Power
  let prevB2Power = if state.t == 0: 1'f32 else: state.beta2Power
  let b1t = prevB1Power * opt.beta1
  let b2t = prevB2Power * opt.beta2
  let corrM = scalarF32(1'f32 / (1'f32 - b1t))
  let corrV = scalarF32(1'f32 / (1'f32 - b2t))
  let epsScalar = scalarF32(opt.eps)
  let wdScalar = scalarF32(opt.weightDecay)
  var newM = newSeq[Tensor](pl.len)
  var newV = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    var bdims: seq[int] = @[]
    let b1B = broadcastTo(b1, pl[i].shape, bdims)
    let b2B = broadcastTo(b2, pl[i].shape, bdims)
    let oneMinusB1B = broadcastTo(oneMinusB1, pl[i].shape, bdims)
    let oneMinusB2B = broadcastTo(oneMinusB2, pl[i].shape, bdims)
    newM[i] = add(mul(b1B, state.m[i]), mul(oneMinusB1B, gl[i]))
    let gSq = mul(gl[i], gl[i])
    newV[i] = add(mul(b2B, state.v[i]), mul(oneMinusB2B, gSq))
    let corrMB = broadcastTo(corrM, pl[i].shape, bdims)
    let corrVB = broadcastTo(corrV, pl[i].shape, bdims)
    let mHat = mul(corrMB, newM[i])
    let vHat = mul(corrVB, newV[i])
    let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
    let epsB = broadcastTo(epsScalar, pl[i].shape, bdims)
    let denom = add(sqrt(vHat), epsB)
    let adamUpdate = divide(mHat, denom)
    # Decoupled weight decay: p = p * (1 - lr * wd) - lr * adam_update
    let wdB = broadcastTo(wdScalar, pl[i].shape, bdims)
    let decay = mul(lrB, wdB)
    let one = scalarF32(1'f32)
    let oneB = broadcastTo(one, pl[i].shape, bdims)
    let decayFactor = sub(oneB, decay)
    let decayed = mul(decayFactor, pl[i])
    updated[i] = sub(decayed, mul(lrB, adamUpdate))
  let newParams = treeUnflatten(params, updated)
  let newState = AdamState(m: newM, v: newV, t: t,
    beta1Power: b1t, beta2Power: b2t)
  (newParams, newState)
