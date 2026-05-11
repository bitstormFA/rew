## Adam optimizer — adaptive moment estimation.
##
## Functional: `step` returns `(newParams, newState)`. No mutation.
## Implements the algorithm from Kingma & Ba (2014) with bias correction.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg
from ../eager import zerosLikeEager

type
  AdamState* = object
    ## Optimizer state for Adam: first moment (m), second moment (v),
    ## step count (t), and running beta powers used for bias correction.
    ## Each moment is a flattened seq of tensors matching the parameter
    ## leaves.
    m*: seq[Tensor]
    v*: seq[Tensor]
    t*: int
    beta1Power*: float32
    beta2Power*: float32

  Adam* = object
    ## Adam optimizer hyperparameters.
    lr*: Tensor      ## 0-d scalar learning rate
    beta1*: float32  ## First moment decay (default 0.9)
    beta2*: float32  ## Second moment decay (default 0.999)
    eps*: float32    ## Numerical stability term (default 1e-8)

proc initAdam*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32): Adam =
  ## Constructs an Adam optimizer. `lr` must be a 0-d scalar tensor.
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initAdam: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  Adam(lr: lr, beta1: beta1, beta2: beta2, eps: eps)

proc initAdamState*[P](params: P): AdamState =
  ## Initializes Adam state (zeros) from a parameter structure.
  ## Must be called inside trace mode since it uses `scalarF32`/`broadcastTo`.
  let pl = treeFlatten(params)
  var m = newSeq[Tensor](pl.len)
  var v = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    if pl[i].isEager:
      m[i] = zerosLikeEager(pl[i])
      v[i] = zerosLikeEager(pl[i])
    else:
      let zData = newSeq[float32](pl[i].numElements)
      m[i] = constantF32(pl[i].shape, zData)
      v[i] = constantF32(pl[i].shape, zData)
  AdamState(m: m, v: v, t: 0, beta1Power: 1'f32, beta2Power: 1'f32)

proc step*[P](opt: Adam; params: P; grads: P;
    state: AdamState): (P, AdamState) =
  ## Performs one Adam update step. Returns `(newParams, newState)`.
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "Adam.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  if state.m.len != pl.len:
    raise newException(PytreeError,
      "Adam.step: state size mismatch")
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
  var newM = newSeq[Tensor](pl.len)
  var newV = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    var bdims: seq[int] = @[]
    let b1B = broadcastTo(b1, pl[i].shape, bdims)
    let b2B = broadcastTo(b2, pl[i].shape, bdims)
    let oneMinusB1B = broadcastTo(oneMinusB1, pl[i].shape, bdims)
    let oneMinusB2B = broadcastTo(oneMinusB2, pl[i].shape, bdims)
    # m_t = beta1 * m_{t-1} + (1-beta1) * g
    newM[i] = add(mul(b1B, state.m[i]), mul(oneMinusB1B, gl[i]))
    # v_t = beta2 * v_{t-1} + (1-beta2) * g^2
    let gSq = mul(gl[i], gl[i])
    newV[i] = add(mul(b2B, state.v[i]), mul(oneMinusB2B, gSq))
    # Bias-corrected estimates.
    let corrMB = broadcastTo(corrM, pl[i].shape, bdims)
    let corrVB = broadcastTo(corrV, pl[i].shape, bdims)
    let mHat = mul(corrMB, newM[i])
    let vHat = mul(corrVB, newV[i])
    # Update: p = p - lr * mHat / (sqrt(vHat) + eps)
    let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
    let epsB = broadcastTo(epsScalar, pl[i].shape, bdims)
    let denom = add(sqrt(vHat), epsB)
    let update = divide(mHat, denom)
    updated[i] = sub(pl[i], mul(lrB, update))
  let newParams = treeUnflatten(params, updated)
  let newState = AdamState(m: newM, v: newV, t: t,
    beta1Power: b1t, beta2Power: b2t)
  (newParams, newState)
