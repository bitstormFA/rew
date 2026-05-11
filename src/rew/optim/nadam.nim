## NAdam optimizer — Adam with Nesterov momentum.
##
## Implements Dozat (2016). The Nesterov momentum applied to Adam
## looks ahead before applying the update step.
##
## Functional: `step` returns `(newParams, newState)`. No mutation.

import std/math
import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  NAdamState* = object
    m*: seq[Tensor]
    v*: seq[Tensor]
    t*: int
    beta1Power*: float32
    beta2Power*: float32

  NAdam* = object
    lr*: Tensor
    beta1*: float32   ## First moment decay (default 0.9)
    beta2*: float32   ## Second moment decay (default 0.999)
    eps*: float32     ## Numerical stability (default 1e-8)

proc initNAdam*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32): NAdam =
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initNAdam: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  NAdam(lr: lr, beta1: beta1, beta2: beta2, eps: eps)

proc initNAdamState*[P](params: P): NAdamState =
  let pl = treeFlatten(params)
  var m = newSeq[Tensor](pl.len)
  var v = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    m[i] = constantF32(pl[i].shape, zData)
    v[i] = constantF32(pl[i].shape, zData)
  NAdamState(m: m, v: v, t: 0, beta1Power: 1'f32, beta2Power: 1'f32)

proc step*[P](opt: NAdam; params: P; grads: P;
    state: NAdamState): (P, NAdamState) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "NAdam.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  if state.m.len != pl.len:
    raise newException(PytreeError,
      "NAdam.step: state size mismatch")
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
  # Nesterov correction: the term added to the gradient
  let nesTerm = scalarF32(opt.beta1 * (1'f32 - 0.5'f32 * float32(pow(0.96'f64, float64(t)))))
  let b1Nes = mul(nesTerm, oneMinusB1)
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
    newM[i] = add(mul(b1B, state.m[i]), mul(oneMinusB1B, gl[i]))
    let gSq = mul(gl[i], gl[i])
    newV[i] = add(mul(b2B, state.v[i]), mul(oneMinusB2B, gSq))
    # Nesterov-aware first moment estimate
    let b1NesB = broadcastTo(b1Nes, pl[i].shape, bdims)
    let mNes = add(mul(b1B, newM[i]), mul(b1NesB, gl[i]))
    let corrMB = broadcastTo(corrM, pl[i].shape, bdims)
    let corrVB = broadcastTo(corrV, pl[i].shape, bdims)
    let mHat = mul(corrMB, mNes)
    let vHat = mul(corrVB, newV[i])
    let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
    let epsB = broadcastTo(epsScalar, pl[i].shape, bdims)
    let denom = add(sqrt(vHat), epsB)
    let update = divide(mHat, denom)
    updated[i] = sub(pl[i], mul(lrB, update))
  let newParams = treeUnflatten(params, updated)
  let newState = NAdamState(m: newM, v: newV, t: t,
    beta1Power: b1t, beta2Power: b2t)
  (newParams, newState)
