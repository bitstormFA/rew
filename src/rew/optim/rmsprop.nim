## RMSprop optimizer — adaptive learning rate with optional momentum and
## centred variant.
##
## Functional: `step` returns `(newParams, newState)`. No mutation.
## Implements the algorithms from Tieleman & Hinton (2012) lecture slides
## and Graves (2013).

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg

type
  RmspropState* = object
    squareAvg*: seq[Tensor]
    momentumBuf*: seq[Tensor]   ## Used only when momentum != 0
    gradAvg*: seq[Tensor]       ## Used only when centred = true

  Rmsprop* = object
    lr*: Tensor
    alpha*: float32       ## Smoothing constant (default 0.99)
    eps*: float32         ## Numerical stability (default 1e-8)
    momentum*: float32    ## Momentum coefficient (default 0, disabled)
    centred*: bool        ## Use centred RMSprop

proc initRmsprop*(lr: Tensor; alpha: float32 = 0.99'f32;
    eps: float32 = 1e-8'f32; momentum: float32 = 0'f32;
    centred: bool = false): Rmsprop =
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initRmsprop: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  Rmsprop(lr: lr, alpha: alpha, eps: eps, momentum: momentum,
          centred: centred)

proc initRmspropState*[P](params: P; momentum: float32 = 0'f32;
    centred: bool = false): RmspropState =
  let pl = treeFlatten(params)
  var sqAvg = newSeq[Tensor](pl.len)
  var momBuf: seq[Tensor] = @[]
  var gradAvg: seq[Tensor] = @[]
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    sqAvg[i] = constantF32(pl[i].shape, zData)
  if momentum != 0'f32:
    momBuf = newSeq[Tensor](pl.len)
    for i in 0 ..< pl.len:
      let zData = newSeq[float32](pl[i].numElements)
      momBuf[i] = constantF32(pl[i].shape, zData)
  if centred:
    gradAvg = newSeq[Tensor](pl.len)
    for i in 0 ..< pl.len:
      let zData = newSeq[float32](pl[i].numElements)
      gradAvg[i] = constantF32(pl[i].shape, zData)
  RmspropState(squareAvg: sqAvg, momentumBuf: momBuf, gradAvg: gradAvg)

proc step*[P](opt: Rmsprop; params: P; grads: P;
    state: RmspropState): (P, RmspropState) =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "Rmsprop.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  let alphaScalar = scalarF32(opt.alpha)
  let oneMinusAlpha = scalarF32(1'f32 - opt.alpha)
  let epsScalar = scalarF32(opt.eps)
  var newSqAvg = newSeq[Tensor](pl.len)
  var newMomBuf: seq[Tensor] = @[]
  var newGradAvg: seq[Tensor] = @[]
  if opt.momentum != 0'f32:
    newMomBuf = newSeq[Tensor](pl.len)
  if opt.centred:
    newGradAvg = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  let muScalar = scalarF32(opt.momentum)
  for i in 0 ..< pl.len:
    var bdims: seq[int] = @[]
    let alphaB = broadcastTo(alphaScalar, pl[i].shape, bdims)
    let oneMinusAlphaB = broadcastTo(oneMinusAlpha, pl[i].shape, bdims)
    let gSq = mul(gl[i], gl[i])
    newSqAvg[i] = add(mul(alphaB, state.squareAvg[i]),
                       mul(oneMinusAlphaB, gSq))
    let epsB = broadcastTo(epsScalar, pl[i].shape, bdims)
    if opt.centred:
      newGradAvg[i] = add(mul(alphaB, state.gradAvg[i]),
                           mul(oneMinusAlphaB, gl[i]))
      let varTerm = sub(newSqAvg[i],
                         mul(newGradAvg[i], newGradAvg[i]))
      let denom = add(sqrt(varTerm), epsB)
      var update = divide(gl[i], denom)
      if opt.momentum != 0'f32:
        let muB = broadcastTo(muScalar, pl[i].shape, bdims)
        newMomBuf[i] = add(mul(muB, state.momentumBuf[i]), update)
        update = newMomBuf[i]
      let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
      updated[i] = sub(pl[i], mul(lrB, update))
    else:
      let denom = add(sqrt(newSqAvg[i]), epsB)
      var update = divide(gl[i], denom)
      if opt.momentum != 0'f32:
        let muB = broadcastTo(muScalar, pl[i].shape, bdims)
        newMomBuf[i] = add(mul(muB, state.momentumBuf[i]), update)
        update = newMomBuf[i]
      let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
      updated[i] = sub(pl[i], mul(lrB, update))
  let newParams = treeUnflatten(params, updated)
  let newState = RmspropState(squareAvg: newSqAvg,
    momentumBuf: newMomBuf, gradAvg: newGradAvg)
  (newParams, newState)
