## SGD with momentum.
##
## Functional: `step` returns `(newParams, newVelocity)`. No mutation.
## Implements classical momentum: `v = mu * v + g; p = p - lr * v`.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/linalg

type
  MomentumState* = object
    ## Velocity buffers matching parameter leaves.
    velocity*: seq[Tensor]

  MomentumSgd* = object
    ## SGD with momentum hyperparameters.
    lr*: Tensor       ## 0-d scalar learning rate
    momentum*: float32  ## Momentum coefficient (default 0.9)

proc initMomentumSgd*(lr: Tensor; momentum: float32 = 0.9'f32): MomentumSgd =
  ## Constructs a momentum SGD optimizer. `lr` must be a 0-d scalar tensor.
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initMomentumSgd: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  MomentumSgd(lr: lr, momentum: momentum)

proc initMomentumState*[P](params: P): MomentumState =
  ## Initializes momentum state (zeros) from a parameter structure.
  let pl = treeFlatten(params)
  var vel = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    let zData = newSeq[float32](pl[i].numElements)
    vel[i] = constantF32(pl[i].shape, zData)
  MomentumState(velocity: vel)

proc step*[P](opt: MomentumSgd; params: P; grads: P;
    state: MomentumState): (P, MomentumState) =
  ## Performs one momentum SGD step: `v = mu*v + g; p = p - lr*v`.
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "MomentumSgd.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  if state.velocity.len != pl.len:
    raise newException(PytreeError,
      "MomentumSgd.step: state size mismatch")
  let mu = scalarF32(opt.momentum)
  var newVel = newSeq[Tensor](pl.len)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    var bdims: seq[int] = @[]
    let muB = broadcastTo(mu, pl[i].shape, bdims)
    let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
    # v = mu * v_prev + g
    newVel[i] = add(mul(muB, state.velocity[i]), gl[i])
    # p = p - lr * v
    updated[i] = sub(pl[i], mul(lrB, newVel[i]))
  let newParams = treeUnflatten(params, updated)
  (newParams, MomentumState(velocity: newVel))
