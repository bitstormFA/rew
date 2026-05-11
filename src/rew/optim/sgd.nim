## Vanilla SGD optimizer.
##
## Pure functional: `step` returns new parameters; nothing is mutated.
## State-bearing optimizers (Adam, momentum-SGD, \u2026) use the
## `(newParams, newState)` tuple shape called out in the layer-8 rules.
## Plain SGD has no state, so its `step` returns just the params.

import ../tensor
import ../pytree
import ../ops/arith
import ../ops/linalg

type
  Sgd* = object
    ## Vanilla stochastic gradient descent. `lr` is a 0-d float32 tensor
    ## so it composes naturally with the dispatcher (no host scalar
    ## leaking into traced programs).
    lr*: Tensor

proc initSgd*(lr: Tensor): Sgd =
  ## Constructs an `Sgd` instance. `lr` must be a 0-d (scalar) float32
  ## tensor; the same value is reused for every parameter leaf.
  if lr.shape.len != 0:
    raise newException(TensorError,
      "initSgd: lr must be a 0-d (scalar) tensor, got shape " & $lr.shape)
  Sgd(lr: lr)

proc step*[P](opt: Sgd; params: P; grads: P): P =
  ## Returns `params - lr * grads` leaf-wise. `params` and `grads` must
  ## flatten to the same number of tensors with the same shapes.
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  if pl.len != gl.len:
    raise newException(PytreeError,
      "Sgd.step: leaf-count mismatch (" & $pl.len & " vs " & $gl.len & ")")
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    if pl[i].shape != gl[i].shape:
      raise newException(TensorError,
        "Sgd.step: leaf #" & $i & " shape mismatch (" &
          $pl[i].shape & " vs " & $gl[i].shape & ")")
    var bdims: seq[int] = @[]
    let lrB = broadcastTo(opt.lr, pl[i].shape, bdims)
    let scaled = mul(lrB, gl[i])
    updated[i] = sub(pl[i], scaled)
  treeUnflatten(params, updated)
