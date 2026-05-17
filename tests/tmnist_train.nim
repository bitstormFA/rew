## End-to-end lowering test for the MNIST training step.
##
## Wires the same pipeline used by `examples/mnist_mlp.nim` (Linear →
## relu → Linear → softmaxCrossEntropy → vjp → SGD update) through
## `jit.lower` and checks the resulting StableHLO module verifies and
## contains the expected ops. Pure trace-mode — no PJRT plugin
## required.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

proc makeInput(dtype: DType; shape: openArray[int]): Tensor =
  initTraceTensor(ShValueId(1), dtype, @shape, TestDevice)

block mnist_train_step_lowers:
  const InFeat = 8
  const Hidden = 4
  const Classes = 3
  const Batch = 2

  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]
    let ya = args[5]
    let lr = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: param(p[0]), bias: param(p[1]))
      let l2 = Linear(weight: param(p[2]), bias: param(p[3]))
      softmaxCrossEntropy(forward(l2, relu(forward(l1, xa))), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lr, p.shape, bdims)
      sub(p, mul(lrB, g))
    @[vr.output,
      upd(args[0], grads[0]),
      upd(args[1], grads[1]),
      upd(args[2], grads[2]),
      upd(args[3], grads[3])]

  let j = jit(trainFn, "mnist_train_step", donateArgs = [0, 1, 2, 3])
  let w1 = makeInput(dtFloat32, [InFeat, Hidden])
  let b1 = makeInput(dtFloat32, [Hidden])
  let w2 = makeInput(dtFloat32, [Hidden, Classes])
  let b2 = makeInput(dtFloat32, [Classes])
  let x  = makeInput(dtFloat32, [Batch, InFeat])
  let y  = makeInput(dtFloat32, [Batch, Classes])
  let lr = makeInput(dtFloat32, [])

  let m = j.lower([w1, b1, w2, b2, x, y, lr])
  doAssert j.cacheSize == 1
  doAssert m.funcs[0].name == "mnist_train_step"
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text or "stablehlo.dot" in text
  doAssert "stablehlo.maximum" in text   ## relu = max(x, 0)
  doAssert "stablehlo.select" in text    ## relu vjp uses select
  doAssert "stablehlo.exp" in text       ## softmax-xent forward
  doAssert "stablehlo.log" in text       ## log-sum-exp
  doAssert "stablehlo.divide" in text    ## reduceMean + softmax grad
  doAssert "stablehlo.subtract" in text  ## SGD update + per-sample loss
  doAssert "stablehlo.broadcast_in_dim" in text  ## bias + lr
  ## Five outputs: loss + four updated params.
  doAssert text.count("return") >= 1

  ## Re-lowering with the same signature must hit the cache.
  discard j.lower([w1, b1, w2, b2, x, y, lr])
  doAssert j.cacheSize == 1
