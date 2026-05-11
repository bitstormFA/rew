## Tests for `reduceMean` (composite over reduceSum/divide) and
## `softmaxCrossEntropy` loss. Both are differentiable end-to-end via
## the existing primitive vjp rules.

import rew
import std/strutils

let TestDevice = cpu(0)

block reduce_mean_lowers:
  ## reduceMean = reduceSum + divide-by-broadcast(scalar).
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let m = reduceMean(inputs[0], [0, 1])
    doAssert m.shape == @[]
    ctx.traceReturn(@[m])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reduce" in text
  doAssert "stablehlo.divide" in text
  doAssert "stablehlo.constant" in text  ## the divisor scalar

block reduce_mean_partial:
  ## Reducing only one axis must broadcast the divisor to the
  ## surviving shape.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 5]])
    let m = reduceMean(inputs[0], [1])
    doAssert m.shape == @[4]
    ctx.traceReturn(@[m])
  verify(ctx.builder.build())

block reduce_mean_grad:
  ## d/dx mean(x) = 1/n * 1 \u2014 grad must lower without error.
  let lossFn = proc(args: openArray[Tensor]): Tensor =
    reduceMean(args[0], [0])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    grad(lossFn, args), "mean_grad")
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[5], TestDevice)
  let text = emitText(trainStep.lower([x]))
  doAssert "stablehlo.divide" in text
  doAssert "stablehlo.reduce" in text

block softmax_xent_lowers:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
        @[@[2, 3], @[2, 3]])
    let l = softmaxCrossEntropy(inputs[0], inputs[1])
    doAssert l.shape == @[]
    ctx.traceReturn(@[l])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.exp" in text
  doAssert "stablehlo.log" in text
  doAssert "stablehlo.reduce" in text
  doAssert "stablehlo.subtract" in text

block softmax_xent_grad:
  ## End-to-end: grad of softmax-xent w.r.t. logits must lower cleanly.
  let lossFn = proc(args: openArray[Tensor]): Tensor =
    softmaxCrossEntropy(args[0], args[1])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    @[grad(lossFn, args)[0]], "xent_grad")
  let logits = initTraceTensor(ShValueId(1), dtFloat32, @[2, 3], TestDevice)
  let labels = initTraceTensor(ShValueId(2), dtFloat32, @[2, 3], TestDevice)
  let text = emitText(trainStep.lower([logits, labels]))
  doAssert "stablehlo.exp" in text
  doAssert "stablehlo.log" in text
  doAssert "stablehlo.divide" in text
