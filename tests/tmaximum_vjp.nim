## Tests for the `maximum` / `minimum` vjp rules and their composition
## with `relu` (= `maximum(x, 0)`). Pure trace-mode — no PJRT plugin
## needed.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block maximum_vjp_lowers:
  ## d/da max(a, b) = (a >= b) ? cot : 0; rules sum to cot.
  let lossFn = proc(args: openArray[Tensor]): Tensor =
    let m = maximum(args[0], args[1])
    reduceSum(m, [0])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    grad(lossFn, args), "max_grad")
  let a = initTraceTensor(ShValueId(1), dtFloat32, @[3], TestDevice)
  let b = initTraceTensor(ShValueId(2), dtFloat32, @[3], TestDevice)
  let m = trainStep.lower([a, b])
  doAssert m.funcs[0].name == "max_grad"
  let text = emitText(m)
  doAssert "stablehlo.maximum" in text
  doAssert "stablehlo.compare" in text
  doAssert " GE," in text or "GE\"" in text or "GE " in text
  doAssert "stablehlo.select" in text
  doAssert "stablehlo.subtract" in text  ## dB = cot - dA

block minimum_vjp_lowers:
  let lossFn = proc(args: openArray[Tensor]): Tensor =
    reduceSum(minimum(args[0], args[1]), [0])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    grad(lossFn, args), "min_grad")
  let a = initTraceTensor(ShValueId(1), dtFloat32, @[3], TestDevice)
  let b = initTraceTensor(ShValueId(2), dtFloat32, @[3], TestDevice)
  let text = emitText(trainStep.lower([a, b]))
  doAssert "stablehlo.minimum" in text
  doAssert "stablehlo.compare" in text
  doAssert "stablehlo.select" in text
  doAssert " LE," in text or "LE\"" in text or "LE " in text

block relu_grad_via_max:
  ## `relu` is composite (max(x, 0)); its grad must reuse the max rule.
  let lossFn = proc(args: openArray[Tensor]): Tensor =
    reduceSum(relu(args[0]), [0])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    grad(lossFn, args), "relu_grad")
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[4], TestDevice)
  let text = emitText(trainStep.lower([x]))
  doAssert "stablehlo.maximum" in text
  doAssert "stablehlo.select" in text
  doAssert "stablehlo.compare" in text
