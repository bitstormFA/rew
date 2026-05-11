## Phase 6 \u2014 autograd vjp registry, tape, grad/vjp transform.

import rew
import std/strutils

let TestDevice = cpu(0)

block rules_installed:
  for op in ["add", "sub", "mul", "neg", "divide", "exp", "log",
      "sqrt", "tanh", "cbrt", "expm1", "log1p", "logistic", "tan",
      "atan2", "power", "optimizationBarrier",
      "reshape", "transpose", "reverse", "reduceSum", "reduceProd",
      "dot", "matmul", "broadcastTo"]:
    doAssert hasVjpRule(op), "missing rule: " & op
    doAssert vjpRuleStatus(op) == vrsInstalled
  for op in ["maximum", "minimum", "abs", "reduceMax", "reduceMin",
      "dotGeneral"]:
    doAssert hasVjpRule(op), "missing rule: " & op
    doAssert vjpRuleStatus(op) == vrsInstalled
  doAssert missingVjpRules().len == 0

block grad_of_mul_self:
  ## d/dx (x*x) = 2x \u2014 the gradient subgraph should add `x + x`.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let x = args[0]
      let sq = mul(x, x)
      reduceSum(sq, [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  # Two `mul`s show up: the forward x*x and the backward cot * x (twice).
  doAssert text.count("stablehlo.multiply") >= 2

block grad_of_add:
  ## d/dx (x+y) wrt x = 1, wrt y = 1.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[2], @[2]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(add(args[0], args[1]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[2]
    doAssert grads[1].shape == @[2]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block grad_through_reduce_prod:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(reduceProd(args[0], [1]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[2, 3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert text.count("stablehlo.multiply") >= 1

block value_and_grad_returns_value_and_grads:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(mul(args[0], args[0]), [0])
    let vg = valueAndGrad(f, inputs)
    doAssert vg.value.shape.len == 0
    doAssert vg.grads.len == 1
    doAssert vg.grads[0].shape == @[3]
    ctx.traceReturn(@[vg.value] & vg.grads)
  let m = ctx.builder.build()
  verify(m)

block vjp_returns_pullback:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      exp(args[0])
    let v = vjp(f, inputs)
    doAssert v.output.shape == @[2, 3]
    let cot = constantF32([2, 3], @[1'f32, 1, 1, 1, 1, 1])
    let grads = v.pullback(cot)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[2, 3]
    ctx.traceReturn(@[v.output] & grads)
  let m = ctx.builder.build()
  verify(m)

block grad_openxla_unary_math:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let x = args[0]
      let y = add(cbrt(x), expm1(x))
      let z = add(log1p(y), logistic(y))
      reduceSum(add(z, tan(y)), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["stablehlo.cbrt", "stablehlo.exponential_minus_one",
                 "stablehlo.log_plus_one", "stablehlo.logistic",
                 "stablehlo.tan"]:
    doAssert needle in text, text

block grad_openxla_binary_math:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32], @[@[3], @[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = args[0]
      let x = args[1]
      reduceSum(add(atan2(y, x), power(x, y)), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[3]
    doAssert grads[1].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in ["stablehlo.atan2", "stablehlo.power", "stablehlo.log"]:
    doAssert needle in text, text

block grad_through_optimization_barrier:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(optimizationBarrier(args[0]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.optimization_barrier" in text, text

block grad_through_stop_gradient_is_zero:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(stopGradient(args[0]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.constant" in text, text

block grad_through_matmul:
  ## d/dW of sum(x @ W). Should produce a matmul(transpose(x), ones).
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[3, 5]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = matmul(args[0], args[1])
      reduceSum(y, [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[4, 3]
    doAssert grads[1].shape == @[3, 5]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text  # forward + backward matmuls
  doAssert "stablehlo.transpose" in text

block grad_through_dot:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[3, 5]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = dot(args[0], args[1])
      reduceSum(y, [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[4, 3]
    doAssert grads[1].shape == @[3, 5]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot" in text
  doAssert "stablehlo.transpose" in text

block grad_through_broadcast:
  ## d/db of sum(broadcast(b)) reduces back to b's shape via reduceSum.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let big = broadcastTo(args[0], [4, 3], [1])
      reduceSum(big, [0, 1])
    let grads = grad(f, inputs)
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block grad_through_reverse:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(reverse(args[0], [0, 1]), [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[2, 3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reverse" in text, text

block grad_through_strided_slice:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[5]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = slice(args[0], [1], [5], [2])
      reduceSum(y, [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[5]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.slice" in text, text
  doAssert "stablehlo.concatenate" in text, text

block grad_of_linear_loss:
  ## End-to-end: grad of MSE-loss wrt the Linear layer's weight & bias.
  ## Confirms that composite layers (Linear.forward, mseLoss) are
  ## differentiated via their primitive decomposition.
  let key = initKey(7u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 4], @[2, 3]])
    let x = inputs[0]
    let target = inputs[1]
    let layer = initLinear(key, 4, 3)
    let f = proc(args: openArray[Tensor]): Tensor =
      let w = args[0]
      let b = args[1]
      let xw = matmul(x, w)
      let bias = broadcastTo(b, xw.shape, [1])
      let pred = add(xw, bias)
      mseLoss(pred, target)
    let grads = grad(f, [layer.weight, layer.bias])
    doAssert grads.len == 2
    doAssert grads[0].shape == @[4, 3]
    doAssert grads[1].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block grad_requires_scalar_output:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor = args[0]
    doAssertRaises(GradError):
      discard grad(f, inputs)
    ctx.traceReturn(inputs)

block abs_vjp_works:
  ## abs VJP is implemented — verify it produces a gradient tensor.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(abs(args[0]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[2]
    ctx.traceReturn(inputs)

block conv2d_forward_shape:
  ## conv2d shape inference: NHWC `[1, 5, 5, 2]` x OIHW `[3, 2, 3, 3]`
  ## with padding=1, stride=1 produces `[1, 5, 5, 3]` (SAME).
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[1, 5, 5, 2], @[3, 2, 3, 3]])
    let y = conv2d(inputs[0], inputs[1], [1, 1], [[1, 1], [1, 1]])
    doAssert y.shape == @[1, 5, 5, 3]
    ctx.traceReturn(@[y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.convolution" in text

block conv2d_vjp_shapes:
  ## conv2d VJP must produce gradients with the same shape as primals.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 4, 4, 1], @[2, 1, 3, 3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = conv2d(args[0], args[1], [1, 1], [[1, 1], [1, 1]])
      reduceSum(y, [0, 1, 2, 3])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[2, 4, 4, 1]
    doAssert grads[1].shape == @[2, 1, 3, 3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block conv2d_vjp_strided_dilated_shapes:
  ## Stride and kernel dilation are both represented in the backward
  ## convolutions rather than rejected by the rule.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[1, 7, 7, 1], @[2, 1, 3, 3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = conv2d(args[0], args[1], [2, 2], [[2, 1], [1, 2]], [2, 1])
      reduceSum(y, [0, 1, 2, 3])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[1, 7, 7, 1]
    doAssert grads[1].shape == @[2, 1, 3, 3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "lhs_dilate = [2, 2]" in text, text
  doAssert "rhs_dilate = [2, 1]" in text or
    "rhs_dilate = [2, 2]" in text, text

block max_pool_forward_shape:
  ## maxPool2d shape: `[1, 4, 4, 1]` with kernel=2, stride=2 → `[1, 2, 2, 1]`.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 4, 1]])
    let y = maxPool2d(inputs[0], [2, 2], [2, 2])
    doAssert y.shape == @[1, 2, 2, 1]
    ctx.traceReturn(@[y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.reduce_window" in text

block max_pool_vjp_overlapping_padded_shapes:
  ## Overlapping and padded windows lower through select_and_scatter.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 3, 3, 1]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = maxPool2d(args[0], [2, 2], [1, 1], [[1, 0], [1, 0]])
      reduceSum(y, [0, 1, 2, 3])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[1, 3, 3, 1]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.select_and_scatter" in text, text

block max_pool_vjp_shapes:
  ## maxPool2d VJP returns a gradient tensor with the input shape.
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 4, 2]])
    let f = proc(args: openArray[Tensor]): Tensor =
      let y = maxPool2d(args[0], [2, 2], [2, 2])
      reduceSum(y, [0, 1, 2, 3])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[1, 4, 4, 2]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
