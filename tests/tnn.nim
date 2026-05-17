## Phase 5d \u2014 nn (Linear, ReLU, MSE) end-to-end inside trace mode.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block linear_forward_shape:
  let key = initKey(0xC0FFEEu64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[8, 4]])
    let layer = initLinear(key, 4, 16)
    doAssert layer.weight.shape == @[4, 16]
    doAssert layer.bias.shape == @[16]
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[8, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.broadcast_in_dim" in text
  doAssert "stablehlo.add" in text
  doAssert "stablehlo.constant" in text

block lora_merge_trace:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 2]])
    let base = Linear(
      weight: param(constantF32([2, 3], [
        1'f32, 2'f32, 3'f32,
        4'f32, 5'f32, 6'f32,
      ])),
      bias: param(constantF32([3], [0.5'f32, 1'f32, -1'f32])),
    )
    var layer = initLoraLinear(initKey(3u64), base, rank = 1,
      alpha = 2'f32)
    layer.A = param(constantF32([1, 2], [0.25'f32, -0.5'f32]))
    layer.B = param(constantF32([3, 1], [2'f32, -1'f32, 0.5'f32]))
    let before = layer.forward(inputs[0])
    layer.merge()
    doAssert layer.merged
    doAssert layer.base.weight.shape == @[2, 3]
    let after = layer.forward(inputs[0])
    doAssert before.shape == @[2, 3]
    doAssert after.shape == @[2, 3]
    layer.merge()
    doAssert layer.base.weight.shape == @[2, 3]
    ctx.traceReturn([before, after, layer.base.weight.value])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.transpose" in text
  doAssert "stablehlo.add" in text

block relu_lowers_to_max_with_zero:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 5]])
    let y = relu(inputs[0])
    doAssert y.shape == @[3, 5]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.maximum" in text
  doAssert "stablehlo.broadcast_in_dim" in text
  doAssert "stablehlo.constant" in text

block mse_loss_is_scalar:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 2], @[4, 2]])
    let l = mseLoss(inputs[0], inputs[1])
    doAssert l.shape.len == 0, "loss should be scalar, got " & $l.shape
    doAssert l.dtype == dtFloat32
    ctx.traceReturn([l])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.subtract" in text
  doAssert "stablehlo.multiply" in text
  doAssert "stablehlo.reduce" in text

block linear_relu_chain:
  ## Realistic mini-MLP forward + loss inside one trace.
  let key = initKey(42u64)
  let keys = split(key, 2)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[8, 4], @[8, 3]])
    let l1 = initLinear(keys[0], 4, 16)
    let l2 = initLinear(keys[1], 16, 3)
    let h = relu(l1.forward(inputs[0]))
    let pred = l2.forward(h)
    let loss = mseLoss(pred, inputs[1])
    doAssert loss.shape.len == 0
    ctx.traceReturn([loss])
  let m = ctx.builder.build()
  verify(m)

block linear_zero_in_features_raises:
  let key = initKey(1u64)
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1, 1]])
    doAssertRaises(TensorError):
      discard initLinear(key, 0, 4)
    let zero = scalarF32(0'f32)
    ctx.traceReturn([zero])

block mse_shape_mismatch_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[2, 3], @[2, 4]])
    doAssertRaises(TensorError):
      discard mseLoss(inputs[0], inputs[1])
    ctx.traceReturn([inputs[0]])
