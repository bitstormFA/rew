## Phase 5d \u2014 SGD optimizer step over a pytree of params/grads.

import rew

let TestDevice = cpu(0)

type
  MiniMlp = object
    l1: Linear
    l2: Linear

block sgd_step_single_linear:
  let key = initKey(7u64)
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1, 4]])
    let layer = initLinear(key, 4, 8)
    let grads = layer  # pretend the grads have the same structure
    let opt = initSgd(scalarF32(0.01'f32))
    let updated = opt.step(layer, grads)
    doAssert updated.weight.shape == @[4, 8]
    doAssert updated.bias.shape == @[8]
    ctx.traceReturn([updated.weight, updated.bias])
  let m = ctx.builder.build()
  verify(m)

block sgd_step_nested_pytree:
  let key = initKey(123u64)
  let keys = split(key, 2)
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1, 4]])
    let net = MiniMlp(
      l1: initLinear(keys[0], 4, 16),
      l2: initLinear(keys[1], 16, 3),
    )
    let opt = initSgd(scalarF32(0.05'f32))
    let updated = opt.step(net, net)
    doAssert updated.l1.weight.shape == net.l1.weight.shape
    doAssert updated.l2.bias.shape == net.l2.bias.shape
    ctx.traceReturn([updated.l1.weight, updated.l2.bias])
  let m = ctx.builder.build()
  verify(m)

block sgd_lr_must_be_scalar:
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1]])
    let badLr = constantF32([2], [0.1'f32, 0.2'f32])
    doAssertRaises(TensorError):
      discard initSgd(badLr)
    ctx.traceReturn([badLr])
