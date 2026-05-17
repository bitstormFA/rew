## Tests for SwiGLU and GatedFeedForward.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block swiglu_trace:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 16]])
    let y = swiglu(inputs[0])
    doAssert y.shape == @[1, 4, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.logistic" in text or "stablehlo.multiply" in text

block gated_ffn_trace:
  let key = initKey(42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 8, 32]])
    let layer = initGatedFeedForward(key, 32, 64)
    doAssert layer.hiddenDim == 64
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[2, 8, 32]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block gated_ffn_no_dropout:
  let key = initKey(99u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 3, 8]])
    let layer = initGatedFeedForward(key, 8, 16, dropout = 0.0'f32)
    doAssert not layer.useDropout
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 3, 8]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block gated_ffn_with_dropout:
  let key = initKey(7u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2, 12]])
    let layer = initGatedFeedForward(key, 12, 24, dropout = 0.1'f32)
    doAssert layer.useDropout
    let y = layer.forward(inputs[0], key, training = true)
    doAssert y.shape == @[1, 2, 12]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block gated_ffn_jit_lower:
  let key = initKey(33u64)
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let layer = initGatedFeedForward(key, 16, 32)
    @[layer.forward(args[0])]
  let jitted = jit(f)
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[1, 3, 16], TestDevice)
  let m = jitted.lower([x])
  doAssert m.funcs.len == 1

echo "All GatedFFN tests passed"
