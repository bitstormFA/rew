## Tests for Grouped Query Attention (GQA / MQA).

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

# --- GQA basic ---

block gqa_trace_shape:
  let key = initKey(123u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 8, 64], @[2, 6, 64], @[2, 6, 64]])
    let layer = initGroupedQueryAttention(key, 64, 8, 2)
    doAssert layer.numHeads == 8
    doAssert layer.numKVHeads == 2
    doAssert layer.headDim == 8
    let y = layer.forward(inputs[0], inputs[1], inputs[2])
    doAssert y.shape == @[2, 8, 64]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.broadcast_in_dim" in text

# --- MHA mode (numKVHeads == numHeads) ---

block gqa_as_mha:
  let key = initKey(42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[1, 4, 32], @[1, 4, 32], @[1, 4, 32]])
    let layer = initGroupedQueryAttention(key, 32, 4, 4)
    doAssert layer.numHeads == 4
    doAssert layer.numKVHeads == 4
    let y = layer.forward(inputs[0], inputs[1], inputs[2])
    doAssert y.shape == @[1, 4, 32]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- MQA mode (numKVHeads == 1) ---

block gqa_as_mqa:
  let key = initKey(77u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 5, 48], @[2, 5, 48], @[2, 5, 48]])
    let layer = initGroupedQueryAttention(key, 48, 6, 1)
    doAssert layer.numHeads == 6
    doAssert layer.numKVHeads == 1
    let y = layer.forward(inputs[0], inputs[1], inputs[2])
    doAssert y.shape == @[2, 5, 48]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- GQA with RoPE ---

block gqa_with_rope:
  let key = initKey(99u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[1, 3, 32], @[1, 3, 32], @[1, 3, 32]])
    let rope = initRotaryPositionEncoding(8, 1024)
    let layer = initGroupedQueryAttention(key, 32, 4, 2, rope = rope)
    doAssert layer.hasRope
    let y = layer.forward(inputs[0], inputs[1], inputs[2])
    doAssert y.shape == @[1, 3, 32]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- GQA with causal masking ---

block gqa_causal:
  let key = initKey(11u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[1, 4, 16], @[1, 4, 16], @[1, 4, 16]])
    let layer = initGroupedQueryAttention(key, 16, 2, 1)
    let y = layer.forward(inputs[0], inputs[1], inputs[2], causal = true)
    doAssert y.shape == @[1, 4, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- GQA with different seq/kv lengths ---

block gqa_cross_lengths:
  let key = initKey(55u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32, dtFloat32],
      @[@[2, 10, 24], @[2, 7, 24], @[2, 7, 24]])
    let layer = initGroupedQueryAttention(key, 24, 3, 3)
    let y = layer.forward(inputs[0], inputs[1], inputs[2])
    doAssert y.shape == @[2, 10, 24]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- GQA jit lowering ---

block gqa_jit_lower:
  let key = initKey(33u64)
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let layer = initGroupedQueryAttention(key, 32, 4, 2)
    @[layer.forward(args[0], args[1], args[2])]
  let jitted = jit(f)
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[1, 4, 32], TestDevice)
  let y = initTraceTensor(ShValueId(2), dtFloat32, @[1, 4, 32], TestDevice)
  let z = initTraceTensor(ShValueId(3), dtFloat32, @[1, 4, 32], TestDevice)
  let m = jitted.lower([x, y, z])
  doAssert m.funcs.len == 1

echo "All GQA tests passed"
