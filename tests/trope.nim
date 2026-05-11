## Tests for Rotary Position Encodings (RoPE + YaRN + ALiBi).

import rew
import std/strutils

let TestDevice = cpu(0)

# --- RotaryPositionEncoding ---

block rope_basic_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 4, 8, 16]])
    let layer = initRotaryPositionEncoding(16, 1024)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[2, 4, 8, 16]
    doAssert y.dtype == dtFloat32
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block rope_with_offset:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2, 3, 8]])
    let layer = initRotaryPositionEncoding(8, 2048)
    let y = layer.forward(inputs[0], offset = 5)
    doAssert y.shape == @[1, 2, 3, 8]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block rope_small_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 5, 6]])
    let layer = initRotaryPositionEncoding(6, 1024)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 5, 6]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block rope_head_dim_2:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 3, 2]])
    let layer = initRotaryPositionEncoding(2, 1024)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 3, 2]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block rope_jit_equivalent:
  let key = initKey(42u64)
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let layer = initRotaryPositionEncoding(key, 8, 128)
    @[layer.forward(args[0])]
  let jitted = jit(f)
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[1, 2, 4, 8], TestDevice)
  let m = jitted.lower([x])
  doAssert m.funcs.len == 1
  let text = emitText(m)
  doAssert "stablehlo.slice" in text

# --- YaRN ---

block yarn_basic_shape:
  let config = initYarnConfig(2048, 8192, alpha = 2.0'f32)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2, 3, 16]])
    let layer = initYarnRotaryPositionEncoding(16, config)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 2, 3, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block yarn_long_sequence:
  let config = initYarnConfig(2048, 32768, alpha = 4.0'f32)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 1, 1000, 4]])
    let layer = initYarnRotaryPositionEncoding(4, config)
    let y = layer.forward(inputs[0])
    doAssert y.shape == @[1, 1, 1000, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- ALiBi ---

block alibi_basic:
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1]])
    let layer = initAlibiBias(4, 128)
    doAssert layer.biases.shape == @[4, 128, 128]
    let bias = layer.forward(32)
    doAssert bias.shape == @[4, 32, 32]
    ctx.traceReturn([bias])
  let m = ctx.builder.build()
  verify(m)

block alibi_kv_different:
  withTrace ctx, "main", TestDevice:
    discard ctx.traceInputs(@[dtFloat32], @[@[1]])
    let layer = initAlibiBias(8, 256)
    let bias = layer.forward(16, kvLen = 32)
    doAssert bias.shape == @[8, 16, 32]

block alibi_jit_shape:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let layer = initAlibiBias(2, 64)
    @[layer.forward(4)]
  let jitted = jit(f)
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[2, 4, 4], TestDevice)
  let m = jitted.lower([x])
  doAssert m.funcs.len == 1
  let text = emitText(m)
  doAssert "stablehlo.slice" in text

echo "All RoPE/YaRN/ALiBi tests passed"
