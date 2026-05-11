## Gather-family op behavior checks.

import rew
import std/strutils

block index_select_trace_shape_and_ir:
  withTrace ctx, "index_select", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 3], @[4]])
    let y = indexSelect(inputs[0], inputs[1], dim = 0)
    doAssert y.shape == @[4, 3]
    doAssert y.dtype == dtFloat32
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let txt = emitText(m)
  doAssert "stablehlo.torch_index_select" in txt

echo "tgather: OK"
