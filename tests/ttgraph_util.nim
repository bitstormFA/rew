## Graph utilities — trace-mode shape and IR checks.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block degree_computes_per_node_counts:
  withTrace ctx, "main", TestDevice:
    # edge_index row [0, 1, 2, 0, 1] → degrees {0:2, 1:2, 2:1, 3:0}
    let inputs = ctx.traceInputs(@[dtInt32], @[@[5]])
    let deg = degree(inputs[0], 4)
    doAssert deg.shape == @[4]
    doAssert deg.dtype == dtFloat32
    ctx.traceReturn([deg])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.scatter" in text

block add_self_loops_shape:
  withTrace ctx, "main", TestDevice:
    # edge_index [2, 3], 5 nodes → [2, 3+5]
    let inputs = ctx.traceInputs(@[dtInt32], @[@[2, 3]])
    let ei = addSelfLoops(inputs[0], 5)
    doAssert ei.shape == @[2, 8]
    doAssert ei.dtype == dtInt32
    ctx.traceReturn([ei])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.concatenate" in text

block normalize_edge_index_has_norms:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtInt32], @[@[2, 4]])
    let (eiLoop, normW) = normalizeEdgeIndex(inputs[0], 6)
    doAssert eiLoop.shape[0] == 2
    doAssert eiLoop.shape[1] == 10  # 4 + 6 self-loops
    doAssert normW.shape[0] == eiLoop.shape[1]
    ctx.traceReturn([eiLoop, normW])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.concatenate" in text
  doAssert "stablehlo.log" in text
  doAssert "stablehlo.exponential" in text

block softmax_per_segment_shape:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[6, 3], @[6]])
    let y = softmaxPerSegment(dataIn[0], dataIn[1], 3)
    doAssert y.shape == @[6, 3]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
