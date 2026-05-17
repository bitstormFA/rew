## Segment reduction ops — trace-mode shape and IR checks.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

block segment_sum_1d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[6], @[6]])
    let y = segmentSum(dataIn[0], dataIn[1], 3)
    doAssert y.shape == @[3]
    doAssert y.dtype == dtFloat32
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.scatter" in text, "expected scatter op in segmentSum IR"

block segment_sum_2d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[8, 4], @[8]])
    let y = segmentSum(dataIn[0], dataIn[1], 5)
    doAssert y.shape == @[5, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block segment_max_1d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[6], @[6]])
    let y = segmentMax(dataIn[0], dataIn[1], 3)
    doAssert y.shape == @[3]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.scatter" in text

block segment_max_2d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[8, 4], @[8]])
    let y = segmentMax(dataIn[0], dataIn[1], 5)
    doAssert y.shape == @[5, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block segment_mean_1d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[6], @[6]])
    let y = segmentMean(dataIn[0], dataIn[1], 3)
    doAssert y.shape == @[3]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block segment_mean_2d:
  withTrace ctx, "main", TestDevice:
    let dataIn = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[8, 4], @[8]])
    let y = segmentMean(dataIn[0], dataIn[1], 5)
    doAssert y.shape == @[5, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
