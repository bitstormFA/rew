## Full pinned StableHLO surface coverage for the late OpenXLA ops.

import rew
import rew/xla
import std/strutils

template assertContains(haystack, needle: string) =
  doAssert needle in haystack,
    "expected " & needle.escape & " in:\n" & haystack

let
  f32Scalar = initTensorType(dtFloat32, [])
  f32Vec2 = initTensorType(dtFloat32, [2])
  f32Vec5 = initTensorType(dtFloat32, [5])
  f32_2x2 = initTensorType(dtFloat32, [2, 2])
  f32_3x4x2 = initTensorType(dtFloat32, [3, 4, 2])
  f32_2x3x2x2 = initTensorType(dtFloat32, [2, 3, 2, 2])
  i32Vec2 = initTensorType(dtInt32, [2])
  i32Vec3 = initTensorType(dtInt32, [3])
  i32_2x2 = initTensorType(dtInt32, [2, 2])
  i32_2x3x2 = initTensorType(dtInt32, [2, 3, 2])
  u64Vec2 = initTensorType(dtUint64, [2])

let gatherDims = GatherDimensionNumbers(
  offsetDims: @[2, 3],
  collapsedSliceDims: @[0],
  startIndexMap: @[0, 2],
  indexVectorDim: 2)

let scatterDims = ScatterDimensionNumbers(
  updateWindowDims: @[2, 3],
  insertedWindowDims: @[0],
  scatterDimsToOperandDims: @[0, 2],
  indexVectorDim: 2)

block random_index_region_ops:
  var b = initBuilder("full_tensor")
  let outputs = [
    initTensorType(dtFloat32, [2, 3]),
    u64Vec2,
    f32_2x2,
    f32_2x3x2x2,
    f32_2x3x2x2,
    f32Vec5,
    f32Vec5,
    f32_3x4x2,
    initTensorType(dtFloat32, [4, 2]),
  ]
  let args = b.beginFunc("main", [
    f32Scalar, f32Scalar, i32Vec2, u64Vec2,
    f32_3x4x2, i32_2x3x2, i32Vec3,
    f32Vec5, f32Vec5, f32_2x3x2x2,
    initTensorType(dtFloat32, [4, 2]),
    f32_2x2,
  ], outputs)

  let random = b.rng(args[0], args[1], args[2], rdNormal, [2, 3])
  let bits = b.rngBitGenerator(args[3], raThreeFry, f32_2x2)
  let gathered = b.gather(args[4], args[5], gatherDims, [1, 2, 2],
    [2, 3, 2, 2])
  let dynamicGathered = b.dynamicGather(args[4], args[5], args[6],
    gatherDims, [2, 3, 2, 2])
  let sorted = b.sort([args[7]], dimension = 0, isStable = true,
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.compare(xs[0], xs[1], "LT")])
  let mapped = b.mapOp([args[7], args[8]], [dtFloat32], [0],
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.add(xs[0], xs[1])])
  let scattered = b.scatter([args[4]], args[5], [args[9]], scatterDims,
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.add(xs[0], xs[1])])
  let selected = b.selectAndScatter(args[10], args[11], args[0],
    [2, 1], [1, 1], [[0, 0], [0, 0]],
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.compare(xs[0], xs[1], "GE")],
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.add(xs[0], xs[1])])
  b.returnOp([random, bits[0], bits[1], gathered, dynamicGathered,
    sorted[0], mapped[0], scattered[0], selected])
  b.endFunc()

  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in [
    "stablehlo.rng",
    "stablehlo.rng_bit_generator",
    "stablehlo.gather",
    "stablehlo.dynamic_gather",
    "stablehlo.sort",
    "stablehlo.map",
    "stablehlo.scatter",
    "stablehlo.select_and_scatter",
  ]:
    assertContains text, needle

block collective_dynamic_conv_quantize:
  var b = initBuilder("full_collective")
  let lhs = initTensorType(dtFloat32, [1, 4, 4, 1])
  let rhs = initTensorType(dtFloat32, [1, 1, 3, 3])
  let outputs = [
    initTensorType(dtFloat32, [2, 4]),
    f32_2x2,
    initTensorType(dtFloat32, [4, 1]),
    initTensorType(dtFloat32, [2, 1]),
    f32_2x2,
    f32_2x2,
    f32_2x2,
    initTensorType(dtFloat32, [1, 2, 2, 1]),
    initTensorType(dtInt8, [2, 2]),
    f32_2x2,
  ]
  let args = b.beginFunc("main", [f32_2x2, lhs, rhs, i32_2x2],
    outputs)
  let groups = @[@[0, 1]]
  let allGathered = b.allGather([args[0]], 1, @[@[2, 4]], groups)
  let allReduced = b.allReduce([args[0]], groups,
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.add(xs[0], xs[1])])
  let allToAll = b.allToAll([args[0]], 1, 0, 2, @[@[4, 1]], groups)
  let reduceScatter = b.reduceScatter(args[0], 1, [2, 1], groups,
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.add(xs[0], xs[1])])
  let broadcasted = b.collectiveBroadcast(args[0], groups)
  let permuted = b.collectivePermute(args[0], [[0, 1], [1, 0]])
  let summed = b.crossReplicaSum(args[0], groups)
  let conv = b.dynamicConv(args[1], args[2], args[3],
    [1, 1], [1, 1], [1, 1], nhwcOIHWConvDims(2), [1, 2, 2, 1])
  let quantized = b.uniformQuantize(args[0], initTensorType(dtInt8, [2, 2]))
  let dequantized = b.uniformDequantize(quantized, dtFloat32)
  b.returnOp([allGathered[0], allReduced[0], allToAll[0],
    reduceScatter, broadcasted, permuted, summed, conv, quantized,
    dequantized])
  b.endFunc()

  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in [
    "stablehlo.all_gather",
    "stablehlo.all_reduce",
    "stablehlo.all_to_all",
    "stablehlo.reduce_scatter",
    "stablehlo.collective_broadcast",
    "stablehlo.collective_permute",
    "stablehlo.cross-replica-sum",
    "stablehlo.dynamic_conv",
    "stablehlo.uniform_quantize",
    "stablehlo.uniform_dequantize",
  ]:
    assertContains text, needle

block tokens_async_custom_case:
  var b = initBuilder("full_value")
  let tensorVal = initValueType(f32Vec2)
  let tokenVal = initTokenType()
  let outputs = [
    tensorVal,
    tokenVal,
    tensorVal,
    tensorVal,
    tensorVal,
    tokenVal,
  ]
  let args = b.beginValueFunc("main", [initValueType(initTensorType(dtInt32, []))],
    outputs)
  let token = b.createToken()
  let fed = b.infeed(token, [tensorVal, tokenVal])
  let outToken = b.outfeed([fed[0]], fed[1])
  let sent = b.send([fed[0]], outToken, initChannelHandle(0, 1))
  let received = b.recv(sent, [tensorVal, tokenVal], initChannelHandle(0, 1))
  let future = b.asyncStart([received[0]],
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[b.abs(xs[0])])
  let done = b.asyncDone(future)
  let called = b.customCall([done[0]], [tensorVal], "rew.test")
  let comp = b.composite([called[0]], [tensorVal], "rew.composite",
    "rew_decompose")
  let branches: array[2, ShRegionBuilder] = [
    proc(b: var ShBuilder): seq[ShValueId] {.closure.} =
      @[b.constant(dtFloat32, [2], f32Bytes([1'f32, 2'f32]))],
    proc(b: var ShBuilder): seq[ShValueId] {.closure.} =
      @[b.constant(dtFloat32, [2], f32Bytes([3'f32, 4'f32]))],
  ]
  let branched = b.caseOp(args[0], branches)
  b.returnOp([done[0], received[1], called[0], comp[0], branched[0],
    fed[1]])
  b.endFunc()

  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in [
    "stablehlo.infeed",
    "stablehlo.outfeed",
    "stablehlo.send",
    "stablehlo.recv",
    "stablehlo.async_start",
    "stablehlo.async_done",
    "stablehlo.custom_call",
    "stablehlo.composite",
    "stablehlo.case",
  ]:
    assertContains text, needle

echo "topenxla_ops_full: OK"
