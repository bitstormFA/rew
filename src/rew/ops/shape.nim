## Shape-changing ops — `reshape`, `transpose`, `reverse`.
##
## These do not change dtype or device; only the layout/shape. Trace mode
## emits the matching StableHLO op; eager mode delegates to the registered
## backend (no broadcasting in v1 — broadcasts arrive in a later phase).

import ../tensor
import ../dtype
import ../device
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker
import ./concat

proc reshape*(a: Tensor; newShape: openArray[int]): Tensor {.rewOp.} =
  ## Reshape `a` to `newShape`. The element count of the new shape must
  ## match the operand. Negative or zero dims are rejected here (the
  ## v1 IR has only static shapes).
  for d in newShape:
    if d <= 0:
      raise newException(TensorError,
        "reshape: shape contains non-positive dim " & $d)
  var produced = 1
  for d in newShape: produced *= d
  if produced != a.numElements:
    raise newException(TensorError,
      "reshape: element count mismatch — operand has " & $a.numElements &
        " elements, target shape has " & $produced)
  case currentMode()
  of dmTrace:
    requireTrace(a, "reshape")
    let ctx = currentTraceContext()
    let id = shops.reshape(ctx.builder, a.traceId, newShape)
    result = initTraceTensor(id, a.dtype, newShape, a.device, a.sharding)
    recordTraceOp("reshape", [a], result,
      @[("shape", @newShape)])
  of dmEager:
    requireEager(a, "reshape")
    let outs = dispatchEager("reshape", [a],
      [("shape", $(@newShape))])
    doAssert outs.len == 1, "reshape: eager backend returned wrong arity"
    result = outs[0]

proc transpose*(a: Tensor; permutation: openArray[int]): Tensor {.rewOp.} =
  ## Permute the axes of `a`. `permutation` must be a length-rank list of
  ## distinct indices in `0 ..< rank`.
  if permutation.len != a.shape.len:
    raise newException(TensorError,
      "transpose: permutation length " & $permutation.len &
        " does not match operand rank " & $a.shape.len)
  var seenAxes = newSeq[bool](a.shape.len)
  var newShape = newSeq[int](a.shape.len)
  for i, p in permutation:
    if p < 0 or p >= a.shape.len:
      raise newException(TensorError,
        "transpose: permutation index " & $p & " out of range")
    if seenAxes[p]:
      raise newException(TensorError,
        "transpose: permutation index " & $p & " repeated")
    seenAxes[p] = true
    newShape[i] = a.shape[p]
  case currentMode()
  of dmTrace:
    requireTrace(a, "transpose")
    let ctx = currentTraceContext()
    let id = shops.transpose(ctx.builder, a.traceId, permutation)
    result = initTraceTensor(id, a.dtype, newShape, a.device, a.sharding)
    recordTraceOp("transpose", [a], result,
      @[("permutation", @permutation)])
  of dmEager:
    requireEager(a, "transpose")
    let outs = dispatchEager("transpose", [a],
      [("permutation", $(@permutation))])
    doAssert outs.len == 1, "transpose: eager backend returned wrong arity"
    result = outs[0]

proc reverse*(a: Tensor; dimensions: openArray[int]): Tensor {.rewOp.} =
  ## Reverse `a` along each axis in `dimensions`. Dimensions must be
  ## distinct indices in `0 ..< rank`; dtype and shape are unchanged.
  if dimensions.len == 0:
    raise newException(TensorError,
      "reverse: dimensions must not be empty")
  var seenAxes = newSeq[bool](a.shape.len)
  for d in dimensions:
    if d < 0 or d >= a.shape.len:
      raise newException(TensorError,
        "reverse: dimension " & $d & " out of range")
    if seenAxes[d]:
      raise newException(TensorError,
        "reverse: dimension " & $d & " repeated")
    seenAxes[d] = true
  case currentMode()
  of dmTrace:
    requireTrace(a, "reverse")
    let ctx = currentTraceContext()
    let id = shops.reverse(ctx.builder, a.traceId, dimensions)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("reverse", [a], result,
      @[("dimensions", @dimensions)])
  of dmEager:
    requireEager(a, "reverse")
    let outs = dispatchEager("reverse", [a],
      [("dimensions", $(@dimensions))])
    doAssert outs.len == 1, "reverse: eager backend returned wrong arity"
    result = outs[0]

proc getDimensionSize*(a: Tensor; dimension: int): Tensor {.rewOp.} =
  ## Return the static size of `a.shape[dimension]` as a scalar int32
  ## StableHLO tensor.
  if dimension < 0 or dimension >= a.shape.len:
    raise newException(TensorError,
      "getDimensionSize: dimension " & $dimension &
        " out of range for rank " & $a.shape.len)
  case currentMode()
  of dmTrace:
    requireTrace(a, "getDimensionSize")
    let ctx = currentTraceContext()
    let id = shops.getDimensionSize(ctx.builder, a.traceId, dimension)
    result = initTraceTensor(id, dtInt32, [], a.device, a.sharding)
    recordTraceOp("getDimensionSize", [a], result,
      @[("dimension", @[dimension])])
  of dmEager:
    requireEager(a, "getDimensionSize")
    let outs = dispatchEager("getDimensionSize", [a],
      [("dimension", $dimension)])
    doAssert outs.len == 1,
      "getDimensionSize: eager backend returned wrong arity"
    result = outs[0]

proc padOutputShape(a: Tensor; edgePaddingLow, edgePaddingHigh,
    interiorPadding: openArray[int]): seq[int] =
  try:
    shops.padOutputShape(a.shape, edgePaddingLow, edgePaddingHigh,
      interiorPadding)
  except ShBuilderError as e:
    raise newException(TensorError, e.msg)

proc pad*(a, paddingValue: Tensor; edgePaddingLow, edgePaddingHigh,
    interiorPadding: openArray[int]): Tensor {.rewOp.} =
  ## Pad `a` with a scalar `paddingValue` using StableHLO `pad`.
  requireSameMode(a, paddingValue, "pad")
  requireSameDevice(a, paddingValue, "pad")
  if paddingValue.dtype != a.dtype or paddingValue.shape.len != 0:
    raise newException(TensorError,
      "pad: paddingValue must be scalar with dtype " & $a.dtype &
        ", got " & $paddingValue.dtype & $paddingValue.shape)
  let outShape = padOutputShape(a, edgePaddingLow, edgePaddingHigh,
    interiorPadding)
  case currentMode()
  of dmTrace:
    requireTrace(a, "pad")
    requireTrace(paddingValue, "pad")
    let ctx = currentTraceContext()
    let id = shops.pad(ctx.builder, a.traceId, paddingValue.traceId,
      edgePaddingLow, edgePaddingHigh, interiorPadding)
    result = initTraceTensor(id, a.dtype, outShape, a.device, a.sharding)
    recordTraceOp("pad", [a, paddingValue], result, @[
      ("edgePaddingLow", @edgePaddingLow),
      ("edgePaddingHigh", @edgePaddingHigh),
      ("interiorPadding", @interiorPadding),
    ])
  of dmEager:
    requireEager(a, "pad")
    requireEager(paddingValue, "pad")
    let outs = dispatchEager("pad", [a, paddingValue], [
      ("edge_padding_low", $(@edgePaddingLow)),
      ("edge_padding_high", $(@edgePaddingHigh)),
      ("interior_padding", $(@interiorPadding)),
    ])
    doAssert outs.len == 1, "pad: eager backend returned wrong arity"
    result = outs[0]

proc broadcast*(a: Tensor; broadcastSizes: openArray[int]): Tensor {.rewOp.} =
  ## Legacy StableHLO broadcast. The result shape is
  ## `broadcastSizes & a.shape`.
  for i, d in broadcastSizes:
    if d < 0:
      raise newException(TensorError,
        "broadcast: broadcast size #" & $i & " must be non-negative")
  var outShape = @broadcastSizes
  outShape.add a.shape
  case currentMode()
  of dmTrace:
    requireTrace(a, "broadcast")
    let ctx = currentTraceContext()
    let id = shops.broadcast(ctx.builder, a.traceId, broadcastSizes)
    result = initTraceTensor(id, a.dtype, outShape, a.device, a.sharding)
    recordTraceOp("broadcast", [a], result,
      @[("broadcastSizes", @broadcastSizes)])
  of dmEager:
    requireEager(a, "broadcast")
    let outs = dispatchEager("broadcast", [a],
      [("broadcast_sizes", $(@broadcastSizes))])
    doAssert outs.len == 1, "broadcast: eager backend returned wrong arity"
    result = outs[0]

proc requireDynamicStartIndices(a: Tensor; startIndices: openArray[Tensor];
    opName: string) =
  if startIndices.len != a.shape.len:
    raise newException(TensorError,
      opName & ": start index count " & $startIndices.len &
        " must match operand rank " & $a.shape.len)
  for i, idx in startIndices:
    requireSameMode(a, idx, opName)
    requireSameDevice(a, idx, opName)
    if not (idx.dtype.isSignedInt or idx.dtype.isUnsignedInt) or
        idx.shape.len != 0:
      raise newException(TensorError,
        opName & ": start index #" & $i &
          " must be an integer scalar, got " & $idx.dtype & $idx.shape)

proc dynamicSlice*(a: Tensor; startIndices: openArray[Tensor];
    sliceSizes: openArray[int]): Tensor {.rewOp.} =
  ## Dynamic StableHLO slice with scalar tensor start indices.
  requireDynamicStartIndices(a, startIndices, "dynamicSlice")
  if sliceSizes.len != a.shape.len:
    raise newException(TensorError,
      "dynamicSlice: sliceSizes length " & $sliceSizes.len &
        " must match operand rank " & $a.shape.len)
  for i, size in sliceSizes:
    if size < 0 or size > a.shape[i]:
      raise newException(TensorError,
        "dynamicSlice: slice size " & $size &
          " out of range for dim " & $i)
  var operands = @[a]
  operands.add startIndices
  case currentMode()
  of dmTrace:
    requireTrace(a, "dynamicSlice")
    for idx in startIndices:
      requireTrace(idx, "dynamicSlice")
    let ctx = currentTraceContext()
    var ids = newSeq[typeof(a.traceId)](startIndices.len)
    for i, idx in startIndices:
      ids[i] = idx.traceId
    let id = shops.dynamicSlice(ctx.builder, a.traceId, ids, sliceSizes)
    result = initTraceTensor(id, a.dtype, sliceSizes, a.device, a.sharding)
    recordTraceOp("dynamicSlice", operands, result,
      @[("sliceSizes", @sliceSizes)])
  of dmEager:
    requireEager(a, "dynamicSlice")
    for idx in startIndices:
      requireEager(idx, "dynamicSlice")
    let outs = dispatchEager("dynamicSlice", operands,
      [("slice_sizes", $(@sliceSizes))])
    doAssert outs.len == 1,
      "dynamicSlice: eager backend returned wrong arity"
    result = outs[0]

proc dynamicUpdateSlice*(a, update: Tensor;
    startIndices: openArray[Tensor]): Tensor {.rewOp.} =
  ## Dynamic StableHLO update-slice with scalar tensor start indices.
  requireSameMode(a, update, "dynamicUpdateSlice")
  requireSameDevice(a, update, "dynamicUpdateSlice")
  if update.dtype != a.dtype:
    raise newException(TensorError,
      "dynamicUpdateSlice: update dtype " & $update.dtype &
        " differs from operand dtype " & $a.dtype)
  if update.shape.len != a.shape.len:
    raise newException(TensorError,
      "dynamicUpdateSlice: update rank must match operand rank")
  for i in 0 ..< a.shape.len:
    if update.shape[i] > a.shape[i]:
      raise newException(TensorError,
        "dynamicUpdateSlice: update dim " & $i &
          " is larger than operand dim")
  requireDynamicStartIndices(a, startIndices, "dynamicUpdateSlice")
  var operands = @[a, update]
  operands.add startIndices
  case currentMode()
  of dmTrace:
    requireTrace(a, "dynamicUpdateSlice")
    requireTrace(update, "dynamicUpdateSlice")
    for idx in startIndices:
      requireTrace(idx, "dynamicUpdateSlice")
    let ctx = currentTraceContext()
    var ids = newSeq[typeof(a.traceId)](startIndices.len)
    for i, idx in startIndices:
      ids[i] = idx.traceId
    let id = shops.dynamicUpdateSlice(ctx.builder, a.traceId,
      update.traceId, ids)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("dynamicUpdateSlice", operands, result)
  of dmEager:
    requireEager(a, "dynamicUpdateSlice")
    requireEager(update, "dynamicUpdateSlice")
    for idx in startIndices:
      requireEager(idx, "dynamicUpdateSlice")
    let outs = dispatchEager("dynamicUpdateSlice", operands)
    doAssert outs.len == 1,
      "dynamicUpdateSlice: eager backend returned wrong arity"
    result = outs[0]

proc iota*(dtype: DType; shape: openArray[int]; dimension: int;
    device: Device = defaultDevice()): Tensor {.rewOp.} =
  ## Build a StableHLO iota tensor on `device`.
  if dtype == dtBool:
    raise newException(TensorError, "iota: bool dtype is not supported")
  if dimension < 0 or dimension >= shape.len:
    raise newException(TensorError,
      "iota: dimension " & $dimension & " out of range for rank " &
        $shape.len)
  for i, d in shape:
    if d < 0:
      raise newException(TensorError,
        "iota: shape dimension #" & $i & " must be non-negative")
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.iota(ctx.builder, dtype, shape, dimension)
    result = initTraceTensor(id, dtype, shape, ctx.device)
    recordTraceOp("iota", [], result,
      @[("shape", @shape), ("dimension", @[dimension])])
  of dmEager:
    let outs = dispatchEager("iota", [], [
      ("dtype", $dtype),
      ("shape", $(@shape)),
      ("dimension", $dimension),
      ("device", $device),
    ])
    doAssert outs.len == 1, "iota: eager backend returned wrong arity"
    result = outs[0]

proc setDimensionSize*(a, size: Tensor; dimension: int): Tensor {.rewOp.} =
  ## Attach a dynamic dimension size to `a`.
  requireSameMode(a, size, "setDimensionSize")
  requireSameDevice(a, size, "setDimensionSize")
  if dimension < 0 or dimension >= a.shape.len:
    raise newException(TensorError,
      "setDimensionSize: dimension " & $dimension &
        " out of range for rank " & $a.shape.len)
  if not (size.dtype.isSignedInt or size.dtype.isUnsignedInt) or
      size.shape.len != 0:
    raise newException(TensorError,
      "setDimensionSize: size must be an integer scalar, got " &
        $size.dtype & $size.shape)
  case currentMode()
  of dmTrace:
    requireTrace(a, "setDimensionSize")
    requireTrace(size, "setDimensionSize")
    let ctx = currentTraceContext()
    let id = shops.setDimensionSize(ctx.builder, a.traceId,
      size.traceId, dimension)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("setDimensionSize", [a, size], result,
      @[("dimension", @[dimension])])
  of dmEager:
    requireEager(a, "setDimensionSize")
    requireEager(size, "setDimensionSize")
    let outs = dispatchEager("setDimensionSize", [a, size],
      [("dimension", $dimension)])
    doAssert outs.len == 1,
      "setDimensionSize: eager backend returned wrong arity"
    result = outs[0]

proc dynamicReshape*(a, outputShape: Tensor;
    resultShape: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO dynamic reshape with explicit static result metadata.
  requireSameMode(a, outputShape, "dynamicReshape")
  requireSameDevice(a, outputShape, "dynamicReshape")
  if not (outputShape.dtype.isSignedInt or outputShape.dtype.isUnsignedInt) or
      outputShape.shape != @[resultShape.len]:
    raise newException(TensorError,
      "dynamicReshape: outputShape must be an integer vector of length " &
        $resultShape.len)
  var n = 1
  for d in resultShape:
    if d < 0:
      raise newException(TensorError,
        "dynamicReshape: result shape contains negative dim " & $d)
    n *= d
  if n != a.numElements:
    raise newException(TensorError,
      "dynamicReshape: result element count does not match operand")
  case currentMode()
  of dmTrace:
    requireTrace(a, "dynamicReshape")
    requireTrace(outputShape, "dynamicReshape")
    let ctx = currentTraceContext()
    let id = shops.dynamicReshape(ctx.builder, a.traceId,
      outputShape.traceId, resultShape)
    result = initTraceTensor(id, a.dtype, resultShape, a.device, a.sharding)
    recordTraceOp("dynamicReshape", [a, outputShape], result,
      @[("resultShape", @resultShape)])
  of dmEager:
    requireEager(a, "dynamicReshape")
    requireEager(outputShape, "dynamicReshape")
    let outs = dispatchEager("dynamicReshape", [a, outputShape],
      [("result_shape", $(@resultShape))])
    doAssert outs.len == 1,
      "dynamicReshape: eager backend returned wrong arity"
    result = outs[0]

proc dynamicPad*(a, paddingValue, edgePaddingLow, edgePaddingHigh,
    interiorPadding: Tensor; resultShape: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO dynamic pad with explicit static result metadata.
  for input in [paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding]:
    requireSameMode(a, input, "dynamicPad")
    requireSameDevice(a, input, "dynamicPad")
  if paddingValue.dtype != a.dtype or paddingValue.shape.len != 0:
    raise newException(TensorError,
      "dynamicPad: paddingValue must be scalar with dtype " & $a.dtype)
  for (inputName, input) in [("edgePaddingLow", edgePaddingLow),
                            ("edgePaddingHigh", edgePaddingHigh),
                            ("interiorPadding", interiorPadding)]:
    if not (input.dtype.isSignedInt or input.dtype.isUnsignedInt) or
        input.shape != @[a.shape.len]:
      raise newException(TensorError,
        "dynamicPad: " & inputName &
          " must be an integer vector of length " & $a.shape.len)
  if resultShape.len != a.shape.len:
    raise newException(TensorError,
      "dynamicPad: result rank must match operand rank")
  case currentMode()
  of dmTrace:
    requireTrace(a, "dynamicPad")
    requireTrace(paddingValue, "dynamicPad")
    requireTrace(edgePaddingLow, "dynamicPad")
    requireTrace(edgePaddingHigh, "dynamicPad")
    requireTrace(interiorPadding, "dynamicPad")
    let ctx = currentTraceContext()
    let id = shops.dynamicPad(ctx.builder, a.traceId, paddingValue.traceId,
      edgePaddingLow.traceId, edgePaddingHigh.traceId,
      interiorPadding.traceId, resultShape)
    result = initTraceTensor(id, a.dtype, resultShape, a.device, a.sharding)
    recordTraceOp("dynamicPad",
      [a, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding],
      result, @[("resultShape", @resultShape)])
  of dmEager:
    requireEager(a, "dynamicPad")
    requireEager(paddingValue, "dynamicPad")
    requireEager(edgePaddingLow, "dynamicPad")
    requireEager(edgePaddingHigh, "dynamicPad")
    requireEager(interiorPadding, "dynamicPad")
    let outs = dispatchEager("dynamicPad",
      [a, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding],
      [("result_shape", $(@resultShape))])
    doAssert outs.len == 1,
      "dynamicPad: eager backend returned wrong arity"
    result = outs[0]

proc dynamicIota*(dtype: DType; outputShape: Tensor;
    resultShape: openArray[int]; dimension: int): Tensor {.rewOp.} =
  ## StableHLO dynamic iota with explicit static result metadata.
  if dtype == dtBool:
    raise newException(TensorError,
      "dynamicIota: bool dtype is not supported")
  if not (outputShape.dtype.isSignedInt or outputShape.dtype.isUnsignedInt) or
      outputShape.shape != @[resultShape.len]:
    raise newException(TensorError,
      "dynamicIota: outputShape must be an integer vector of length " &
        $resultShape.len)
  if dimension < 0 or dimension >= resultShape.len:
    raise newException(TensorError,
      "dynamicIota: dimension " & $dimension &
        " out of range for rank " & $resultShape.len)
  case currentMode()
  of dmTrace:
    requireTrace(outputShape, "dynamicIota")
    let ctx = currentTraceContext()
    let id = shops.dynamicIota(ctx.builder, dtype, outputShape.traceId,
      resultShape, dimension)
    result = initTraceTensor(id, dtype, resultShape,
      outputShape.device, outputShape.sharding)
    recordTraceOp("dynamicIota", [outputShape], result,
      @[("resultShape", @resultShape), ("dimension", @[dimension])])
  of dmEager:
    requireEager(outputShape, "dynamicIota")
    let outs = dispatchEager("dynamicIota", [outputShape], [
      ("dtype", $dtype),
      ("result_shape", $(@resultShape)),
      ("dimension", $dimension),
    ])
    doAssert outs.len == 1,
      "dynamicIota: eager backend returned wrong arity"
    result = outs[0]

proc realDynamicSlice*(a, startIndices, limitIndices, strides: Tensor;
    resultShape: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO real_dynamic_slice with explicit static result metadata.
  for input in [startIndices, limitIndices, strides]:
    requireSameMode(a, input, "realDynamicSlice")
    requireSameDevice(a, input, "realDynamicSlice")
    if not (input.dtype.isSignedInt or input.dtype.isUnsignedInt) or
        input.shape != @[a.shape.len]:
      raise newException(TensorError,
        "realDynamicSlice: slice bounds must be integer vectors of length " &
          $a.shape.len)
  if resultShape.len != a.shape.len:
    raise newException(TensorError,
      "realDynamicSlice: result rank must match operand rank")
  case currentMode()
  of dmTrace:
    requireTrace(a, "realDynamicSlice")
    requireTrace(startIndices, "realDynamicSlice")
    requireTrace(limitIndices, "realDynamicSlice")
    requireTrace(strides, "realDynamicSlice")
    let ctx = currentTraceContext()
    let id = shops.realDynamicSlice(ctx.builder, a.traceId,
      startIndices.traceId, limitIndices.traceId, strides.traceId,
      resultShape)
    result = initTraceTensor(id, a.dtype, resultShape, a.device, a.sharding)
    recordTraceOp("realDynamicSlice",
      [a, startIndices, limitIndices, strides], result,
      @[("resultShape", @resultShape)])
  of dmEager:
    requireEager(a, "realDynamicSlice")
    requireEager(startIndices, "realDynamicSlice")
    requireEager(limitIndices, "realDynamicSlice")
    requireEager(strides, "realDynamicSlice")
    let outs = dispatchEager("realDynamicSlice",
      [a, startIndices, limitIndices, strides],
      [("result_shape", $(@resultShape))])
    doAssert outs.len == 1,
      "realDynamicSlice: eager backend returned wrong arity"
    result = outs[0]

# ---- shape-manipulation composites ---------------------------------------

proc squeeze*(a: Tensor; dim: int): Tensor =
  ## Remove a size-1 dimension at `dim`. The dimension must have size 1.
  if dim < 0 or dim >= a.shape.len:
    raise newException(TensorError,
      "squeeze: dim " & $dim & " out of range for rank " & $a.shape.len)
  if a.shape[dim] != 1:
    raise newException(TensorError,
      "squeeze: dim " & $dim & " has size " & $a.shape[dim] & ", expected 1")
  var newShape: seq[int] = @[]
  for i, s in a.shape:
    if i != dim: newShape.add s
  reshape(a, newShape)

proc squeeze*(a: Tensor): Tensor =
  ## Remove all size-1 dimensions.
  var newShape: seq[int] = @[]
  for s in a.shape:
    if s != 1: newShape.add s
  if newShape.len == 0:
    return reshape(a, [1])
  reshape(a, newShape)

proc unsqueeze*(a: Tensor; dim: int): Tensor =
  ## Insert a size-1 dimension at position `dim`.
  let rank = a.shape.len
  let pos = if dim < 0: rank + dim + 1 else: dim
  if pos < 0 or pos > rank:
    raise newException(TensorError,
      "unsqueeze: dim " & $dim & " out of range for rank " & $rank)
  var newShape: seq[int] = @[]
  for i in 0 ..< pos: newShape.add a.shape[i]
  newShape.add 1
  for i in pos ..< rank: newShape.add a.shape[i]
  reshape(a, newShape)

proc flatten*(a: Tensor; startDim, endDim: int): Tensor =
  ## Flatten a contiguous range of dimensions. Negative indices are relative
  ## to the rank.
  let rank = a.shape.len
  let s = if startDim < 0: rank + startDim else: startDim
  let e = if endDim < 0: rank + endDim else: endDim
  if s < 0 or s >= rank or e < 0 or e >= rank or s > e:
    raise newException(TensorError,
      "flatten: invalid dim range [" & $startDim & ", " & $endDim &
        "] for rank " & $rank)
  var flatSize = 1
  for i in s .. e: flatSize *= a.shape[i]
  var newShape: seq[int] = @[]
  for i in 0 ..< s: newShape.add a.shape[i]
  newShape.add flatSize
  for i in e + 1 ..< rank: newShape.add a.shape[i]
  reshape(a, newShape)

proc stack*(tensors: openArray[Tensor]; dim = 0): Tensor =
  ## Join a sequence of tensors along a new `dim`. All must have the same shape.
  if tensors.len == 0:
    raise newException(TensorError, "stack: empty tensor list")
  let first = tensors[0]
  for i, t in tensors:
    if t.shape != first.shape:
      raise newException(TensorError,
        "stack: shape mismatch at index " & $i & " (" & $t.shape &
          " vs " & $first.shape & ")")
  var expanded: seq[Tensor] = @[]
  for t in tensors:
    expanded.add unsqueeze(t, dim)
  concat(expanded, dim)

proc split*(a: Tensor; sizes: openArray[int]; dim = 0): seq[Tensor] =
  ## Split `a` along `dim` into pieces with the given `sizes`.
  let pos = if dim < 0: a.shape.len + dim else: dim
  if pos < 0 or pos >= a.shape.len:
    raise newException(TensorError,
      "split: dim " & $dim & " out of range for rank " & $a.shape.len)
  var total = 0
  for s in sizes: total += s
  if total != a.shape[pos]:
    raise newException(TensorError,
      "split: sum of sizes " & $total &
        " does not match dim size " & $a.shape[pos])
  result = @[]
  var offset = 0
  for s in sizes:
    var starts = newSeq[int](a.shape.len)
    var limits = newSeq[int](a.shape.len)
    var strides = newSeq[int](a.shape.len)
    for j in 0 ..< a.shape.len:
      starts[j] = 0
      limits[j] = a.shape[j]
      strides[j] = 1
    starts[pos] = offset
    limits[pos] = offset + s
    result.add slice(a, starts, limits, strides)
    offset += s

proc chunk*(a: Tensor; chunks: int; dim = 0): seq[Tensor] =
  ## Split `a` into `chunks` pieces along `dim`. All pieces have the same
  ## size except possibly the last.
  let pos = if dim < 0: a.shape.len + dim else: dim
  if pos < 0 or pos >= a.shape.len:
    raise newException(TensorError,
      "chunk: dim " & $dim & " out of range for rank " & $a.shape.len)
  if chunks <= 0:
    raise newException(TensorError, "chunk: chunks must be > 0")
  let dimSize = a.shape[pos]
  var sizes: seq[int] = @[]
  let baseSize = dimSize div chunks
  let remainder = dimSize mod chunks
  for i in 0 ..< chunks:
    sizes.add baseSize + (if i < remainder: 1 else: 0)
  split(a, sizes, dim)

proc unbind*(a: Tensor; dim = 0): seq[Tensor] =
  ## Unbind `a` along `dim` into one slice per index.
  let pos = if dim < 0: a.shape.len + dim else: dim
  var sizes = newSeq[int](a.shape[pos])
  for i in 0 ..< a.shape[pos]: sizes[i] = 1
  split(a, sizes, dim)

proc roll*(a: Tensor; shifts: openArray[int]; dims: openArray[int]): Tensor =
  ## Roll tensor elements along `dims` by `shifts`. Positive shifts roll
  ## forward; negative shifts roll backward.
  if shifts.len != dims.len:
    raise newException(TensorError,
      "roll: shifts and dims must have the same length")
  result = a
  for k in 0 ..< shifts.len:
    let d = if dims[k] < 0: a.shape.len + dims[k] else: dims[k]
    let size = a.shape[d]
    if size <= 1: continue
    let s = shifts[k] mod size
    if s == 0: continue
    let splitPt = if s > 0: size - s else: -s
    let rank = result.shape.len
    var starts1 = newSeq[int](rank)
    var limits1 = newSeq[int](rank)
    var strides1 = newSeq[int](rank)
    var starts2 = newSeq[int](rank)
    var limits2 = newSeq[int](rank)
    var strides2 = newSeq[int](rank)
    for i in 0 ..< rank:
      starts1[i] = 0
      limits1[i] = (if i == d: splitPt else: result.shape[i])
      strides1[i] = 1
      starts2[i] = (if i == d: splitPt else: 0)
      limits2[i] = result.shape[i]
      strides2[i] = 1
    let part1 = slice(result, starts1, limits1, strides1)
    let part2 = slice(result, starts2, limits2, strides2)
    result = concat([part2, part1], d)

proc roll*(a: Tensor; shift: int; dim: int): Tensor =
  roll(a, [shift], [dim])

proc roll*(a: Tensor; shift: int): Tensor =
  ## Roll a flattened view of `a` by `shift` elements.
  let flat = flatten(a, 0, -1)
  roll(flat, shift, 0).reshape(a.shape)

proc flip*(a: Tensor; dims: openArray[int]): Tensor =
  ## Reverse `a` along the given dimensions. Convenience alias for
  ## `reverse` when you want the PyTorch-style name.
  reverse(a, dims)

proc flip*(a: Tensor; dim: int): Tensor =
  reverse(a, [dim])

proc rot90*(a: Tensor; k = 1; dims: array[2, int] = [0, 1]): Tensor =
  ## Rotate `a` by `k * 90` degrees in the plane defined by `dims`.
  if a.shape.len < 2:
    raise newException(TensorError,
      "rot90: operand must have rank >= 2, got " & $a.shape.len)
  if dims[0] == dims[1]:
    raise newException(TensorError, "rot90: dims must be distinct")
  let m = k mod 4
  if m == 0:
    return a
  result = a
  for _ in 1 .. m:
    var perm = newSeq[int](result.shape.len)
    for i in 0 ..< result.shape.len: perm[i] = i
    perm[dims[0]] = dims[1]
    perm[dims[1]] = dims[0]
    result = transpose(result, perm)
    result = flip(result, dims[0])

proc permute*(a: Tensor; dims: openArray[int]): Tensor =
  ## Permute the axes of `a`. Alias for `transpose` matching the PyTorch
  ## API surface.
  transpose(a, dims)

proc swapaxes*(a: Tensor; axis1, axis2: int): Tensor =
  ## Swap two axes of `a`.
  let rank = a.shape.len
  let ax1 = if axis1 < 0: rank + axis1 else: axis1
  let ax2 = if axis2 < 0: rank + axis2 else: axis2
  if ax1 < 0 or ax1 >= rank or ax2 < 0 or ax2 >= rank:
    raise newException(TensorError,
      "swapaxes: axis out of range for rank " & $rank)
  var perm = newSeq[int](rank)
  for i in 0 ..< rank: perm[i] = i
  swap(perm[ax1], perm[ax2])
  transpose(a, perm)

proc moveaxis*(a: Tensor; source, dest: int): Tensor =
  ## Move axis `source` to position `dest`.
  let rank = a.shape.len
  let src = if source < 0: rank + source else: source
  let dst = if dest < 0: rank + dest else: dest
  if src < 0 or src >= rank or dst < 0 or dst >= rank:
    raise newException(TensorError,
      "moveaxis: axis out of range for rank " & $rank)
  var perm: seq[int] = @[]
  for i in 0 ..< rank:
    if i != src: perm.add i
  perm.insert(src, dst)
  transpose(a, perm)

proc ravel*(a: Tensor): Tensor =
  ## Flatten `a` to a 1-D tensor.
  var n = 1
  for d in a.shape: n *= d
  reshape(a, @[n])

proc hstack*(tensors: varargs[Tensor]): Tensor =
  ## Stack tensors horizontally (column-wise). For 1-D inputs each
  ## is treated as a row `[1, N]`.
  if tensors.len == 0:
    raise newException(TensorError, "hstack: at least one tensor required")
  var parts: seq[Tensor] = @[]
  for t in tensors:
    if t.shape.len == 1:
      parts.add unsqueeze(t, 0)
    else:
      parts.add t
  concat(parts, 1)

proc vstack*(tensors: varargs[Tensor]): Tensor =
  ## Stack tensors vertically (row-wise). For 1-D inputs each is
  ## treated as a row `[1, N]`.
  if tensors.len == 0:
    raise newException(TensorError, "vstack: at least one tensor required")
  var parts: seq[Tensor] = @[]
  for t in tensors:
    if t.shape.len == 1:
      parts.add unsqueeze(t, 0)
    else:
      parts.add t
  concat(parts, 0)
