## Spatial pooling ops \u2014 `maxPool2d`, `avgPool2d`.
##
## Both wrap `stablehlo.reduce_window` with a fixed reducer body
## (maximum or sum + divide). NHWC layout is assumed for the input;
## the window only spans the spatial axes (H, W).
##
## v1 supports `dtFloat32` only (the reducer init constants are
## materialised as float32). Broaden when other dtypes need pooling.

import std/math
import ../tensor
import ../dispatch
import ../dtype
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker
import ./literal
import ./linalg
import ./arith
import ./compare
import ./concat
import ./factory
import ./ternary

type
  MaxPool2dResult* = object
    ## Max-pool output values plus flattened input-spatial indices.
    values*: Tensor
    indices*: Tensor

proc joinInts(xs: openArray[int]): string =
  result = "["
  for i, x in xs:
    if i > 0: result.add ','
    result.add $x
  result.add ']'

proc joinPadding(p: openArray[array[2, int]]): string =
  result = "["
  for i, pair in p:
    if i > 0: result.add ','
    result.add '['
    result.add $pair[0]
    result.add ','
    result.add $pair[1]
    result.add ']'
  result.add ']'

proc float32Bytes(v: float32): seq[byte] =
  let bits = cast[uint32](v)
  result = newSeq[byte](4)
  result[0] = byte(bits and 0xFF'u32)
  result[1] = byte((bits shr 8) and 0xFF'u32)
  result[2] = byte((bits shr 16) and 0xFF'u32)
  result[3] = byte((bits shr 24) and 0xFF'u32)

proc actualPoolStrides(strides, kernelSize: array[2, int];
    opName: string): array[2, int] =
  result = strides
  if result[0] == 0 and result[1] == 0:
    result = kernelSize
  if result[0] <= 0 or result[1] <= 0:
    raise newException(TensorError, opName & ": strides must be positive")

proc validateMaxPoolInput(input: Tensor; kernelSize, strides: array[2, int];
    opName: string): array[2, int] =
  if input.shape.len != 4:
    raise newException(TensorError,
      opName & ": input must be NHWC rank-4, got " & $input.shape)
  if input.dtype != dtFloat32:
    raise newException(TensorError,
      opName & ": v1 supports only float32 (got " & $input.dtype & ")")
  if kernelSize[0] <= 0 or kernelSize[1] <= 0:
    raise newException(TensorError, opName & ": kernelSize must be positive")
  result = actualPoolStrides(strides, kernelSize, opName)

proc poolOutputShape(input: Tensor; kernelSize, strides: array[2, int];
    padding: array[2, array[2, int]]): seq[int] =
  let windowDims = [1, kernelSize[0], kernelSize[1], 1]
  let windowStrides = [1, strides[0], strides[1], 1]
  let pad: array[4, array[2, int]] =
    [[0, 0], padding[0], padding[1], [0, 0]]
  let dilations = [1, 1, 1, 1]
  result = reduceWindowOutputShape(input.shape, windowDims, windowStrides, pad,
    dilations, dilations)

proc maxPool2d*(input: Tensor;
    kernelSize: array[2, int];
    strides: array[2, int] = [0, 0];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]]): Tensor {.rewOp.} =
  ## 2-D max pooling over an NHWC input. `kernelSize` is `[kH, kW]`,
  ## `strides` is `[sH, sW]` (defaults to `kernelSize` when both entries
  ## are zero), and `padding` covers spatial axes only.
  let actualStrides = validateMaxPoolInput(input, kernelSize, strides,
    "maxPool2d")
  let windowDims = [1, kernelSize[0], kernelSize[1], 1]
  let windowStrides = [1, actualStrides[0], actualStrides[1], 1]
  let pad: array[4, array[2, int]] =
    [[0, 0], padding[0], padding[1], [0, 0]]
  let dilations = [1, 1, 1, 1]
  let outShape = poolOutputShape(input, kernelSize, actualStrides, padding)
  case currentMode()
  of dmTrace:
    requireTrace(input, "maxPool2d")
    let ctx = currentTraceContext()
    let initBytes = float32Bytes(NegInf.float32)
    let initId = ctx.builder.constant(input.dtype, [], initBytes)
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      shops.maximum(b, lhs, rhs)
    let id = ctx.builder.reduceWindow(input.traceId, initId,
      windowDims, windowStrides, pad, dilations, dilations, body)
    result = initTraceTensor(id, input.dtype, outShape, input.device,
      input.sharding)
    recordTraceOp("maxPool2d", [input], result, @[
      ("kernelSize", @kernelSize),
      ("strides", @actualStrides),
      ("padding", @[padding[0][0], padding[0][1],
        padding[1][0], padding[1][1]]),
    ])
  of dmEager:
    requireEager(input, "maxPool2d")
    let attrs = @[
      ("kernel_size", joinInts(kernelSize)),
      ("strides", joinInts(actualStrides)),
      ("padding", joinPadding(padding)),
    ]
    let outs = dispatchEager("maxPool2d", [input], attrs)
    doAssert outs.len == 1, "maxPool2d: eager backend returned wrong arity"
    result = outs[0]

proc maxPool2dWithIndices*(input: Tensor;
    kernelSize: array[2, int];
    strides: array[2, int] = [0, 0];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]]): MaxPool2dResult =
  ## 2-D max pooling over NHWC input, returning both values and flattened
  ## input-spatial indices suitable for `maxUnpool2d`.
  let actualStrides = validateMaxPoolInput(input, kernelSize, strides,
    "maxPool2dWithIndices")
  let outShape = poolOutputShape(input, kernelSize, actualStrides, padding)
  let n = input.shape[0]
  let hIn = input.shape[1]
  let wIn = input.shape[2]
  let c = input.shape[3]
  let hOut = outShape[1]
  let wOut = outShape[2]
  var valueRows: seq[Tensor] = @[]
  var indexRows: seq[Tensor] = @[]
  for oh in 0 ..< hOut:
    var valueCols: seq[Tensor] = @[]
    var indexCols: seq[Tensor] = @[]
    for ow in 0 ..< wOut:
      var bestValue = full([n, 1, 1, c], NegInf.float32, input.dtype,
        input.device)
      var bestIndex = zeros([n, 1, 1, c], dtInt32, input.device)
      let hStart = oh * actualStrides[0] - padding[0][0]
      let wStart = ow * actualStrides[1] - padding[1][0]
      for kh in 0 ..< kernelSize[0]:
        let srcH = hStart + kh
        if srcH >= 0 and srcH < hIn:
          for kw in 0 ..< kernelSize[1]:
            let srcW = wStart + kw
            if srcW >= 0 and srcW < wIn:
              let candidate = slice(input, [0, srcH, srcW, 0],
                [n, srcH + 1, srcW + 1, c], [1, 1, 1, 1])
              let replace = compare(candidate, bestValue, "GT")
              let candidateIndex = full([n, 1, 1, c],
                float32(srcH * wIn + srcW), dtInt32, input.device)
              bestIndex = select(replace, candidateIndex, bestIndex)
              bestValue = select(replace, candidate, bestValue)
      valueCols.add bestValue
      indexCols.add bestIndex
    valueRows.add concat(valueCols, 2)
    indexRows.add concat(indexCols, 2)
  MaxPool2dResult(
    values: concat(valueRows, 1),
    indices: concat(indexRows, 1),
  )


proc avgPool2d*(input: Tensor;
    kernelSize: array[2, int];
    strides: array[2, int] = [0, 0];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]]): Tensor =
  ## 2-D average pooling over an NHWC input. Uses `reduceWindow` with
  ## sum body, then divides by the window size.
  if input.shape.len != 4:
    raise newException(TensorError,
      "avgPool2d: input must be NHWC rank-4, got " & $input.shape)
  if input.dtype != dtFloat32:
    raise newException(TensorError,
      "avgPool2d: v1 supports only float32 (got " & $input.dtype & ")")
  if kernelSize[0] <= 0 or kernelSize[1] <= 0:
    raise newException(TensorError, "avgPool2d: kernelSize must be positive")
  var actualStrides = strides
  if actualStrides[0] == 0 and actualStrides[1] == 0:
    actualStrides = kernelSize
  if actualStrides[0] <= 0 or actualStrides[1] <= 0:
    raise newException(TensorError, "avgPool2d: strides must be positive")
  let windowDims = [1, kernelSize[0], kernelSize[1], 1]
  let windowStrides = [1, actualStrides[0], actualStrides[1], 1]
  let pad: array[4, array[2, int]] =
    [[0, 0], padding[0], padding[1], [0, 0]]
  let dilations = [1, 1, 1, 1]
  let outShape = reduceWindowOutputShape(input.shape,
    windowDims, windowStrides, pad, dilations, dilations)
  case currentMode()
  of dmTrace:
    requireTrace(input, "avgPool2d")
    let ctx = currentTraceContext()
    let initBytes = float32Bytes(0'f32)
    let initId = ctx.builder.constant(input.dtype, [], initBytes)
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      shops.add(b, lhs, rhs)
    let id = ctx.builder.reduceWindow(input.traceId, initId,
      windowDims, windowStrides, pad, dilations, dilations, body)
    let summed = initTraceTensor(id, input.dtype, outShape, input.device,
      input.sharding)
    let windowCount = scalarF32(float32(kernelSize[0] * kernelSize[1]))
    var zeroDims: seq[int] = @[]
    let divisor = broadcastTo(windowCount, outShape, zeroDims)
    result = divide(summed, divisor)
    recordTraceOp("avgPool2d", [input], result, @[
      ("kernelSize", @kernelSize),
      ("strides", @actualStrides),
      ("padding", @[padding[0][0], padding[0][1],
        padding[1][0], padding[1][1]]),
    ])
  of dmEager:
    requireEager(input, "avgPool2d")
    let attrs = @[
      ("kernel_size", joinInts(kernelSize)),
      ("strides", joinInts(actualStrides)),
      ("padding", joinPadding(padding)),
    ]
    let outs = dispatchEager("avgPool2d", [input], attrs)
    doAssert outs.len == 1, "avgPool2d: eager backend returned wrong arity"
    result = outs[0]

proc adaptiveAvgPool2d*(input: Tensor;
    outputSize: array[2, int]): Tensor =
  ## Adaptive average pooling. Computes the kernel size and strides from
  ## the input spatial dimensions and the target `outputSize`.
  if input.shape.len != 4:
    raise newException(TensorError,
      "adaptiveAvgPool2d: input must be NHWC rank-4, got " & $input.shape)
  let hIn = input.shape[1]
  let wIn = input.shape[2]
  let hOut = outputSize[0]
  let wOut = outputSize[1]
  if hOut <= 0 or wOut <= 0:
    raise newException(TensorError,
      "adaptiveAvgPool2d: output size must be positive")
  let kH = (hIn + hOut - 1) div hOut
  let kW = (wIn + wOut - 1) div wOut
  let sH = hIn div hOut
  let sW = wIn div wOut
  avgPool2d(input, [kH, kW], [sH, sW])

proc adaptiveMaxPool2d*(input: Tensor;
    outputSize: array[2, int]): Tensor =
  ## Adaptive max pooling. Computes the kernel size and strides from
  ## the input spatial dimensions and the target `outputSize`.
  if input.shape.len != 4:
    raise newException(TensorError,
      "adaptiveMaxPool2d: input must be NHWC rank-4, got " & $input.shape)
  let hIn = input.shape[1]
  let wIn = input.shape[2]
  let hOut = outputSize[0]
  let wOut = outputSize[1]
  if hOut <= 0 or wOut <= 0:
    raise newException(TensorError,
      "adaptiveMaxPool2d: output size must be positive")
  let kH = (hIn + hOut - 1) div hOut
  let kW = (wIn + wOut - 1) div wOut
  let sH = hIn div hOut
  let sW = wIn div wOut
  maxPool2d(input, [kH, kW], [sH, sW])
