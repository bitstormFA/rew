## 2-D convolution dispatch op (`conv2d`).
##
## ## Layout
## - input: NHWC `[N, H, W, C_in]`
## - kernel: OIHW `[C_out, C_in, kH, kW]`
## - output: NHWC `[N, H', W', C_out]`
##
## Trace mode lowers to `stablehlo.convolution` with the matching
## dimension numbers; eager mode routes through the registered backend
## (which itself emits a single-op StableHLO module). The vjp rule for
## `conv2d` lives in `src/rew/autograd/rules.nim` and decomposes a
## convolution gradient into two more `stablehlo.convolution` calls.
##
## v1 supports only stride/padding/dilation expressed as concrete
## integers; dynamic shapes are out of scope.

import ../tensor
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./shape
import ./marker

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

proc conv2d*(input, kernel: Tensor;
    strides: array[2, int] = [1, 1];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]];
    dilation: array[2, int] = [1, 1]): Tensor {.rewOp.} =
  ## Computes a 2-D convolution. `input` is NHWC `[N, H, W, C_in]`,
  ## `kernel` is OIHW `[C_out, C_in, kH, kW]`, output is NHWC
  ## `[N, H', W', C_out]` where the spatial sizes follow the standard
  ## SAME/VALID formulas with explicit padding.
  if input.shape.len != 4:
    raise newException(TensorError,
      "conv2d: input must be NHWC rank-4, got " & $input.shape)
  if kernel.shape.len != 4:
    raise newException(TensorError,
      "conv2d: kernel must be OIHW rank-4, got " & $kernel.shape)
  if input.dtype != kernel.dtype:
    raise newException(TensorError,
      "conv2d: dtype mismatch (" & $input.dtype & " vs " & $kernel.dtype & ")")
  if input.shape[3] != kernel.shape[1]:
    raise newException(TensorError,
      "conv2d: input channels " & $input.shape[3] &
        " do not match kernel input channels " & $kernel.shape[1])
  if strides[0] <= 0 or strides[1] <= 0:
    raise newException(TensorError, "conv2d: strides must be positive")
  if dilation[0] <= 0 or dilation[1] <= 0:
    raise newException(TensorError, "conv2d: dilation must be positive")

  let dims = nhwcOIHWConvDims(2)
  let outShape = convolutionOutputShape(input.shape, kernel.shape,
    strides, padding, [1, 1], dilation, dims, 1, 1)

  case currentMode()
  of dmTrace:
    requireTrace(input, "conv2d")
    requireTrace(kernel, "conv2d")
    let ctx = currentTraceContext()
    let id = shops.convolution(ctx.builder, input.traceId, kernel.traceId,
      strides, padding, [1, 1], dilation, dims, 1, 1)
    result = initTraceTensor(id, input.dtype, outShape, input.device,
      input.sharding)
    recordTraceOp("conv2d", [input, kernel], result, @[
      ("strides", @strides),
      ("padding", @[padding[0][0], padding[0][1],
        padding[1][0], padding[1][1]]),
      ("dilation", @dilation),
    ])
  of dmEager:
    requireEager(input, "conv2d")
    requireEager(kernel, "conv2d")
    requireSameDevice(input, kernel, "conv2d")
    let attrs = @[
      ("strides", joinInts(strides)),
      ("padding", joinPadding(padding)),
      ("dilation", joinInts(dilation)),
    ]
    let outs = dispatchEager("conv2d", [input, kernel], attrs)
    doAssert outs.len == 1, "conv2d: eager backend returned wrong arity"
    result = outs[0]

proc dynamicConv*(input, kernel, padding: Tensor;
    windowStrides, lhsDilation, rhsDilation: openArray[int];
    dims: ConvDimensionNumbers; resultShape: openArray[int];
    featureGroupCount = 1; batchGroupCount = 1;
    windowReversal: openArray[bool] = []): Tensor {.rewOp.} =
  ## Dynamic convolution where `padding` is a runtime 2-D tensor `[N, 2]`
  ## instead of a compile-time constant. `resultShape` must be supplied
  ## explicitly.
  case currentMode()
  of dmTrace:
    requireTrace(input, "dynamicConv")
    requireTrace(kernel, "dynamicConv")
    requireTrace(padding, "dynamicConv")
    let ctx = currentTraceContext()
    let id = shops.dynamicConv(ctx.builder, input.traceId,
      kernel.traceId, padding.traceId, windowStrides, lhsDilation,
      rhsDilation, dims, resultShape, featureGroupCount, batchGroupCount,
      windowReversal)
    result = initTraceTensor(id, input.dtype, resultShape, input.device,
      input.sharding)
    recordTraceOp("dynamicConv", [input, kernel, padding], result)
  of dmEager:
    raise newException(TensorError,
      "dynamicConv: only supported in trace/jit mode")

# ---- conv composites -----------------------------------------------------

proc conv1d*(input, kernel: Tensor;
    stride = 1; padding: array[2, int] = [0, 0];
    dilation = 1): Tensor =
  ## 1-D convolution. `input` is NCW, `kernel` is OIW, output is NCW'.
  ## Implemented by reshaping to NHWC and using conv2d.
  if input.shape.len != 3:
    raise newException(TensorError,
      "conv1d: input must be NCW rank-3, got " & $input.shape)
  if kernel.shape.len != 3:
    raise newException(TensorError,
      "conv1d: kernel must be OIW rank-3, got " & $kernel.shape)
  let input4d = unsqueeze(input, 2)
  let kernel4d = unsqueeze(kernel, 2)
  let convResult = conv2d(input4d, kernel4d,
    strides=[1, stride],
    padding=[[0, 0], padding],
    dilation=[1, dilation])
  squeeze(convResult, 2)

proc convTranspose2d*(input, kernel: Tensor;
    strides: array[2, int] = [1, 1];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]];
    outputPadding: array[2, int] = [0, 0]): Tensor =
  ## 2-D transposed convolution. `input` is NHWC, `kernel` is OIHW,
  ## output is NHWC. Implemented via `stablehlo.convolution` with
  ## window reversal.
  if input.shape.len != 4:
    raise newException(TensorError,
      "convTranspose2d: input must be NHWC rank-4, got " & $input.shape)
  if kernel.shape.len != 4:
    raise newException(TensorError,
      "convTranspose2d: kernel must be OIHW rank-4, got " & $kernel.shape)
  if input.dtype != kernel.dtype:
    raise newException(TensorError,
      "convTranspose2d: dtype mismatch")
  if input.shape[3] != kernel.shape[0]:
    raise newException(TensorError,
      "convTranspose2d: input C_in " & $input.shape[3] &
        " != kernel C_out " & $kernel.shape[0])
  let cIn = kernel.shape[1]
  let kH = kernel.shape[2]
  let kW = kernel.shape[3]
  let sH = strides[0]
  let sW = strides[1]
  let pH0 = padding[0][0]
  let pH1 = padding[0][1]
  let pW0 = padding[1][0]
  let pW1 = padding[1][1]
  # Output spatial dims: H' = sH*(H-1) + kH - 2*pH0 - 2*pH1 + outputPadding
  let hOut = sH * (input.shape[1] - 1) + kH - pH0 - pH1 + outputPadding[0]
  let wOut = sW * (input.shape[2] - 1) + kW - pW0 - pW1 + outputPadding[1]
  let outShape = [input.shape[0], hOut, wOut, cIn]
  # Transposed conv is a forward conv with input=C_in, kernel transposed,
  # window_reversal=true, and adjusted padding.
  # Transposed conv uses swapped kernel I/O axes and window reversal.
  let dims = ConvDimensionNumbers(
    inputBatch: 0, inputFeature: 3, inputSpatial: @[1, 2],
    kernelInputFeature: 0, kernelOutputFeature: 1,
    kernelSpatial: @[2, 3],
    outputBatch: 0, outputFeature: 3, outputSpatial: @[1, 2])
  let transposedPadding: seq[array[2, int]] = @[
    [kH - 1 - pH0, kH - 1 - pH1],
    [kW - 1 - pW0, kW - 1 - pW1],
  ]
  case currentMode()
  of dmTrace:
    requireTrace(input, "convTranspose2d")
    requireTrace(kernel, "convTranspose2d")
    let ctx = currentTraceContext()
    let id = shops.convolution(ctx.builder, input.traceId,
      kernel.traceId, strides, transposedPadding, [1, 1], [1, 1],
      dims, 1, 1, [true, true])
    result = initTraceTensor(id, input.dtype, outShape,
      input.device, input.sharding)
    recordTraceOp("convTranspose2d", [input, kernel], result)
  of dmEager:
    raise newException(TensorError,
      "convTranspose2d: only supported in trace/jit mode")

proc conv3d*(input, kernel: Tensor;
    strides: array[3, int] = [1, 1, 1];
    padding: array[3, array[2, int]] = [[0, 0], [0, 0], [0, 0]];
    dilation: array[3, int] = [1, 1, 1]): Tensor =
  ## 3-D convolution. `input` is NDHWC, `kernel` is OIDHW, output is
  ## NDHWC. Implemented by reshaping to NHWC and using conv2d, then
  ## reshaping back.
  if input.shape.len != 5:
    raise newException(TensorError,
      "conv3d: input must be NDHWC rank-5, got " & $input.shape)
  if kernel.shape.len != 5:
    raise newException(TensorError,
      "conv3d: kernel must be OIDHW rank-5, got " & $kernel.shape)
  # Merge D and H into a single spatial dimension: N, D*H, W, C
  let n = input.shape[0]
  let d = input.shape[1]
  let h = input.shape[2]
  let w = input.shape[3]
  let cIn = input.shape[4]
  let cOut = kernel.shape[0]
  let kd = kernel.shape[2]
  let kh = kernel.shape[3]
  let kw = kernel.shape[4]
  # Reshape input to NHWC: [N, D*H, W, C]
  let input4d = reshape(input, [n, d * h, w, cIn])
  # Reshape kernel to OIHW: [C_out, C_in, kD*kH, kW]
  let kernel4d = reshape(kernel, [cOut, cIn, kd * kh, kw])
  let convResult = conv2d(input4d, kernel4d,
    strides = [strides[0] * strides[1], strides[2]],
    padding = [[padding[0][0] + padding[1][0],
      padding[0][1] + padding[1][1]], padding[2]],
    dilation = [dilation[0] * dilation[1], dilation[2]])
  # Reshape output back to NDHWC.
  # Determine output spatial dims.
  let hOut = convResult.shape[1]
  let wOut = convResult.shape[2]
  let dOut = hOut div h
  reshape(convResult, [n, dOut, h, wOut, cOut])
