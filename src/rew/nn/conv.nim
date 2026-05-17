## `Conv2d` \u2014 2-D convolutional layer.
##
## Pure value type carrying its weight and bias. `weight` has OIHW
## layout `[outChannels, inChannels, kernelH, kernelW]`; `bias` has
## shape `[outChannels]`. The forward pass uses NHWC inputs.
##
## `initConv2d` materialises the parameters via `stablehlo.constant`
## (trace mode); for eager mode use `initConv2dEager` from the example.

import std/math
import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ../ops/shape
import ../ops/conv
import ./init

type
  Conv2d* = object
    ## 2-D convolution layer with optional bias. The forward pass
    ## treats the leading axis of `x` as batch and the trailing axis
    ## as channels (NHWC).
    weight*: Param[Tensor]   ## shape `[outChannels, inChannels, kernelH, kernelW]`
    bias*: Param[Tensor]     ## shape `[outChannels]`
    stride*: array[2, int]
    padding*: array[2, array[2, int]]
    dilation*: array[2, int]

proc initConv2d*(key: Key; inChannels, outChannels: int;
    kernelSize: array[2, int];
    stride: array[2, int] = [1, 1];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]];
    dilation: array[2, int] = [1, 1]): Conv2d =
  ## Constructs a `Conv2d` with Kaiming uniform initialization on
  ## `weight` (bound = `sqrt(1 / (inChannels * kH * kW))`) and a zero
  ## `bias`. Trace-mode only (uses `constantF32`).
  if inChannels <= 0 or outChannels <= 0:
    raise newException(TensorError,
      "initConv2d: channel counts must be positive (got " &
        $inChannels & ", " & $outChannels & ")")
  if kernelSize[0] <= 0 or kernelSize[1] <= 0:
    raise newException(TensorError,
      "initConv2d: kernelSize must be positive")
  let keys = split(key, 2)
  let fanIn = inChannels * kernelSize[0] * kernelSize[1]
  let bound = sqrt(1.0f32 / float32(fanIn))
  let wCount = outChannels * inChannels * kernelSize[0] * kernelSize[1]
  let wData = uniformF32(keys[0], wCount, -bound, bound)
  let bData = newSeq[float32](outChannels)
  result = Conv2d(
    weight: param(constantF32(
      [outChannels, inChannels, kernelSize[0], kernelSize[1]], wData)),
    bias: param(constantF32([outChannels], bData)),
    stride: stride,
    padding: padding,
    dilation: dilation,
  )

proc forward*(layer: Conv2d; x: Tensor): Tensor =
  ## Applies the layer to an NHWC input `x`
  ## `[N, H, W, inChannels]`, returning `[N, H', W', outChannels]`.
  let y = conv2d(x, layer.weight, layer.stride, layer.padding,
    layer.dilation)
  let biasB = broadcastTo(layer.bias, y.shape, [3])
  add(y, biasB)

proc flatten*(t: Tensor; startDim: int = 1): Tensor =
  ## Flattens all axes from `startDim` onwards into a single trailing
  ## axis. With the default `startDim = 1`, an input of shape
  ## `[batch, ...]` becomes `[batch, prod(...)]`. Composite over
  ## `reshape`.
  if startDim < 0 or startDim >= t.shape.len:
    raise newException(TensorError,
      "flatten: startDim " & $startDim & " out of range for rank " &
        $t.shape.len)
  var newShape = newSeq[int](startDim + 1)
  for i in 0 ..< startDim: newShape[i] = t.shape[i]
  var trailing = 1
  for i in startDim ..< t.shape.len: trailing *= t.shape[i]
  newShape[startDim] = trailing
  reshape(t, newShape)
