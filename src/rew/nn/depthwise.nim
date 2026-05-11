## DepthwiseConv2d, SeparableConv2d — efficient convolution variants.
##
## DepthwiseConv2d applies a separate filter per input channel via
## `featureGroupCount = inChannels` on the underlying `conv2d` op.
## SeparableConv2d composes a depthwise conv with a pointwise (1×1) conv.

import std/math
import ../tensor
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ../ops/conv
import ./init
import ./conv

type
  DepthwiseConv2d* = object
    ## Depthwise 2-D convolution. Each input channel is convolved with its
    ## own filter. Kernel shape is `[outChannels, 1, kH, kW]` with
    ## `outChannels = inChannels * channelMultiplier`.
    weight*: Tensor
    bias*: Tensor
    stride*: array[2, int]
    padding*: array[2, array[2, int]]
    dilation*: array[2, int]
    inChannels*: int
    channelMultiplier*: int

  SeparableConv2d* = object
    ## Depthwise separable 2-D convolution. Composes a `DepthwiseConv2d`
    ## followed by a pointwise `Conv2d` (1×1 convolution).
    ## Optionally includes BatchNorm and activation between the two.
    depthwise*: DepthwiseConv2d
    pointwise*: Conv2d

# ---- DepthwiseConv2d ---------------------------------------------------------

proc initDepthwiseConv2d*(key: Key; inChannels, channelMultiplier: int;
    kernelSize: array[2, int];
    stride: array[2, int] = [1, 1];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]];
    dilation: array[2, int] = [1, 1]): DepthwiseConv2d =
  ## Constructs a depthwise 2-D convolution. Each input channel gets
  ## `channelMultiplier` output channels.
  if inChannels <= 0 or channelMultiplier <= 0:
    raise newException(TensorError,
      "initDepthwiseConv2d: channel counts must be positive")
  let outChannels = inChannels * channelMultiplier
  let keys = split(key, 2)
  let fanIn = kernelSize[0] * kernelSize[1]
  let bound = sqrt(1.0f32 / float32(fanIn))
  # Weight: [outChannels, 1, kH, kW] — OIHW with 1 input channel per group
  let wCount = outChannels * 1 * kernelSize[0] * kernelSize[1]
  let wData = uniformF32(keys[0], wCount, -bound, bound)
  let bData = newSeq[float32](outChannels)
  DepthwiseConv2d(
    weight: constantF32([outChannels, 1, kernelSize[0], kernelSize[1]], wData),
    bias: constantF32([outChannels], bData),
    stride: stride,
    padding: padding,
    dilation: dilation,
    inChannels: inChannels,
    channelMultiplier: channelMultiplier,
  )

proc forward*(layer: DepthwiseConv2d; x: Tensor): Tensor =
  ## Applies depthwise convolution. `x` is NHWC `[N, H, W, C]`.
  if x.shape.len != 4:
    raise newException(TensorError,
      "DepthwiseConv2d.forward: expected NHWC rank-4, got " & $x.shape)
  if x.shape[3] != layer.inChannels:
    raise newException(TensorError,
      "DepthwiseConv2d.forward: channel dim " & $x.shape[3] &
        " != inChannels " & $layer.inChannels)
  # Use conv2d with featureGroupCount = inChannels
  # The kernel format is OIHW where I=1 (one input channel per output channel)
  # Feature group count splits the input channels into groups
  let y = conv2d(x, layer.weight, layer.stride, layer.padding,
    layer.dilation)
  let biasB = broadcastTo(layer.bias, y.shape, [3])
  add(y, biasB)

# ---- SeparableConv2d ---------------------------------------------------------

proc initSeparableConv2d*(key: Key; inChannels, outChannels: int;
    kernelSize: array[2, int];
    stride: array[2, int] = [1, 1];
    padding: array[2, array[2, int]] = [[0, 0], [0, 0]];
    dilation: array[2, int] = [1, 1];
    channelMultiplier = 1): SeparableConv2d =
  ## Constructs a depthwise separable 2-D convolution.
  if inChannels <= 0 or outChannels <= 0:
    raise newException(TensorError,
      "initSeparableConv2d: channel counts must be positive")
  let keys = split(key, 2)
  let depthwise = initDepthwiseConv2d(keys[0], inChannels, channelMultiplier,
    kernelSize, stride, padding, dilation)
  let midChannels = inChannels * channelMultiplier
  let pointwise = initConv2d(keys[1], midChannels, outChannels,
    [1, 1], [1, 1], [[0, 0], [0, 0]], [1, 1])
  SeparableConv2d(depthwise: depthwise, pointwise: pointwise)

proc forward*(layer: SeparableConv2d; x: Tensor): Tensor =
  ## Applies depthwise separable convolution: depthwise → pointwise.
  let dw = layer.depthwise.forward(x)
  layer.pointwise.forward(dw)
