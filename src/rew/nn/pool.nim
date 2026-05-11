## Pooling layers — global pooling, adaptive 1-D pooling, MaxUnpool2d.
##
## Global pooling reduces all spatial dimensions to size 1.
## Adaptive 1-D pooling mirrors the existing 2-D variants.

import ../tensor
import ../dtype
import ../ops/reduce
import ../ops/shape
import ../ops/pool
import ../ops/concat
import ../ops/compare
import ../ops/factory
import ../ops/ternary

# ---- Global pooling ----------------------------------------------------------

proc globalAvgPool1d*(x: Tensor): Tensor =
  ## Global average pooling over the spatial dim (dim 1 for NLC input).
  if x.shape.len < 2:
    raise newException(TensorError,
      "globalAvgPool1d: input must have at least 2 dims")
  reduceMean(x, @[1])

proc globalAvgPool2d*(x: Tensor): Tensor =
  ## Global average pooling over spatial dims. For NHWC `[N, H, W, C]`,
  ## reduces over [1, 2] yielding `[N, C]`.
  if x.shape.len < 3:
    raise newException(TensorError,
      "globalAvgPool2d: input must have at least 3 dims")
  var spatialDims: seq[int] = @[]
  for i in 1 ..< x.shape.len - 1: spatialDims.add i
  reduceMean(x, spatialDims)

proc globalAvgPool3d*(x: Tensor): Tensor =
  ## Global average pooling over spatial dims. For NDHWC `[N, D, H, W, C]`,
  ## reduces over [1, 2, 3] yielding `[N, C]`.
  if x.shape.len < 4:
    raise newException(TensorError,
      "globalAvgPool3d: input must have at least 4 dims")
  var spatialDims: seq[int] = @[]
  for i in 1 ..< x.shape.len - 1: spatialDims.add i
  reduceMean(x, spatialDims)

proc globalMaxPool1d*(x: Tensor): Tensor =
  ## Global max pooling over the spatial dim (dim 1 for NLC input).
  if x.shape.len < 2:
    raise newException(TensorError,
      "globalMaxPool1d: input must have at least 2 dims")
  reduceMax(x, @[1])

proc globalMaxPool2d*(x: Tensor): Tensor =
  ## Global max pooling over spatial dims. For NHWC `[N, H, W, C]`,
  ## reduces over [1, 2] yielding `[N, C]`.
  if x.shape.len < 3:
    raise newException(TensorError,
      "globalMaxPool2d: input must have at least 3 dims")
  var spatialDims: seq[int] = @[]
  for i in 1 ..< x.shape.len - 1: spatialDims.add i
  reduceMax(x, spatialDims)

proc globalMaxPool3d*(x: Tensor): Tensor =
  ## Global max pooling over spatial dims. For NDHWC `[N, D, H, W, C]`,
  ## reduces over [1, 2, 3] yielding `[N, C]`.
  if x.shape.len < 4:
    raise newException(TensorError,
      "globalMaxPool3d: input must have at least 4 dims")
  var spatialDims: seq[int] = @[]
  for i in 1 ..< x.shape.len - 1: spatialDims.add i
  reduceMax(x, spatialDims)

# ---- Adaptive 1-D pooling ----------------------------------------------------

proc adaptiveAvgPool1d*(x: Tensor; outputSize: int): Tensor =
  ## Adaptive average pooling over the temporal/spatial axis.
  ## `x` shape: `[N, L, C]`. Output: `[N, outputSize, C]`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "adaptiveAvgPool1d: expected NLC rank-3, got " & $x.shape)
  if outputSize <= 0:
    raise newException(TensorError,
      "adaptiveAvgPool1d: output size must be positive")
  let nhwc = unsqueeze(x, 2)
  let pooled = adaptiveAvgPool2d(nhwc, [outputSize, 1])
  squeeze(pooled, 2)

proc adaptiveMaxPool1d*(x: Tensor; outputSize: int): Tensor =
  ## Adaptive max pooling over the temporal/spatial axis.
  ## `x` shape: `[N, L, C]`. Output: `[N, outputSize, C]`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "adaptiveMaxPool1d: expected NLC rank-3, got " & $x.shape)
  if outputSize <= 0:
    raise newException(TensorError,
      "adaptiveMaxPool1d: output size must be positive")
  let nhwc = unsqueeze(x, 2)
  let pooled = adaptiveMaxPool2d(nhwc, [outputSize, 1])
  squeeze(pooled, 2)

# ---- MaxUnpool2d -------------------------------------------------------------

proc maxUnpool2d*(x: Tensor; indices: Tensor;
    outputSize: array[2, int]): Tensor =
  ## Max unpooling: scatters `x` values into a zero tensor of shape
  ## `[N, outputSize[0], outputSize[1], C]` at positions given by
  ## `indices`. `indices` has the same shape as `x` and contains flat
  ## (row-major) indices into the output spatial grid.
  ## Duplicate indices are accumulated, matching scatter-add semantics.
  requireSameMode(x, indices, "maxUnpool2d")
  requireSameDevice(x, indices, "maxUnpool2d")
  if x.shape != indices.shape:
    raise newException(TensorError,
      "maxUnpool2d: x and indices must have the same shape")
  if x.shape.len != 4:
    raise newException(TensorError,
      "maxUnpool2d: expected NHWC rank-4, got " & $x.shape)
  if not (indices.dtype.isSignedInt or indices.dtype.isUnsignedInt):
    raise newException(TensorError,
      "maxUnpool2d: indices must be integer, got " & $indices.dtype)
  if outputSize[0] <= 0 or outputSize[1] <= 0:
    raise newException(TensorError,
      "maxUnpool2d: output size must be positive")

  let n = x.shape[0]
  let c = x.shape[3]
  let hOut = outputSize[0]
  let wOut = outputSize[1]
  var rows: seq[Tensor] = @[]
  for oh in 0 ..< hOut:
    var cols: seq[Tensor] = @[]
    for ow in 0 ..< wOut:
      let flatIndex = oh * wOut + ow
      let target = full(indices.shape, float32(flatIndex), indices.dtype,
        indices.device)
      let mask = compare(indices, target, "EQ")
      let selected = select(mask, x, zerosLike(x))
      cols.add reshape(reduceSum(selected, @[1, 2]), [n, 1, 1, c])
    rows.add concat(cols, 2)
  concat(rows, 1)
