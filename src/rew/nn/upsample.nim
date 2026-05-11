## Upsample layer — spatial upsampling for vision models.
##
## Thin wrapper over `interpolate` / `upsampleNearest2d`.

import ../tensor
import ../ops/interpolate

type
  Upsample* = object
    ## Spatial upsampling layer. Supports nearest-neighbor and bilinear
    ## modes for 2-D inputs (NHWC).
    scaleFactor*: array[2, int]
    mode*: string

proc initUpsample*(scaleFactor: array[2, int];
    mode = "nearest"): Upsample =
  ## Constructs an upsampling layer. `mode` is "nearest" or "bilinear".
  if scaleFactor[0] <= 0 or scaleFactor[1] <= 0:
    raise newException(TensorError,
      "initUpsample: scale factors must be positive")
  if mode notin ["nearest", "bilinear"]:
    raise newException(TensorError,
      "initUpsample: mode must be 'nearest' or 'bilinear', got '" & mode & "'")
  Upsample(scaleFactor: scaleFactor, mode: mode)

proc forward*(layer: Upsample; x: Tensor): Tensor =
  ## Upsamples `x` by `scaleFactor`. For NHWC input, outputs
  ## `[N, H*sH, W*sW, C]`.
  if x.shape.len != 4:
    raise newException(TensorError,
      "Upsample.forward: expected NHWC rank-4, got " & $x.shape)
  case layer.mode:
  of "nearest":
    upsampleNearest2d(x, layer.scaleFactor)
  of "bilinear":
    let hOut = x.shape[1] * layer.scaleFactor[0]
    let wOut = x.shape[2] * layer.scaleFactor[1]
    interpolate(x, [hOut, wOut], ipBilinear)
  else:
    raise newException(TensorError,
      "Upsample.forward: unknown mode '" & layer.mode & "'")
