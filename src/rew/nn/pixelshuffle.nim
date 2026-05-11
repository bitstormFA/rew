## PixelShuffle / PixelUnshuffle — spatial-channel rearrangement.
##
## PixelShuffle rearranges a `[N, C*r^2, H, W]` tensor to `[N, C, H*r, W*r]`
## (NCHW layout). PixelUnshuffle is the inverse.

import ../tensor
import ../ops/shape

proc pixelShuffle*(x: Tensor; upscaleFactor: int): Tensor =
  ## Rearranges elements from `[N, C*r^2, H, W]` to `[N, C, H*r, W*r]`.
  ## NCHW layout (standard for PixelShuffle in all frameworks).
  if x.shape.len != 4:
    raise newException(TensorError,
      "pixelShuffle: expected NCHW rank-4, got " & $x.shape)
  let r = upscaleFactor
  if r <= 1:
    raise newException(TensorError,
      "pixelShuffle: upscale factor must be >= 2, got " & $r)
  let n = x.shape[0]
  let cr2 = x.shape[1]
  let h = x.shape[2]
  let w = x.shape[3]
  if cr2 mod (r * r) != 0:
    raise newException(TensorError,
      "pixelShuffle: channels " & $cr2 &
        " must be divisible by r^2 = " & $(r * r))
  let c = cr2 div (r * r)
  # Reshape: [N, C, r, r, H, W]
  let reshaped = reshape(x, [n, c, r, r, h, w])
  # Transpose: [N, C, H, r, W, r]
  let transposed = transpose(reshaped, [0, 1, 4, 2, 5, 3])
  # Reshape: [N, C, H*r, W*r]
  reshape(transposed, [n, c, h * r, w * r])

proc pixelUnshuffle*(x: Tensor; downscaleFactor: int): Tensor =
  ## Rearranges elements from `[N, C, H*r, W*r]` to `[N, C*r^2, H, W]`.
  ## NCHW layout. Inverse of pixelShuffle.
  if x.shape.len != 4:
    raise newException(TensorError,
      "pixelUnshuffle: expected NCHW rank-4, got " & $x.shape)
  let r = downscaleFactor
  if r <= 1:
    raise newException(TensorError,
      "pixelUnshuffle: downscale factor must be >= 2, got " & $r)
  let n = x.shape[0]
  let c = x.shape[1]
  let hR = x.shape[2]
  let wR = x.shape[3]
  if hR mod r != 0 or wR mod r != 0:
    raise newException(TensorError,
      "pixelUnshuffle: spatial dims " & $hR & "x" & $wR &
        " must be divisible by " & $r)
  let h = hR div r
  let w = wR div r
  # Reshape: [N, C, H, r, W, r]
  let reshaped = reshape(x, [n, c, h, r, w, r])
  # Transpose: [N, C, r, r, H, W]
  let transposed = transpose(reshaped, [0, 1, 3, 5, 2, 4])
  # Reshape: [N, C*r^2, H, W]
  reshape(transposed, [n, c * r * r, h, w])
