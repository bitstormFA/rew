## fold / unfold — im2col and col2im operations.
##
## `unfold` extracts sliding local blocks from a batched input tensor
## (equivalent to `torch.nn.functional.unfold`).
## `fold` combines an array of sliding local blocks into a tensor
## (equivalent to `torch.nn.functional.fold`).
##
## Composite over reshape, slice, concat, pad, and elementwise addition.

import ../tensor
import ../ops/shape
import ../ops/concat
import ../ops/factory
import ../ops/arith

proc requirePositivePair(name: string; value: array[2, int]; opName: string) =
  if value[0] <= 0 or value[1] <= 0:
    raise newException(TensorError,
      opName & ": " & name & " must be positive, got " & $value)

proc requireNonNegativePair(name: string; value: array[2, int];
    opName: string) =
  if value[0] < 0 or value[1] < 0:
    raise newException(TensorError,
      opName & ": " & name & " must be non-negative, got " & $value)

proc windowOutputSize(inputSize, kernel, stride, padding, dilation: int): int =
  (inputSize + 2 * padding - dilation * (kernel - 1) - 1) div stride + 1

proc zeroLikeShape(t: Tensor; shape: openArray[int]): Tensor =
  zeros(shape, t.dtype, t.device)

proc zeroScalarLike(t: Tensor): Tensor =
  zeros([], t.dtype, t.device)

proc slicePixel(x: Tensor; h, w: int): Tensor =
  slice(x, [0, 0, h, w], [x.shape[0], x.shape[1], h + 1, w + 1],
    [1, 1, 1, 1])

proc padPixelToOutput(pixel: Tensor; h, w, outH, outW: int): Tensor =
  pad(pixel, zeroScalarLike(pixel), [0, 0, h, w],
    [0, 0, outH - h - 1, outW - w - 1], [0, 0, 0, 0])

proc unfold*(x: Tensor; kernelSize: array[2, int];
    stride: array[2, int] = [1, 1];
    padding: array[2, int] = [0, 0];
    dilation: array[2, int] = [1, 1]): Tensor =
  ## Extracts sliding local blocks from an NCHW input.
  ## `x` is `[N, C, H, W]`. Returns `[N, C*kH*kW, L]` where `L` is the
  ## number of patches.
  if x.shape.len != 4:
    raise newException(TensorError,
      "unfold: expected NCHW rank-4, got " & $x.shape)
  requirePositivePair("kernelSize", kernelSize, "unfold")
  requirePositivePair("stride", stride, "unfold")
  requireNonNegativePair("padding", padding, "unfold")
  requirePositivePair("dilation", dilation, "unfold")
  let n = x.shape[0]
  let c = x.shape[1]
  let h = x.shape[2]
  let w = x.shape[3]
  let kH = kernelSize[0]
  let kW = kernelSize[1]
  let sH = stride[0]
  let sW = stride[1]
  let pH = padding[0]
  let pW = padding[1]
  let dH = dilation[0]
  let dW = dilation[1]
  let hOut = windowOutputSize(h, kH, sH, pH, dH)
  let wOut = windowOutputSize(w, kW, sW, pW, dW)
  if hOut <= 0 or wOut <= 0:
    raise newException(TensorError,
      "unfold: kernel too large for input size")

  var patches: seq[Tensor] = @[]
  for hi in countup(0, hOut - 1):
    for wi in countup(0, wOut - 1):
      let hStart = hi * sH - pH
      let wStart = wi * sW - pW
      var rows: seq[Tensor] = @[]
      for kh in 0 ..< kH:
        var cols: seq[Tensor] = @[]
        let srcH = hStart + kh * dH
        for kw in 0 ..< kW:
          let srcW = wStart + kw * dW
          if srcH >= 0 and srcH < h and srcW >= 0 and srcW < w:
            cols.add slicePixel(x, srcH, srcW)
          else:
            cols.add zeroLikeShape(x, [n, c, 1, 1])
        rows.add concat(cols, 3)
      let patch = concat(rows, 2)
      patches.add reshape(patch, [n, c * kH * kW])

  let stacked = stack(patches, 2)
  reshape(stacked, [n, c * kH * kW, hOut * wOut])

proc fold*(x: Tensor; outputSize: array[2, int];
    kernelSize: array[2, int];
    stride: array[2, int] = [1, 1];
    padding: array[2, int] = [0, 0];
    dilation: array[2, int] = [1, 1]): Tensor =
  ## Combines an array of sliding local blocks into a tensor.
  ## `x` is `[N, C*kH*kW, L]`. Returns `[N, C, H_out, W_out]`.
  ## Inverse of `unfold`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "fold: expected rank-3 [N, C*kH*kW, L], got " & $x.shape)
  requirePositivePair("outputSize", outputSize, "fold")
  requirePositivePair("kernelSize", kernelSize, "fold")
  requirePositivePair("stride", stride, "fold")
  requireNonNegativePair("padding", padding, "fold")
  requirePositivePair("dilation", dilation, "fold")

  let n = x.shape[0]
  let h = outputSize[0]
  let w = outputSize[1]
  let kH = kernelSize[0]
  let kW = kernelSize[1]
  let patchSize = kH * kW
  if x.shape[1] mod patchSize != 0:
    raise newException(TensorError,
      "fold: channel dimension " & $x.shape[1] &
        " is not divisible by kernel area " & $patchSize)
  let c = x.shape[1] div patchSize
  let hOut = windowOutputSize(h, kH, stride[0], padding[0], dilation[0])
  let wOut = windowOutputSize(w, kW, stride[1], padding[1], dilation[1])
  if hOut <= 0 or wOut <= 0:
    raise newException(TensorError,
      "fold: kernel too large for output size")
  let expectedL = hOut * wOut
  if x.shape[2] != expectedL:
    raise newException(TensorError,
      "fold: patch count " & $x.shape[2] &
        " does not match expected " & $expectedL)

  var acc = zeroLikeShape(x, [n, c, h, w])
  var patchIndex = 0
  for hi in 0 ..< hOut:
    for wi in 0 ..< wOut:
      let patch = reshape(slice(x, [0, 0, patchIndex],
        [n, x.shape[1], patchIndex + 1], [1, 1, 1]), [n, c, kH, kW])
      let hStart = hi * stride[0] - padding[0]
      let wStart = wi * stride[1] - padding[1]
      for kh in 0 ..< kH:
        let dstH = hStart + kh * dilation[0]
        if dstH >= 0 and dstH < h:
          for kw in 0 ..< kW:
            let dstW = wStart + kw * dilation[1]
            if dstW >= 0 and dstW < w:
              let pixel = slicePixel(patch, kh, kw)
              acc = add(acc, padPixelToOutput(pixel, dstH, dstW, h, w))
      inc patchIndex
  acc
