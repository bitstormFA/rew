## Interpolation / resizing ops for tensor data.
##
## Composite ops for spatial upsampling and downsampling. Nearest-neighbor,
## bilinear, and bicubic resizing are implemented as composites of reshape,
## broadcast, slice, concat, and arithmetic ops.

import std/math
import ../dtype
import ../tensor
import ../ops/shape
import ../ops/linalg
import ../ops/arith
import ../ops/literal
import ../ops/concat

type
  InterpolationMode* = enum
    ipNearest
    ipBilinear
    ipBicubic

proc nearestSourceIndex(outIndex, inSize, outSize: int): int =
  result = (outIndex * inSize) div outSize
  if result >= inSize:
    result = inSize - 1

proc requireResizeDim(x: Tensor; dim, outSize: int; opName: string) =
  if dim < 0 or dim >= x.shape.len:
    raise newException(TensorError,
      opName & ": dim " & $dim & " out of range for rank " &
        $x.shape.len)
  if x.shape[dim] <= 0:
    raise newException(TensorError,
      opName & ": input dim " & $dim & " must be positive")
  if outSize <= 0:
    raise newException(TensorError,
      opName & ": output size must be positive")

proc requireInterpolatedDType(x: Tensor; opName: string) =
  if not (x.dtype.isFloat or x.dtype.isComplex):
    raise newException(TensorError,
      opName & ": interpolation requires floating or complex dtype, got " &
        x.dtype.name)

func clampIndex(index, size: int): int =
  if index < 0:
    0
  elif index >= size:
    size - 1
  else:
    index

func sourceCoord(outIndex, inSize, outSize: int): float32 =
  float32(outIndex) * float32(inSize) / float32(outSize)

func linearSample(outIndex, inSize, outSize: int):
    tuple[lo, hi: int; frac: float32] =
  let src = sourceCoord(outIndex, inSize, outSize)
  let base = int(floor(src))
  result.lo = clampIndex(base, inSize)
  result.hi = clampIndex(base + 1, inSize)
  result.frac = src - float32(base)

func cubicWeight(x: float32): float32 =
  let ax = abs(x)
  if ax <= 1.0'f32:
    (1.5'f32 * ax - 2.5'f32) * ax * ax + 1.0'f32
  elif ax < 2.0'f32:
    ((-0.5'f32 * ax + 2.5'f32) * ax - 4.0'f32) * ax + 2.0'f32
  else:
    0.0'f32

proc sliceIndex(x: Tensor; dim, index: int): Tensor =
  var starts = newSeq[int](x.shape.len)
  var limits = x.shape
  var strides = newSeq[int](x.shape.len)
  for i in 0 ..< x.shape.len:
    strides[i] = 1
  starts[dim] = index
  limits[dim] = index + 1
  slice(x, starts, limits, strides)

proc scalarLike(x: Tensor; value: float32): Tensor =
  constant(x.dtype, [], scalarBytes(x.dtype, value), x.device)

proc scaleTensor(x: Tensor; weight: float32): Tensor =
  if weight == 1.0'f32:
    return x
  let scalar = scalarLike(x, weight)
  mul(broadcastTo(scalar, x.shape, @[]), x)

proc addWeightedTerm(acc: var Tensor; hasAcc: var bool; x: Tensor;
    weight: float32) =
  if weight != 0.0'f32:
    let term = scaleTensor(x, weight)
    if hasAcc:
      acc = add(acc, term)
    else:
      acc = term
      hasAcc = true

proc blendLinear(a, b: Tensor; frac: float32): Tensor =
  var acc: Tensor
  var hasAcc = false
  addWeightedTerm(acc, hasAcc, a, 1.0'f32 - frac)
  addWeightedTerm(acc, hasAcc, b, frac)
  if hasAcc:
    acc
  else:
    scaleTensor(a, 0.0'f32)

proc resizeNearestDim(x: Tensor; dim, outSize: int): Tensor =
  requireResizeDim(x, dim, outSize, "resizeNearestDim")
  let inSize = x.shape[dim]
  if outSize == inSize:
    return x
  var pieces: seq[Tensor] = @[]
  for outIndex in 0 ..< outSize:
    let srcIndex = nearestSourceIndex(outIndex, inSize, outSize)
    var starts = newSeq[int](x.shape.len)
    var limits = x.shape
    var strides = newSeq[int](x.shape.len)
    for i in 0 ..< x.shape.len:
      strides[i] = 1
    starts[dim] = srcIndex
    limits[dim] = srcIndex + 1
    pieces.add slice(x, starts, limits, strides)
  if pieces.len == 1:
    pieces[0]
  else:
    concat(pieces, dim)

proc resizeLinearDim(x: Tensor; dim, outSize: int): Tensor =
  requireResizeDim(x, dim, outSize, "resizeLinearDim")
  requireInterpolatedDType(x, "resizeLinearDim")
  let inSize = x.shape[dim]
  if outSize == inSize:
    return x
  var pieces: seq[Tensor] = @[]
  for outIndex in 0 ..< outSize:
    let (lo, hi, frac) = linearSample(outIndex, inSize, outSize)
    let loSlice = sliceIndex(x, dim, lo)
    if lo == hi or frac == 0.0'f32:
      pieces.add loSlice
    else:
      pieces.add blendLinear(loSlice, sliceIndex(x, dim, hi), frac)
  if pieces.len == 1:
    pieces[0]
  else:
    concat(pieces, dim)

proc resizeCubicDim(x: Tensor; dim, outSize: int): Tensor =
  requireResizeDim(x, dim, outSize, "resizeCubicDim")
  requireInterpolatedDType(x, "resizeCubicDim")
  let inSize = x.shape[dim]
  if outSize == inSize:
    return x
  var pieces: seq[Tensor] = @[]
  for outIndex in 0 ..< outSize:
    let src = sourceCoord(outIndex, inSize, outSize)
    let base = int(floor(src))
    let frac = src - float32(base)
    var acc: Tensor
    var hasAcc = false
    for offset in -1 .. 2:
      let index = clampIndex(base + offset, inSize)
      let weight = cubicWeight(float32(offset) - frac)
      addWeightedTerm(acc, hasAcc, sliceIndex(x, dim, index), weight)
    if hasAcc:
      pieces.add acc
    else:
      pieces.add scaleTensor(sliceIndex(x, dim, 0), 0.0'f32)
  if pieces.len == 1:
    pieces[0]
  else:
    concat(pieces, dim)

proc upsampleNearest2d*(x: Tensor; scaleFactor: array[2, int]): Tensor =
  ## Upsamples NHWC `x` by repeating each spatial element
  ## `scaleFactor[0]` times along H and `scaleFactor[1]` times along W.
  if x.shape.len != 4:
    raise newException(TensorError,
      "upsampleNearest2d: expected NHWC rank-4, got " & $x.shape)
  let h = x.shape[1]
  let w = x.shape[2]
  let c = x.shape[3]
  let sh = scaleFactor[0]
  let sw = scaleFactor[1]
  if sh <= 0 or sw <= 0:
    raise newException(TensorError,
      "upsampleNearest2d: scale factors must be positive")
  # Reshape to [N, H, 1, W, 1, C], broadcast to [N, H, sh, W, sw, C],
  # reshape to [N, H*sh, W*sw, C]
  let expanded = reshape(x, [x.shape[0], h, 1, w, 1, c])
  let bcast = broadcastTo(expanded, [x.shape[0], h, sh, w, sw, c],
    @[0, 1, 3, 5])
  reshape(bcast, [x.shape[0], h * sh, w * sw, c])

proc upsampleBilinear2d*(x: Tensor; scaleFactor: array[2, int]): Tensor =
  ## Bilinear upsample of NHWC `x` by integer `scaleFactor`.
  ##
  ## Computes bilinear interpolation between source pixels using
  ## a 4-corner decomposition: the input is padded by one pixel on
  ## the right and bottom (edge replication), four shifted views are
  ## extracted, blended with bilinear weights, and interleaved into
  ## the output via reshape + transpose (pixel-shuffle style).
  if x.shape.len != 4:
    raise newException(TensorError,
      "upsampleBilinear2d: expected NHWC rank-4, got " & $x.shape)
  requireInterpolatedDType(x, "upsampleBilinear2d")
  let sh = scaleFactor[0]
  let sw = scaleFactor[1]
  if sh <= 0 or sw <= 0:
    raise newException(TensorError,
      "upsampleBilinear2d: scale factors must be positive")
  let n = x.shape[0]
  let h = x.shape[1]
  let w = x.shape[2]
  let c = x.shape[3]
  # Pad right and bottom by replicating the last row/column.
  let lastCol = slice(x, @[0, 0, w - 1, 0], @[n, h, w, c],
    @[1, 1, 1, 1])
  let rightPadded = concat(@[x, lastCol], 2)
  let lastRow = slice(rightPadded, @[0, h - 1, 0, 0],
    @[n, h, w + 1, c], @[1, 1, 1, 1])
  let padded = concat(@[rightPadded, lastRow], 1)
  # Four corner views: [N, H, W, C] each.
  let v00 = slice(padded, @[0, 0, 0, 0], @[n, h, w, c],
    @[1, 1, 1, 1])
  let v01 = slice(padded, @[0, 0, 1, 0], @[n, h, w + 1, c],
    @[1, 1, 1, 1])
  let v10 = slice(padded, @[0, 1, 0, 0], @[n, h + 1, w, c],
    @[1, 1, 1, 1])
  let v11 = slice(padded, @[0, 1, 1, 0], @[n, h + 1, w + 1, c],
    @[1, 1, 1, 1])
  # Build the 4 bilinear outputs per source pixel.
  # For offsets (i, j) in 0..sh-1 × 0..sw-1, weights are:
  #   w00=(1-i/sh)*(1-j/sw), w01=(1-i/sh)*(j/sw),
  #   w10=(i/sh)*(1-j/sw),   w11=(i/sh)*(j/sw)
  # We build per-offset results and interleave via reshape/transpose.
  let vs = float32(sh)
  let ws = float32(sw)
  var tiles: seq[Tensor] = @[]
  for i in 0 ..< sh:
    let alpha = float32(i) / vs
    let omAlpha = 1'f32 - alpha
    for j in 0 ..< sw:
      let beta = float32(j) / ws
      let omBeta = 1'f32 - beta
      let w00 = scalarLike(x, omAlpha * omBeta)
      let w01 = scalarLike(x, omAlpha * beta)
      let w10 = scalarLike(x, alpha * omBeta)
      let w11 = scalarLike(x, alpha * beta)
      var tile: Tensor
      if alpha == 0'f32 and beta == 0'f32:
        tile = v00
      elif alpha == 0'f32 and beta != 0'f32:
        tile = add(mul(broadcastTo(w00, v00.shape, @[]), v00),
                   mul(broadcastTo(w01, v01.shape, @[]), v01))
      elif alpha != 0'f32 and beta == 0'f32:
        tile = add(mul(broadcastTo(w00, v00.shape, @[]), v00),
                   mul(broadcastTo(w10, v10.shape, @[]), v10))
      else:
        let t0 = add(mul(broadcastTo(w00, v00.shape, @[]), v00),
                     mul(broadcastTo(w01, v01.shape, @[]), v01))
        let t1 = add(mul(broadcastTo(w10, v10.shape, @[]), v10),
                     mul(broadcastTo(w11, v11.shape, @[]), v11))
        tile = add(t0, t1)
      tiles.add unsqueeze(tile, 3)
  # Interleave: concat along dim 3 gives [N, H, W, sh*sw, C].
  let combined = concat(tiles, 3)
  # Reshape to [N, H, W, sh, sw, C], transpose to [N, H, sh, W, sw, C].
  let reshaped = reshape(combined, @[n, h, w, sh, sw, c])
  let transposed = transpose(reshaped, @[0, 1, 3, 2, 4, 5])
  reshape(transposed, @[n, h * sh, w * sw, c])

proc interpolate*(x: Tensor; size: openArray[int];
    mode: InterpolationMode = ipNearest): Tensor =
  ## Resize `x` to `size`.
  ##
  ## `ipNearest` supports static 1-D resize along the last dimension and
  ## static 2-D resize of NHWC rank-4 tensors. `ipBilinear` and `ipBicubic`
  ## support static 1-D resize along the last dimension and static 2-D
  ## resize of NHWC rank-4 tensors for floating or complex dtypes.
  let ndim = size.len
  if ndim < 1 or ndim > 2:
    raise newException(TensorError,
      "interpolate: 1-D or 2-D resize only, got size " & $size)
  let rank = x.shape.len
  if rank < ndim:
    raise newException(TensorError,
      "interpolate: input rank " & $rank & " < resize dims " & $ndim)
  let outH = size[0]
  let outW = if ndim == 2: size[1] else: outH
  if outH <= 0 or outW <= 0:
    raise newException(TensorError,
      "interpolate: output sizes must be positive")
  case mode:
  of ipNearest:
    if ndim == 1:
      return resizeNearestDim(x, rank - 1, outH)
    if x.shape.len != 4:
      raise newException(TensorError,
        "interpolate: nearest 2-D resize requires NHWC rank-4, got " &
          $x.shape)
    let inH = x.shape[1]
    let inW = x.shape[2]
    if outH >= inH and outW >= inW and outH mod inH == 0 and
        outW mod inW == 0:
      let sH = outH div inH
      let sW = outW div inW
      return upsampleNearest2d(x, [sH, sW])
    return resizeNearestDim(resizeNearestDim(x, 1, outH), 2, outW)
  of ipBilinear:
    if ndim == 1:
      return resizeLinearDim(x, rank - 1, outH)
    if rank != 4:
      raise newException(TensorError,
        "interpolate: bilinear requires NHWC rank-4, got " & $x.shape)
    let inH = x.shape[1]
    let inW = x.shape[2]
    if outH >= inH and outW >= inW and outH mod inH == 0 and
        outW mod inW == 0:
      let sH = outH div inH
      let sW = outW div inW
      return upsampleBilinear2d(x, [sH, sW])
    return resizeLinearDim(resizeLinearDim(x, 1, outH), 2, outW)
  of ipBicubic:
    if ndim == 1:
      return resizeCubicDim(x, rank - 1, outH)
    if rank != 4:
      raise newException(TensorError,
        "interpolate: bicubic requires NHWC rank-4, got " & $x.shape)
    return resizeCubicDim(resizeCubicDim(x, 1, outH), 2, outW)
