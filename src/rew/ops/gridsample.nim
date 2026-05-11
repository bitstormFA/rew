## grid_sample — spatial sampling with normalized coordinates.
##
## Samples an NCHW image at arbitrary spatial locations given by a grid.
## The grid specifies normalized coordinates in [-1, 1] for each output
## pixel. Supports bilinear interpolation and zero-padding fill modes.

import ../tensor
import ../dtype
import ./arith
import ./compare
import ./concat
import ./factory
import ./linalg
import ./literal
import ./shape
import ./ternary
import ./unary

type
  GridSampleMode* = enum
    gsBilinear
    gsNearest

proc scalarLike(t: Tensor; value: float32): Tensor =
  full([], value, t.dtype, t.device)

proc gridScalar(grid: Tensor; n, h, w, component: int): Tensor =
  reshape(slice(grid, [n, h, w, component],
    [n + 1, h + 1, w + 1, component + 1], [1, 1, 1, 1]), [])

proc normalizedToIndex(coord: Tensor; size: int; alignCorners: bool): Tensor =
  let shifted = add(coord, scalarLike(coord, 1'f32))
  if alignCorners:
    mul(shifted, scalarLike(coord, 0.5'f32 * float32(size - 1)))
  else:
    sub(mul(shifted, scalarLike(coord, 0.5'f32 * float32(size))),
      scalarLike(coord, 0.5'f32))

proc clampIndex(index: Tensor; size: int): Tensor =
  clamp(scalarI32(0'i32, index.device), index,
    scalarI32(int32(size - 1), index.device))

proc validIndex(index: Tensor; size: int): Tensor =
  bitwiseAnd(
    compare(index, scalarI32(0'i32, index.device), "GE"),
    compare(index, scalarI32(int32(size), index.device), "LT"))

proc validPair(hIndex, wIndex: Tensor; height, width: int): Tensor =
  bitwiseAnd(validIndex(hIndex, height), validIndex(wIndex, width))

proc broadcastScalar(t: Tensor; shape: openArray[int]): Tensor =
  var dims: seq[int] = @[]
  broadcastTo(t, shape, dims)

proc maskedSample(x: Tensor; n: int; hIndex, wIndex, valid: Tensor): Tensor =
  let sample = dynamicSlice(x, [
    scalarI32(int32(n), x.device),
    scalarI32(0'i32, x.device),
    clampIndex(hIndex, x.shape[2]),
    clampIndex(wIndex, x.shape[3]),
  ], [1, x.shape[1], 1, 1])
  let mask = broadcastScalar(valid, sample.shape)
  select(mask, sample, zerosLike(sample))

proc scaleSample(sample, weight: Tensor): Tensor =
  mul(sample, broadcastScalar(weight, sample.shape))

proc gridSample*(x: Tensor; grid: Tensor;
    mode: GridSampleMode = gsBilinear; alignCorners = false): Tensor =
  ## Samples `x` (NCHW, `[N, C, H, W]`) at locations specified by `grid`
  ## (`[N, H_out, W_out, 2]`). Grid coordinates are normalized to [-1, 1].
  ##
  ## `gsNearest` selects the nearest valid source pixel. `gsBilinear`
  ## performs zero-padded bilinear interpolation and requires `x` and
  ## `grid` to have the same floating dtype. With `alignCorners = false`,
  ## coordinates follow the common half-pixel convention; with
  ## `alignCorners = true`, `-1` and `1` map exactly to the first and last
  ## source pixels.
  if x.shape.len != 4:
    raise newException(TensorError,
      "gridSample: input must be NCHW rank-4, got " & $x.shape)
  if grid.shape.len != 4 or grid.shape[3] != 2:
    raise newException(TensorError,
      "gridSample: grid must be [N, H, W, 2], got " & $grid.shape)
  requireSameMode(x, grid, "gridSample")
  requireSameDevice(x, grid, "gridSample")
  if grid.dtype.isFloat == false:
    raise newException(TensorError,
      "gridSample: grid dtype must be floating, got " & grid.dtype.name)
  if grid.shape[0] != x.shape[0]:
    raise newException(TensorError,
      "gridSample: grid batch " & $grid.shape[0] &
        " does not match input batch " & $x.shape[0])
  if x.shape[2] <= 0 or x.shape[3] <= 0 or
      grid.shape[1] <= 0 or grid.shape[2] <= 0:
    raise newException(TensorError,
      "gridSample: spatial dimensions must be positive")
  if mode == gsBilinear and (x.dtype != grid.dtype or not x.dtype.isFloat):
    raise newException(TensorError,
      "gridSample: bilinear mode requires input and grid to share a " &
        "floating dtype")

  let nBatch = x.shape[0]
  let hOut = grid.shape[1]
  let wOut = grid.shape[2]
  let inH = x.shape[2]
  let inW = x.shape[3]
  var batches: seq[Tensor] = @[]
  for n in 0 ..< nBatch:
    var rows: seq[Tensor] = @[]
    for oh in 0 ..< hOut:
      var cols: seq[Tensor] = @[]
      for ow in 0 ..< wOut:
        let gx = gridScalar(grid, n, oh, ow, 0)
        let gy = gridScalar(grid, n, oh, ow, 1)
        let xCoord = normalizedToIndex(gx, inW, alignCorners)
        let yCoord = normalizedToIndex(gy, inH, alignCorners)
        case mode
        of gsNearest:
          let ix = astype(floor(add(xCoord, scalarLike(xCoord, 0.5'f32))),
            dtInt32)
          let iy = astype(floor(add(yCoord, scalarLike(yCoord, 0.5'f32))),
            dtInt32)
          cols.add maskedSample(x, n, iy, ix, validPair(iy, ix, inH, inW))
        of gsBilinear:
          let x0f = floor(xCoord)
          let y0f = floor(yCoord)
          let x1f = add(x0f, scalarLike(x0f, 1'f32))
          let y1f = add(y0f, scalarLike(y0f, 1'f32))
          let x0 = astype(x0f, dtInt32)
          let x1 = astype(x1f, dtInt32)
          let y0 = astype(y0f, dtInt32)
          let y1 = astype(y1f, dtInt32)
          let wx0 = sub(x1f, xCoord)
          let wx1 = sub(xCoord, x0f)
          let wy0 = sub(y1f, yCoord)
          let wy1 = sub(yCoord, y0f)
          let s00 = maskedSample(x, n, y0, x0, validPair(y0, x0, inH, inW))
          let s01 = maskedSample(x, n, y0, x1, validPair(y0, x1, inH, inW))
          let s10 = maskedSample(x, n, y1, x0, validPair(y1, x0, inH, inW))
          let s11 = maskedSample(x, n, y1, x1, validPair(y1, x1, inH, inW))
          let top = add(scaleSample(s00, mul(wy0, wx0)),
            scaleSample(s01, mul(wy0, wx1)))
          let bottom = add(scaleSample(s10, mul(wy1, wx0)),
            scaleSample(s11, mul(wy1, wx1)))
          cols.add add(top, bottom)
      rows.add concat(cols, 3)
    batches.add concat(rows, 2)
  concat(batches, 0)
