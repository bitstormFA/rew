## Random sampling and initialization helpers for nn layers.
##
## These bridge the rng layer (which deliberately knows nothing about
## tensors) and the constant op. Bytes are generated host-side via
## Threefry, then handed to `constant` so the resulting `Tensor` plays
## nicely with the dispatcher.

import std/math
import ../device
import ../eager
import ../rng
import ../tensor

func u32ToUnitFloat(x: uint32): float32 {.inline.} =
  ## Maps a `uint32` uniformly to `[0, 1)` with 24 bits of mantissa
  ## resolution.
  float32(x shr 8) * (1'f32 / 16777216'f32)

proc uniformF32*(key: Key; count: int; lo, hi: float32): seq[float32] =
  ## Generates `count` independent samples from `Uniform(lo, hi)` keyed
  ## off `key`. Each sample uses a `foldIn(key, i)` so callers do not
  ## need to pre-split.
  result = newSeq[float32](count)
  let scale = hi - lo
  for i in 0 ..< count:
    let k = foldIn(key, uint64(i))
    result[i] = lo + scale * u32ToUnitFloat(k.a)

proc normalF32*(key: Key; count: int; mean: float32 = 0'f32;
    std: float32 = 1'f32): seq[float32] =
  ## Generates `count` samples from `Normal(mean, std)` using
  ## Box-Muller transform on pairs of uniform samples.
  result = newSeq[float32](count)
  let pairs = (count + 1) div 2
  for i in 0 ..< pairs:
    let k1 = foldIn(key, uint64(2 * i))
    let k2 = foldIn(key, uint64(2 * i + 1))
    let u1 = u32ToUnitFloat(k1.a)
    let u2 = u32ToUnitFloat(k2.a)
    # Avoid log(0) by clamping u1 away from zero.
    let safeU1 = max(u1, 1e-10'f32)
    let r = sqrt(-2'f32 * ln(safeU1))
    let theta = 2'f32 * PI.float32 * u2
    let z0 = r * cos(theta)
    let z1 = r * sin(theta)
    result[2 * i] = mean + std * z0
    if 2 * i + 1 < count:
      result[2 * i + 1] = mean + std * z1

proc zerosF32*(count: int): seq[float32] =
  ## Returns `count` zero-valued float32 samples.
  newSeq[float32](count)

proc onesF32*(count: int): seq[float32] =
  ## Returns `count` one-valued float32 samples.
  result = newSeq[float32](count)
  for i in 0 ..< count: result[i] = 1'f32

proc shapeElementCount(shape: openArray[int]; opName: string): int =
  result = 1
  for dim in shape:
    if dim < 0:
      raise newException(TensorError,
        opName & ": shape contains negative dimension " & $dim)
    result *= dim

proc uniformF32*(d: Device; key: Key; shape: openArray[int];
    lo: float32 = 0'f32; hi: float32 = 1'f32): Tensor =
  ## Generates a keyed uniform-random `float32` tensor on `d`.
  ## Randomness is host-side and explicit: the caller owns the `Key` and
  ## decides how to split or fold it before calling.
  let count = shapeElementCount(shape, "uniformF32")
  let data = uniformF32(key, count, lo, hi)
  fromHostF32(d, data, shape)

proc normalF32*(d: Device; key: Key; shape: openArray[int];
    mean: float32 = 0'f32; std: float32 = 1'f32): Tensor =
  ## Generates a keyed normal-random `float32` tensor on `d`.
  let count = shapeElementCount(shape, "normalF32")
  let data = normalF32(key, count, mean, std)
  fromHostF32(d, data, shape)

proc zerosF32*(d: Device; shape: openArray[int]): Tensor =
  ## Creates a zero-filled `float32` tensor on `d`.
  let data = zerosF32(shapeElementCount(shape, "zerosF32"))
  fromHostF32(d, data, shape)

proc onesF32*(d: Device; shape: openArray[int]): Tensor =
  ## Creates a one-filled `float32` tensor on `d`.
  let data = onesF32(shapeElementCount(shape, "onesF32"))
  fromHostF32(d, data, shape)

proc xavierUniformF32*(key: Key; fanIn, fanOut: int): seq[float32] =
  ## Xavier/Glorot uniform initialization: `U(-bound, bound)` where
  ## `bound = sqrt(6 / (fanIn + fanOut))`.
  let bound = sqrt(6'f32 / float32(fanIn + fanOut))
  uniformF32(key, fanIn * fanOut, -bound, bound)

proc xavierNormalF32*(key: Key; fanIn, fanOut: int): seq[float32] =
  ## Xavier/Glorot normal initialization: `N(0, std)` where
  ## `std = sqrt(2 / (fanIn + fanOut))`.
  let std = sqrt(2'f32 / float32(fanIn + fanOut))
  normalF32(key, fanIn * fanOut, 0'f32, std)

proc kaimingUniformF32*(key: Key; fanIn: int; count: int): seq[float32] =
  ## Kaiming/He uniform initialization: `U(-bound, bound)` where
  ## `bound = sqrt(3 / fanIn)`. Appropriate for ReLU activations.
  let bound = sqrt(3'f32 / float32(fanIn))
  uniformF32(key, count, -bound, bound)

proc kaimingNormalF32*(key: Key; fanIn: int; count: int): seq[float32] =
  ## Kaiming/He normal initialization: `N(0, std)` where
  ## `std = sqrt(2 / fanIn)`. Appropriate for ReLU activations.
  let std = sqrt(2'f32 / float32(fanIn))
  normalF32(key, count, 0'f32, std)

proc orthogonalF32*(key: Key; rows, cols: int): seq[float32] =
  ## Orthogonal matrix initialization via QR decomposition of a random matrix.
  ## Returns a flat `seq[float32]` of length `rows * cols` in row-major order.
  ## For `rows < cols`, the matrix has orthogonal rows.
  if rows <= 0 or cols <= 0:
    raise newException(TensorError,
      "orthogonalF32: rows and cols must be positive")
  let count = rows * cols
  var a = normalF32(key, count, 0'f32, 1'f32)
  # Modified Gram-Schmidt orthogonalization on the row vectors.
  for i in 0 ..< rows:
    # Normalize row i
    var norm = 0'f32
    for j in 0 ..< cols:
      norm += a[i * cols + j] * a[i * cols + j]
    let invNorm = 1'f32 / sqrt(max(norm, 1e-10'f32))
    for j in 0 ..< cols:
      a[i * cols + j] *= invNorm
    # Orthogonalize subsequent rows against row i
    for k in i + 1 ..< rows:
      var dot = 0'f32
      for j in 0 ..< cols:
        dot += a[i * cols + j] * a[k * cols + j]
      for j in 0 ..< cols:
        a[k * cols + j] -= dot * a[i * cols + j]
  result = a

proc orthogonalF32*(d: Device; key: Key; shape: openArray[int]): Tensor =
  ## Creates an orthogonally-initialized `float32` tensor on `d`.
  ## For a 2-D tensor shape `[rows, cols]`, each row is orthogonal.
  if shape.len != 2:
    raise newException(TensorError,
      "orthogonalF32: tensor must be 2-D (got rank " & $shape.len & ")")
  let rows = shape[0]
  let cols = shape[1]
  let data = orthogonalF32(key, rows, cols)
  fromHostF32(d, data, shape)

proc truncatedNormalF32*(key: Key; count: int; mean: float32 = 0'f32;
    std: float32 = 1'f32; bound: float32 = 2'f32): seq[float32] =
  ## Truncated normal: samples from N(mean, std) clamped to
  ## `[mean - bound*std, mean + bound*std]`. Resamples out-of-range values
  ## by rejection.
  result = newSeq[float32](count)
  let lo = mean - bound * std
  let hi = mean + bound * std
  for i in 0 ..< count:
    var sample: float32
    while true:
      let k = foldIn(key, uint64(i))
      # Use Box-Muller with foldIn-based keys
      let k1 = foldIn(k, 0'u64)
      let k2 = foldIn(k, 1'u64)
      let u1 = u32ToUnitFloat(k1.a)
      let u2 = u32ToUnitFloat(k2.a)
      let safeU1 = max(u1, 1e-10'f32)
      let r = sqrt(-2'f32 * ln(safeU1))
      let theta = 2'f32 * PI.float32 * u2
      sample = mean + std * r * cos(theta)
      if sample >= lo and sample <= hi:
        break
    result[i] = sample

proc constantF32*(value: float32; shape: openArray[int]): seq[float32] =
  ## Returns a flat `seq[float32]` filled with `value`.
  var count = 1
  for d in shape:
    if d < 0:
      raise newException(TensorError,
        "constantF32: shape contains negative dimension " & $d)
    count *= d
  result = newSeq[float32](count)
  for i in 0 ..< count:
    result[i] = value

proc truncatedNormalF32*(d: Device; key: Key; shape: openArray[int];
    mean: float32 = 0'f32; std: float32 = 1'f32;
    bound: float32 = 2'f32): Tensor =
  ## Creates a truncated-normal `float32` tensor on `d`.
  let count = shapeElementCount(shape, "truncatedNormalF32")
  let data = truncatedNormalF32(key, count, mean, std, bound)
  fromHostF32(d, data, shape)
