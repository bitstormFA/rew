## Tensor-creation convenience functions — `zeros`, `ones`, `full`, `eye`,
## `arange`, `linspace`.
##
## These are composites over existing primitives. No dedicated VJP rules
## are needed — autodiff decomposes through the primitives automatically.

import ../tensor
import ../dtype
import ../device
import ./literal
import ./linalg
import ./shape
import ./unary
import ./compare
import ./arith

proc zeros*(shape: openArray[int]; dtype: DType = dtFloat32;
    device: Device = defaultDevice()): Tensor =
  ## Return a tensor filled with zeros.
  let scalar = constant(dtype, [], scalarBytes(dtype, 0'f32), device)
  if shape.len == 0:
    return scalar
  var bdims: seq[int] = @[]
  broadcastTo(scalar, @shape, bdims)

proc zerosLike*(t: Tensor; dtype: DType = t.dtype): Tensor =
  ## Return a tensor of zeros with the same shape and device as `t`.
  zeros(t.shape, dtype, t.device)

proc ones*(shape: openArray[int]; dtype: DType = dtFloat32;
    device: Device = defaultDevice()): Tensor =
  ## Return a tensor filled with ones.
  let scalar = constant(dtype, [], scalarBytes(dtype, 1'f32), device)
  if shape.len == 0:
    return scalar
  var bdims: seq[int] = @[]
  broadcastTo(scalar, @shape, bdims)

proc onesLike*(t: Tensor; dtype: DType = t.dtype): Tensor =
  ## Return a tensor of ones with the same shape and device as `t`.
  ones(t.shape, dtype, t.device)

proc full*(shape: openArray[int]; value: float32;
    dtype: DType = dtFloat32; device: Device = defaultDevice()): Tensor =
  ## Return a tensor filled with `value`.
  let scalar = constant(dtype, [], scalarBytes(dtype, value), device)
  if shape.len == 0:
    return scalar
  var bdims: seq[int] = @[]
  broadcastTo(scalar, @shape, bdims)

proc fullLike*(t: Tensor; value: float32; dtype: DType = t.dtype): Tensor =
  ## Return a tensor filled with `value` with the same shape and device as `t`.
  full(t.shape, value, dtype, t.device)

proc eye*(n: int; m: int = 0; dtype: DType = dtFloat32;
    device: Device = defaultDevice()): Tensor =
  ## Return a 2-D tensor with ones on the diagonal and zeros elsewhere.
  let cols = if m == 0: n else: m
  let rows = iota(dtInt32, [n, cols], 0, device)
  let colsIota = iota(dtInt32, [n, cols], 1, device)
  let mask = compare(rows, colsIota, "EQ")
  astype(mask, dtype)

proc arange*(start, stop: int; step = 1; dtype: DType = dtInt32;
    device: Device = defaultDevice()): Tensor =
  ## Return a 1-D tensor with values from `start` to `stop` (exclusive)
  ## stepping by `step`.
  if step == 0:
    raise newException(TensorError, "arange: step must not be zero")
  if (step > 0 and start >= stop) or (step < 0 and start <= stop):
    raise newException(TensorError,
      "arange: invalid range [" & $start & " ..< " & $stop & "] with step " &
        $step)
  let n = (stop - start + step - (if step > 0: 1 else: -1)) div step
  let indices = iota(dtInt32, [n], 0, device)
  if step != 1:
    let scaled = mul(indices,
      constant(dtInt32, [], i32Bytes([int32(step)]), device))
    if start != 0:
      let offset = constant(dtInt32, [], i32Bytes([int32(start)]), device)
      result = add(scaled, offset)
    else:
      result = scaled
  elif start != 0:
    let offset = constant(dtInt32, [], i32Bytes([int32(start)]), device)
    result = add(indices, offset)
  else:
    result = indices
  if dtype != dtInt32:
    result = astype(result, dtype)

proc arange*(stop: int; dtype: DType = dtInt32;
    device: Device = defaultDevice()): Tensor =
  ## Return a 1-D tensor with values `0 ..< stop`.
  arange(0, stop, 1, dtype, device)

proc linspace*(start, stop: float32; n: int; dtype: DType = dtFloat32;
    device: Device = defaultDevice()): Tensor =
  ## Return a 1-D tensor of `n` evenly-spaced values from `start` to
  ## `stop` (inclusive).
  if n < 1:
    raise newException(TensorError, "linspace: n must be >= 1")
  if n == 1:
    return full([1], start, dtype, device)
  let indices = arange(0, n, 1, dtFloat32, device)
  let step = (stop - start) / float32(n - 1)
  let stepScalar = scalarF32(step, device)
  var bdims: seq[int] = @[]
  let stepBcast = broadcastTo(stepScalar, [n], bdims)
  let scaled = mul(indices, stepBcast)
  let startScalar = scalarF32(start, device)
  let startBcast = broadcastTo(startScalar, [n], bdims)
  add(scaled, startBcast)
