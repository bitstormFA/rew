## Constant-tensor dispatch ops \u2014 the trace-mode entry point for inline
## literals.
##
## In trace mode literals emit `stablehlo.constant`. In eager mode they are
## copied to the requested device as ordinary device tensors.

import ../tensor
import ../dispatch
import ../dtype
import ../device
import ../stablehlo/ops as shops
from ../eager import transferToDevice

proc shapeElementCount(shape: openArray[int]; opName: string): int =
  result = 1
  for d in shape:
    if d < 0:
      raise newException(TensorError,
        opName & ": shape contains negative dimension " & $d)
    result *= d

proc shapeDims(shape: openArray[int]): seq[int64] =
  result = newSeq[int64](shape.len)
  for i, d in shape:
    result[i] = int64(d)

proc validateConstantBytes(dtype: DType; shape: openArray[int];
    dataLen: int; opName: string): int =
  let n = shapeElementCount(shape, opName)
  result = n * dtype.byteSize
  if dataLen != result:
    raise newException(TensorError,
      opName & ": data length " & $dataLen &
        " byte(s) does not match shape product " & $n &
        " * dtype size " & $dtype.byteSize & " for " & dtype.name)

proc f32Bytes*(xs: openArray[float32]): seq[byte] =
  ## Little-endian raw bytes of `xs`. Exposed so callers (notably nn-init
  ## helpers and the SGD constructor) can hand a buffer straight to
  ## `constant`.
  result = newSeq[byte](xs.len * 4)
  for i, v in xs:
    let bits = cast[uint32](v)
    result[i*4 + 0] = byte(bits and 0xFF'u32)
    result[i*4 + 1] = byte((bits shr 8) and 0xFF'u32)
    result[i*4 + 2] = byte((bits shr 16) and 0xFF'u32)
    result[i*4 + 3] = byte((bits shr 24) and 0xFF'u32)

proc constant*(dtype: DType; shape: openArray[int];
    data: openArray[byte]; device: Device = defaultDevice()): Tensor =
  ## Literal tensor with raw little-endian element bytes.
  ##
  ## Trace mode emits `stablehlo.constant` in the active trace. Eager mode
  ## copies the bytes to `device`. `data.len` must equal
  ## `prod(shape) * dtype.byteSize`.
  let byteLen = validateConstantBytes(dtype, shape, data.len, "constant")
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = ctx.builder.constant(dtype, shape, data)
    result = initTraceTensor(id, dtype, shape, ctx.device)
  of dmEager:
    var local = @data
    let ptrIn = if local.len == 0: nil else: addr local[0]
    let h = transferToDevice(device, ptrIn, dtype, shapeDims(shape), byteLen)
    result = initEagerTensor(h, dtype, shape, device)

proc scalarF32*(v: float32; device: Device = defaultDevice()): Tensor =
  ## 0-d `float32` constant tensor.
  constant(dtFloat32, [], f32Bytes([v]), device)

proc constantF32*(shape: openArray[int]; data: openArray[float32];
    device: Device = defaultDevice()): Tensor =
  ## Float32 constant tensor of the given `shape`. `data.len` must equal
  ## the product of `shape`.
  var n = 1
  for d in shape: n *= d
  if data.len != n:
    raise newException(TensorError,
      "constantF32: data length " & $data.len &
        " does not match shape product " & $n)
  constant(dtFloat32, shape, f32Bytes(data), device)

proc scalarBool*(v: bool; device: Device = defaultDevice()): Tensor =
  ## 0-d `bool` constant tensor. Used to seed `cond` in tests and as a
  ## general predicate literal in trace mode.
  let b = if v: 1'u8 else: 0'u8
  constant(dtBool, [], @[byte(b)], device)

proc i32Bytes*(xs: openArray[int32]): seq[byte] =
  ## Little-endian raw bytes of `xs`.
  result = newSeq[byte](xs.len * 4)
  for i, v in xs:
    let bits = cast[uint32](v)
    result[i*4 + 0] = byte(bits and 0xFF'u32)
    result[i*4 + 1] = byte((bits shr 8) and 0xFF'u32)
    result[i*4 + 2] = byte((bits shr 16) and 0xFF'u32)
    result[i*4 + 3] = byte((bits shr 24) and 0xFF'u32)

proc rawLeBytes(bits: uint64; width: int): seq[byte] =
  result = newSeq[byte](width)
  for i in 0 ..< width:
    result[i] = byte((bits shr (8 * i)) and 0xFF'u64)

proc f32ToBfloat16Bits(v: float32): uint16 =
  var bits = cast[uint32](v)
  let roundBias = 0x7FFF'u32 + ((bits shr 16) and 1'u32)
  bits += roundBias
  uint16(bits shr 16)

proc f32ToFloat16Bits(v: float32): uint16 =
  let bits = cast[uint32](v)
  let sign = (bits shr 16) and 0x8000'u32
  let exp = int((bits shr 23) and 0xFF'u32)
  let mant = bits and 0x7FFFFF'u32
  if exp == 0xFF:
    let payload = if mant == 0'u32: 0x7C00'u32 else: 0x7E00'u32
    return uint16(sign or payload)

  var halfExp = exp - 127 + 15
  if halfExp >= 31:
    return uint16(sign or 0x7C00'u32)
  if halfExp <= 0:
    if halfExp < -10:
      return uint16(sign)
    let m = mant or 0x800000'u32
    let shift = 14 - halfExp
    var halfMant = m shr shift
    let roundBit = (m shr (shift - 1)) and 1'u32
    let sticky = m and ((1'u32 shl (shift - 1)) - 1'u32)
    if roundBit != 0'u32 and (sticky != 0'u32 or
        (halfMant and 1'u32) != 0'u32):
      inc halfMant
    return uint16(sign or halfMant)

  var halfMant = mant shr 13
  let roundBit = (mant shr 12) and 1'u32
  let sticky = mant and 0xFFF'u32
  if roundBit != 0'u32 and (sticky != 0'u32 or
      (halfMant and 1'u32) != 0'u32):
    inc halfMant
    if halfMant == 0x400'u32:
      halfMant = 0
      inc halfExp
      if halfExp >= 31:
        return uint16(sign or 0x7C00'u32)
  uint16(sign or (uint32(halfExp) shl 10) or halfMant)

proc scalarBytes*(dtype: DType; value: float32): seq[byte] =
  ## Raw little-endian bytes for a scalar literal of `dtype`.
  case dtype
  of dtBool:
    return @[byte(value != 0'f32)]
  of dtInt4, dtInt8:
    return rawLeBytes(uint64(cast[uint8](int8(value))), 1)
  of dtInt16:
    return rawLeBytes(uint64(cast[uint16](int16(value))), 2)
  of dtInt32:
    return rawLeBytes(uint64(cast[uint32](int32(value))), 4)
  of dtInt64:
    return rawLeBytes(cast[uint64](int64(value)), 8)
  of dtUint4, dtUint8, dtNF4, dtFloat8E4M3Fn, dtFloat8E5M2:
    return rawLeBytes(uint64(uint8(value)), 1)
  of dtUint16:
    return rawLeBytes(uint64(uint16(value)), 2)
  of dtUint32:
    return rawLeBytes(uint64(uint32(value)), 4)
  of dtUint64:
    return rawLeBytes(uint64(value), 8)
  of dtFloat16:
    return rawLeBytes(uint64(f32ToFloat16Bits(value)), 2)
  of dtBFloat16:
    return rawLeBytes(uint64(f32ToBfloat16Bits(value)), 2)
  of dtFloat32:
    return f32Bytes([value])
  of dtFloat64:
    return rawLeBytes(cast[uint64](float64(value)), 8)
  of dtComplex64:
    return f32Bytes([value, 0'f32])
  of dtComplex128:
    result = rawLeBytes(cast[uint64](float64(value)), 8)
    result.add rawLeBytes(cast[uint64](0.0), 8)

proc scalarI32*(v: int32; device: Device = defaultDevice()): Tensor =
  ## 0-d `int32` constant tensor. Used by `fori` to seed loop counters
  ## in trace mode.
  constant(dtInt32, [], i32Bytes([v]), device)
