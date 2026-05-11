## GGUF format reader and writer.
##
## GGUF is the standard format for llama.cpp and many quantized LLMs.
## Format: magic bytes "GGUF" + version + tensor count + metadata KV pairs
## + tensor infos + padding + tensor data.
##
## Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

import std/[streams, endians]
import ./dtype

type
  GgufError* = object of CatchableError

  GgufValueType* = enum
    gvtUint8 = 0,    gvtInt8  = 1,    gvtUint16 = 2,   gvtInt16  = 3,
    gvtUint32 = 4,   gvtInt32 = 5,    gvtFloat32 = 6,  gvtBool    = 7,
    gvtString = 8,   gvtArray = 9,    gvtUint64 = 10,  gvtInt64  = 11,
    gvtFloat64 = 12

  GgufMetadataValue* = object
    case kind*: GgufValueType
    of gvtUint8:    u8*: uint8
    of gvtInt8:     i8*: int8
    of gvtUint16:   u16*: uint16
    of gvtInt16:    i16*: int16
    of gvtUint32:   u32*: uint32
    of gvtInt32:    i32*: int32
    of gvtFloat32:  f32*: float32
    of gvtBool:     b*: bool
    of gvtString:   s*: string
    of gvtArray:    arr*: seq[GgufMetadataValue]
    of gvtUint64:   u64*: uint64
    of gvtInt64:    i64*: int64
    of gvtFloat64:  f64*: float64

  GgufTensorInfo* = object
    name*: string
    nDims*: uint32
    dims*: seq[uint64]
    dtype*: DType
    offset*: uint64

  GgufHeader* = object
    version*: uint32
    tensorCount*: uint64
    metadataKvCount*: uint64
    metadata*: seq[tuple[key: string; value: GgufMetadataValue]]
    tensors*: seq[GgufTensorInfo]
    alignment*: int
    dataOffset*: uint64

const GgufMagic = 0x46554747'u32  ## "GGUF" in little-endian

func ggufTypeToDtype(t: uint32): DType =
  case t
  of 0: dtFloat32
  of 1: dtFloat16
  of 19: dtBFloat16
  else: dtInt8  ## quantized → stored as int8

# ---- reader -----------------------------------------------------------------

proc readLe32(s: Stream): uint32 =
  var buf: array[4, byte]
  if s.readData(addr buf[0], 4) != 4:
    raise newException(GgufError, "truncated GGUF file")
  littleEndian32(addr result, addr buf[0])

proc readLe64(s: Stream): uint64 =
  var buf: array[8, byte]
  if s.readData(addr buf[0], 8) != 8:
    raise newException(GgufError, "truncated GGUF file")
  littleEndian64(addr result, addr buf[0])

proc readString(s: Stream): string =
  let len = readLe64(s)
  if len > 10_000_000'u64:
    raise newException(GgufError, "GGUF string too long: " & $len)
  result = s.readStr(int(len))

proc readValue(s: Stream; vt: GgufValueType): GgufMetadataValue =
  case vt
  of gvtUint8:   GgufMetadataValue(kind: gvtUint8, u8: s.readUint8())
  of gvtInt8:    GgufMetadataValue(kind: gvtInt8, i8: int8(s.readInt8()))
  of gvtUint16:  GgufMetadataValue(kind: gvtUint16, u16: uint16(readLe32(s)))
  of gvtInt16:   GgufMetadataValue(kind: gvtInt16, i16: int16(readLe32(s)))
  of gvtUint32:  GgufMetadataValue(kind: gvtUint32, u32: readLe32(s))
  of gvtInt32:   GgufMetadataValue(kind: gvtInt32, i32: int32(readLe32(s)))
  of gvtFloat32:
    var v: float32
    let raw = readLe32(s)
    copyMem(addr v, addr raw, 4)
    GgufMetadataValue(kind: gvtFloat32, f32: v)
  of gvtBool:    GgufMetadataValue(kind: gvtBool, b: s.readUint8() != 0)
  of gvtString:  GgufMetadataValue(kind: gvtString, s: s.readString())
  of gvtArray:
    let arrType = GgufValueType(s.readUint32())
    let arrLen = readLe64(s)
    var arr = newSeq[GgufMetadataValue](int(arrLen))
    for i in 0 ..< int(arrLen):
      arr[i] = s.readValue(arrType)
    GgufMetadataValue(kind: gvtArray, arr: arr)
  of gvtUint64:  GgufMetadataValue(kind: gvtUint64, u64: readLe64(s))
  of gvtInt64:   GgufMetadataValue(kind: gvtInt64, i64: int64(readLe64(s)))
  of gvtFloat64:
    var v: float64
    var buf: array[8, byte]
    if s.readData(addr buf[0], 8) != 8:
      raise newException(GgufError, "truncated GGUF float64")
    littleEndian64(addr v, addr buf[0])
    GgufMetadataValue(kind: gvtFloat64, f64: v)

proc loadGguf*(s: Stream): GgufHeader =
  let magic = readLe32(s)
  if magic != GgufMagic:
    raise newException(GgufError, "not a GGUF file (bad magic)")
  result.version = readLe32(s)
  result.tensorCount = readLe64(s)
  result.metadataKvCount = readLe64(s)
  # Read metadata KV pairs.
  for i in 0 ..< result.metadataKvCount:
    var key = s.readString()
    let vt = GgufValueType(s.readUint32())
    let value = s.readValue(vt)
    result.metadata.add((key, value))
  # Read tensor infos.
  for i in 0 ..< result.tensorCount:
    var ti: GgufTensorInfo
    ti.name = s.readString()
    ti.nDims = readLe32(s)
    ti.dims = newSeq[uint64](ti.nDims)
    for j in 0 ..< ti.nDims:
      ti.dims[j] = readLe64(s)
    let typeVal = readLe32(s)
    ti.dtype = ggufTypeToDtype(typeVal)
    ti.offset = readLe64(s)
    result.tensors.add ti
  # Align to boundary (default 32 bytes).
  result.alignment = 32
  result.dataOffset = uint64(s.getPosition())
  # Align position.
  let pos = s.getPosition()
  let pad = (result.alignment - (pos mod result.alignment)) mod result.alignment
  if pad > 0:
    discard s.readStr(pad)

proc loadGguf*(path: string): GgufHeader =
  let s = newFileStream(path, fmRead)
  if s.isNil:
    raise newException(IOError, "loadGguf: cannot open '" & path & "'")
  defer: s.close()
  loadGguf(s)

# ---- writer -----------------------------------------------------------------

proc writeLe32(s: Stream; value: uint32) =
  var buf = value
  var outVal: uint32
  littleEndian32(addr outVal, addr buf)
  s.writeData(addr outVal, 4)

proc writeLe64(s: Stream; value: uint64) =
  var buf = value
  var outVal: uint64
  littleEndian64(addr outVal, addr buf)
  s.writeData(addr outVal, 8)

proc saveGguf*(s: Stream; header: GgufHeader; tensorData: openArray[seq[byte]]) =
  writeLe32(s, GgufMagic)
  writeLe32(s, header.version)
  writeLe64(s, uint64(header.tensors.len))
  writeLe64(s, uint64(header.metadata.len))
  # Write metadata.
  for (key, value) in header.metadata:
    s.write(key)
    s.write(key.len.uint64)
    s.write(ord(value.kind).uint32)
    discard  ## full metadata write deferred
  # Write tensor infos.
  for i, ti in header.tensors:
    s.write(ti.name)
    s.write(ti.name.len.uint64)
    writeLe32(s, ti.nDims)
    for dim in ti.dims:
      writeLe64(s, dim)
    writeLe32(s, 0'u32)  ## dtype = F32
    writeLe64(s, 0'u64)  ## offset placeholder
  # Write tensor data.
  for data in tensorData:
    s.write(cast[string](data))

proc saveGguf*(path: string; header: GgufHeader;
    tensorData: openArray[seq[byte]]) =
  let s = newFileStream(path, fmWrite)
  if s.isNil:
    raise newException(IOError, "saveGguf: cannot open '" & path & "'")
  defer: s.close()
  saveGguf(s, header, tensorData)
