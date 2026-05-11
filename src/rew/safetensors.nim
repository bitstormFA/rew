## safetensors format reader and writer.
##
## The safetensors format is the standard weight serialization for
## HuggingFace models. It consists of an 8-byte little-endian header
## length prefix, a JSON metadata header, and concatenated raw tensor
## data.
##
## Reference: https://github.com/huggingface/safetensors

import std/[json, os, streams, tables]
import ./dtype

type
  SafeTensorError* = object of CatchableError
    ## Raised on malformed safetensors data.

  SafeTensorInfo* = object
    ## Per-tensor metadata.
    dtype*: DType
    shape*: seq[int]
    dataOffsets*: tuple[start, `end`: int]  ## byte offsets into the data section

  SafeTensorFile* = object
    ## Parsed safetensors file metadata.
    tensors*: Table[string, SafeTensorInfo]
    rawData*: string  ## raw concatenated tensor bytes

  SafeTensorMetadata* = object
    ## Metadata-only view of a safetensors file.
    path*: string
    dataStart*: int
    dataLen*: int
    tensors*: Table[string, SafeTensorInfo]

# ---- dtype mapping ----------------------------------------------------------

func dtypeOfSafeTensor(dt: string): DType =
  case dt
  of "F32": dtFloat32
  of "F16": dtFloat16
  of "BF16": dtBFloat16
  of "F64": dtFloat64
  of "I8": dtInt8
  of "I16": dtInt16
  of "I32": dtInt32
  of "I64": dtInt64
  of "U8": dtUint8
  of "U16": dtUint16
  of "U32": dtUint32
  of "U64": dtUint64
  of "BOOL": dtBool
  of "F8_E4M3FN": dtFloat8E4M3Fn
  of "F8_E5M2": dtFloat8E5M2
  else:
    raise newException(SafeTensorError,
      "unsupported safetensors dtype: " & dt)

func safeTensorDtypeName(dt: DType): string =
  case dt
  of dtFloat32: "F32"
  of dtFloat16: "F16"
  of dtBFloat16: "BF16"
  of dtFloat64: "F64"
  of dtInt8: "I8"
  of dtInt16: "I16"
  of dtInt32: "I32"
  of dtInt64: "I64"
  of dtUint8: "U8"
  of dtUint16: "U16"
  of dtUint32: "U32"
  of dtUint64: "U64"
  of dtBool: "BOOL"
  of dtFloat8E4M3Fn: "F8_E4M3FN"
  of dtFloat8E5M2: "F8_E5M2"
  else:
    raise newException(SafeTensorError,
      "cannot serialize " & dt.name & " to safetensors")

# ---- parser ----------------------------------------------------------------

proc le64At(data: string; pos: int): uint64 =
  for i in 0 ..< 8:
    result = result or (uint64(ord(data[pos + i])) shl (i * 8))

proc parseHeader(headerJson: string; dataLen: int): SafeTensorFile =
  let node = parseJson(headerJson)
  if node.kind != JObject:
    raise newException(SafeTensorError,
      "safetensors header is not a JSON object")
  for name, info in node.pairs:
    if info.kind != JObject:
      raise newException(SafeTensorError,
        "safetensors tensor '" & name & "' entry is not an object")
    var stInfo: SafeTensorInfo
    if not info.hasKey("dtype") or info["dtype"].kind != JString:
      raise newException(SafeTensorError,
        "safetensors tensor '" & name & "' missing dtype")
    stInfo.dtype = dtypeOfSafeTensor(info["dtype"].getStr())
    if not info.hasKey("shape") or info["shape"].kind != JArray:
      raise newException(SafeTensorError,
        "safetensors tensor '" & name & "' missing shape")
    for dim in info["shape"]:
      stInfo.shape.add dim.getInt()
    if not info.hasKey("data_offsets") or info["data_offsets"].kind != JArray:
      raise newException(SafeTensorError,
        "safetensors tensor '" & name & "' missing data_offsets")
    let offs = info["data_offsets"]
    stInfo.dataOffsets = (offs[0].getInt(), offs[1].getInt())
    if stInfo.dataOffsets.start < 0 or stInfo.dataOffsets.`end` > dataLen or
       stInfo.dataOffsets.start > stInfo.dataOffsets.`end`:
      raise newException(SafeTensorError,
        "safetensors tensor '" & name & "' has invalid data_offsets")
    result.tensors[name] = stInfo

# ---- file I/O ---------------------------------------------------------------

proc loadSafeTensors*(s: Stream): SafeTensorFile =
  ## Reads a safetensors file from `s`.
  let headerLen = le64At(s.readStr(8), 0)
  if headerLen > 100_000_000'u64:
    raise newException(SafeTensorError,
      "safetensors header too large: " & $headerLen)
  let headerJson = s.readStr(int(headerLen))
  let rawData = s.readAll()
  result = parseHeader(headerJson, rawData.len)
  result.rawData = rawData

proc loadSafeTensors*(path: string): SafeTensorFile =
  ## Reads a safetensors file from `path`.
  let s = newFileStream(path, fmRead)
  if s.isNil:
    raise newException(IOError, "loadSafeTensors: cannot open '" & path & "'")
  defer: s.close()
  loadSafeTensors(s)

proc loadSafeTensorMetadata*(path: string): SafeTensorMetadata =
  ## Reads only the safetensors header from `path`.
  let s = newFileStream(path, fmRead)
  if s.isNil:
    raise newException(IOError,
      "loadSafeTensorMetadata: cannot open '" & path & "'")
  defer: s.close()
  let prefix = s.readStr(8)
  if prefix.len != 8:
    raise newException(SafeTensorError,
      "safetensors file is too short for a header prefix")
  let headerLen = le64At(prefix, 0)
  if headerLen > 100_000_000'u64:
    raise newException(SafeTensorError,
      "safetensors header too large: " & $headerLen)
  let headerJson = s.readStr(int(headerLen))
  if headerJson.len != int(headerLen):
    raise newException(SafeTensorError,
      "safetensors file ended inside the header")
  let dataStart = 8 + int(headerLen)
  let fileLen = getFileSize(path).int
  if fileLen < dataStart:
    raise newException(SafeTensorError,
      "safetensors file ended before the data section")
  let parsed = parseHeader(headerJson, fileLen - dataStart)
  SafeTensorMetadata(
    path: path,
    dataStart: dataStart,
    dataLen: fileLen - dataStart,
    tensors: parsed.tensors,
  )

proc hasTensor*(st: SafeTensorFile; name: string): bool =
  ## Returns true when `name` exists in the loaded safetensors file.
  st.tensors.hasKey(name)

proc hasTensor*(meta: SafeTensorMetadata; name: string): bool =
  ## Returns true when `name` exists in the metadata header.
  meta.tensors.hasKey(name)

proc tensorData*(st: SafeTensorFile; name: string): seq[byte] =
  ## Extracts the raw bytes for a tensor by name.
  if not st.tensors.hasKey(name):
    raise newException(SafeTensorError, "tensor '" & name & "' not found")
  let info = st.tensors[name]
  let start = info.dataOffsets.start
  let len = info.dataOffsets.`end` - start
  result = newSeq[byte](len)
  if len > 0:
    copyMem(addr result[0], addr st.rawData[start], len)

proc tensorData*(meta: SafeTensorMetadata; name: string): seq[byte] =
  ## Reads a tensor byte slice by name without loading the whole file.
  if not meta.tensors.hasKey(name):
    raise newException(SafeTensorError, "tensor '" & name & "' not found")
  let info = meta.tensors[name]
  let len = info.dataOffsets.`end` - info.dataOffsets.start
  result = newSeq[byte](len)
  if len == 0:
    return
  let s = newFileStream(meta.path, fmRead)
  if s.isNil:
    raise newException(IOError,
      "tensorData: cannot open '" & meta.path & "'")
  defer: s.close()
  s.setPosition(meta.dataStart + info.dataOffsets.start)
  let readBytes = s.readData(addr result[0], len)
  if readBytes != len:
    raise newException(SafeTensorError,
      "tensor '" & name & "' ended before its data_offsets range")

iterator tensorPairs*(st: SafeTensorFile):
    tuple[name: string; dtype: DType; shape: seq[int]; data: seq[byte]] =
  ## Yields each tensor's name, dtype, shape, and raw bytes.
  for name, info in st.tensors:
    yield (name, info.dtype, info.shape, st.tensorData(name))

proc listTensors*(st: SafeTensorFile): seq[string] =
  ## Returns all tensor names in the file.
  for name in st.tensors.keys:
    result.add name

proc listTensors*(meta: SafeTensorMetadata): seq[string] =
  ## Returns all tensor names in a metadata-only file view.
  for name in meta.tensors.keys:
    result.add name

# ---- writer -----------------------------------------------------------------

proc saveSafeTensors*(s: Stream; tensors: Table[string,
    tuple[dtype: DType; shape: seq[int]; data: seq[byte]]]) =
  ## Writes tensors as a safetensors file to `s`.
  var header = newJObject()
  var offset = 0
  for name, tensor in tensors:
    let byteLen = tensor.data.len
    header[name] = %* {
      "dtype": safeTensorDtypeName(tensor.dtype),
      "shape": tensor.shape,
      "data_offsets": [offset, offset + byteLen],
    }
    offset += byteLen
  let headerStr = $header
  let headerLen = uint64(headerStr.len)
  # Write 8-byte little-endian header length.
  for i in 0 ..< 8:
    s.write char((headerLen shr (i * 8)) and 0xFF'u64)
  s.write(headerStr)
  for name, tensor in tensors:
    s.write(cast[string](tensor.data))

proc saveSafeTensors*(path: string; tensors: Table[string,
    tuple[dtype: DType; shape: seq[int]; data: seq[byte]]]) =
  ## Writes tensors as a safetensors file to `path`.
  let s = newFileStream(path, fmWrite)
  if s.isNil:
    raise newException(IOError, "saveSafeTensors: cannot open '" & path & "'")
  defer: s.close()
  saveSafeTensors(s, tensors)
