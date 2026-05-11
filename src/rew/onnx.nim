## ONNX model import/export.
##
## This module implements the small protobuf surface needed to read and
## write ONNX `ModelProto` files without adding a runtime dependency on
## Python, protoc, or a protobuf package. The public representation is a
## plain value tree (`OnnxModel` -> `OnnxGraph` -> nodes/tensors), plus
## conversion helpers for the StableHLO subset rew can represent safely.

import std/[sets, streams, tables]
import ./dtype
import ./stablehlo/ir
import ./stablehlo/ops as shops

type
  OnnxError* = object of CatchableError
    ## Raised for malformed ONNX protobuf data or unsupported ONNX
    ## features during conversion to/from rew StableHLO.

  OnnxAttributeKind* = enum
    ## Supported ONNX attribute payload kinds.
    oakUndefined
    oakFloat
    oakInt
    oakString
    oakTensor
    oakFloats
    oakInts
    oakStrings

  OnnxTensor* = object
    ## Tensor initializer or tensor-valued attribute.
    ##
    ## `data` stores raw little-endian element bytes in row-major order.
    name*: string
    dtype*: DType
    shape*: seq[int]
    data*: seq[byte]

  OnnxAttribute* = object
    ## A named ONNX node attribute.
    name*: string
    case kind*: OnnxAttributeKind
    of oakUndefined:
      discard
    of oakFloat:
      f*: float32
    of oakInt:
      i*: int64
    of oakString:
      s*: string
    of oakTensor:
      t*: OnnxTensor
    of oakFloats:
      floats*: seq[float32]
    of oakInts:
      ints*: seq[int64]
    of oakStrings:
      strings*: seq[string]

  OnnxValueInfo* = object
    ## Name, dtype, and shape metadata for a graph value.
    ##
    ## Static dimensions are stored in `shape`. Unknown or symbolic
    ## dimensions are stored as `-1`; when the symbolic name is known,
    ## the same index in `dimParams` stores it. `toShModule` requires
    ## all dimensions to be static.
    name*: string
    dtype*: DType
    shape*: seq[int]
    dimParams*: seq[string]
    hasType*: bool

  OnnxNode* = object
    ## One ONNX graph node.
    opType*: string
    inputs*: seq[string]
    outputs*: seq[string]
    name*: string
    domain*: string
    attributes*: seq[OnnxAttribute]

  OnnxGraph* = object
    ## A top-level ONNX computation graph.
    name*: string
    docString*: string
    inputs*: seq[OnnxValueInfo]
    outputs*: seq[OnnxValueInfo]
    valueInfo*: seq[OnnxValueInfo]
    initializers*: seq[OnnxTensor]
    nodes*: seq[OnnxNode]

  OnnxOpset* = object
    ## Imported ONNX operator-set version for a domain.
    domain*: string
    version*: int64

  OnnxModel* = object
    ## Top-level ONNX model container.
    irVersion*: int64
    producerName*: string
    producerVersion*: string
    domain*: string
    modelVersion*: int64
    docString*: string
    opsets*: seq[OnnxOpset]
    graph*: OnnxGraph

const
  OnnxDefaultIrVersion* = 9'i64
    ## Default IR version used by `initOnnxModel`.
  OnnxDefaultOpsetVersion* = 18'i64
    ## Default standard-domain opset used by `initOnnxModel`.

  AttrFloat = 1
  AttrInt = 2
  AttrString = 3
  AttrTensor = 4
  AttrFloats = 6
  AttrInts = 7
  AttrStrings = 8

# ---- dtype mapping --------------------------------------------------------

func toOnnxDataType*(dt: DType): int =
  ## Returns the ONNX `TensorProto.DataType` enum value for `dt`.
  case dt
  of dtFloat32: 1
  of dtUint8: 2
  of dtInt8: 3
  of dtUint16: 4
  of dtInt16: 5
  of dtInt32: 6
  of dtInt64: 7
  of dtBool: 9
  of dtFloat16: 10
  of dtFloat64: 11
  of dtUint32: 12
  of dtUint64: 13
  of dtComplex64: 14
  of dtComplex128: 15
  of dtBFloat16: 16
  of dtInt4, dtUint4, dtNF4, dtFloat8E4M3Fn, dtFloat8E5M2:
    raise newException(OnnxError,
      "ONNX dtype '" & dt.name & "' is not supported")

func dtypeOfOnnxDataType*(dataType: int): DType =
  ## Converts an ONNX `TensorProto.DataType` enum value to `DType`.
  case dataType
  of 1: dtFloat32
  of 2: dtUint8
  of 3: dtInt8
  of 4: dtUint16
  of 5: dtInt16
  of 6: dtInt32
  of 7: dtInt64
  of 9: dtBool
  of 10: dtFloat16
  of 11: dtFloat64
  of 12: dtUint32
  of 13: dtUint64
  of 14: dtComplex64
  of 15: dtComplex128
  of 16: dtBFloat16
  else:
    raise newException(OnnxError,
      "ONNX dtype " & $dataType & " is not supported by rew")

func elementCount(shape: openArray[int]): int =
  result = 1
  for d in shape:
    if d < 0:
      raise newException(OnnxError,
        "ONNX tensor has non-static dimension " & $d)
    result *= d

proc validateTensorBytes(dtype: DType; shape: openArray[int];
    dataLen: int; context: string) =
  let expected = elementCount(shape) * dtype.byteSize
  if dataLen != expected:
    raise newException(OnnxError,
      context & ": data length " & $dataLen &
        " does not match dtype/shape (" & $expected & ")")

# ---- public constructors --------------------------------------------------

func initOnnxTensor*(name: string; dtype: DType; shape: openArray[int];
    data: sink seq[byte]): OnnxTensor =
  ## Builds an ONNX tensor from raw little-endian bytes.
  validateTensorBytes(dtype, shape, data.len, "initOnnxTensor")
  OnnxTensor(name: name, dtype: dtype, shape: @shape, data: data)

func initOnnxValueInfo*(name: string; dtype: DType;
    shape: openArray[int]): OnnxValueInfo =
  ## Builds static tensor metadata for a graph input, output, or value.
  OnnxValueInfo(name: name, dtype: dtype, shape: @shape,
    dimParams: newSeq[string](shape.len), hasType: true)

func initOnnxAttribute*(name: string; value: float32): OnnxAttribute =
  ## Builds a float attribute.
  OnnxAttribute(name: name, kind: oakFloat, f: value)

func initOnnxAttribute*(name: string; value: int64): OnnxAttribute =
  ## Builds an integer attribute.
  OnnxAttribute(name: name, kind: oakInt, i: value)

func initOnnxAttribute*(name: string; value: string): OnnxAttribute =
  ## Builds a byte/string attribute.
  OnnxAttribute(name: name, kind: oakString, s: value)

func initOnnxAttribute*(name: string; value: OnnxTensor): OnnxAttribute =
  ## Builds a tensor-valued attribute.
  OnnxAttribute(name: name, kind: oakTensor, t: value)

func initOnnxAttribute*(name: string;
    values: openArray[float32]): OnnxAttribute =
  ## Builds a repeated-float attribute.
  OnnxAttribute(name: name, kind: oakFloats, floats: @values)

func initOnnxAttribute*(name: string;
    values: openArray[int64]): OnnxAttribute =
  ## Builds a repeated-integer attribute.
  OnnxAttribute(name: name, kind: oakInts, ints: @values)

func initOnnxAttribute*(name: string;
    values: openArray[string]): OnnxAttribute =
  ## Builds a repeated-string attribute.
  OnnxAttribute(name: name, kind: oakStrings, strings: @values)

func initOnnxNode*(opType: string; inputs, outputs: openArray[string];
    name = ""; attributes: openArray[OnnxAttribute] = [];
    domain = ""): OnnxNode =
  ## Builds an ONNX graph node.
  OnnxNode(opType: opType, inputs: @inputs, outputs: @outputs, name: name,
    domain: domain, attributes: @attributes)

func initOnnxGraph*(name: string; inputs: openArray[OnnxValueInfo] = [];
    outputs: openArray[OnnxValueInfo] = [];
    nodes: openArray[OnnxNode] = [];
    initializers: openArray[OnnxTensor] = []): OnnxGraph =
  ## Builds an ONNX graph.
  OnnxGraph(name: name, inputs: @inputs, outputs: @outputs, nodes: @nodes,
    initializers: @initializers)

func initOnnxModel*(graph: OnnxGraph;
    opsetVersion: int64 = OnnxDefaultOpsetVersion): OnnxModel =
  ## Builds a minimal ONNX model around `graph`.
  OnnxModel(
    irVersion: OnnxDefaultIrVersion,
    producerName: "rew",
    opsets: @[OnnxOpset(domain: "", version: opsetVersion)],
    graph: graph,
  )

# ---- byte helpers ---------------------------------------------------------

proc bytesToString(data: openArray[byte]): string =
  result = newString(data.len)
  for i, b in data:
    result[i] = char(b)

proc stringToBytes(data: string): seq[byte] =
  result = newSeq[byte](data.len)
  for i, ch in data:
    result[i] = byte(ord(ch))

proc appendLe16(dst: var seq[byte]; value: uint64) =
  dst.add byte(value and 0xff'u64)
  dst.add byte((value shr 8) and 0xff'u64)

proc appendLe32(dst: var seq[byte]; value: uint64) =
  for shift in [0, 8, 16, 24]:
    dst.add byte((value shr shift) and 0xff'u64)

proc appendLe64(dst: var seq[byte]; value: uint64) =
  for shift in [0, 8, 16, 24, 32, 40, 48, 56]:
    dst.add byte((value shr shift) and 0xff'u64)

proc le32(data: string; pos: int): uint32 =
  uint32(ord(data[pos])) or
    (uint32(ord(data[pos + 1])) shl 8) or
    (uint32(ord(data[pos + 2])) shl 16) or
    (uint32(ord(data[pos + 3])) shl 24)

proc le64(data: string; pos: int): uint64 =
  result = 0
  for i in 0 ..< 8:
    result = result or (uint64(ord(data[pos + i])) shl (i * 8))

proc le32(data: openArray[byte]; pos: int): uint32 =
  uint32(data[pos]) or
    (uint32(data[pos + 1]) shl 8) or
    (uint32(data[pos + 2]) shl 16) or
    (uint32(data[pos + 3]) shl 24)

proc le64(data: openArray[byte]; pos: int): uint64 =
  result = 0
  for i in 0 ..< 8:
    result = result or (uint64(data[pos + i]) shl (i * 8))

proc f32Bytes(value: float32): seq[byte] =
  result = @[]
  appendLe32(result, uint64(cast[uint32](value)))

proc f64Bytes(value: float64): seq[byte] =
  result = @[]
  appendLe64(result, cast[uint64](value))

proc int64Bytes(values: openArray[int64]): seq[byte] =
  result = @[]
  for value in values:
    appendLe64(result, cast[uint64](value))

proc int64Values(t: OnnxTensor; context: string): seq[int64] =
  if t.dtype notin {dtInt64, dtInt32}:
    raise newException(OnnxError,
      context & ": expected int64/int32 shape tensor, got " & t.dtype.name)
  let width = t.dtype.byteSize
  if t.data.len mod width != 0:
    raise newException(OnnxError,
      context & ": malformed integer tensor byte length")
  for offset in countup(0, t.data.len - width, width):
    if t.dtype == dtInt64:
      result.add cast[int64](le64(t.data, offset))
    else:
      result.add int64(cast[int32](le32(t.data, offset)))

# ---- protobuf reader ------------------------------------------------------

proc pbError(context: string): ref OnnxError =
  newException(OnnxError, "loadOnnx: " & context)

proc readVarint(data: string; pos: var int; context: string): uint64 =
  var shift = 0
  while true:
    if pos >= data.len:
      raise pbError(context & ": truncated varint")
    let b = uint64(ord(data[pos]))
    inc pos
    result = result or ((b and 0x7f'u64) shl shift)
    if (b and 0x80'u64) == 0:
      break
    shift += 7
    if shift > 63:
      raise pbError(context & ": varint too long")

proc readFixed32(data: string; pos: var int; context: string): uint32 =
  if pos + 4 > data.len:
    raise pbError(context & ": truncated fixed32")
  result = le32(data, pos)
  pos += 4

proc readFixed64(data: string; pos: var int; context: string): uint64 =
  if pos + 8 > data.len:
    raise pbError(context & ": truncated fixed64")
  result = le64(data, pos)
  pos += 8

proc readBytes(data: string; pos: var int; context: string): string =
  let n = int(readVarint(data, pos, context & ": length"))
  if n < 0 or pos + n > data.len:
    raise pbError(context & ": truncated length-delimited field")
  result = data[pos ..< pos + n]
  pos += n

proc nextField(data: string; pos: var int;
    context: string): tuple[number, wire: int] =
  let tag = readVarint(data, pos, context & ": tag")
  result.number = int(tag shr 3)
  result.wire = int(tag and 0x7'u64)
  if result.number <= 0:
    raise pbError(context & ": invalid field number " & $result.number)

proc skipField(data: string; pos: var int; wire: int; context: string) =
  case wire
  of 0:
    discard readVarint(data, pos, context)
  of 1:
    if pos + 8 > data.len: raise pbError(context & ": truncated fixed64")
    pos += 8
  of 2:
    discard readBytes(data, pos, context)
  of 5:
    if pos + 4 > data.len: raise pbError(context & ": truncated fixed32")
    pos += 4
  else:
    raise pbError(context & ": unsupported protobuf wire type " & $wire)

proc requireWire(wire, expected: int; context: string) =
  if wire != expected:
    raise pbError(context & ": expected wire type " & $expected &
      ", got " & $wire)

proc readRepeatedVarints(data: string; wire: int; pos: var int;
    context: string): seq[uint64] =
  case wire
  of 0:
    result.add readVarint(data, pos, context)
  of 2:
    let payload = readBytes(data, pos, context)
    var p = 0
    while p < payload.len:
      result.add readVarint(payload, p, context & ": packed")
  else:
    raise pbError(context & ": expected varint or packed varint")

proc readRepeatedFixed32(data: string; wire: int; pos: var int;
    context: string): seq[uint32] =
  case wire
  of 5:
    result.add readFixed32(data, pos, context)
  of 2:
    let payload = readBytes(data, pos, context)
    if payload.len mod 4 != 0:
      raise pbError(context & ": packed fixed32 length is not a multiple of 4")
    var p = 0
    while p < payload.len:
      result.add readFixed32(payload, p, context & ": packed")
  else:
    raise pbError(context & ": expected fixed32 or packed fixed32")

proc readRepeatedFixed64(data: string; wire: int; pos: var int;
    context: string): seq[uint64] =
  case wire
  of 1:
    result.add readFixed64(data, pos, context)
  of 2:
    let payload = readBytes(data, pos, context)
    if payload.len mod 8 != 0:
      raise pbError(context & ": packed fixed64 length is not a multiple of 8")
    var p = 0
    while p < payload.len:
      result.add readFixed64(payload, p, context & ": packed")
  else:
    raise pbError(context & ": expected fixed64 or packed fixed64")

proc parseTensor(data: string): OnnxTensor
proc parseAttribute(data: string): OnnxAttribute
proc parseGraph(data: string): OnnxGraph

proc tensorDataFromInt32(values: openArray[int32]; dtype: DType): seq[byte] =
  for value in values:
    let bits = uint64(cast[uint32](value))
    case dtype
    of dtBool, dtInt8, dtUint8:
      result.add byte(bits and 0xff'u64)
    of dtInt16, dtUint16, dtFloat16, dtBFloat16:
      appendLe16(result, bits)
    of dtInt32, dtUint32:
      appendLe32(result, bits)
    else:
      raise newException(OnnxError,
        "loadOnnx: int32_data cannot represent dtype " & dtype.name)

proc tensorDataFromInt64(values: openArray[int64]; dtype: DType): seq[byte] =
  if dtype != dtInt64:
    raise newException(OnnxError,
      "loadOnnx: int64_data cannot represent dtype " & dtype.name)
  for value in values:
    appendLe64(result, cast[uint64](value))

proc tensorDataFromUint64(values: openArray[uint64]; dtype: DType): seq[byte] =
  case dtype
  of dtUint32:
    for value in values: appendLe32(result, value)
  of dtUint64:
    for value in values: appendLe64(result, value)
  else:
    raise newException(OnnxError,
      "loadOnnx: uint64_data cannot represent dtype " & dtype.name)

proc parseTensor(data: string): OnnxTensor =
  var pos = 0
  var dataType = 0
  var rawData: seq[byte] = @[]
  var hasRawData = false
  var floatBytesBuf: seq[byte] = @[]
  var doubleBytesBuf: seq[byte] = @[]
  var int32Vals: seq[int32] = @[]
  var int64Vals: seq[int64] = @[]
  var uint64Vals: seq[uint64] = @[]

  while pos < data.len:
    let field = nextField(data, pos, "TensorProto")
    case field.number
    of 1:
      for value in readRepeatedVarints(data, field.wire, pos,
          "TensorProto.dims"):
        let dim = cast[int64](value)
        if dim < 0:
          raise pbError("TensorProto.dims: negative dimension " & $dim)
        result.shape.add int(dim)
    of 2:
      requireWire(field.wire, 0, "TensorProto.data_type")
      dataType = int(readVarint(data, pos, "TensorProto.data_type"))
    of 4:
      for bits in readRepeatedFixed32(data, field.wire, pos,
          "TensorProto.float_data"):
        appendLe32(floatBytesBuf, uint64(bits))
    of 5:
      for value in readRepeatedVarints(data, field.wire, pos,
          "TensorProto.int32_data"):
        int32Vals.add cast[int32](uint32(value))
    of 7:
      for value in readRepeatedVarints(data, field.wire, pos,
          "TensorProto.int64_data"):
        int64Vals.add cast[int64](value)
    of 8:
      requireWire(field.wire, 2, "TensorProto.name")
      result.name = readBytes(data, pos, "TensorProto.name")
    of 9:
      requireWire(field.wire, 2, "TensorProto.raw_data")
      rawData = stringToBytes(readBytes(data, pos, "TensorProto.raw_data"))
      hasRawData = true
    of 10:
      for bits in readRepeatedFixed64(data, field.wire, pos,
          "TensorProto.double_data"):
        appendLe64(doubleBytesBuf, bits)
    of 11:
      for value in readRepeatedVarints(data, field.wire, pos,
          "TensorProto.uint64_data"):
        uint64Vals.add value
    of 14:
      requireWire(field.wire, 0, "TensorProto.data_location")
      let loc = int(readVarint(data, pos, "TensorProto.data_location"))
      if loc != 0:
        raise newException(OnnxError,
          "loadOnnx: external tensor data is not supported")
    else:
      skipField(data, pos, field.wire, "TensorProto." & $field.number)

  result.dtype = dtypeOfOnnxDataType(dataType)
  if hasRawData:
    result.data = rawData
  elif floatBytesBuf.len > 0:
    result.data = floatBytesBuf
  elif doubleBytesBuf.len > 0:
    result.data = doubleBytesBuf
  elif int32Vals.len > 0:
    result.data = tensorDataFromInt32(int32Vals, result.dtype)
  elif int64Vals.len > 0:
    result.data = tensorDataFromInt64(int64Vals, result.dtype)
  elif uint64Vals.len > 0:
    result.data = tensorDataFromUint64(uint64Vals, result.dtype)
  else:
    result.data = @[]
  validateTensorBytes(result.dtype, result.shape, result.data.len,
    "loadOnnx: tensor '" & result.name & "'")

proc parseTensorShape(data: string): tuple[shape: seq[int]; params: seq[string]] =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "TensorShapeProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "TensorShapeProto.dim")
      let dimData = readBytes(data, pos, "TensorShapeProto.dim")
      var dpos = 0
      var hasValue = false
      var value = -1
      var param = ""
      while dpos < dimData.len:
        let dimField = nextField(dimData, dpos, "TensorShapeProto.Dimension")
        case dimField.number
        of 1:
          requireWire(dimField.wire, 0, "TensorShapeProto.Dimension.dim_value")
          value = int(cast[int64](readVarint(dimData, dpos,
            "TensorShapeProto.Dimension.dim_value")))
          hasValue = true
        of 2:
          requireWire(dimField.wire, 2, "TensorShapeProto.Dimension.dim_param")
          param = readBytes(dimData, dpos,
            "TensorShapeProto.Dimension.dim_param")
        else:
          skipField(dimData, dpos, dimField.wire,
            "TensorShapeProto.Dimension." & $dimField.number)
      if hasValue:
        result.shape.add value
        result.params.add ""
      else:
        result.shape.add -1
        result.params.add param
    else:
      skipField(data, pos, field.wire, "TensorShapeProto." & $field.number)

proc parseType(data: string): tuple[hasType: bool; dtype: DType;
    shape: seq[int]; params: seq[string]] =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "TypeProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "TypeProto.tensor_type")
      let tensorData = readBytes(data, pos, "TypeProto.tensor_type")
      var tpos = 0
      var elemType = 0
      while tpos < tensorData.len:
        let tfield = nextField(tensorData, tpos, "TypeProto.Tensor")
        case tfield.number
        of 1:
          requireWire(tfield.wire, 0, "TypeProto.Tensor.elem_type")
          elemType = int(readVarint(tensorData, tpos,
            "TypeProto.Tensor.elem_type"))
        of 2:
          requireWire(tfield.wire, 2, "TypeProto.Tensor.shape")
          let parsed = parseTensorShape(readBytes(tensorData, tpos,
            "TypeProto.Tensor.shape"))
          result.shape = parsed.shape
          result.params = parsed.params
        else:
          skipField(tensorData, tpos, tfield.wire,
            "TypeProto.Tensor." & $tfield.number)
      result.hasType = true
      result.dtype = dtypeOfOnnxDataType(elemType)
    else:
      skipField(data, pos, field.wire, "TypeProto." & $field.number)

proc parseValueInfo(data: string): OnnxValueInfo =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "ValueInfoProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "ValueInfoProto.name")
      result.name = readBytes(data, pos, "ValueInfoProto.name")
    of 2:
      requireWire(field.wire, 2, "ValueInfoProto.type")
      let parsed = parseType(readBytes(data, pos, "ValueInfoProto.type"))
      result.hasType = parsed.hasType
      result.dtype = parsed.dtype
      result.shape = parsed.shape
      result.dimParams = parsed.params
    else:
      skipField(data, pos, field.wire, "ValueInfoProto." & $field.number)

proc parseAttribute(data: string): OnnxAttribute =
  var pos = 0
  var attrType = 0
  var hasFloat = false
  var floatVal = 0'f32
  var hasInt = false
  var intVal = 0'i64
  var hasString = false
  var stringVal = ""
  var hasTensor = false
  var tensorVal = OnnxTensor()
  var floatVals: seq[float32] = @[]
  var intVals: seq[int64] = @[]
  var stringVals: seq[string] = @[]

  while pos < data.len:
    let field = nextField(data, pos, "AttributeProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "AttributeProto.name")
      result.name = readBytes(data, pos, "AttributeProto.name")
    of 2:
      for bits in readRepeatedFixed32(data, field.wire, pos,
          "AttributeProto.f"):
        floatVal = cast[float32](bits)
        hasFloat = true
    of 3:
      requireWire(field.wire, 0, "AttributeProto.i")
      intVal = cast[int64](readVarint(data, pos, "AttributeProto.i"))
      hasInt = true
    of 4:
      requireWire(field.wire, 2, "AttributeProto.s")
      stringVal = readBytes(data, pos, "AttributeProto.s")
      hasString = true
    of 5:
      requireWire(field.wire, 2, "AttributeProto.t")
      tensorVal = parseTensor(readBytes(data, pos, "AttributeProto.t"))
      hasTensor = true
    of 7:
      for bits in readRepeatedFixed32(data, field.wire, pos,
          "AttributeProto.floats"):
        floatVals.add cast[float32](bits)
    of 8:
      for value in readRepeatedVarints(data, field.wire, pos,
          "AttributeProto.ints"):
        intVals.add cast[int64](value)
    of 9:
      requireWire(field.wire, 2, "AttributeProto.strings")
      stringVals.add readBytes(data, pos, "AttributeProto.strings")
    of 20:
      requireWire(field.wire, 0, "AttributeProto.type")
      attrType = int(readVarint(data, pos, "AttributeProto.type"))
    else:
      skipField(data, pos, field.wire, "AttributeProto." & $field.number)

  if attrType == 0:
    if hasFloat: attrType = AttrFloat
    elif hasInt: attrType = AttrInt
    elif hasString: attrType = AttrString
    elif hasTensor: attrType = AttrTensor
    elif floatVals.len > 0: attrType = AttrFloats
    elif intVals.len > 0: attrType = AttrInts
    elif stringVals.len > 0: attrType = AttrStrings

  case attrType
  of AttrFloat:
    result = OnnxAttribute(name: result.name, kind: oakFloat, f: floatVal)
  of AttrInt:
    result = OnnxAttribute(name: result.name, kind: oakInt, i: intVal)
  of AttrString:
    result = OnnxAttribute(name: result.name, kind: oakString, s: stringVal)
  of AttrTensor:
    result = OnnxAttribute(name: result.name, kind: oakTensor, t: tensorVal)
  of AttrFloats:
    result = OnnxAttribute(name: result.name, kind: oakFloats,
      floats: floatVals)
  of AttrInts:
    result = OnnxAttribute(name: result.name, kind: oakInts, ints: intVals)
  of AttrStrings:
    result = OnnxAttribute(name: result.name, kind: oakStrings,
      strings: stringVals)
  else:
    result = OnnxAttribute(name: result.name, kind: oakUndefined)

proc parseNode(data: string): OnnxNode =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "NodeProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "NodeProto.input")
      result.inputs.add readBytes(data, pos, "NodeProto.input")
    of 2:
      requireWire(field.wire, 2, "NodeProto.output")
      result.outputs.add readBytes(data, pos, "NodeProto.output")
    of 3:
      requireWire(field.wire, 2, "NodeProto.name")
      result.name = readBytes(data, pos, "NodeProto.name")
    of 4:
      requireWire(field.wire, 2, "NodeProto.op_type")
      result.opType = readBytes(data, pos, "NodeProto.op_type")
    of 5:
      requireWire(field.wire, 2, "NodeProto.attribute")
      result.attributes.add parseAttribute(readBytes(data, pos,
        "NodeProto.attribute"))
    of 7:
      requireWire(field.wire, 2, "NodeProto.domain")
      result.domain = readBytes(data, pos, "NodeProto.domain")
    else:
      skipField(data, pos, field.wire, "NodeProto." & $field.number)

proc parseGraph(data: string): OnnxGraph =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "GraphProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "GraphProto.node")
      result.nodes.add parseNode(readBytes(data, pos, "GraphProto.node"))
    of 2:
      requireWire(field.wire, 2, "GraphProto.name")
      result.name = readBytes(data, pos, "GraphProto.name")
    of 5:
      requireWire(field.wire, 2, "GraphProto.initializer")
      result.initializers.add parseTensor(readBytes(data, pos,
        "GraphProto.initializer"))
    of 10:
      requireWire(field.wire, 2, "GraphProto.doc_string")
      result.docString = readBytes(data, pos, "GraphProto.doc_string")
    of 11:
      requireWire(field.wire, 2, "GraphProto.input")
      result.inputs.add parseValueInfo(readBytes(data, pos,
        "GraphProto.input"))
    of 12:
      requireWire(field.wire, 2, "GraphProto.output")
      result.outputs.add parseValueInfo(readBytes(data, pos,
        "GraphProto.output"))
    of 13:
      requireWire(field.wire, 2, "GraphProto.value_info")
      result.valueInfo.add parseValueInfo(readBytes(data, pos,
        "GraphProto.value_info"))
    else:
      skipField(data, pos, field.wire, "GraphProto." & $field.number)

proc parseOpset(data: string): OnnxOpset =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "OperatorSetIdProto")
    case field.number
    of 1:
      requireWire(field.wire, 2, "OperatorSetIdProto.domain")
      result.domain = readBytes(data, pos, "OperatorSetIdProto.domain")
    of 2:
      requireWire(field.wire, 0, "OperatorSetIdProto.version")
      result.version = cast[int64](readVarint(data, pos,
        "OperatorSetIdProto.version"))
    else:
      skipField(data, pos, field.wire,
        "OperatorSetIdProto." & $field.number)

proc parseModel(data: string): OnnxModel =
  var pos = 0
  while pos < data.len:
    let field = nextField(data, pos, "ModelProto")
    case field.number
    of 1:
      requireWire(field.wire, 0, "ModelProto.ir_version")
      result.irVersion = cast[int64](readVarint(data, pos,
        "ModelProto.ir_version"))
    of 2:
      requireWire(field.wire, 2, "ModelProto.producer_name")
      result.producerName = readBytes(data, pos, "ModelProto.producer_name")
    of 3:
      requireWire(field.wire, 2, "ModelProto.producer_version")
      result.producerVersion = readBytes(data, pos,
        "ModelProto.producer_version")
    of 4:
      requireWire(field.wire, 2, "ModelProto.domain")
      result.domain = readBytes(data, pos, "ModelProto.domain")
    of 5:
      requireWire(field.wire, 0, "ModelProto.model_version")
      result.modelVersion = cast[int64](readVarint(data, pos,
        "ModelProto.model_version"))
    of 6:
      requireWire(field.wire, 2, "ModelProto.doc_string")
      result.docString = readBytes(data, pos, "ModelProto.doc_string")
    of 7:
      requireWire(field.wire, 2, "ModelProto.graph")
      result.graph = parseGraph(readBytes(data, pos, "ModelProto.graph"))
    of 8:
      requireWire(field.wire, 2, "ModelProto.opset_import")
      result.opsets.add parseOpset(readBytes(data, pos,
        "ModelProto.opset_import"))
    else:
      skipField(data, pos, field.wire, "ModelProto." & $field.number)

# ---- protobuf writer ------------------------------------------------------

proc addVarint(dst: var string; value: uint64) =
  var x = value
  while x >= 0x80'u64:
    dst.add char((x and 0x7f'u64) or 0x80'u64)
    x = x shr 7
  dst.add char(x)

proc addKey(dst: var string; field, wire: int) =
  addVarint(dst, uint64((field shl 3) or wire))

proc addInt64Field(dst: var string; field: int; value: int64) =
  addKey(dst, field, 0)
  addVarint(dst, cast[uint64](value))

proc addInt32Field(dst: var string; field: int; value: int) =
  addKey(dst, field, 0)
  addVarint(dst, uint64(value))

proc addBytesField(dst: var string; field: int; value: string) =
  addKey(dst, field, 2)
  addVarint(dst, uint64(value.len))
  dst.add value

proc addStringField(dst: var string; field: int; value: string) =
  addBytesField(dst, field, value)

proc addFixed32Field(dst: var string; field: int; bits: uint32) =
  addKey(dst, field, 5)
  for shift in [0, 8, 16, 24]:
    dst.add char((bits shr shift) and 0xff'u32)

proc addMessageField(dst: var string; field: int; message: string) =
  addBytesField(dst, field, message)

proc encodeTensorShape(shape: openArray[int];
    dimParams: openArray[string]): string =
  for i, dim in shape:
    var dimMsg = ""
    if dim >= 0:
      addInt64Field(dimMsg, 1, int64(dim))
    elif i < dimParams.len and dimParams[i].len > 0:
      addStringField(dimMsg, 2, dimParams[i])
    addMessageField(result, 1, dimMsg)

proc encodeType(info: OnnxValueInfo): string =
  var tensorType = ""
  addInt32Field(tensorType, 1, toOnnxDataType(info.dtype))
  addMessageField(tensorType, 2, encodeTensorShape(info.shape,
    info.dimParams))
  addMessageField(result, 1, tensorType)

proc encodeValueInfo(info: OnnxValueInfo): string =
  addStringField(result, 1, info.name)
  if info.hasType:
    addMessageField(result, 2, encodeType(info))

proc encodeTensor(t: OnnxTensor): string =
  validateTensorBytes(t.dtype, t.shape, t.data.len,
    "saveOnnx: tensor '" & t.name & "'")
  for dim in t.shape:
    addInt64Field(result, 1, int64(dim))
  addInt32Field(result, 2, toOnnxDataType(t.dtype))
  if t.name.len > 0:
    addStringField(result, 8, t.name)
  if t.data.len > 0:
    addBytesField(result, 9, bytesToString(t.data))

proc encodeAttribute(attr: OnnxAttribute): string =
  addStringField(result, 1, attr.name)
  case attr.kind
  of oakUndefined:
    addInt32Field(result, 20, 0)
  of oakFloat:
    addFixed32Field(result, 2, cast[uint32](attr.f))
    addInt32Field(result, 20, AttrFloat)
  of oakInt:
    addInt64Field(result, 3, attr.i)
    addInt32Field(result, 20, AttrInt)
  of oakString:
    addBytesField(result, 4, attr.s)
    addInt32Field(result, 20, AttrString)
  of oakTensor:
    addMessageField(result, 5, encodeTensor(attr.t))
    addInt32Field(result, 20, AttrTensor)
  of oakFloats:
    for value in attr.floats:
      addFixed32Field(result, 7, cast[uint32](value))
    addInt32Field(result, 20, AttrFloats)
  of oakInts:
    for value in attr.ints:
      addInt64Field(result, 8, value)
    addInt32Field(result, 20, AttrInts)
  of oakStrings:
    for value in attr.strings:
      addBytesField(result, 9, value)
    addInt32Field(result, 20, AttrStrings)

proc encodeNode(node: OnnxNode): string =
  for value in node.inputs:
    addStringField(result, 1, value)
  for value in node.outputs:
    addStringField(result, 2, value)
  if node.name.len > 0:
    addStringField(result, 3, node.name)
  addStringField(result, 4, node.opType)
  for attr in node.attributes:
    addMessageField(result, 5, encodeAttribute(attr))
  if node.domain.len > 0:
    addStringField(result, 7, node.domain)

proc encodeGraph(graph: OnnxGraph): string =
  for node in graph.nodes:
    addMessageField(result, 1, encodeNode(node))
  if graph.name.len > 0:
    addStringField(result, 2, graph.name)
  for tensor in graph.initializers:
    addMessageField(result, 5, encodeTensor(tensor))
  if graph.docString.len > 0:
    addStringField(result, 10, graph.docString)
  for input in graph.inputs:
    addMessageField(result, 11, encodeValueInfo(input))
  for output in graph.outputs:
    addMessageField(result, 12, encodeValueInfo(output))
  for info in graph.valueInfo:
    addMessageField(result, 13, encodeValueInfo(info))

proc encodeOpset(opset: OnnxOpset): string =
  if opset.domain.len > 0:
    addStringField(result, 1, opset.domain)
  addInt64Field(result, 2, opset.version)

proc encodeModel(model: OnnxModel): string =
  addInt64Field(result, 1, model.irVersion)
  if model.producerName.len > 0:
    addStringField(result, 2, model.producerName)
  if model.producerVersion.len > 0:
    addStringField(result, 3, model.producerVersion)
  if model.domain.len > 0:
    addStringField(result, 4, model.domain)
  if model.modelVersion != 0:
    addInt64Field(result, 5, model.modelVersion)
  if model.docString.len > 0:
    addStringField(result, 6, model.docString)
  addMessageField(result, 7, encodeGraph(model.graph))
  if model.opsets.len == 0:
    addMessageField(result, 8, encodeOpset(OnnxOpset(
      domain: "", version: OnnxDefaultOpsetVersion)))
  else:
    for opset in model.opsets:
      addMessageField(result, 8, encodeOpset(opset))

# ---- file I/O -------------------------------------------------------------

proc loadOnnx*(s: Stream): OnnxModel =
  ## Reads an ONNX `ModelProto` from `s`.
  parseModel(s.readAll())

proc loadOnnx*(path: string): OnnxModel =
  ## Reads an ONNX `ModelProto` from `path`.
  let s = newFileStream(path, fmRead)
  if s.isNil:
    raise newException(IOError, "loadOnnx: cannot open '" & path & "'")
  defer: s.close()
  loadOnnx(s)

proc saveOnnx*(s: Stream; model: OnnxModel) =
  ## Writes `model` as an ONNX `ModelProto` to `s`.
  s.write(encodeModel(model))

proc saveOnnx*(path: string; model: OnnxModel) =
  ## Writes `model` as an ONNX `ModelProto` to `path`.
  let s = newFileStream(path, fmWrite)
  if s.isNil:
    raise newException(IOError, "saveOnnx: cannot open '" & path & "'")
  defer: s.close()
  saveOnnx(s, model)

# ---- ONNX -> StableHLO ----------------------------------------------------

proc staticType(info: OnnxValueInfo; context: string): ShTensorType =
  if not info.hasType:
    raise newException(OnnxError, context & ": missing tensor type")
  for i, dim in info.shape:
    if dim < 0:
      let suffix =
        if i < info.dimParams.len and info.dimParams[i].len > 0:
          " ('" & info.dimParams[i] & "')"
        else:
          ""
      raise newException(OnnxError,
        context & ": dimension " & $i & " is dynamic" & suffix)
  initTensorType(info.dtype, info.shape)

proc attrIndex(node: OnnxNode; name: string): int =
  for i, attr in node.attributes:
    if attr.name == name:
      return i
  -1

proc attrInt(node: OnnxNode; name: string; default: int64): int64 =
  let idx = attrIndex(node, name)
  if idx < 0: return default
  let attr = node.attributes[idx]
  if attr.kind != oakInt:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': attribute '" & name &
        "' must be an integer")
  attr.i

proc attrFloat(node: OnnxNode; name: string; default: float32): float32 =
  let idx = attrIndex(node, name)
  if idx < 0: return default
  let attr = node.attributes[idx]
  if attr.kind != oakFloat:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': attribute '" & name &
        "' must be a float")
  attr.f

proc attrString(node: OnnxNode; name, default: string): string =
  let idx = attrIndex(node, name)
  if idx < 0: return default
  let attr = node.attributes[idx]
  if attr.kind != oakString:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': attribute '" & name &
        "' must be a string")
  attr.s

proc attrInts(node: OnnxNode; name: string): seq[int64] =
  let idx = attrIndex(node, name)
  if idx < 0: return @[]
  let attr = node.attributes[idx]
  if attr.kind != oakInts:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': attribute '" & name &
        "' must be an integer list")
  attr.ints

proc attrTensor(node: OnnxNode; name: string): OnnxTensor =
  let idx = attrIndex(node, name)
  if idx < 0:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': missing tensor attribute '" &
        name & "'")
  let attr = node.attributes[idx]
  if attr.kind != oakTensor:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': attribute '" & name &
        "' must be a tensor")
  attr.t

proc requireOutput(node: OnnxNode): string =
  if node.outputs.len != 1 or node.outputs[0].len == 0:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "' must have exactly one output")
  node.outputs[0]

proc requireInput(values: Table[string, ShValueId]; node: OnnxNode;
    index: int): ShValueId =
  if index >= node.inputs.len or node.inputs[index].len == 0:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': missing input #" & $index)
  let name = node.inputs[index]
  if not values.hasKey(name):
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': unknown input '" & name & "'")
  values[name]

proc broadcastToShape(builder: var ShBuilder; id: ShValueId;
    outputShape: openArray[int]; context: string): ShValueId =
  let ty = builder.getType(id)
  if ty.shape == @outputShape:
    return id
  if ty.shape.len > outputShape.len:
    raise newException(OnnxError,
      context & ": cannot broadcast " & $ty & " to rank " &
        $outputShape.len)
  let offset = outputShape.len - ty.shape.len
  var dims = newSeq[int](ty.shape.len)
  for i, dim in ty.shape:
    let outAxis = offset + i
    if dim != 1 and dim != outputShape[outAxis]:
      raise newException(OnnxError,
        context & ": cannot broadcast dimension " & $dim &
          " to " & $outputShape[outAxis])
    dims[i] = outAxis
  shops.broadcastInDim(builder, id, outputShape, dims)

proc broadcastPair(builder: var ShBuilder; lhs, rhs: ShValueId;
    context: string): tuple[lhs, rhs: ShValueId] =
  let lt = builder.getType(lhs)
  let rt = builder.getType(rhs)
  if lt.dtype != rt.dtype:
    raise newException(OnnxError,
      context & ": dtype mismatch " & lt.dtype.name & " vs " &
        rt.dtype.name)
  let rank = max(lt.shape.len, rt.shape.len)
  var outShape = newSeq[int](rank)
  for axis in 0 ..< rank:
    let li = axis - (rank - lt.shape.len)
    let ri = axis - (rank - rt.shape.len)
    let ld = if li < 0: 1 else: lt.shape[li]
    let rd = if ri < 0: 1 else: rt.shape[ri]
    if ld == rd:
      outShape[axis] = ld
    elif ld == 1:
      outShape[axis] = rd
    elif rd == 1:
      outShape[axis] = ld
    else:
      raise newException(OnnxError,
        context & ": cannot broadcast dimensions " & $ld & " and " & $rd)
  result.lhs = broadcastToShape(builder, lhs, outShape, context)
  result.rhs = broadcastToShape(builder, rhs, outShape, context)

proc bindOutput(values: var Table[string, ShValueId]; node: OnnxNode;
    id: ShValueId) =
  values[requireOutput(node)] = id

proc zeroScalar(builder: var ShBuilder; dtype: DType): ShValueId =
  shops.constant(builder, dtype, [], newSeq[byte](dtype.byteSize))

proc scalar(builder: var ShBuilder; dtype: DType; value: float32;
    context: string): ShValueId =
  let data =
    case dtype
    of dtFloat32: f32Bytes(value)
    of dtFloat64: f64Bytes(float64(value))
    else:
      if value == 0'f32:
        newSeq[byte](dtype.byteSize)
      else:
        raise newException(OnnxError,
          context & ": scalar multiply for dtype " & dtype.name &
            " is not supported")
  shops.constant(builder, dtype, [], data)

proc scale(builder: var ShBuilder; id: ShValueId; value: float32;
    context: string): ShValueId =
  if value > 0.999999'f32 and value < 1.000001'f32:
    return id
  let ty = builder.getType(id)
  let s = scalar(builder, ty.dtype, value, context)
  let sb = shops.broadcastInDim(builder, s, ty.shape, [])
  shops.mul(builder, id, sb)

proc lowerMatMul(builder: var ShBuilder; lhs, rhs: ShValueId;
    context: string): ShValueId =
  let lt = builder.getType(lhs)
  let rt = builder.getType(rhs)
  if lt.shape.len < 2 or rt.shape.len < 2:
    raise newException(OnnxError,
      context & ": rew imports MatMul for rank >= 2 tensors")
  let lhsBatch = lt.shape.len - 2
  let rhsBatch = rt.shape.len - 2
  if lhsBatch != rhsBatch:
    raise newException(OnnxError,
      context & ": MatMul batch broadcasting is not supported")
  var lhsBatching = newSeq[int](lhsBatch)
  var rhsBatching = newSeq[int](rhsBatch)
  for i in 0 ..< lhsBatch:
    if lt.shape[i] != rt.shape[i]:
      raise newException(OnnxError,
        context & ": MatMul batch dimension mismatch")
    lhsBatching[i] = i
    rhsBatching[i] = i
  shops.dotGeneral(builder, lhs, rhs, lhsBatching, rhsBatching,
    [lt.shape.len - 1], [rt.shape.len - 2])

proc resolveReshape(inputShape: openArray[int]; requested: openArray[int64];
    allowZero: bool): seq[int] =
  result = newSeq[int](requested.len)
  var infer = -1
  var known = 1
  for i, rawDim in requested:
    if rawDim == -1:
      if infer >= 0:
        raise newException(OnnxError,
          "ONNX Reshape: only one inferred dimension is allowed")
      infer = i
      result[i] = 1
    elif rawDim == 0 and not allowZero:
      if i >= inputShape.len:
        raise newException(OnnxError,
          "ONNX Reshape: 0 dimension has no matching input axis")
      result[i] = inputShape[i]
      known *= result[i]
    elif rawDim < -1:
      raise newException(OnnxError,
        "ONNX Reshape: invalid requested dimension " & $rawDim)
    else:
      result[i] = int(rawDim)
      known *= result[i]
  if infer >= 0:
    let total = elementCount(inputShape)
    if known == 0 or total mod known != 0:
      raise newException(OnnxError,
        "ONNX Reshape: cannot infer dimension for element count " & $total)
    result[infer] = total div known

proc lowerConv(builder: var ShBuilder; values: Table[string, ShValueId];
    node: OnnxNode): ShValueId =
  let x = requireInput(values, node, 0)
  let w = requireInput(values, node, 1)
  let xt = builder.getType(x)
  let wt = builder.getType(w)
  if xt.shape.len < 3 or wt.shape.len != xt.shape.len:
    raise newException(OnnxError,
      "ONNX Conv: rew imports NCHW/OIHW convolutions only")
  let spatialRank = xt.shape.len - 2
  var strides = newSeq[int](spatialRank)
  for i in 0 ..< strides.len: strides[i] = 1
  let strideAttr = attrInts(node, "strides")
  if strideAttr.len > 0:
    if strideAttr.len != spatialRank:
      raise newException(OnnxError, "ONNX Conv: bad strides rank")
    for i, v in strideAttr: strides[i] = int(v)

  var dilations = newSeq[int](spatialRank)
  for i in 0 ..< dilations.len: dilations[i] = 1
  let dilationAttr = attrInts(node, "dilations")
  if dilationAttr.len > 0:
    if dilationAttr.len != spatialRank:
      raise newException(OnnxError, "ONNX Conv: bad dilations rank")
    for i, v in dilationAttr: dilations[i] = int(v)

  var padding = newSeq[array[2, int]](spatialRank)
  let pads = attrInts(node, "pads")
  if pads.len > 0:
    if pads.len != spatialRank * 2:
      raise newException(OnnxError, "ONNX Conv: bad pads rank")
    for i in 0 ..< spatialRank:
      padding[i] = [int(pads[i]), int(pads[i + spatialRank])]

  let autoPad = attrString(node, "auto_pad", "NOTSET")
  if autoPad.len > 0 and autoPad != "NOTSET":
    raise newException(OnnxError,
      "ONNX Conv: auto_pad='" & autoPad & "' is not supported")

  var dims = shops.ConvDimensionNumbers(
    inputBatch: 0,
    inputFeature: 1,
    kernelInputFeature: 1,
    kernelOutputFeature: 0,
    outputBatch: 0,
    outputFeature: 1,
  )
  for i in 0 ..< spatialRank:
    dims.inputSpatial.add 2 + i
    dims.kernelSpatial.add 2 + i
    dims.outputSpatial.add 2 + i
  var ones = newSeq[int](spatialRank)
  for i in 0 ..< ones.len:
    ones[i] = 1
  let group = int(attrInt(node, "group", 1))
  result = shops.convolution(builder, x, w, strides, padding, ones,
    dilations, dims, featureGroupCount = group)
  if node.inputs.len > 2 and node.inputs[2].len > 0:
    let bias = requireInput(values, node, 2)
    let outTy = builder.getType(result)
    let biasB = shops.broadcastInDim(builder, bias, outTy.shape, [1])
    result = shops.add(builder, result, biasB)

proc lowerNode(builder: var ShBuilder; values: var Table[string, ShValueId];
    consts: var Table[string, OnnxTensor]; node: OnnxNode) =
  if node.domain.len > 0:
    raise newException(OnnxError,
      "ONNX node '" & node.opType & "': domain '" & node.domain &
        "' is not supported")

  case node.opType
  of "Constant":
    let tensor = attrTensor(node, "value")
    let id = shops.constant(builder, tensor.dtype, tensor.shape, tensor.data)
    let outName = requireOutput(node)
    values[outName] = id
    consts[outName] = OnnxTensor(name: outName, dtype: tensor.dtype,
      shape: tensor.shape, data: tensor.data)
  of "Identity":
    values[requireOutput(node)] = requireInput(values, node, 0)
  of "Add", "Sub", "Mul", "Div", "Max", "Min":
    let pair = broadcastPair(builder, requireInput(values, node, 0),
      requireInput(values, node, 1), "ONNX " & node.opType)
    let id =
      case node.opType
      of "Add": shops.add(builder, pair.lhs, pair.rhs)
      of "Sub": shops.sub(builder, pair.lhs, pair.rhs)
      of "Mul": shops.mul(builder, pair.lhs, pair.rhs)
      of "Div": shops.divide(builder, pair.lhs, pair.rhs)
      of "Max": shops.maximum(builder, pair.lhs, pair.rhs)
      else: shops.minimum(builder, pair.lhs, pair.rhs)
    bindOutput(values, node, id)
  of "Neg":
    bindOutput(values, node, shops.neg(builder, requireInput(values, node, 0)))
  of "Exp":
    bindOutput(values, node, shops.exponential(builder,
      requireInput(values, node, 0)))
  of "Log":
    bindOutput(values, node, shops.log(builder, requireInput(values, node, 0)))
  of "Sqrt":
    bindOutput(values, node, shops.sqrt(builder, requireInput(values, node, 0)))
  of "Abs":
    bindOutput(values, node, shops.abs(builder, requireInput(values, node, 0)))
  of "Tanh":
    bindOutput(values, node, shops.tanh(builder, requireInput(values, node, 0)))
  of "Sigmoid":
    bindOutput(values, node, shops.logistic(builder,
      requireInput(values, node, 0)))
  of "Relu":
    let x = requireInput(values, node, 0)
    let ty = builder.getType(x)
    let z = zeroScalar(builder, ty.dtype)
    let zb = shops.broadcastInDim(builder, z, ty.shape, [])
    bindOutput(values, node, shops.maximum(builder, x, zb))
  of "MatMul":
    bindOutput(values, node, lowerMatMul(builder, requireInput(values, node, 0),
      requireInput(values, node, 1), "ONNX MatMul"))
  of "Gemm":
    var a = requireInput(values, node, 0)
    var b = requireInput(values, node, 1)
    if attrInt(node, "transA", 0) != 0:
      a = shops.transpose(builder, a, [1, 0])
    if attrInt(node, "transB", 0) != 0:
      b = shops.transpose(builder, b, [1, 0])
    var y = lowerMatMul(builder, a, b, "ONNX Gemm")
    y = scale(builder, y, attrFloat(node, "alpha", 1'f32), "ONNX Gemm")
    if node.inputs.len > 2 and node.inputs[2].len > 0:
      var c = requireInput(values, node, 2)
      c = scale(builder, c, attrFloat(node, "beta", 1'f32), "ONNX Gemm")
      c = broadcastToShape(builder, c, builder.getType(y).shape,
        "ONNX Gemm bias")
      y = shops.add(builder, y, c)
    bindOutput(values, node, y)
  of "Reshape":
    let x = requireInput(values, node, 0)
    if node.inputs.len < 2 or not consts.hasKey(node.inputs[1]):
      raise newException(OnnxError,
        "ONNX Reshape: shape input must be a constant initializer")
    let requested = int64Values(consts[node.inputs[1]], "ONNX Reshape")
    let shape = resolveReshape(builder.getType(x).shape, requested,
      attrInt(node, "allowzero", 0) != 0)
    bindOutput(values, node, shops.reshape(builder, x, shape))
  of "Transpose":
    let x = requireInput(values, node, 0)
    var permAttr = attrInts(node, "perm")
    let rank = builder.getType(x).shape.len
    var perm = newSeq[int](rank)
    if permAttr.len == 0:
      for i in 0 ..< rank: perm[i] = rank - 1 - i
    else:
      if permAttr.len != rank:
        raise newException(OnnxError, "ONNX Transpose: bad perm rank")
      for i, v in permAttr:
        perm[i] = int(if v < 0: int64(rank) + v else: v)
    bindOutput(values, node, shops.transpose(builder, x, perm))
  of "Concat":
    if node.inputs.len == 0:
      raise newException(OnnxError, "ONNX Concat: no inputs")
    var ids = newSeq[ShValueId](node.inputs.len)
    for i in 0 ..< node.inputs.len:
      ids[i] = requireInput(values, node, i)
    let rank = builder.getType(ids[0]).shape.len
    var axis = int(attrInt(node, "axis", 0))
    if axis < 0: axis += rank
    bindOutput(values, node, shops.concatenate(builder, ids, axis))
  of "Cast":
    let toType = int(attrInt(node, "to", 0))
    bindOutput(values, node, shops.convert(builder, requireInput(values, node, 0),
      dtypeOfOnnxDataType(toType)))
  of "Conv":
    bindOutput(values, node, lowerConv(builder, values, node))
  else:
    raise newException(OnnxError,
      "ONNX op '" & node.opType & "' is not supported by toShModule")

proc toShModule*(model: OnnxModel): ShModule =
  ## Converts a supported ONNX graph into rew's StableHLO IR.
  ##
  ## Supported imports include common tensor constants, elementwise ops
  ## with NumPy-style broadcasting, MatMul/Gemm, Reshape with constant
  ## shape input, Transpose, Concat, Cast, Relu/Sigmoid, and NCHW Conv.
  ## Unsupported ONNX features raise `OnnxError`.
  var initializerNames = initHashSet[string]()
  for tensor in model.graph.initializers:
    initializerNames.incl tensor.name

  var inputTypes: seq[ShTensorType] = @[]
  var inputNames: seq[string] = @[]
  for input in model.graph.inputs:
    if initializerNames.contains(input.name):
      continue
    inputTypes.add staticType(input, "ONNX input '" & input.name & "'")
    inputNames.add input.name

  var builder = shops.initBuilder(
    if model.graph.name.len > 0: model.graph.name else: "onnx_module")
  let args = builder.beginFunc("main", inputTypes, [])
  var values = initTable[string, ShValueId]()
  var consts = initTable[string, OnnxTensor]()
  for i, name in inputNames:
    values[name] = args[i]
  for tensor in model.graph.initializers:
    let id = shops.constant(builder, tensor.dtype, tensor.shape, tensor.data)
    values[tensor.name] = id
    consts[tensor.name] = tensor
  for node in model.graph.nodes:
    lowerNode(builder, values, consts, node)

  var outputIds: seq[ShValueId] = @[]
  var outputTypes: seq[ShTensorType] = @[]
  for output in model.graph.outputs:
    if not values.hasKey(output.name):
      raise newException(OnnxError,
        "ONNX output '" & output.name & "' is not produced by the graph")
    let id = values[output.name]
    outputIds.add id
    outputTypes.add builder.getType(id)
  builder.setCurrentOutputTypes(outputTypes)
  builder.returnOp(outputIds)
  builder.endFunc()
  builder.build()

proc importOnnx*(s: Stream): ShModule =
  ## Reads an ONNX model from `s` and converts it to StableHLO IR.
  loadOnnx(s).toShModule()

proc importOnnx*(path: string): ShModule =
  ## Reads an ONNX model from `path` and converts it to StableHLO IR.
  loadOnnx(path).toShModule()

# ---- StableHLO -> ONNX ----------------------------------------------------

proc tensorType(fn: ShFunction; id: ShValueId; context: string): ShTensorType =
  if id.int <= 0 or id.int >= fn.types.len:
    raise newException(OnnxError, context & ": invalid value id " & $id)
  fn.types[id.int]

proc attr(op: ShOp; name: string; context: string): ShAttr =
  for entry in op.attrs:
    if entry.name == name:
      return entry.value
  raise newException(OnnxError, context & ": missing attribute '" & name & "'")

proc attrI64(op: ShOp; name: string; default: int64): int64 =
  for entry in op.attrs:
    if entry.name == name:
      if entry.value.kind != akI64:
        raise newException(OnnxError,
          "toOnnxModel: attribute '" & name & "' is not i64")
      return entry.value.i64
  default

proc attrI64s(op: ShOp; name: string): seq[int64] =
  for entry in op.attrs:
    if entry.name == name:
      if entry.value.kind != akI64Array:
        raise newException(OnnxError,
          "toOnnxModel: attribute '" & name & "' is not an i64 array")
      return entry.value.i64s
  @[]

func idName(id: ShValueId): string =
  "v" & $id.int

proc valueName(names: Table[int, string]; id: ShValueId): string =
  if names.hasKey(id.int): names[id.int] else: idName(id)

proc shapeTensor(name: string; shape: openArray[int]): OnnxTensor =
  var dims = newSeq[int64](shape.len)
  for i, dim in shape:
    dims[i] = int64(dim)
  initOnnxTensor(name, dtInt64, [shape.len], int64Bytes(dims))

proc allEqual(xs, ys: openArray[int64]): bool =
  if xs.len != ys.len: return false
  for i in 0 ..< xs.len:
    if xs[i] != ys[i]: return false
  true

proc isStandardMatMul(fn: ShFunction; op: ShOp): bool =
  if op.operands.len != 2:
    return false
  let lhsTy = tensorType(fn, op.operands[0], "toOnnxModel MatMul lhs")
  let rhsTy = tensorType(fn, op.operands[1], "toOnnxModel MatMul rhs")
  if lhsTy.shape.len < 2 or rhsTy.shape.len < 2:
    return false
  let dims = attr(op, "dot_dimension_numbers", "toOnnxModel MatMul")
  if dims.kind != akDotDims:
    return false
  if not allEqual(dims.lhsContractingDims,
      [int64(lhsTy.shape.len - 1)]):
    return false
  if not allEqual(dims.rhsContractingDims,
      [int64(rhsTy.shape.len - 2)]):
    return false
  let batchRank = lhsTy.shape.len - 2
  if rhsTy.shape.len - 2 != batchRank:
    return false
  var expected = newSeq[int64](batchRank)
  for i in 0 ..< batchRank: expected[i] = int64(i)
  allEqual(dims.lhsBatchingDims, expected) and
    allEqual(dims.rhsBatchingDims, expected)

proc denseTensorFromConstant(op: ShOp; name: string): OnnxTensor =
  let value = attr(op, "value", "toOnnxModel constant")
  if value.kind != akDenseElements:
    raise newException(OnnxError,
      "toOnnxModel: stablehlo.constant is not dense elements")
  initOnnxTensor(name, value.denseDtype, value.denseShape, value.denseBytes)

proc isTrailingBroadcast(fn: ShFunction; op: ShOp): bool =
  if op.operands.len != 1 or op.results.len != 1:
    return false
  let inRank = tensorType(fn, op.operands[0],
    "toOnnxModel broadcast input").shape.len
  let outRank = op.results[0].ty.shape.len
  let dims = attrI64s(op, "broadcast_dimensions")
  if dims.len != inRank:
    return false
  for i, dim in dims:
    if dim != int64(outRank - inRank + i):
      return false
  true

proc convDimsAreOnnx(dims: ShAttr; spatialRank: int): bool =
  if dims.kind != akConvDims:
    return false
  if dims.inputBatchDim != 0 or dims.inputFeatureDim != 1 or
      dims.kernelOutputFeatureDim != 0 or dims.kernelInputFeatureDim != 1 or
      dims.outputBatchDim != 0 or dims.outputFeatureDim != 1:
    return false
  if dims.inputSpatialDims.len != spatialRank or
      dims.kernelSpatialDims.len != spatialRank or
      dims.outputSpatialDims.len != spatialRank:
    return false
  for i in 0 ..< spatialRank:
    if dims.inputSpatialDims[i] != int64(2 + i) or
        dims.kernelSpatialDims[i] != int64(2 + i) or
        dims.outputSpatialDims[i] != int64(2 + i):
      return false
  true

proc convNode(fn: ShFunction; op: ShOp;
    names: Table[int, string]): OnnxNode =
  let lhsTy = tensorType(fn, op.operands[0], "toOnnxModel Conv")
  let spatialRank = lhsTy.shape.len - 2
  let dims = attr(op, "dimension_numbers", "toOnnxModel Conv")
  if not convDimsAreOnnx(dims, spatialRank):
    raise newException(OnnxError,
      "toOnnxModel: only NCHW/OIHW StableHLO convolutions export as ONNX Conv")
  if attrI64(op, "batch_group_count", 1) != 1:
    raise newException(OnnxError,
      "toOnnxModel: batch_group_count is not representable as ONNX Conv")
  let lhsDilation = attrI64s(op, "lhs_dilation")
  for value in lhsDilation:
    if value != 1:
      raise newException(OnnxError,
        "toOnnxModel: lhs_dilation is not representable as ONNX Conv")
  let strides = attrI64s(op, "window_strides")
  let rhsDilation = attrI64s(op, "rhs_dilation")
  let padding = attr(op, "padding", "toOnnxModel Conv")
  if padding.kind != akI64Matrix or padding.matRows != spatialRank or
      padding.matCols != 2:
    raise newException(OnnxError, "toOnnxModel: bad convolution padding")
  var pads = newSeq[int64](spatialRank * 2)
  for i in 0 ..< spatialRank:
    pads[i] = padding.matVals[i * 2]
    pads[i + spatialRank] = padding.matVals[i * 2 + 1]
  result = initOnnxNode("Conv",
    [valueName(names, op.operands[0]), valueName(names, op.operands[1])],
    [valueName(names, op.results[0].id)],
    attributes = [
      initOnnxAttribute("strides", strides),
      initOnnxAttribute("pads", pads),
      initOnnxAttribute("dilations", rhsDilation),
      initOnnxAttribute("group", attrI64(op, "feature_group_count", 1)),
    ])

proc toOnnxModel*(module: ShModule; graphName = "";
    opsetVersion: int64 = OnnxDefaultOpsetVersion): OnnxModel =
  ## Converts a supported rew StableHLO module into an ONNX model.
  ##
  ## Supported exports include constants, elementwise ops, MatMul-shaped
  ## dot_general, Reshape, Transpose, Concat, Cast, trailing
  ## broadcast_in_dim as Expand, and NCHW/OIHW Conv. Unsupported IR raises
  ## `OnnxError`.
  if module.funcs.len == 0:
    raise newException(OnnxError, "toOnnxModel: module has no functions")
  let fn = module.funcs[0]
  var names = initTable[int, string]()
  var graph = OnnxGraph(name:
    if graphName.len > 0: graphName
    elif module.name.len > 0: module.name
    else: fn.name)
  for i, arg in fn.args:
    let name = "arg" & $i
    names[arg.id.int] = name
    graph.inputs.add initOnnxValueInfo(name, arg.ty.dtype, arg.ty.shape)

  var returnOperands: seq[ShValueId] = @[]
  var shapeCounter = 0
  var valueInfos: seq[OnnxValueInfo] = @[]

  proc addResultInfos(op: ShOp) =
    for res in op.results:
      names[res.id.int] = idName(res.id)
      valueInfos.add initOnnxValueInfo(idName(res.id), res.ty.dtype,
        res.ty.shape)

  for op in fn.ops:
    case op.kind
    of okReturn:
      returnOperands = op.operands
    of okConstant:
      if op.results.len != 1:
        raise newException(OnnxError,
          "toOnnxModel: multi-result constants are unsupported")
      names[op.results[0].id.int] = idName(op.results[0].id)
      graph.initializers.add denseTensorFromConstant(op, idName(op.results[0].id))
    of okAdd, okSub, okMul, okDiv, okMax, okMin,
       okNeg, okExp, okLog, okSqrt, okAbs, okTanh, okLogistic:
      addResultInfos(op)
      let opType =
        case op.kind
        of okAdd: "Add"
        of okSub: "Sub"
        of okMul: "Mul"
        of okDiv: "Div"
        of okMax: "Max"
        of okMin: "Min"
        of okNeg: "Neg"
        of okExp: "Exp"
        of okLog: "Log"
        of okSqrt: "Sqrt"
        of okAbs: "Abs"
        of okTanh: "Tanh"
        else: "Sigmoid"
      var inputs = newSeq[string](op.operands.len)
      for i, id in op.operands:
        inputs[i] = valueName(names, id)
      graph.nodes.add initOnnxNode(opType, inputs,
        [valueName(names, op.results[0].id)])
    of okDot:
      addResultInfos(op)
      graph.nodes.add initOnnxNode("MatMul",
        [valueName(names, op.operands[0]), valueName(names, op.operands[1])],
        [valueName(names, op.results[0].id)])
    of okDotGeneral:
      if not isStandardMatMul(fn, op):
        raise newException(OnnxError,
          "toOnnxModel: dot_general is not a standard ONNX MatMul")
      addResultInfos(op)
      graph.nodes.add initOnnxNode("MatMul",
        [valueName(names, op.operands[0]), valueName(names, op.operands[1])],
        [valueName(names, op.results[0].id)])
    of okReshape:
      addResultInfos(op)
      let shapeName = idName(op.results[0].id) & "_shape"
      graph.initializers.add shapeTensor(shapeName, op.results[0].ty.shape)
      graph.nodes.add initOnnxNode("Reshape",
        [valueName(names, op.operands[0]), shapeName],
        [valueName(names, op.results[0].id)])
    of okTranspose:
      addResultInfos(op)
      graph.nodes.add initOnnxNode("Transpose",
        [valueName(names, op.operands[0])],
        [valueName(names, op.results[0].id)],
        attributes = [initOnnxAttribute("perm", attrI64s(op, "permutation"))])
    of okConcatenate:
      addResultInfos(op)
      var inputs = newSeq[string](op.operands.len)
      for i, id in op.operands:
        inputs[i] = valueName(names, id)
      graph.nodes.add initOnnxNode("Concat", inputs,
        [valueName(names, op.results[0].id)],
        attributes = [initOnnxAttribute("axis", attrI64(op, "dimension", 0))])
    of okConvert:
      addResultInfos(op)
      graph.nodes.add initOnnxNode("Cast",
        [valueName(names, op.operands[0])],
        [valueName(names, op.results[0].id)],
        attributes = [initOnnxAttribute("to",
          int64(toOnnxDataType(op.results[0].ty.dtype)))])
    of okBroadcastInDim:
      if not isTrailingBroadcast(fn, op):
        raise newException(OnnxError,
          "toOnnxModel: only trailing broadcast_in_dim exports as ONNX Expand")
      addResultInfos(op)
      let shapeName = "shape_" & $shapeCounter
      inc shapeCounter
      graph.initializers.add shapeTensor(shapeName, op.results[0].ty.shape)
      graph.nodes.add initOnnxNode("Expand",
        [valueName(names, op.operands[0]), shapeName],
        [valueName(names, op.results[0].id)])
    of okConvolution:
      addResultInfos(op)
      graph.nodes.add convNode(fn, op, names)
    else:
      raise newException(OnnxError,
        "toOnnxModel: StableHLO op " & $op.kind &
          " is not supported by ONNX export")

  if returnOperands.len == 0 and fn.outputTypes.len > 0:
    raise newException(OnnxError,
      "toOnnxModel: function has outputs but no return op")
  var outputNames = initHashSet[string]()
  for i, id in returnOperands:
    let ty = tensorType(fn, id, "toOnnxModel output")
    let name = valueName(names, id)
    outputNames.incl name
    graph.outputs.add initOnnxValueInfo(name, ty.dtype, ty.shape)
  var inputNames = initHashSet[string]()
  for input in graph.inputs:
    inputNames.incl input.name
  var initializerNames = initHashSet[string]()
  for tensor in graph.initializers:
    initializerNames.incl tensor.name
  for info in valueInfos:
    if not outputNames.contains(info.name) and
        not inputNames.contains(info.name) and
        not initializerNames.contains(info.name):
      graph.valueInfo.add info
  initOnnxModel(graph, opsetVersion)

proc saveOnnx*(s: Stream; module: ShModule; graphName = "";
    opsetVersion: int64 = OnnxDefaultOpsetVersion) =
  ## Converts `module` to ONNX and writes it to `s`.
  saveOnnx(s, module.toOnnxModel(graphName, opsetVersion))

proc saveOnnx*(path: string; module: ShModule; graphName = "";
    opsetVersion: int64 = OnnxDefaultOpsetVersion) =
  ## Converts `module` to ONNX and writes it to `path`.
  saveOnnx(path, module.toOnnxModel(graphName, opsetVersion))

proc exportOnnx*(s: Stream; module: ShModule; graphName = "";
    opsetVersion: int64 = OnnxDefaultOpsetVersion) =
  ## Alias for `saveOnnx(s, module, ...)`.
  saveOnnx(s, module, graphName, opsetVersion)

proc exportOnnx*(path: string; module: ShModule; graphName = "";
    opsetVersion: int64 = OnnxDefaultOpsetVersion) =
  ## Alias for `saveOnnx(path, module, ...)`.
  saveOnnx(path, module, graphName, opsetVersion)
