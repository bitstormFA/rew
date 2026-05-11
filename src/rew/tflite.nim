## TensorFlow Lite model export.
##
## TFLite stores models as FlatBuffers with the `TFL3` file identifier. This
## module writes the small schema subset needed to export rew StableHLO graphs
## without adding a dependency on TensorFlow, flatc, or a FlatBuffers package.

import std/[streams, tables]
import ./dtype
import ./stablehlo/ir

type
  TfliteError* = object of CatchableError
    ## Raised when a StableHLO graph uses a feature this exporter cannot
    ## represent as a TensorFlow Lite FlatBuffer.

  TfliteOperatorCode* = object
    ## One TFLite operator-code entry. Operators reference these by index.
    builtinCode*: int32
    version*: int

  TfliteTensor* = object
    ## Tensor metadata in a TFLite subgraph.
    name*: string
    dtype*: DType
    shape*: seq[int]
    buffer*: int

  TfliteBuffer* = object
    ## Raw little-endian data for constant tensors.
    data*: seq[byte]

  TfliteBuiltinOptionsKind* = enum
    ## Supported builtin-options payloads.
    tbokNone
    tbokAdd
    tbokMul
    tbokSub
    tbokDiv
    tbokConcatenation
    tbokReshape
    tbokCast
    tbokBatchMatMul
    tbokBroadcastTo
    tbokTranspose
    tbokExp
    tbokMaximumMinimum
    tbokNeg
    tbokAbs

  TfliteBuiltinOptions* = object
    ## Operator-specific options for builtins that need them.
    case kind*: TfliteBuiltinOptionsKind
    of tbokConcatenation:
      axis*: int
    of tbokReshape:
      newShape*: seq[int]
    of tbokCast:
      inDType*: DType
      outDType*: DType
    of tbokNone, tbokAdd, tbokMul, tbokSub, tbokDiv, tbokBatchMatMul,
       tbokBroadcastTo, tbokTranspose, tbokExp, tbokMaximumMinimum,
       tbokNeg, tbokAbs:
      discard

  TfliteOperator* = object
    ## One TFLite operator invocation.
    opcodeIndex*: int
    inputs*: seq[int]
    outputs*: seq[int]
    options*: TfliteBuiltinOptions

  TfliteSubgraph* = object
    ## TFLite subgraph. Rew exports one main subgraph.
    name*: string
    tensors*: seq[TfliteTensor]
    inputs*: seq[int]
    outputs*: seq[int]
    operators*: seq[TfliteOperator]

  TfliteModel* = object
    ## Top-level TFLite model container.
    version*: int
    operatorCodes*: seq[TfliteOperatorCode]
    subgraphs*: seq[TfliteSubgraph]
    description*: string
    buffers*: seq[TfliteBuffer]

const
  TfliteSchemaVersion* = 3
    ## TFLite schema version written by this exporter.

  TfliteBuiltinAdd* = 0'i32
  TfliteBuiltinConcatenation* = 2'i32
  TfliteBuiltinFullyConnected* = 9'i32
  TfliteBuiltinLogistic* = 14'i32
  TfliteBuiltinMul* = 18'i32
  TfliteBuiltinReshape* = 22'i32
  TfliteBuiltinTanh* = 28'i32
  TfliteBuiltinTranspose* = 39'i32
  TfliteBuiltinSub* = 41'i32
  TfliteBuiltinDiv* = 42'i32
  TfliteBuiltinExp* = 47'i32
  TfliteBuiltinCast* = 53'i32
  TfliteBuiltinMaximum* = 55'i32
  TfliteBuiltinMinimum* = 57'i32
  TfliteBuiltinNeg* = 59'i32
  TfliteBuiltinSin* = 66'i32
  TfliteBuiltinLog* = 73'i32
  TfliteBuiltinSqrt* = 75'i32
  TfliteBuiltinRsqrt* = 76'i32
  TfliteBuiltinPow* = 78'i32
  TfliteBuiltinAbs* = 101'i32
  TfliteBuiltinCeil* = 104'i32
  TfliteBuiltinCos* = 108'i32
  TfliteBuiltinBatchMatMul* = 126'i32
  TfliteBuiltinBroadcastTo* = 130'i32
  TfliteBuiltinAtan2* = 156'i32
  TfliteBuiltinSign* = 158'i32

  TflitePlaceholderForGreaterOpCodes = 127'i32

  OptConcatenation = 10
  OptAdd = 11
  OptMul = 21
  OptTranspose = 26
  OptSub = 28
  OptDiv = 29
  OptExp = 33
  OptCast = 37
  OptMaximumMinimum = 39
  OptNeg = 42
  OptAbs = 78
  OptBatchMatMul = 101
  OptBroadcastTo = 104
  OptReshape = 17

# ---- public constructors --------------------------------------------------

func initTfliteBuiltinOptions*(): TfliteBuiltinOptions =
  ## Builds an empty builtin-options payload.
  TfliteBuiltinOptions(kind: tbokNone)

func initTfliteConcatenationOptions*(axis: int): TfliteBuiltinOptions =
  ## Builds TFLite ConcatenationOptions.
  TfliteBuiltinOptions(kind: tbokConcatenation, axis: axis)

func initTfliteReshapeOptions*(newShape: openArray[int]): TfliteBuiltinOptions =
  ## Builds TFLite ReshapeOptions.
  TfliteBuiltinOptions(kind: tbokReshape, newShape: @newShape)

func initTfliteCastOptions*(inDType, outDType: DType): TfliteBuiltinOptions =
  ## Builds TFLite CastOptions.
  TfliteBuiltinOptions(kind: tbokCast, inDType: inDType, outDType: outDType)

func initTfliteOperator*(opcodeIndex: int; inputs, outputs: openArray[int];
    options = initTfliteBuiltinOptions()): TfliteOperator =
  ## Builds a TFLite operator invocation.
  TfliteOperator(opcodeIndex: opcodeIndex, inputs: @inputs,
    outputs: @outputs, options: options)

func initTfliteTensor*(name: string; dtype: DType; shape: openArray[int];
    buffer = 0): TfliteTensor =
  ## Builds TFLite tensor metadata.
  TfliteTensor(name: name, dtype: dtype, shape: @shape, buffer: buffer)

func initTfliteBuffer*(data: sink seq[byte]): TfliteBuffer =
  ## Builds a TFLite data buffer.
  TfliteBuffer(data: data)

func initTfliteSubgraph*(name: string;
    tensors: openArray[TfliteTensor] = [];
    inputs: openArray[int] = [];
    outputs: openArray[int] = [];
    operators: openArray[TfliteOperator] = []): TfliteSubgraph =
  ## Builds a TFLite subgraph.
  TfliteSubgraph(name: name, tensors: @tensors, inputs: @inputs,
    outputs: @outputs, operators: @operators)

func initTfliteModel*(subgraph: TfliteSubgraph;
    description = "rew"): TfliteModel =
  ## Builds a TFLite model around one main subgraph.
  TfliteModel(
    version: TfliteSchemaVersion,
    subgraphs: @[subgraph],
    description: description,
    buffers: @[TfliteBuffer()],
  )

func toTfliteTensorType*(dt: DType): int =
  ## Returns the TFLite `TensorType` enum value for `dt`.
  case dt
  of dtFloat32: 0
  of dtFloat16: 1
  of dtInt32: 2
  of dtUint8: 3
  of dtInt64: 4
  of dtBool: 6
  of dtInt16: 7
  of dtComplex64: 8
  of dtInt8: 9
  of dtFloat64: 10
  of dtComplex128: 11
  of dtUint64: 12
  of dtUint32: 15
  of dtUint16: 16
  of dtBFloat16: 18
  of dtInt4, dtUint4, dtNF4, dtFloat8E4M3Fn, dtFloat8E5M2:
    raise newException(TfliteError,
      "TFLite dtype '" & dt.name & "' is not supported")

# ---- little-endian helpers ------------------------------------------------

proc bytesToString(data: openArray[byte]): string =
  result = newString(data.len)
  for i, b in data:
    result[i] = char(b)

proc appendLe32(dst: var seq[byte]; value: uint32) =
  for shift in [0, 8, 16, 24]:
    dst.add byte((value shr shift) and 0xff'u32)

proc int32Bytes(values: openArray[int]): seq[byte] =
  for value in values:
    appendLe32(result, cast[uint32](int32(value)))

# ---- FlatBuffer writer ----------------------------------------------------

type
  FbFieldKind = enum
    fkByte
    fkBool
    fkInt32
    fkUInt32
    fkOffset

  FbField = object
    present: bool
    kind: FbFieldKind

  FbTable = object
    start: int
    fields: seq[int]

  FbWriter = object
    buf: seq[byte]

func field(kind: FbFieldKind): FbField =
  FbField(present: true, kind: kind)

func absentField(): FbField =
  FbField()

func alignUp(value, alignment: int): int =
  let rem = value mod alignment
  if rem == 0: value else: value + alignment - rem

func fieldAlign(kind: FbFieldKind): int =
  case kind
  of fkByte, fkBool: 1
  of fkInt32, fkUInt32, fkOffset: 4

func fieldSize(kind: FbFieldKind): int =
  case kind
  of fkByte, fkBool: 1
  of fkInt32, fkUInt32, fkOffset: 4

proc alignTo(w: var FbWriter; alignment: int) =
  while w.buf.len mod alignment != 0:
    w.buf.add 0

proc addU16(w: var FbWriter; value: uint16) =
  w.buf.add byte(value and 0xff'u16)
  w.buf.add byte((value shr 8) and 0xff'u16)

proc addU32(w: var FbWriter; value: uint32) =
  appendLe32(w.buf, value)

proc patchU8(w: var FbWriter; pos: int; value: int) =
  if value < 0 or value > 255:
    raise newException(TfliteError, "TFLite FlatBuffer byte value out of range")
  w.buf[pos] = byte(value)

proc patchBool(w: var FbWriter; pos: int; value: bool) =
  w.buf[pos] = if value: 1'u8 else: 0'u8

proc patchU32(w: var FbWriter; pos: int; value: uint32) =
  for i, shift in [0, 8, 16, 24]:
    w.buf[pos + i] = byte((value shr shift) and 0xff'u32)

proc patchI32(w: var FbWriter; pos: int; value: int32) =
  patchU32(w, pos, cast[uint32](value))

proc patchOffset(w: var FbWriter; pos, target: int) =
  if pos < 0:
    raise newException(TfliteError, "TFLite FlatBuffer offset field is absent")
  if target <= pos:
    raise newException(TfliteError,
      "TFLite FlatBuffer writer attempted to point backward")
  patchU32(w, pos, uint32(target - pos))

proc beginTable(w: var FbWriter; fields: openArray[FbField]): FbTable =
  var fieldOffsets = newSeq[int](fields.len)
  var cursor = 4
  for i, f in fields:
    if f.present:
      cursor = alignUp(cursor, fieldAlign(f.kind))
      fieldOffsets[i] = cursor
      cursor += fieldSize(f.kind)
    else:
      fieldOffsets[i] = 0
  let objectSize = alignUp(cursor, 4)
  let vtableSize = 4 + fields.len * 2

  w.alignTo(2)
  let vtableStart = w.buf.len
  w.addU16(uint16(vtableSize))
  w.addU16(uint16(objectSize))
  for offset in fieldOffsets:
    w.addU16(uint16(offset))

  w.alignTo(4)
  let tableStart = w.buf.len
  let oldLen = w.buf.len
  w.buf.setLen(oldLen + objectSize)
  w.patchI32(tableStart, int32(tableStart - vtableStart))

  result = FbTable(start: tableStart, fields: newSeq[int](fields.len))
  for i, offset in fieldOffsets:
    result.fields[i] = if offset == 0: -1 else: tableStart + offset

proc addString(w: var FbWriter; value: string): int =
  w.alignTo(4)
  result = w.buf.len
  w.addU32(uint32(value.len))
  for ch in value:
    w.buf.add byte(ord(ch))
  w.buf.add 0

proc addIntVector(w: var FbWriter; values: openArray[int]): int =
  w.alignTo(4)
  result = w.buf.len
  w.addU32(uint32(values.len))
  for value in values:
    appendLe32(w.buf, cast[uint32](int32(value)))

proc addByteVector(w: var FbWriter; values: openArray[byte]): int =
  while (w.buf.len + 4) mod 16 != 0:
    w.buf.add 0
  result = w.buf.len
  w.addU32(uint32(values.len))
  for value in values:
    w.buf.add value

proc beginOffsetVector(w: var FbWriter; count: int): tuple[start: int;
    slots: seq[int]] =
  w.alignTo(4)
  result.start = w.buf.len
  w.addU32(uint32(count))
  result.slots = newSeq[int](count)
  for i in 0 ..< count:
    result.slots[i] = w.buf.len
    w.addU32(0)

proc setByte(w: var FbWriter; table: FbTable; fieldIndex, value: int) =
  w.patchU8(table.fields[fieldIndex], value)

proc setBool(w: var FbWriter; table: FbTable; fieldIndex: int; value: bool) =
  w.patchBool(table.fields[fieldIndex], value)

proc setI32(w: var FbWriter; table: FbTable; fieldIndex: int; value: int32) =
  w.patchI32(table.fields[fieldIndex], value)

proc setU32(w: var FbWriter; table: FbTable; fieldIndex: int; value: uint32) =
  w.patchU32(table.fields[fieldIndex], value)

proc patchFieldOffset(w: var FbWriter; table: FbTable; fieldIndex: int;
    target: int) =
  w.patchOffset(table.fields[fieldIndex], target)

# ---- TFLite FlatBuffer tables --------------------------------------------

proc optionsUnionType(kind: TfliteBuiltinOptionsKind): int =
  case kind
  of tbokNone: 0
  of tbokAdd: OptAdd
  of tbokMul: OptMul
  of tbokSub: OptSub
  of tbokDiv: OptDiv
  of tbokConcatenation: OptConcatenation
  of tbokReshape: OptReshape
  of tbokCast: OptCast
  of tbokBatchMatMul: OptBatchMatMul
  of tbokBroadcastTo: OptBroadcastTo
  of tbokTranspose: OptTranspose
  of tbokExp: OptExp
  of tbokMaximumMinimum: OptMaximumMinimum
  of tbokNeg: OptNeg
  of tbokAbs: OptAbs

proc buildOptions(w: var FbWriter; options: TfliteBuiltinOptions): int =
  case options.kind
  of tbokNone:
    raise newException(TfliteError,
      "TFLite FlatBuffer writer received empty builtin options")
  of tbokConcatenation:
    let tab = w.beginTable([field(fkInt32)])
    w.setI32(tab, 0, int32(options.axis))
    result = tab.start
  of tbokReshape:
    let tab = w.beginTable([field(fkOffset)])
    let shape = w.addIntVector(options.newShape)
    w.patchFieldOffset(tab, 0, shape)
    result = tab.start
  of tbokCast:
    let tab = w.beginTable([field(fkByte), field(fkByte)])
    w.setByte(tab, 0, options.inDType.toTfliteTensorType)
    w.setByte(tab, 1, options.outDType.toTfliteTensorType)
    result = tab.start
  of tbokAdd, tbokMul, tbokSub, tbokDiv, tbokBatchMatMul,
     tbokBroadcastTo, tbokTranspose, tbokExp, tbokMaximumMinimum,
     tbokNeg, tbokAbs:
    result = w.beginTable(newSeq[FbField]()).start

proc buildOperatorCode(w: var FbWriter; code: TfliteOperatorCode): int =
  let tab = w.beginTable([
    field(fkByte), absentField(), field(fkInt32), field(fkInt32)])
  let deprecatedCode =
    if code.builtinCode > TflitePlaceholderForGreaterOpCodes:
      TflitePlaceholderForGreaterOpCodes
    else:
      code.builtinCode
  w.setByte(tab, 0, int(deprecatedCode))
  w.setI32(tab, 2, int32(if code.version <= 0: 1 else: code.version))
  w.setI32(tab, 3, code.builtinCode)
  tab.start

proc buildTensor(w: var FbWriter; tensor: TfliteTensor): int =
  let tab = w.beginTable([
    field(fkOffset), field(fkByte), field(fkUInt32), field(fkOffset),
    absentField(), absentField(), absentField(), absentField(),
    field(fkBool)])
  w.setByte(tab, 1, tensor.dtype.toTfliteTensorType)
  w.setU32(tab, 2, uint32(tensor.buffer))
  w.setBool(tab, 8, true)
  let shape = w.addIntVector(tensor.shape)
  w.patchFieldOffset(tab, 0, shape)
  let name = w.addString(tensor.name)
  w.patchFieldOffset(tab, 3, name)
  tab.start

proc buildOperator(w: var FbWriter; op: TfliteOperator): int =
  let hasOptions = op.options.kind != tbokNone
  var fields = @[
    field(fkUInt32), field(fkOffset), field(fkOffset)]
  if hasOptions:
    fields.add field(fkByte)
    fields.add field(fkOffset)
  let tab = w.beginTable(fields)
  w.setU32(tab, 0, uint32(op.opcodeIndex))
  let inputs = w.addIntVector(op.inputs)
  w.patchFieldOffset(tab, 1, inputs)
  let outputs = w.addIntVector(op.outputs)
  w.patchFieldOffset(tab, 2, outputs)
  if hasOptions:
    w.setByte(tab, 3, optionsUnionType(op.options.kind))
    let options = w.buildOptions(op.options)
    w.patchFieldOffset(tab, 4, options)
  tab.start

proc buildBuffer(w: var FbWriter; buffer: TfliteBuffer): int =
  if buffer.data.len == 0:
    return w.beginTable(newSeq[FbField]()).start
  let tab = w.beginTable([field(fkOffset)])
  let data = w.addByteVector(buffer.data)
  w.patchFieldOffset(tab, 0, data)
  tab.start

proc buildSubgraph(w: var FbWriter; graph: TfliteSubgraph): int =
  let tab = w.beginTable([
    field(fkOffset), field(fkOffset), field(fkOffset), field(fkOffset),
    field(fkOffset)])

  let tensors = w.beginOffsetVector(graph.tensors.len)
  w.patchFieldOffset(tab, 0, tensors.start)
  for i, tensor in graph.tensors:
    let target = w.buildTensor(tensor)
    w.patchOffset(tensors.slots[i], target)

  let inputs = w.addIntVector(graph.inputs)
  w.patchFieldOffset(tab, 1, inputs)
  let outputs = w.addIntVector(graph.outputs)
  w.patchFieldOffset(tab, 2, outputs)

  let operators = w.beginOffsetVector(graph.operators.len)
  w.patchFieldOffset(tab, 3, operators.start)
  for i, op in graph.operators:
    let target = w.buildOperator(op)
    w.patchOffset(operators.slots[i], target)

  let name = w.addString(graph.name)
  w.patchFieldOffset(tab, 4, name)
  tab.start

proc validateModel(model: TfliteModel) =
  if model.subgraphs.len == 0:
    raise newException(TfliteError, "encodeTflite: model has no subgraphs")
  if model.buffers.len == 0:
    raise newException(TfliteError,
      "encodeTflite: model buffers must start with an empty sentinel")
  if model.buffers[0].data.len != 0:
    raise newException(TfliteError,
      "encodeTflite: model buffer #0 must be empty")

proc encodeTflite*(model: TfliteModel): seq[byte] =
  ## Encodes `model` as a TensorFlow Lite FlatBuffer.
  validateModel(model)
  var w = FbWriter(buf: @[])
  w.addU32(0)
  w.buf.add byte(ord('T'))
  w.buf.add byte(ord('F'))
  w.buf.add byte(ord('L'))
  w.buf.add byte(ord('3'))

  let tab = w.beginTable([
    field(fkUInt32), field(fkOffset), field(fkOffset), field(fkOffset),
    field(fkOffset)])
  w.patchU32(0, uint32(tab.start))
  w.setU32(tab, 0, uint32(
    if model.version == 0: TfliteSchemaVersion else: model.version))

  let opCodes = w.beginOffsetVector(model.operatorCodes.len)
  w.patchFieldOffset(tab, 1, opCodes.start)
  for i, code in model.operatorCodes:
    let target = w.buildOperatorCode(code)
    w.patchOffset(opCodes.slots[i], target)

  let subgraphs = w.beginOffsetVector(model.subgraphs.len)
  w.patchFieldOffset(tab, 2, subgraphs.start)
  for i, graph in model.subgraphs:
    let target = w.buildSubgraph(graph)
    w.patchOffset(subgraphs.slots[i], target)

  let description = w.addString(model.description)
  w.patchFieldOffset(tab, 3, description)

  let buffers = w.beginOffsetVector(model.buffers.len)
  w.patchFieldOffset(tab, 4, buffers.start)
  for i, buffer in model.buffers:
    let target = w.buildBuffer(buffer)
    w.patchOffset(buffers.slots[i], target)

  w.buf

# ---- StableHLO -> TFLite --------------------------------------------------

proc tensorType(fn: ShFunction; id: ShValueId; context: string): ShTensorType =
  if id.int <= 0 or id.int >= fn.types.len:
    raise newException(TfliteError, context & ": invalid value id " & $id)
  fn.types[id.int]

proc attr(op: ShOp; name: string; context: string): ShAttr =
  for entry in op.attrs:
    if entry.name == name:
      return entry.value
  raise newException(TfliteError, context & ": missing attribute '" & name & "'")

proc attrI64(op: ShOp; name: string; default: int64): int64 =
  for entry in op.attrs:
    if entry.name == name:
      if entry.value.kind != akI64:
        raise newException(TfliteError,
          "toTfliteModel: attribute '" & name & "' is not i64")
      return entry.value.i64
  default

proc attrI64s(op: ShOp; name: string): seq[int64] =
  for entry in op.attrs:
    if entry.name == name:
      if entry.value.kind != akI64Array:
        raise newException(TfliteError,
          "toTfliteModel: attribute '" & name & "' is not an i64 array")
      return entry.value.i64s
  @[]

func idName(id: ShValueId): string =
  "v" & $id.int

func allEqual(xs, ys: openArray[int64]): bool =
  if xs.len != ys.len:
    return false
  for i in 0 ..< xs.len:
    if xs[i] != ys[i]:
      return false
  true

proc isStandardMatMul(fn: ShFunction; op: ShOp): bool =
  if op.operands.len != 2:
    return false
  let lhsTy = tensorType(fn, op.operands[0], "toTfliteModel BatchMatMul lhs")
  let rhsTy = tensorType(fn, op.operands[1], "toTfliteModel BatchMatMul rhs")
  if lhsTy.shape.len < 2 or rhsTy.shape.len < 2:
    return false
  let dims = attr(op, "dot_dimension_numbers", "toTfliteModel BatchMatMul")
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
  for i in 0 ..< batchRank:
    expected[i] = int64(i)
  allEqual(dims.lhsBatchingDims, expected) and
    allEqual(dims.rhsBatchingDims, expected)

proc isTrailingBroadcast(fn: ShFunction; op: ShOp): bool =
  if op.operands.len != 1 or op.results.len != 1:
    return false
  let inRank = tensorType(fn, op.operands[0],
    "toTfliteModel broadcast input").shape.len
  let outRank = op.results[0].ty.shape.len
  let dims = attrI64s(op, "broadcast_dimensions")
  if dims.len != inRank:
    return false
  for i, dim in dims:
    if dim != int64(outRank - inRank + i):
      return false
  true

proc opcodeIndex(model: var TfliteModel; builtinCode: int32;
    version = 1): int =
  for i, code in model.operatorCodes:
    if code.builtinCode == builtinCode and code.version == version:
      return i
  result = model.operatorCodes.len
  model.operatorCodes.add TfliteOperatorCode(
    builtinCode: builtinCode,
    version: version,
  )

proc addTensor(graph: var TfliteSubgraph; name: string; ty: ShTensorType;
    buffer = 0): int =
  result = graph.tensors.len
  graph.tensors.add initTfliteTensor(name, ty.dtype, ty.shape, buffer)

proc addBuffer(model: var TfliteModel; data: openArray[byte]): int =
  result = model.buffers.len
  model.buffers.add TfliteBuffer(data: @data)

proc addConstTensor(model: var TfliteModel; graph: var TfliteSubgraph;
    name: string; dtype: DType; shape: openArray[int];
    data: openArray[byte]): int =
  let buffer = model.addBuffer(data)
  result = graph.tensors.len
  graph.tensors.add initTfliteTensor(name, dtype, shape, buffer)

proc addShapeTensor(model: var TfliteModel; graph: var TfliteSubgraph;
    baseName: string; values: openArray[int]): int =
  model.addConstTensor(graph, baseName, dtInt32, [values.len],
    int32Bytes(values))

proc requireTensor(tensors: Table[int, int]; id: ShValueId;
    context: string): int =
  if not tensors.hasKey(id.int):
    raise newException(TfliteError,
      context & ": value " & $id & " has no TFLite tensor")
  tensors[id.int]

proc addOperator(model: var TfliteModel; graph: var TfliteSubgraph;
    builtinCode: int32; inputs, outputs: openArray[int];
    options = initTfliteBuiltinOptions(); version = 1) =
  let idx = model.opcodeIndex(builtinCode, version)
  graph.operators.add initTfliteOperator(idx, inputs, outputs, options)

proc denseConstant(op: ShOp; context: string): ShAttr =
  let value = attr(op, "value", context)
  if value.kind != akDenseElements:
    raise newException(TfliteError,
      context & ": stablehlo.constant is not dense elements")
  value

proc singleOutput(op: ShOp; context: string): ShValueId =
  if op.results.len != 1:
    raise newException(TfliteError,
      context & ": only single-result ops are supported")
  op.results[0].id

proc addOutputTensor(graph: var TfliteSubgraph;
    valueTensors: var Table[int, int]; op: ShOp; context: string): int =
  let id = singleOutput(op, context)
  result = graph.addTensor(idName(id), op.results[0].ty)
  valueTensors[id.int] = result

proc binaryCode(kind: ShOpKind): tuple[code: int32; options: TfliteBuiltinOptions] =
  case kind
  of okAdd:
    result = (TfliteBuiltinAdd, TfliteBuiltinOptions(kind: tbokAdd))
  of okSub:
    result = (TfliteBuiltinSub, TfliteBuiltinOptions(kind: tbokSub))
  of okMul:
    result = (TfliteBuiltinMul, TfliteBuiltinOptions(kind: tbokMul))
  of okDiv:
    result = (TfliteBuiltinDiv, TfliteBuiltinOptions(kind: tbokDiv))
  of okMax:
    result = (TfliteBuiltinMaximum,
      TfliteBuiltinOptions(kind: tbokMaximumMinimum))
  of okMin:
    result = (TfliteBuiltinMinimum,
      TfliteBuiltinOptions(kind: tbokMaximumMinimum))
  of okAtan2:
    result = (TfliteBuiltinAtan2, initTfliteBuiltinOptions())
  of okPower:
    result = (TfliteBuiltinPow, initTfliteBuiltinOptions())
  else:
    raise newException(TfliteError,
      "toTfliteModel: StableHLO op " & $kind &
        " is not a supported binary TFLite op")

proc unaryCode(kind: ShOpKind): tuple[code: int32; options: TfliteBuiltinOptions] =
  case kind
  of okNeg:
    result = (TfliteBuiltinNeg, TfliteBuiltinOptions(kind: tbokNeg))
  of okExp:
    result = (TfliteBuiltinExp, TfliteBuiltinOptions(kind: tbokExp))
  of okLog:
    result = (TfliteBuiltinLog, initTfliteBuiltinOptions())
  of okSqrt:
    result = (TfliteBuiltinSqrt, initTfliteBuiltinOptions())
  of okAbs:
    result = (TfliteBuiltinAbs, TfliteBuiltinOptions(kind: tbokAbs))
  of okTanh:
    result = (TfliteBuiltinTanh, initTfliteBuiltinOptions())
  of okLogistic:
    result = (TfliteBuiltinLogistic, initTfliteBuiltinOptions())
  of okSine:
    result = (TfliteBuiltinSin, initTfliteBuiltinOptions())
  of okCosine:
    result = (TfliteBuiltinCos, initTfliteBuiltinOptions())
  of okRsqrt:
    result = (TfliteBuiltinRsqrt, initTfliteBuiltinOptions())
  of okCeil:
    result = (TfliteBuiltinCeil, initTfliteBuiltinOptions())
  of okSign:
    result = (TfliteBuiltinSign, initTfliteBuiltinOptions())
  else:
    raise newException(TfliteError,
      "toTfliteModel: StableHLO op " & $kind &
        " is not a supported unary TFLite op")

proc toIntSeq(values: openArray[int64]): seq[int] =
  result = newSeq[int](values.len)
  for i, value in values:
    result[i] = int(value)

proc toTfliteModel*(module: ShModule; graphName = "";
    description = "rew"): TfliteModel =
  ## Converts a supported rew StableHLO module into a TFLite model.
  ##
  ## Supported exports include dense constants, elementwise ops, standard
  ## `dot_general`/`dot` as `BATCH_MATMUL`, Reshape, Transpose, Concat,
  ## Cast, and trailing `broadcast_in_dim` as `BROADCAST_TO`.
  ## Unsupported IR raises `TfliteError`.
  if module.funcs.len == 0:
    raise newException(TfliteError, "toTfliteModel: module has no functions")

  let fn = module.funcs[0]
  result = TfliteModel(
    version: TfliteSchemaVersion,
    description: description,
    buffers: @[TfliteBuffer()],
  )
  var graph = TfliteSubgraph(name:
    if graphName.len > 0: graphName
    elif module.name.len > 0: module.name
    else: fn.name)
  var valueTensors = initTable[int, int]()

  for i, arg in fn.args:
    let tensor = graph.addTensor("arg" & $i, arg.ty)
    valueTensors[arg.id.int] = tensor
    graph.inputs.add tensor

  var returnOperands: seq[ShValueId] = @[]
  var shapeCounter = 0

  for op in fn.ops:
    case op.kind
    of okReturn:
      returnOperands = op.operands
    of okConstant:
      let value = denseConstant(op, "toTfliteModel constant")
      let id = singleOutput(op, "toTfliteModel constant")
      let tensor = result.addConstTensor(graph, idName(id),
        value.denseDtype, value.denseShape, value.denseBytes)
      valueTensors[id.int] = tensor
    of okAdd, okSub, okMul, okDiv, okMax, okMin, okAtan2, okPower:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel binary op")
      let mapped = binaryCode(op.kind)
      result.addOperator(graph, mapped.code,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel binary lhs"),
         valueTensors.requireTensor(op.operands[1], "toTfliteModel binary rhs")],
        [outTensor], mapped.options)
    of okNeg, okExp, okLog, okSqrt, okAbs, okTanh, okLogistic,
       okSine, okCosine, okRsqrt, okCeil, okSign:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel unary op")
      let mapped = unaryCode(op.kind)
      result.addOperator(graph, mapped.code,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel unary")],
        [outTensor], mapped.options)
    of okDot:
      let lhsTy = tensorType(fn, op.operands[0], "toTfliteModel dot lhs")
      let rhsTy = tensorType(fn, op.operands[1], "toTfliteModel dot rhs")
      if lhsTy.shape.len != 2 or rhsTy.shape.len != 2:
        raise newException(TfliteError,
          "toTfliteModel: stablehlo.dot exports only rank-2 tensors")
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel dot")
      result.addOperator(graph, TfliteBuiltinBatchMatMul,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel dot lhs"),
         valueTensors.requireTensor(op.operands[1], "toTfliteModel dot rhs")],
        [outTensor], TfliteBuiltinOptions(kind: tbokBatchMatMul))
    of okDotGeneral:
      if not isStandardMatMul(fn, op):
        raise newException(TfliteError,
          "toTfliteModel: dot_general is not a standard TFLite BatchMatMul")
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel dot_general")
      result.addOperator(graph, TfliteBuiltinBatchMatMul,
        [valueTensors.requireTensor(op.operands[0],
           "toTfliteModel dot_general lhs"),
         valueTensors.requireTensor(op.operands[1],
           "toTfliteModel dot_general rhs")],
        [outTensor], TfliteBuiltinOptions(kind: tbokBatchMatMul))
    of okReshape:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel reshape")
      let shapeTensor = result.addShapeTensor(graph,
        "shape_reshape_" & $shapeCounter, op.results[0].ty.shape)
      inc shapeCounter
      result.addOperator(graph, TfliteBuiltinReshape,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel reshape"),
         shapeTensor],
        [outTensor], initTfliteReshapeOptions(op.results[0].ty.shape))
    of okTranspose:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel transpose")
      let perm = attrI64s(op, "permutation").toIntSeq
      let permTensor = result.addShapeTensor(graph,
        "shape_transpose_" & $shapeCounter, perm)
      inc shapeCounter
      result.addOperator(graph, TfliteBuiltinTranspose,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel transpose"),
         permTensor],
        [outTensor], TfliteBuiltinOptions(kind: tbokTranspose))
    of okConcatenate:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel concat")
      var inputs = newSeq[int](op.operands.len)
      for i, operand in op.operands:
        inputs[i] = valueTensors.requireTensor(operand, "toTfliteModel concat")
      result.addOperator(graph, TfliteBuiltinConcatenation, inputs,
        [outTensor], initTfliteConcatenationOptions(
          int(attrI64(op, "dimension", 0))))
    of okConvert:
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel cast")
      let inTy = tensorType(fn, op.operands[0], "toTfliteModel cast input")
      result.addOperator(graph, TfliteBuiltinCast,
        [valueTensors.requireTensor(op.operands[0], "toTfliteModel cast")],
        [outTensor], initTfliteCastOptions(inTy.dtype, op.results[0].ty.dtype))
    of okBroadcastInDim:
      if not isTrailingBroadcast(fn, op):
        raise newException(TfliteError,
          "toTfliteModel: only trailing broadcast_in_dim exports as " &
            "TFLite BROADCAST_TO")
      let outTensor = graph.addOutputTensor(valueTensors, op,
        "toTfliteModel broadcast")
      let shapeTensor = result.addShapeTensor(graph,
        "shape_broadcast_" & $shapeCounter, op.results[0].ty.shape)
      inc shapeCounter
      result.addOperator(graph, TfliteBuiltinBroadcastTo,
        [valueTensors.requireTensor(op.operands[0],
           "toTfliteModel broadcast"),
         shapeTensor],
        [outTensor], TfliteBuiltinOptions(kind: tbokBroadcastTo))
    else:
      raise newException(TfliteError,
        "toTfliteModel: StableHLO op " & $op.kind &
          " is not supported by TFLite export")

  if returnOperands.len == 0 and fn.outputTypes.len > 0:
    raise newException(TfliteError,
      "toTfliteModel: function has outputs but no return op")
  for id in returnOperands:
    graph.outputs.add valueTensors.requireTensor(id, "toTfliteModel output")
  result.subgraphs.add graph

# ---- file I/O -------------------------------------------------------------

proc saveTflite*(s: Stream; model: TfliteModel) =
  ## Writes `model` as a `.tflite` FlatBuffer to `s`.
  s.write(bytesToString(model.encodeTflite()))

proc saveTflite*(path: string; model: TfliteModel) =
  ## Writes `model` as a `.tflite` FlatBuffer to `path`.
  let s = newFileStream(path, fmWrite)
  if s.isNil:
    raise newException(IOError, "saveTflite: cannot open '" & path & "'")
  defer: s.close()
  saveTflite(s, model)

proc saveTflite*(s: Stream; module: ShModule; graphName = "";
    description = "rew") =
  ## Converts `module` to TFLite and writes it to `s`.
  saveTflite(s, module.toTfliteModel(graphName, description))

proc saveTflite*(path: string; module: ShModule; graphName = "";
    description = "rew") =
  ## Converts `module` to TFLite and writes it to `path`.
  saveTflite(path, module.toTfliteModel(graphName, description))

proc exportTflite*(s: Stream; module: ShModule; graphName = "";
    description = "rew") =
  ## Alias for `saveTflite(s, module, ...)`.
  saveTflite(s, module, graphName, description)

proc exportTflite*(path: string; module: ShModule; graphName = "";
    description = "rew") =
  ## Alias for `saveTflite(path, module, ...)`.
  saveTflite(path, module, graphName, description)
