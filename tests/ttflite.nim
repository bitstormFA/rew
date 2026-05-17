## Tests for TensorFlow Lite model export.

import std/os
import rew
import rew/xla
import rew/tflite

proc tmpPath(name: string): string =
  getTempDir() / ("rew_tflite_" & name)

proc f32Raw(values: openArray[float32]): seq[byte] =
  result = newSeq[byte](values.len * 4)
  if result.len > 0:
    copyMem(addr result[0], unsafeAddr values[0], result.len)

proc u16(data: string; pos: int): uint16 =
  uint16(ord(data[pos])) or
    (uint16(ord(data[pos + 1])) shl 8)

proc u32(data: string; pos: int): uint32 =
  uint32(ord(data[pos])) or
    (uint32(ord(data[pos + 1])) shl 8) or
    (uint32(ord(data[pos + 2])) shl 16) or
    (uint32(ord(data[pos + 3])) shl 24)

proc i32(data: string; pos: int): int32 =
  cast[int32](u32(data, pos))

proc fieldPos(data: string; table, fieldIndex: int): int =
  let vtable = table - int(i32(data, table))
  let entry = 4 + fieldIndex * 2
  if entry + 2 > int(u16(data, vtable)):
    return -1
  let offset = int(u16(data, vtable + entry))
  if offset == 0:
    return -1
  table + offset

proc tableU32(data: string; table, fieldIndex: int): uint32 =
  let pos = fieldPos(data, table, fieldIndex)
  doAssert pos >= 0
  u32(data, pos)

proc tableU8(data: string; table, fieldIndex: int): int =
  let pos = fieldPos(data, table, fieldIndex)
  doAssert pos >= 0
  ord(data[pos])

proc target(data: string; table, fieldIndex: int): int =
  let pos = fieldPos(data, table, fieldIndex)
  doAssert pos >= 0
  pos + int(u32(data, pos))

proc vectorLen(data: string; vector: int): int =
  int(u32(data, vector))

proc vectorTable(data: string; vector, index: int): int =
  let pos = vector + 4 + index * 4
  pos + int(u32(data, pos))

block tflite_model_writes_valid_flatbuffer_header:
  var builder = initBuilder("tflite_add")
  let args = builder.beginFunc("main",
    [initTensorType(dtFloat32, [2])], [])
  let c = builder.constant(dtFloat32, [2], f32Raw([3.0'f32, 4.0'f32]))
  let y = builder.add(args[0], c)
  builder.setCurrentOutputTypes([builder.getType(y)])
  builder.returnOp([y])
  builder.endFunc()

  let module = builder.build()
  let model = module.toTfliteModel()
  doAssert model.version == TfliteSchemaVersion
  doAssert model.operatorCodes.len == 1
  doAssert model.operatorCodes[0].builtinCode == TfliteBuiltinAdd
  doAssert model.buffers.len == 2

  let path = tmpPath("add.tflite")
  saveTflite(path, model)
  defer: removeFile(path)
  let data = readFile(path)
  doAssert data.len > 8
  doAssert data[4 .. 7] == "TFL3"

  let root = int(u32(data, 0))
  doAssert tableU32(data, root, 0) == uint32(TfliteSchemaVersion)
  let opCodes = target(data, root, 1)
  doAssert vectorLen(data, opCodes) == 1
  let opCode = vectorTable(data, opCodes, 0)
  doAssert tableU8(data, opCode, 0) == int(TfliteBuiltinAdd)
  let subgraphs = target(data, root, 2)
  doAssert vectorLen(data, subgraphs) == 1
  let buffers = target(data, root, 4)
  doAssert vectorLen(data, buffers) == 2

block tflite_export_handles_linear_style_graph:
  var builder = initBuilder("tflite_linear")
  let args = builder.beginFunc("main",
    [initTensorType(dtFloat32, [2, 3])], [])
  let w = builder.constant(dtFloat32, [3, 4],
    f32Raw([0.0'f32, 1.0'f32, 2.0'f32, 3.0'f32,
            4.0'f32, 5.0'f32, 6.0'f32, 7.0'f32,
            8.0'f32, 9.0'f32, 10.0'f32, 11.0'f32]))
  let b = builder.constant(dtFloat32, [4],
    f32Raw([1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]))
  let xw = builder.dotGeneral(args[0], w, [], [], [1], [0])
  let bias = builder.broadcastInDim(b, [2, 4], [1])
  let y = builder.add(xw, bias)
  builder.setCurrentOutputTypes([builder.getType(y)])
  builder.returnOp([y])
  builder.endFunc()

  let model = builder.build().toTfliteModel()
  let graph = model.subgraphs[0]
  doAssert graph.inputs.len == 1
  doAssert graph.outputs.len == 1
  doAssert graph.operators.len == 3
  doAssert model.operatorCodes[graph.operators[0].opcodeIndex].builtinCode ==
    TfliteBuiltinBatchMatMul
  doAssert model.operatorCodes[graph.operators[1].opcodeIndex].builtinCode ==
    TfliteBuiltinBroadcastTo
  doAssert model.operatorCodes[graph.operators[2].opcodeIndex].builtinCode ==
    TfliteBuiltinAdd

block tflite_stream_export_helper:
  var builder = initBuilder("tflite_stream")
  let args = builder.beginFunc("main",
    [initTensorType(dtFloat32, [1, 2])], [])
  let y = builder.reshape(args[0], [2])
  builder.setCurrentOutputTypes([builder.getType(y)])
  builder.returnOp([y])
  builder.endFunc()

  let path = tmpPath("stream.tflite")
  exportTflite(path, builder.build())
  defer: removeFile(path)
  let data = readFile(path)
  doAssert data[4 .. 7] == "TFL3"
  let root = int(u32(data, 0))
  let opCodes = target(data, root, 1)
  let opCode = vectorTable(data, opCodes, 0)
  doAssert tableU8(data, opCode, 0) == int(TfliteBuiltinReshape)
