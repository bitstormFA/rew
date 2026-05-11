## Tests for ONNX model import/export.

import std/os
import rew

proc tmpPath(name: string): string =
  getTempDir() / ("rew_onnx_" & name)

proc f32Raw(values: openArray[float32]): seq[byte] =
  result = newSeq[byte](values.len * 4)
  if result.len > 0:
    copyMem(addr result[0], unsafeAddr values[0], result.len)

block onnx_model_round_trips_through_file:
  let weight = initOnnxTensor("bias", dtFloat32, [2],
    f32Raw([1.0'f32, 2.0'f32]))
  let graph = initOnnxGraph("roundtrip",
    inputs = [initOnnxValueInfo("x", dtFloat32, [2])],
    outputs = [initOnnxValueInfo("y", dtFloat32, [2])],
    nodes = [initOnnxNode("Add", ["x", "bias"], ["y"])],
    initializers = [weight])
  let model = initOnnxModel(graph)
  let path = tmpPath("roundtrip.onnx")
  saveOnnx(path, model)
  defer: removeFile(path)

  let loaded = loadOnnx(path)
  doAssert loaded.irVersion == OnnxDefaultIrVersion
  doAssert loaded.opsets.len == 1
  doAssert loaded.graph.name == "roundtrip"
  doAssert loaded.graph.nodes.len == 1
  doAssert loaded.graph.nodes[0].opType == "Add"
  doAssert loaded.graph.initializers[0].name == "bias"
  doAssert loaded.graph.initializers[0].data == weight.data

block onnx_import_lowers_add_with_initializer:
  let bias = initOnnxTensor("bias", dtFloat32, [2],
    f32Raw([10.0'f32, 20.0'f32]))
  let graph = initOnnxGraph("import_add",
    inputs = [initOnnxValueInfo("x", dtFloat32, [2])],
    outputs = [initOnnxValueInfo("y", dtFloat32, [2])],
    nodes = [initOnnxNode("Add", ["x", "bias"], ["y"])],
    initializers = [bias])
  let module = initOnnxModel(graph).toShModule()
  doAssert module.funcs.len == 1
  let fn = module.funcs[0]
  doAssert fn.inputTypes == @[initTensorType(dtFloat32, [2])]
  doAssert fn.outputTypes == @[initTensorType(dtFloat32, [2])]
  doAssert fn.ops.len == 3
  doAssert fn.ops[0].kind == okConstant
  doAssert fn.ops[1].kind == okAdd
  doAssert fn.ops[2].kind == okReturn

block stablehlo_export_writes_add_and_constant_initializer:
  var builder = initBuilder("export_add")
  let args = builder.beginFunc("main",
    [initTensorType(dtFloat32, [2])], [])
  let c = builder.constant(dtFloat32, [2], f32Raw([3.0'f32, 4.0'f32]))
  let y = builder.add(args[0], c)
  builder.setCurrentOutputTypes([builder.getType(y)])
  builder.returnOp([y])
  builder.endFunc()

  let model = builder.build().toOnnxModel()
  doAssert model.graph.inputs.len == 1
  doAssert model.graph.outputs.len == 1
  doAssert model.graph.initializers.len == 1
  doAssert model.graph.nodes.len == 1
  doAssert model.graph.nodes[0].opType == "Add"

  let path = tmpPath("export_add.onnx")
  saveOnnx(path, model)
  defer: removeFile(path)
  let loaded = loadOnnx(path)
  doAssert loaded.graph.nodes[0].opType == "Add"
  doAssert loaded.graph.initializers[0].data == f32Raw([3.0'f32, 4.0'f32])

block import_and_export_stream_helpers:
  let graph = initOnnxGraph("stream",
    inputs = [initOnnxValueInfo("x", dtFloat32, [1])],
    outputs = [initOnnxValueInfo("y", dtFloat32, [1])],
    nodes = [initOnnxNode("Identity", ["x"], ["y"])])
  let model = initOnnxModel(graph)
  let path = tmpPath("stream.onnx")
  saveOnnx(path, model)
  defer: removeFile(path)

  let imported = importOnnx(path)
  doAssert imported.funcs[0].outputTypes == @[initTensorType(dtFloat32, [1])]

  let outPath = tmpPath("stream_export.onnx")
  exportOnnx(outPath, imported)
  defer: removeFile(outPath)
  let exported = loadOnnx(outPath)
  doAssert exported.graph.outputs[0].shape == @[1]
