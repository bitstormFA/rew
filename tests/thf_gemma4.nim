import std/[json, os, streams, strutils, tables]
import rew

proc tempPath(name: string): string =
  getTempDir() / ("rew_" & $getCurrentProcessId() & "_" & name)

block safetensorsMetadataAndSlices:
  let path = tempPath("tiny.safetensors")
  var tensors = initTable[string, tuple[dtype: DType; shape: seq[int];
    data: seq[byte]]]()
  tensors["x"] = (dtFloat32, @[2], f32Bytes([1'f32, 2'f32]))
  saveSafeTensors(path, tensors)
  let meta = loadSafeTensorMetadata(path)
  doAssert meta.hasTensor("x")
  doAssert meta.tensors["x"].shape == @[2]
  doAssert meta.tensorData("x") == tensors["x"].data
  removeFile(path)

block safetensorsRejectMalformedOffsets:
  let path = tempPath("bad.safetensors")
  let header = """{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"""
  let s = newFileStream(path, fmWrite)
  for i in 0 ..< 8:
    s.write char((uint64(header.len) shr (i * 8)) and 0xFF'u64)
  s.write(header)
  s.close()
  var failed = false
  try:
    discard loadSafeTensorMetadata(path)
  except SafeTensorError:
    failed = true
  doAssert failed
  removeFile(path)

block rowsPayloadParser:
  let payload = %* {
    "num_rows_total": 1,
    "rows": [
      {"row_idx": 0, "row": {"id": "a"}, "truncated_cells": []}
    ]
  }
  let parsed = parseHfRowsPayload(payload)
  doAssert parsed.total == 1
  doAssert parsed.rows.len == 1
  doAssert parsed.rows[0].row["id"].getStr() == "a"

block jsonFileDataset:
  let path = tempPath("data.jsonl")
  writeFile(path, """{"x":1}
{"x":2}
""")
  var sum = 0
  for row in fromHfJsonFile(path):
    sum += row["x"].getInt()
  doAssert sum == 3
  removeFile(path)

block tokenizerMiniBpe:
  const marker = "\226\150\129"
  let path = tempPath("tokenizer.json")
  var vocab = newJObject()
  vocab["<unk>"] = %0
  vocab["<bos>"] = %1
  vocab["<eos>"] = %2
  vocab[marker] = %3
  vocab["H"] = %4
  vocab["i"] = %5
  vocab["Hi"] = %6
  let node = %* {
    "model": {
      "type": "BPE",
      "unk_token": "<unk>",
      "vocab": vocab,
      "merges": ["H i"]
    },
    "added_tokens": [
      {"id": 1, "content": "<bos>", "special": true},
      {"id": 2, "content": "<eos>", "special": true}
    ]
  }
  writeFile(path, $node)
  let tok = loadHfTokenizer(path)
  let ids = tok.encode("Hi", addBos = true, addEos = true)
  doAssert ids == @[1, 6, 2]
  doAssert tok.decode(ids) == "<bos>Hi<eos>"
  removeFile(path)

block gemma4HermesFormatting:
  let tools = %* [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }
  ]
  let row = %* {
    "tools": $tools,
    "conversations": [
      {"from": "system", "value": "Use tools.\n<tools>[]</tools>"},
      {"from": "human", "value": "Weather in Berlin?"},
      {"from": "gpt", "value": "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Berlin\"}}</tool_call>"}
    ]
  }
  let text = formatGemma4ToolSft(row)
  doAssert text.startsWith("<bos><|turn>system")
  doAssert "declaration:get_weather" in text
  doAssert "<|tool_call>call:get_weather{city:<|\"|>Berlin<|\"|>}" in text

block qloraLinearTrace:
  withTrace(ctx, "qlora_linear_trace", defaultDevice()):
    let xs = ctx.traceInputs([dtFloat32], [@[2, 4]])
    let layer = initQloraLinearFromF32(initKey(42),
      @[0.1'f32, 0.2'f32, 0.3'f32,
        0.4'f32, 0.5'f32, 0.6'f32,
        0.7'f32, 0.8'f32, 0.9'f32,
        1.0'f32, 1.1'f32, 1.2'f32],
      4, 3, rank = 2, alpha = 2'f32, groupSize = 4)
    let y = layer.forward(xs[0])
    doAssert y.shape == @[2, 3]
    ctx.traceReturn([y])
    discard ctx.builder.build()

block gemma4TinyForwardTrace:
  withTrace(ctx, "gemma4_tiny_forward", defaultDevice()):
    let xs = ctx.traceInputs([dtFloat32], [@[1, 2, 12]])
    let cfg = initGemma4TextConfig(
      vocabSize = 12,
      hiddenSize = 8,
      hiddenSizePerLayerInput = 0,
      intermediateSize = 16,
      numHiddenLayers = 1,
      numAttentionHeads = 2,
      numKeyValueHeads = 2,
      headDim = 4,
      finalLogitSoftcapping = 0'f32,
      layerTypes = @[g4FullAttention],
    )
    let model = initGemma4TextForCausalLM(initKey(7), cfg)
    let logits = model.forward(xs[0])
    doAssert logits.shape == @[1, 2, 12]
    ctx.traceReturn([logits])
    discard ctx.builder.build()
