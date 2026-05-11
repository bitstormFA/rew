## Tests for Llama architecture.

import rew

let TestDevice = cpu(0)

block llama_config:
  let cfg = initLlamaConfig(
    vocabSize = 32000, hiddenSize = 256, intermediateSize = 512,
    numHiddenLayers = 2, numAttentionHeads = 4, numKeyValueHeads = 2,
    maxPositionEmbeddings = 512, ropeTheta = 10000.0)
  doAssert cfg.hiddenSize == 256
  doAssert cfg.numKeyValueHeads == 2

block llama_decoder_layer_trace:
  ## Test decoder layer directly (no embedding involved).
  let key = initKey(42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 64]])
    let rope = initRotaryPositionEncoding(16, 128,
      theta = 10000.0'f64)
    let layer = initGroupedQueryAttention(key, 64, 4, 2, rope = rope)
    doAssert layer.hasRope
    let y = layer.forward(inputs[0], inputs[0], inputs[0], causal = true)
    doAssert y.shape == @[1, 4, 64]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block llama_model_trace_one_hot:
  ## Full model test with one-hot input.
  let key = initKey(99u64)
  let cfg = initLlamaConfig(
    vocabSize = 50, hiddenSize = 32, intermediateSize = 64,
    numHiddenLayers = 1, numAttentionHeads = 2, numKeyValueHeads = 1,
    maxPositionEmbeddings = 32, ropeTheta = 500000.0)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2, 50]])
    let model = initLlamaModel(key, cfg)
    doAssert model.layers.len == 1
    let y = model.forward(inputs[0], causal = true)
    doAssert y.shape == @[1, 2, 32]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block llama_causal_lm_trace:
  let key = initKey(77u64)
  let cfg = initLlamaConfig(
    vocabSize = 50, hiddenSize = 32, intermediateSize = 64,
    numHiddenLayers = 1, numAttentionHeads = 2, numKeyValueHeads = 1,
    maxPositionEmbeddings = 32, ropeTheta = 10000.0)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 2, 50]])
    let model = initLlamaForCausalLM(key, cfg)
    let y = model.forward(inputs[0], causal = true)
    doAssert y.shape == @[1, 2, cfg.vocabSize]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block llama_jit_lower:
  let key = initKey(7u64)
  let cfg = initLlamaConfig(
    vocabSize = 30, hiddenSize = 16, intermediateSize = 32,
    numHiddenLayers = 1, numAttentionHeads = 2, numKeyValueHeads = 1,
    maxPositionEmbeddings = 16, ropeTheta = 500000.0)
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let model = initLlamaForCausalLM(key, cfg)
    @[model.forward(args[0])]
  let jitted = jit(f)
  let x = initTraceTensor(ShValueId(1), dtFloat32, @[1, 2, cfg.vocabSize],
    TestDevice)
  let m = jitted.lower([x])
  doAssert m.funcs.len == 1

echo "All Llama tests passed"
