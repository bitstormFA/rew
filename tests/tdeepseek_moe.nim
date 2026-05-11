## Tests for DeepSeek MoE routing.

import std/strutils
import rew

let TestDevice = cpu(0)

block deepseek_moe_rejects_invalid_topk:
  var raised = false
  try:
    withTrace ctx, "main", TestDevice:
      discard initDeepSeekMoE(initKey(1'u64), 16, 32, 1, 2, 3)
  except TensorError:
    raised = true
  doAssert raised

block deepseek_moe_topk_routing_trace:
  let key = initKey(7'u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 16]])
    let moe = initDeepSeekMoE(key,
      hiddenSize = 16, intermediateSize = 32,
      numShared = 1, numRouted = 4, numExpertsPerTok = 2)
    let y = moe.forward(inputs[0])
    doAssert y.shape == @[3, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.sort" in text
  doAssert "stablehlo.compare" in text

block deepseek_moe_without_shared_experts:
  let key = initKey(8'u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 8]])
    let moe = initDeepSeekMoE(key,
      hiddenSize = 8, intermediateSize = 16,
      numShared = 0, numRouted = 3, numExpertsPerTok = 1)
    let y = moe.forward(inputs[0])
    doAssert y.shape == @[2, 8]
    ctx.traceReturn([y])
  verify(ctx.builder.build())

echo "All DeepSeek MoE tests passed"
