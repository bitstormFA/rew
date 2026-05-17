## Trace-level tests for multimodal model helpers.

import std/strutils
import rew
import rew/xla

let TestDevice = cpu(0)

block vit_config_helpers:
  let cfg = initViTConfig(
    imageSize = 32, patchSize = 8, numChannels = 3,
    hiddenSize = 64, numLayers = 2, numHeads = 4,
    numClasses = 10)
  doAssert cfg.numPatches == 16
  doAssert cfg.patchDim == 192
  doAssert cfg.mlpHiddenSize == 256

block vit_rejects_invalid_patch_grid:
  var raised = false
  try:
    withTrace ctx, "main", TestDevice:
      discard initViT(initKey(1'u64), initViTConfig(
        imageSize = 30, patchSize = 8, numChannels = 3,
        hiddenSize = 32, numLayers = 1, numHeads = 4,
        numClasses = 5))
  except TensorError:
    raised = true
  doAssert raised

block vit_image_forward_trace:
  let key = initKey(42'u64)
  let cfg = initViTConfig(
    imageSize = 32, patchSize = 8, numChannels = 3,
    hiddenSize = 64, numLayers = 2, numHeads = 4,
    numClasses = 10, dropout = 0'f32)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 32, 32, 3]])
    let model = initViT(key, cfg)
    let patches = model.patchify(inputs[0])
    doAssert patches.shape == @[2, 16, 192]
    let features = model.forwardFeatures(patches, training = false)
    doAssert features.shape == @[2, 64]
    let logits = model.forward(inputs[0], training = false)
    doAssert logits.shape == @[2, 10]
    ctx.traceReturn([logits])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.concatenate" in text
  doAssert "stablehlo.dot_general" in text

block vit_patch_forward_compatibility:
  let key = initKey(43'u64)
  let cfg = initViTConfig(
    imageSize = 8, patchSize = 4, numChannels = 3,
    hiddenSize = 32, numLayers = 1, numHeads = 4,
    numClasses = 7, dropout = 0'f32)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 4, 48]])
    let model = initViT(key, cfg)
    let logits = model.forward(inputs[0], training = false)
    doAssert logits.shape == @[1, 7]
    ctx.traceReturn([logits])
  verify(ctx.builder.build())

block clip_contrastive_trace:
  let key = initKey(99'u64)
  let cfg = initClipConfig(
    visionHiddenSize = 32, textHiddenSize = 24, projectionDim = 16)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[3, 32], @[5, 24]])
    let model = initClipModel(key, cfg)
    let embeds = model.forward(inputs[0], inputs[1])
    doAssert embeds.imageEmbeds.shape == @[3, 16]
    doAssert embeds.textEmbeds.shape == @[5, 16]
    let clipOut = model.forwardContrastive(inputs[0], inputs[1])
    doAssert clipOut.imageEmbeds.shape == @[3, 16]
    doAssert clipOut.textEmbeds.shape == @[5, 16]
    doAssert clipOut.logitsPerImage.shape == @[3, 5]
    doAssert clipOut.logitsPerText.shape == @[5, 3]
    ctx.traceReturn([clipOut.logitsPerImage, clipOut.logitsPerText])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.sqrt" in text

echo "All multimodal tests passed"
