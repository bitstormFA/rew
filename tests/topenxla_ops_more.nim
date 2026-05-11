## Additional OpenXLA op coverage: dynamic broadcast, FFT, einsum,
## triangular_solve, and torch_index_select.

import rew
import std/strutils

let TestDevice = cpu(0)

template assertContains(haystack, needle: string) =
  doAssert needle in haystack,
    "expected " & needle.escape & " in:\n" & haystack

block builder_openxla_more_tensor_ops:
  var b = initBuilder("more")
  let f32_1x3 = initTensorType(dtFloat32, [1, 3])
  let dimsTy = initTensorType(dtInt32, [3])
  let c64x4 = initTensorType(dtComplex64, [4])
  let f32x4 = initTensorType(dtFloat32, [4])
  let f32_3x3 = initTensorType(dtFloat32, [3, 3])
  let f32_4x16 = initTensorType(dtFloat32, [4, 16])
  let f32_16x4 = initTensorType(dtFloat32, [16, 4])
  let selectOperandTy = initTensorType(dtFloat32, [8, 128, 3072, 64])
  let indexTy = initTensorType(dtInt32, [8, 16, 1024])
  let outputs = [
    initTensorType(dtFloat32, [2, 3, 2]),
    c64x4,
    initTensorType(dtComplex64, [3]),
    f32_3x3,
    initTensorType(dtFloat32, [4, 4]),
    f32x4,
    initTensorType(dtFloat32, [8, 128, 16, 1024, 64]),
  ]
  let args = b.beginFunc("main",
    [f32_1x3, dimsTy, c64x4, f32x4, f32_3x3, f32_3x3,
     f32_4x16, f32_16x4, f32_4x16, selectOperandTy, indexTy],
    outputs)
  let dyn = b.dynamicBroadcastInDim(args[0], args[1], [2, 3, 2],
    [2, 1], knownExpandingDimensions = [0],
    knownNonexpandingDimensions = [1])
  let fftc = b.fft(args[2], ftFft, [4])
  let rfft = b.fft(args[3], ftRfft, [4])
  let tri = b.triangularSolve(args[4], args[5],
    transposeA = tkNoTranspose)
  let ein = b.einsum(args[6], args[7], "ab,bc->ac", [4, 4])
  let uein = b.unaryEinsum(args[8], "ab->a", [4])
  let selected = b.torchIndexSelect(args[9], args[10], 2, 1)
  b.returnOp([dyn, fftc, rfft, tri, ein, uein, selected])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  for needle in [
    "stablehlo.dynamic_broadcast_in_dim",
    "stablehlo.fft",
    "stablehlo.triangular_solve",
    "stablehlo.einsum",
    "stablehlo.unary_einsum",
    "stablehlo.torch_index_select",
  ]:
    assertContains text, needle
  assertContains text, "fft_type FFT"
  assertContains text, "einsum_config = \"ab,bc->ac\""
  assertContains text, "tensor<8x128x16x1024x64xf32>"

block trace_openxla_more_tensor_ops:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(
      @[dtFloat32, dtComplex64, dtFloat32, dtFloat32, dtFloat32,
        dtFloat32, dtFloat32, dtFloat32, dtInt32],
      @[@[1, 3], @[4], @[4], @[3, 3], @[3, 3],
        @[4, 16], @[16, 4], @[8, 128, 3072, 64], @[8, 16, 1024]])
    let outDims = constant(dtInt32, [3],
      i32Bytes([2'i32, 3'i32, 2'i32]))
    let dyn = dynamicBroadcastInDim(inputs[0], outDims, [2, 3, 2],
      [2, 1], knownExpandingDimensions = [0],
      knownNonexpandingDimensions = [1])
    let fftc = fft(inputs[1], ftFft, [4])
    let rfft = fft(inputs[2], ftRfft, [4])
    let tri = triangularSolve(inputs[3], inputs[4])
    let ein = einsum(inputs[5], inputs[6], "ab,bc->ac", [4, 4])
    let uein = unaryEinsum(inputs[5], "ab->a", [4])
    let selected = torchIndexSelect(inputs[7], inputs[8], 2, 1)
    doAssert dyn.shape == @[2, 3, 2]
    doAssert fftc.dtype == dtComplex64
    doAssert rfft.dtype == dtComplex64
    doAssert rfft.shape == @[3]
    doAssert tri.shape == @[3, 3]
    doAssert ein.shape == @[4, 4]
    doAssert uein.shape == @[4]
    doAssert selected.shape == @[8, 128, 16, 1024, 64]
    ctx.traceReturn([dyn, fftc, rfft, tri, ein, uein, selected])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  for needle in [
    "stablehlo.dynamic_broadcast_in_dim",
    "stablehlo.fft",
    "stablehlo.triangular_solve",
    "stablehlo.einsum",
    "stablehlo.unary_einsum",
    "stablehlo.torch_index_select",
  ]:
    assertContains text, needle

echo "topenxla_ops_more: OK"
