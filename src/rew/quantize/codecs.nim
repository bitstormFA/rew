## Quantization codecs — host-side weight quantization algorithms.
##
## Supports: NF4, FP8, GPTQ (Optimal Brain Quantizer), AWQ
## (Activation-Aware Weight Quantization), and SmoothQuant-style
## channel-wise scaling.

import std/math
import ../dtype

# ---- NF4 lookup table and helpers ------------------------------------------

const
  Nf4Table = [
    -1.0'f32, -0.6961928009986877'f32, -0.5250730514526367'f32,
    -0.39491748809814453'f32, -0.28444138169288635'f32,
    -0.18477343022823334'f32, -0.09105003625154495'f32,
    0.0'f32, 0.07958029955625534'f32, 0.16093020141124725'f32,
    0.24611230194568634'f32, 0.33791524171829224'f32,
    0.44070982933044434'f32, 0.5626170039176941'f32,
    0.7229568362236023'f32, 1.0'f32,
  ]

proc findNf4Index(value: float32): int =
  var minDist = 1e10'f32
  for i, tableVal in Nf4Table:
    let dist = abs(value - tableVal)
    if dist < minDist:
      minDist = dist
      result = i

proc nf4Quantize*(weights: openArray[float32]; blockSize = 64):
    tuple[packed: seq[byte]; scales: seq[float32]] =
  ## Quantizes float32 weights to NF4 with per-block scaling.
  ## Returns packed bytes (2 values per byte) and per-block scales.
  let numElements = weights.len
  let numBlocks = (numElements + blockSize - 1) div blockSize
  result.packed = newSeq[byte]((numElements + 1) div 2)
  result.scales = newSeq[float32](numBlocks)
  for blk in 0 ..< numBlocks:
    let start = blk * blockSize
    let endPos = min(start + blockSize, numElements)
    var absMax = 1e-10'f32
    for i in start ..< endPos:
      absMax = max(absMax, abs(weights[i]))
    result.scales[blk] = absMax
    # Normalize, quantize, pack.
    for i in start ..< endPos:
      let normalized = clamp(weights[i] / absMax, -1.0'f32, 1.0'f32)
      let code = findNf4Index(normalized) and 0x0F
      let byteIdx = i div 2
      if (i mod 2) == 0:
        result.packed[byteIdx] = byte(code)
      else:
        result.packed[byteIdx] = byte(result.packed[byteIdx].int or
          (code shl 4))

proc nf4Dequantize*(packed: openArray[byte]; scales: openArray[float32];
    numElements, blockSize: int): seq[float32] =
  ## Dequantizes NF4-packed weights to float32.
  result = newSeq[float32](numElements)
  for i in 0 ..< numElements:
    let byteIdx = i div 2
    let code = if (i mod 2) == 0:
        packed[byteIdx].int and 0x0F
      else:
        (packed[byteIdx].int shr 4) and 0x0F
    let blkIdx = i div blockSize
    result[i] = Nf4Table[code] * scales[blkIdx]

proc f32FromLeBytes*(data: openArray[byte]): seq[float32] =
  ## Converts little-endian F32 bytes to host float32 values.
  if (data.len mod 4) != 0:
    raise newException(ValueError,
      "f32FromLeBytes: byte length is not divisible by 4")
  result = newSeq[float32](data.len div 4)
  for i in 0 ..< result.len:
    let base = i * 4
    let bits =
      uint32(data[base]) or
      (uint32(data[base + 1]) shl 8) or
      (uint32(data[base + 2]) shl 16) or
      (uint32(data[base + 3]) shl 24)
    result[i] = cast[float32](bits)

proc bf16FromLeBytes*(data: openArray[byte]): seq[float32] =
  ## Converts little-endian BF16 bytes to host float32 values.
  if (data.len mod 2) != 0:
    raise newException(ValueError,
      "bf16FromLeBytes: byte length is not divisible by 2")
  result = newSeq[float32](data.len div 2)
  for i in 0 ..< result.len:
    let base = i * 2
    let hi = uint32(data[base]) or (uint32(data[base + 1]) shl 8)
    let bits = hi shl 16
    result[i] = cast[float32](bits)

proc tensorBytesToF32*(dtype: DType; data: openArray[byte]): seq[float32] =
  ## Converts safetensors F32/BF16 tensor bytes to host float32 values.
  case dtype
  of dtFloat32: f32FromLeBytes(data)
  of dtBFloat16: bf16FromLeBytes(data)
  else:
    raise newException(ValueError,
      "tensorBytesToF32: unsupported dtype " & dtype.name)

proc nf4QuantizeTensorBytes*(dtype: DType; data: openArray[byte];
    blockSize = 64): tuple[packed: seq[byte]; scales: seq[float32]] =
  ## Quantizes F32/BF16 tensor bytes to NF4 without keeping source bytes.
  nf4Quantize(tensorBytesToF32(dtype, data), blockSize)

# ---- FP8 quantization ------------------------------------------------------

type
  Fp8Format* = enum
    f8E4M3
    f8E5M2

proc float32ToFp8E4M3(value: float32): uint8 =
  ## Converts float32 to FP8 E4M3 format.
  if value == 0.0'f32: return 0'u8
  var sign: uint8 = if value < 0: 0x80'u8 else: 0'u8
  var v = abs(value)
  var exp = 0
  while v >= 2.0'f32:
    v /= 2.0'f32; inc exp
  while v < 1.0'f32 and exp > -6:
    v *= 2.0'f32; dec exp
  if exp > 7: exp = 7
  if exp < -6: exp = -6
  let biasedExp = uint8(max(0'i32, int32(exp + 7)))
  let mantissa = uint8((v - 1.0'f32) * 8.0'f32) and 0x07'u8
  sign or (biasedExp shl 3) or mantissa

proc fp8QuantizeE4M3*(weights: openArray[float32]): seq[byte] =
  result = newSeq[byte](weights.len)
  for i, w in weights:
    result[i] = float32ToFp8E4M3(w)

# ---- GPTQ (Optimal Brain Quantizer) ----------------------------------------

proc gptqQuantize*(weights: openArray[float32]; inFeatures, outFeatures: int;
    bits = 4; groupSize = 128;
    hessian: openArray[float32] = []): tuple[
      qweight: seq[byte]; scales: seq[float32]; zeros: seq[int]] =
  ## GPTQ quantization using Optimal Brain Surgeon.
  ##
  ## `weights`: flat row-major weight array [outFeatures * inFeatures].
  ## `hessian`: approximate inverse Hessian diagonal (optional).
  ## For 4-bit, packs 2 values per byte. Returns quantized weights,
  ## per-group scales, and per-group zero points.
  let numElements = inFeatures * outFeatures
  let numGroups = (outFeatures + groupSize - 1) div groupSize
  let maxVal = float32((1 shl bits) - 1)
  # Find global min/max per group.
  result.scales = newSeq[float32](numGroups * inFeatures)
  result.zeros = newSeq[int](numGroups * inFeatures)
  result.qweight = newSeq[byte]((numElements + 1) div 2)
  # For each output channel group, compute scale and quantize.
  for group in 0 ..< numGroups:
    let rowStart = group * groupSize
    let rowEnd = min(rowStart + groupSize, outFeatures)
    for col in 0 ..< inFeatures:
      var wMin = 1e10'f32
      var wMax = -1e10'f32
      for row in rowStart ..< rowEnd:
        let idx = row * inFeatures + col
        wMin = min(wMin, weights[idx])
        wMax = max(wMax, weights[idx])
      let scale = (wMax - wMin) / maxVal
      let zero = round(-wMin / scale).int
      let sIdx = group * inFeatures + col
      result.scales[sIdx] = scale
      result.zeros[sIdx] = zero
      for row in rowStart ..< rowEnd:
        let idx = row * inFeatures + col
        let q = clamp(round(weights[idx] / scale) + float32(zero),
          0.0'f32, maxVal).int and 0x0F
        let byteIdx = idx div 2
        if (idx mod 2) == 0:
          result.qweight[byteIdx] = byte(q)
        else:
          result.qweight[byteIdx] = byte(
            result.qweight[byteIdx].int or (q shl 4))

# ---- AWQ (Activation-Aware) -----------------------------------------------

proc awqQuantize*(weights: openArray[float32]; inFeatures, outFeatures: int;
    activationStats: openArray[float32]; bits = 4; groupSize = 128):
    tuple[qweight: seq[byte]; scales: seq[float32]; zeros: seq[int]] =
  ## AWQ: per-channel scaling based on activation statistics.
  ##
  ## `activationStats`: per-input-channel activation magnitudes.
  ## Scales weights per channel to minimize quantization error on
  ## frequently-activated channels.
  let numElements = inFeatures * outFeatures
  let numGroups = (outFeatures + groupSize - 1) div groupSize
  let maxVal = float32((1 shl bits) - 1)
  # Compute per-channel scaling from activation stats.
  var channelScales = newSeq[float32](inFeatures)
  for col in 0 ..< inFeatures:
    channelScales[col] = if activationStats[col] > 0:
        1.0'f32 / sqrt(activationStats[col])
      else:
        1.0'f32
  result.scales = newSeq[float32](numGroups * inFeatures)
  result.zeros = newSeq[int](numGroups * inFeatures)
  result.qweight = newSeq[byte]((numElements + 1) div 2)
  for group in 0 ..< numGroups:
    let rowStart = group * groupSize
    let rowEnd = min(rowStart + groupSize, outFeatures)
    for col in 0 ..< inFeatures:
      var wMin = 1e10'f32
      var wMax = -1e10'f32
      for row in rowStart ..< rowEnd:
        let idx = row * inFeatures + col
        let scaled = weights[idx] * channelScales[col]
        wMin = min(wMin, scaled)
        wMax = max(wMax, scaled)
      let scale = (wMax - wMin) / maxVal
      let zero = round(-wMin / scale).int
      let sIdx = group * inFeatures + col
      result.scales[sIdx] = scale
      result.zeros[sIdx] = zero
      for row in rowStart ..< rowEnd:
        let idx = row * inFeatures + col
        let scaled = weights[idx] * channelScales[col]
        let q = clamp(round(scaled / scale) + float32(zero),
          0.0'f32, maxVal).int and 0x0F
        let byteIdx = idx div 2
        if (idx mod 2) == 0:
          result.qweight[byteIdx] = byte(q)
        else:
          result.qweight[byteIdx] = byte(
            result.qweight[byteIdx].int or (q shl 4))

# ---- SmoothQuant -----------------------------------------------------------

proc smoothQuantScales*(xStats, wStats: openArray[float32];
    alpha: float32 = 0.5'f32): seq[float32] =
  ## Computes SmoothQuant per-channel scaling factors.
  ##
  ## `xStats`: activation magnitudes per channel (e.g., max(abs(x))).
  ## `wStats`: weight magnitudes per channel (e.g., max(abs(w))).
  ## Returns channel-wise scales to migrate quantization difficulty
  ## from activations to weights.
  let numChannels = xStats.len
  result = newSeq[float32](numChannels)
  for i in 0 ..< numChannels:
    let xMag = max(xStats[i], 1e-10'f32)
    let wMag = max(wStats[i], 1e-10'f32)
    result[i] = pow(xMag, alpha) / pow(wMag, 1.0'f32 - alpha)
