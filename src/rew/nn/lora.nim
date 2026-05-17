## LoRA (Low-Rank Adaptation) — Hu et al. (2021).
##
## Efficiency finetuning: freeze pretrained weights, add trainable
## low-rank matrices A and B such that the effective weight is
## `W + (alpha/rank) * (B @ A)`.
##
## QLoRA (Dettmers et al., 2023): LoRA on top of NF4-quantized base weights
## for memory-efficient finetuning of large models.
##
## Pure value types following rew's functional nn invariant.

import std/math
import ../tensor
import ../pytree
import ../dtype
import ../rng
import ../ops/shape
import ../ops/linalg
import ../ops/arith
import ../ops/literal
import ../quantize/codecs
import ./linear
import ./init

type
  FrozenLinear* = object
    ## Frozen linear projection used as the base path for LoRA adapters.
    weight*: Buffer[Tensor]
    bias*: Buffer[Tensor]

  LoraLinear* = object
    ## A linear layer augmented with low-rank adaptation.
    ##
    ## `W + (alpha/rank) * (B @ A)` where:
    ##   W  = base weight `[inFeatures, outFeatures]`
    ##   A  = `[rank, inFeatures]`  — down-projection
    ##   B  = `[outFeatures, rank]` — up-projection
    base*: FrozenLinear
    A*: Param[Tensor]   ## shape [rank, inFeatures]
    B*: Param[Tensor]   ## shape [outFeatures, rank]
    rank*: int
    alpha*: float32
    scaling*: float32
    merged*: bool

  LoraConfig* = object
    ## Configuration for applying LoRA to a model.
    rank*: int
    alpha*: float32
    targetModules*: seq[string]  ## e.g. ["q_proj", "v_proj", "o_proj"]

proc freezeLinear*(base: Linear): FrozenLinear =
  ## Converts a trainable `Linear` into a frozen base projection.
  FrozenLinear(
    weight: buffer(base.weight.value),
    bias: buffer(base.bias.value),
  )

proc forward*(layer: FrozenLinear; x: Tensor): Tensor =
  ## Computes the frozen affine projection.
  let xw = matmul(x, layer.weight)
  let biasB = broadcastTo(layer.bias, xw.shape, [1])
  add(xw, biasB)

proc initLoraLinear*(key: Key; base: Linear; rank: int;
    alpha: float32 = 0'f32; useKaiming = true): LoraLinear =
  ## Constructs a `LoraLinear` with A initialized using Kaiming uniform
  ## and B initialized to zero.
  let inFeatures = base.weight.shape[0]
  let outFeatures = base.weight.shape[1]
  let r = rank
  let a = if alpha == 0'f32: float32(rank) else: alpha
  let keys = split(key, 2)
  # A: Kaiming uniform (or normal).
  let aBound = sqrt(1.0'f32 / float32(inFeatures))
  let aData = uniformF32(keys[0], r * inFeatures, -aBound, aBound)
  # B: zero initialization (important for training stability).
  let bData = newSeq[float32](outFeatures * r)
  LoraLinear(
    base: freezeLinear(base),
    A: param(constantF32([r, inFeatures], aData)),
    B: param(constantF32([outFeatures, r], bData)),
    rank: r,
    alpha: a,
    scaling: a / float32(r),
    merged: false,
  )

proc forward*(layer: LoraLinear; x: Tensor): Tensor =
  ## Computes `x @ (W + (alpha/rank) * B @ A) + bias`.
  let baseOut = layer.base.forward(x)
  if layer.merged:
    return baseOut
  # Flatten batch dims for matmul (rank-2 required).
  let isBatched = x.shape.len > 2
  let xFlat = if isBatched:
      let batch = x.shape[0]
      let seqLen = x.shape[x.shape.len - 2]
      let feats = x.shape[x.shape.len - 1]
      reshape(x, [batch * seqLen, feats])
    else:
      x
  # x @ A^T: A is [rank, inFeatures] → transpose to [inFeatures, rank].
  let aT = transpose(layer.A, [1, 0])
  let xA = matmul(xFlat, aT)  # [N, rank]
  # (x @ A^T) @ B^T: B is [outFeatures, rank] → transpose to [rank, outFeatures].
  let bT = transpose(layer.B, [1, 0])
  let delta = matmul(xA, bT)   # [N, outFeatures]
  # Scale and add.
  let resultOut = if layer.scaling != 1'f32:
      let s = scalarF32(layer.scaling)
      var dims: seq[int] = @[]
      let sB = broadcastTo(s, delta.shape, dims)
      add(baseOut, mul(delta, sB))
    else:
      add(baseOut, delta)
  if isBatched:
    reshape(resultOut, x.shape[0..^2] & @[resultOut.shape[1]])
  else:
    resultOut

proc merge*(layer: var LoraLinear) =
  ## Merges LoRA weights into the base weight (for inference).
  ## After merging, forward behaves like a standard Linear.
  if layer.merged: return
  if layer.base.weight.shape.len != 2:
    raise newException(TensorError,
      "LoraLinear.merge: base weight must be rank-2, got " &
        $layer.base.weight.shape)
  if layer.A.shape != @[layer.rank, layer.base.weight.shape[0]]:
    raise newException(TensorError,
      "LoraLinear.merge: A shape " & $layer.A.shape &
        " does not match [rank, inFeatures] " &
        $(@[layer.rank, layer.base.weight.shape[0]]))
  if layer.B.shape != @[layer.base.weight.shape[1], layer.rank]:
    raise newException(TensorError,
      "LoraLinear.merge: B shape " & $layer.B.shape &
        " does not match [outFeatures, rank] " &
        $(@[layer.base.weight.shape[1], layer.rank]))

  let deltaOI = matmul(layer.B, layer.A)
  let delta = transpose(deltaOI, [1, 0])
  let scaledDelta =
    if layer.scaling == 1'f32:
      delta
    else:
      let scale = scalarF32(layer.scaling, layer.base.weight.device)
      var dims: seq[int] = @[]
      mul(delta, broadcastTo(scale, delta.shape, dims))
  layer.base.weight = buffer(add(layer.base.weight, scaledDelta))
  layer.merged = true

# ---- QLoRA ------------------------------------------------------------------

type
  QuantizedLoraLinear* = object
    ## QLoRA: NF4-quantized base weight + fp32 LoRA adapters.
    ##
    ## The base weight is stored quantized; it is dequantized on-the-fly
    ## during the forward pass. LoRA A/B are kept in fp32.
    qweight*: seq[byte]   ## packed NF4 base weight
    qscales*: seq[float32] ## per-group scales
    qzeroPoint*: int       ## zero point (0 for symmetric NF4)
    groupSize*: int
    inFeatures*: int
    outFeatures*: int
    A*: Param[Tensor]
    B*: Param[Tensor]
    rank*: int
    alpha*: float32
    scaling*: float32

  QloraLinear* = object
    ## Tensor-backed QLoRA linear layer.
    ##
    ## `qweight` and `qscales` retain the frozen NF4 representation while
    ## `dequantizedWeight` provides a traceable fp32 matmul path. Gradients
    ## should be requested only for `A` and `B` by optimizers/train loops.
    qweight*: Buffer[Tensor]
    qscales*: Buffer[Tensor]
    dequantizedWeight*: Buffer[Tensor]  ## shape `[inFeatures, outFeatures]`
    bias*: Buffer[Tensor]
    hasBias*: bool
    groupSize*: int
    inFeatures*: int
    outFeatures*: int
    A*: Param[Tensor]
    B*: Param[Tensor]
    rank*: int
    alpha*: float32
    scaling*: float32

  QloraConfig* = object
    ## Default QLoRA fine-tuning hyperparameters.
    rank*: int
    alpha*: float32
    groupSize*: int
    learningRate*: float32
    warmupFraction*: float32
    sequenceLength*: int
    batchSize*: int
    gradientAccumulation*: int

proc defaultQloraConfig*(): QloraConfig =
  ## Returns the Gemma 4 text QLoRA defaults used by the example.
  QloraConfig(
    rank: 16,
    alpha: 16'f32,
    groupSize: 64,
    learningRate: 5e-6'f32,
    warmupFraction: 0.03'f32,
    sequenceLength: 1024,
    batchSize: 1,
    gradientAccumulation: 8,
  )

proc nf4Dequantize*(qweight: openArray[byte]; scales: openArray[float32];
    inFeatures, outFeatures, groupSize: int;
    zeroPoint: int = 0): seq[float32] =
  ## Dequantizes NF4-packed weights to float32.
  ## Returns row-major `[outFeatures * inFeatures]` flat array.
  let numElements = inFeatures * outFeatures
  if groupSize <= 0:
    raise newException(TensorError,
      "nf4Dequantize: groupSize must be positive")
  let numGroups = (numElements + groupSize - 1) div groupSize
  if qweight.len != (numElements + 1) div 2:
    raise newException(TensorError,
      "nf4Dequantize: packed byte length does not match shape")
  if scales.len != numGroups:
    raise newException(TensorError,
      "nf4Dequantize: scale count " & $scales.len &
        " does not match expected " & $numGroups)
  result = newSeq[float32](numElements)
  # NF4 lookup table (16 values).
  const nf4Table = [
    -1.0'f32, -0.6961928009986877'f32, -0.5250730514526367'f32,
    -0.39491748809814453'f32, -0.28444138169288635'f32,
    -0.18477343022823334'f32, -0.09105003625154495'f32,
    0.0'f32, 0.07958029955625534'f32, 0.16093020141124725'f32,
    0.24611230194568634'f32, 0.33791524171829224'f32,
    0.44070982933044434'f32, 0.5626170039176941'f32,
    0.7229568362236023'f32, 1.0'f32]
  for group in 0 ..< numGroups:
    let scale = scales[group]
    let groupStart = group * groupSize
    let byteStart = groupStart div 2
    for i in 0 ..< min(groupSize, numElements - groupStart):
      let byteIdx = byteStart + i div 2
      let nibble = if (i mod 2) == 0:
          qweight[byteIdx].int and 0x0F
        else:
          (qweight[byteIdx].int shr 4) and 0x0F
      let idx = groupStart + i
      if idx < numElements:
        result[idx] = nf4Table[nibble] * scale

proc validateQloraShape(inFeatures, outFeatures, rank, groupSize: int;
    opName: string) =
  if inFeatures <= 0 or outFeatures <= 0:
    raise newException(TensorError,
      opName & ": inFeatures and outFeatures must be positive")
  if rank <= 0:
    raise newException(TensorError, opName & ": rank must be positive")
  if groupSize <= 0:
    raise newException(TensorError, opName & ": groupSize must be positive")

proc expectedNf4Groups(numElements, groupSize: int): int =
  (numElements + groupSize - 1) div groupSize

proc validatePackedNf4(packed: openArray[byte]; scales: openArray[float32];
    numElements, groupSize: int; opName: string) =
  let expectedPacked = (numElements + 1) div 2
  let expectedScales = expectedNf4Groups(numElements, groupSize)
  if packed.len != expectedPacked:
    raise newException(TensorError,
      opName & ": packed byte length " & $packed.len &
        " does not match expected " & $expectedPacked)
  if scales.len != expectedScales:
    raise newException(TensorError,
      opName & ": scale count " & $scales.len &
        " does not match expected " & $expectedScales)

proc initQuantizedLoraLinearFromF32*(key: Key; weight: openArray[float32];
    inFeatures, outFeatures: int; rank: int; groupSize: int = 128;
    alpha: float32 = 0'f32): QuantizedLoraLinear =
  ## Constructs a host-backed QLoRA layer by quantizing fp32 base weights
  ## to NF4 and initializing trainable fp32 LoRA adapters.
  validateQloraShape(inFeatures, outFeatures, rank, groupSize,
    "initQuantizedLoraLinearFromF32")
  let numElements = inFeatures * outFeatures
  if weight.len != numElements:
    raise newException(TensorError,
      "initQuantizedLoraLinearFromF32: weight length " & $weight.len &
        " does not match shape [" & $inFeatures & ", " & $outFeatures & "]")
  let quantized = codecs.nf4Quantize(weight, groupSize)
  let r = rank
  let a = if alpha == 0'f32: float32(rank) else: alpha
  let keys = split(key, 2)
  let aBound = sqrt(1.0'f32 / float32(inFeatures))
  let aData = uniformF32(keys[0], r * inFeatures, -aBound, aBound)
  let bData = newSeq[float32](outFeatures * r)
  QuantizedLoraLinear(
    qweight: quantized.packed,
    qscales: quantized.scales,
    qzeroPoint: 0,
    groupSize: groupSize,
    inFeatures: inFeatures,
    outFeatures: outFeatures,
    A: param(constantF32([r, inFeatures], aData)),
    B: param(constantF32([outFeatures, r], bData)),
    rank: r,
    alpha: a,
    scaling: a / float32(r),
  )

proc initQuantizedLoraLinear*(key: Key; inFeatures, outFeatures: int;
    rank: int; groupSize: int = 128; alpha: float32 = 0'f32):
    QuantizedLoraLinear =
  ## Constructs a `QuantizedLoraLinear` with random fp32 base weights that
  ## are immediately NF4-quantized, plus trainable fp32 LoRA adapters.
  validateQloraShape(inFeatures, outFeatures, rank, groupSize,
    "initQuantizedLoraLinear")
  let numElements = inFeatures * outFeatures
  let keys = split(key, 2)
  let std = 1'f32 / sqrt(float32(inFeatures))
  let base = normalF32(keys[0], numElements, 0'f32, std)
  initQuantizedLoraLinearFromF32(keys[1], base, inFeatures, outFeatures,
    rank, groupSize, alpha)

proc initQloraLinearFromF32*(key: Key; weight: openArray[float32];
    inFeatures, outFeatures: int; bias: openArray[float32] = [];
    rank = 16; alpha = 16'f32; groupSize = 64): QloraLinear =
  ## Builds a tensor-backed QLoRA layer from fp32 `[inFeatures, outFeatures]`
  ## host weights.
  validateQloraShape(inFeatures, outFeatures, rank, groupSize,
    "initQloraLinearFromF32")
  let numElements = inFeatures * outFeatures
  if weight.len != numElements:
    raise newException(TensorError,
      "initQloraLinearFromF32: weight length " & $weight.len &
        " does not match shape [" & $inFeatures & ", " & $outFeatures & "]")
  if bias.len notin [0, outFeatures]:
    raise newException(TensorError,
      "initQloraLinearFromF32: bias length must be 0 or outFeatures")

  let quantized = codecs.nf4Quantize(weight, groupSize)
  let dequantized = codecs.nf4Dequantize(
    quantized.packed, quantized.scales, numElements, groupSize)
  let r = rank
  let a = if alpha == 0'f32: float32(rank) else: alpha
  let keys = split(key, 2)
  let aBound = sqrt(1.0'f32 / float32(inFeatures))
  let aData = uniformF32(keys[0], r * inFeatures, -aBound, aBound)
  let bData = newSeq[float32](outFeatures * r)
  let biasData = if bias.len == 0: newSeq[float32](outFeatures) else: @bias
  QloraLinear(
    qweight: buffer(constant(dtUint8, [quantized.packed.len],
      quantized.packed)),
    qscales: buffer(constantF32([quantized.scales.len], quantized.scales)),
    dequantizedWeight: buffer(
      constantF32([inFeatures, outFeatures], dequantized)),
    bias: buffer(constantF32([outFeatures], biasData)),
    hasBias: true,
    groupSize: groupSize,
    inFeatures: inFeatures,
    outFeatures: outFeatures,
    A: param(constantF32([r, inFeatures], aData)),
    B: param(constantF32([outFeatures, r], bData)),
    rank: r,
    alpha: a,
    scaling: a / float32(r),
  )

proc initQloraLinearFromQuantized*(key: Key; packed: openArray[byte];
    scales: openArray[float32]; inFeatures, outFeatures: int;
    bias: openArray[float32] = []; rank = 16; alpha = 16'f32;
    groupSize = 64): QloraLinear =
  ## Builds a QLoRA layer from existing packed NF4 weights and scales.
  validateQloraShape(inFeatures, outFeatures, rank, groupSize,
    "initQloraLinearFromQuantized")
  let numElements = inFeatures * outFeatures
  validatePackedNf4(packed, scales, numElements, groupSize,
    "initQloraLinearFromQuantized")
  let dequantized = codecs.nf4Dequantize(packed, scales, numElements,
    groupSize)
  let a = if alpha == 0'f32: float32(rank) else: alpha
  let keys = split(key, 2)
  let aBound = sqrt(1.0'f32 / float32(inFeatures))
  let aData = uniformF32(keys[0], rank * inFeatures, -aBound, aBound)
  let bData = newSeq[float32](outFeatures * rank)
  let biasData = if bias.len == 0: newSeq[float32](outFeatures) else: @bias
  QloraLinear(
    qweight: buffer(constant(dtUint8, [packed.len], packed)),
    qscales: buffer(constantF32([scales.len], scales)),
    dequantizedWeight: buffer(
      constantF32([inFeatures, outFeatures], dequantized)),
    bias: buffer(constantF32([outFeatures], biasData)),
    hasBias: true,
    groupSize: groupSize,
    inFeatures: inFeatures,
    outFeatures: outFeatures,
    A: param(constantF32([rank, inFeatures], aData)),
    B: param(constantF32([outFeatures, rank], bData)),
    rank: rank,
    alpha: a,
    scaling: a / float32(rank),
  )

proc forward*(layer: QloraLinear; x: Tensor): Tensor =
  ## Computes the frozen dequantized base projection plus LoRA delta.
  let isBatched = x.shape.len > 2
  let xFlat = if isBatched:
      let leading = x.numElements div x.shape[^1]
      reshape(x, [leading, x.shape[^1]])
    else:
      x
  var baseOut = matmul(xFlat, layer.dequantizedWeight)
  if layer.hasBias:
    let biasB = broadcastTo(layer.bias, baseOut.shape, [1])
    baseOut = add(baseOut, biasB)
  let aT = transpose(layer.A, [1, 0])
  let xA = matmul(xFlat, aT)
  let bT = transpose(layer.B, [1, 0])
  var delta = matmul(xA, bT)
  if layer.scaling != 1'f32:
    let scale = scalarF32(layer.scaling)
    delta = mul(delta, broadcastTo(scale, delta.shape, @[]))
  let outFlat = add(baseOut, delta)
  if isBatched:
    reshape(outFlat, x.shape[0 .. ^2] & @[layer.outFeatures])
  else:
    outFlat
