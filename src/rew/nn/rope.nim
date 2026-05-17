## Rotary Position Embeddings (RoPE) — Su et al. (2021).
##
## Applies a frequency-based rotation to query and key tensors so that
## the dot-product attention score naturally encodes relative position.
##
## Supports:
## - `RotaryPositionEncoding` — standard RoPE with configurable `theta`.
## - YaRN extension (NTK-aware scaling) for context extension.
## - `AlibiBias` — additive linear biases for causal attention.

import std/math
import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/shape
import ../ops/linalg
import ../ops/concat

type
  RotaryPositionEncoding* = object
    ## Rotary Position Embedding layer. Stores precomputed cos/sin tables
    ## for up to `maxSeqLen` positions. The `theta` parameter controls the
    ## base frequency (default 10000.0 as in the original paper; Llama uses
    ## 500000.0).
    cosCached*: Buffer[Tensor]  ## shape [maxSeqLen, headDim]
    sinCached*: Buffer[Tensor]  ## shape [maxSeqLen, headDim]
    theta*: float64
    headDim*: int
    maxSeqLen*: int

  YarnConfig* = object
    ## Configuration for YaRN (Yet another RoPE extensioN) scaling.
    originalMaxSeqLen*: int
    extendedMaxSeqLen*: int
    scale*: float32
    alpha*: float32

  YarnRotaryPositionEncoding* = object
    ## YaRN-extended RoPE. Stores separate cos/sin tables for the
    ## extended context window.
    cosExt*: Buffer[Tensor]
    sinExt*: Buffer[Tensor]
    yarnConfig*: YarnConfig
    headDim*: int

  AlibiBias* = object
    ## ALiBi (Attention with Linear Biases) — Press et al. (2022).
    biases*: Buffer[Tensor]  ## shape [numHeads, maxSeqLen, maxSeqLen]
    numHeads*: int
    maxSeqLen*: int

# ---- Shared rotation helper ------------------------------------------------

proc applyRotate(x: Tensor; cosTbl, sinTbl: Tensor): Tensor =
  ## Applies rotary transform: for each pair (x[2i], x[2i+1]) multiply
  ## by the corresponding cos/sin entry. cosTbl and sinTbl already have
  ## the same shape as x.
  let rank = x.shape.len
  let headDim = x.shape[^1]
  let half = headDim div 2
  # Reshape to [..., half, 2]
  let leadingShape = x.shape[0 ..< rank - 1]
  let pairShape = leadingShape & @[half, 2]
  let xPairs = reshape(x, pairShape)
  # Special-case: handle any leading dims generically using seqs.
  # Build slice coordinates for component 0 and component 1.
  let pairRank = xPairs.shape.len
  var starts0 = newSeq[int](pairRank)
  var limits0 = newSeq[int](pairRank)
  var strides = newSeq[int](pairRank)
  var starts1 = newSeq[int](pairRank)
  var limits1 = newSeq[int](pairRank)
  for i in 0 ..< pairRank - 2:
    starts0[i] = 0; limits0[i] = xPairs.shape[i]; strides[i] = 1
    starts1[i] = 0; limits1[i] = xPairs.shape[i]
  # Second-to-last dim: half
  starts0[pairRank - 2] = 0; limits0[pairRank - 2] = half
  strides[pairRank - 2] = 1
  starts1[pairRank - 2] = 0; limits1[pairRank - 2] = half
  # Last dim: component 0 vs 1
  starts0[pairRank - 1] = 0; limits0[pairRank - 1] = 1
  strides[pairRank - 1] = 1
  starts1[pairRank - 1] = 1; limits1[pairRank - 1] = 2
  let x0 = slice(xPairs, starts0, limits0, strides)
  let x1 = slice(xPairs, starts1, limits1, strides)
  # Remove trailing dim (size 1).
  let flatShape = x0.shape[0 ..< x0.shape.len - 1]
  let x0Flat = reshape(x0, flatShape)
  let x1Flat = reshape(x1, flatShape)
  # Apply same reshape to cosTbl, sinTbl and extract component 0.
  let cosPairs = reshape(cosTbl, pairShape)
  let sinPairs = reshape(sinTbl, pairShape)
  let cos0Slice = slice(cosPairs, starts0, limits0, strides)
  let sin0Slice = slice(sinPairs, starts0, limits0, strides)
  let cos0 = reshape(cos0Slice, flatShape)
  let sin0 = reshape(sin0Slice, flatShape)
  # x0' = cos0 * x0 - sin0 * x1
  # x1' = sin0 * x0 + cos0 * x1
  let x0Rot = sub(mul(cos0, x0Flat), mul(sin0, x1Flat))
  let x1Rot = add(mul(sin0, x0Flat), mul(cos0, x1Flat))
  # Stack back: [..., half, 2]
  let x0r = reshape(x0Rot, flatShape & @[1])
  let x1r = reshape(x1Rot, flatShape & @[1])
  let xRotated = concat([x0r, x1r], x0r.shape.len - 1)
  # Reshape to original x shape.
  reshape(xRotated, x.shape)

# ---- RotaryPositionEncoding ------------------------------------------------

proc initRotaryPositionEncoding*(headDim: int;
    maxSeqLen = 2048; theta: float64 = 10000.0): RotaryPositionEncoding =
  ## Constructs a `RotaryPositionEncoding` with precomputed cos/sin tables.
  ## `headDim` must be even.
  if headDim <= 0 or (headDim mod 2) != 0:
    raise newException(TensorError,
      "initRotaryPositionEncoding: headDim must be positive and even, got " &
        $headDim)
  if maxSeqLen <= 0:
    raise newException(TensorError,
      "initRotaryPositionEncoding: maxSeqLen must be positive")
  let half = headDim div 2
  var invFreq = newSeq[float32](half)
  for i in 0 ..< half:
    let exponent = float64(2 * i) / float64(headDim)
    invFreq[i] = 1.0'f32 / pow(theta, exponent).float32
  var cosVals = newSeq[float32](maxSeqLen * headDim)
  var sinVals = newSeq[float32](maxSeqLen * headDim)
  for pos in 0 ..< maxSeqLen:
    let fp = float32(pos)
    for i in 0 ..< half:
      let angle = fp * invFreq[i]
      let base = pos * headDim + 2 * i
      cosVals[base] = cos(angle)
      cosVals[base + 1] = cos(angle)
      sinVals[base] = sin(angle)
      sinVals[base + 1] = sin(angle)
  RotaryPositionEncoding(
    cosCached: buffer(constantF32([maxSeqLen, headDim], cosVals)),
    sinCached: buffer(constantF32([maxSeqLen, headDim], sinVals)),
    theta: theta,
    headDim: headDim,
    maxSeqLen: maxSeqLen,
  )

proc initRotaryPositionEncoding*(key: Key; headDim: int;
    maxSeqLen = 2048; theta: float64 = 10000.0): RotaryPositionEncoding =
  ## Key-accepting overload for API consistency with other nn layers.
  ## The key is discarded; RoPE has no learnable parameters.
  initRotaryPositionEncoding(headDim, maxSeqLen, theta)

proc forward*(layer: RotaryPositionEncoding; x: Tensor;
    offset: int = 0): Tensor =
  ## Applies rotary position embedding to `x`.
  ## `x` has shape `[..., seqLen, headDim]`. `offset` shifts position
  ## indices (for cached KV inference).
  if x.shape.len < 2:
    raise newException(TensorError,
      "RotaryPositionEncoding.forward: input rank must be at least 2")
  let seqLen = x.shape[^2]
  let headDim = x.shape[^1]
  if headDim != layer.headDim:
    raise newException(TensorError,
      "RotaryPositionEncoding.forward: headDim mismatch " & $headDim &
        " vs " & $layer.headDim)
  let totalLen = offset + seqLen
  if totalLen > layer.maxSeqLen:
    raise newException(TensorError,
      "RotaryPositionEncoding.forward: position " & $totalLen &
        " exceeds maxSeqLen " & $layer.maxSeqLen)
  let rank = x.shape.len
  let cosSlice = slice(layer.cosCached,
    [offset, 0], [totalLen, headDim], [1, 1])
  let sinSlice = slice(layer.sinCached,
    [offset, 0], [totalLen, headDim], [1, 1])
  let cosB = broadcastTo(cosSlice, x.shape, [rank - 2, rank - 1])
  let sinB = broadcastTo(sinSlice, x.shape, [rank - 2, rank - 1])
  applyRotate(x, cosB, sinB)

# ---- YaRN ------------------------------------------------------------------

proc initYarnConfig*(originalMaxSeqLen, extendedMaxSeqLen: int;
    scale: float32 = 0'f32; alpha: float32 = 1.0'f32): YarnConfig =
  let scl = if scale != 0'f32: scale
            else: float32(extendedMaxSeqLen) / float32(originalMaxSeqLen)
  YarnConfig(
    originalMaxSeqLen: originalMaxSeqLen,
    extendedMaxSeqLen: extendedMaxSeqLen,
    scale: scl,
    alpha: alpha,
  )

proc initYarnRotaryPositionEncoding*(headDim: int;
    config: YarnConfig; theta: float64 = 10000.0):
    YarnRotaryPositionEncoding =
  ## Constructs a YaRN-extended RoPE. Uses NTK-aware scaling for the
  ## extended context window.
  if headDim <= 0 or (headDim mod 2) != 0:
    raise newException(TensorError,
      "initYarnRotaryPositionEncoding: headDim must be positive and even")
  let half = headDim div 2
  var cosVals = newSeq[float32](config.extendedMaxSeqLen * headDim)
  var sinVals = newSeq[float32](config.extendedMaxSeqLen * headDim)
  let scaledTheta = theta * float64(config.alpha)
  for pos in 0 ..< config.extendedMaxSeqLen:
    let fp = float32(pos)
    for i in 0 ..< half:
      let exponent = float64(2 * i) / float64(headDim)
      let invFreq = 1.0'f32 / pow(scaledTheta, exponent).float32
      let angle = fp * invFreq
      let base = pos * headDim + 2 * i
      cosVals[base] = cos(angle)
      cosVals[base + 1] = cos(angle)
      sinVals[base] = sin(angle)
      sinVals[base + 1] = sin(angle)
  YarnRotaryPositionEncoding(
    cosExt: buffer(constantF32([config.extendedMaxSeqLen, headDim], cosVals)),
    sinExt: buffer(constantF32([config.extendedMaxSeqLen, headDim], sinVals)),
    yarnConfig: config,
    headDim: headDim,
  )

proc initYarnRotaryPositionEncoding*(key: Key; headDim: int;
    config: YarnConfig; theta: float64 = 10000.0):
    YarnRotaryPositionEncoding =
  initYarnRotaryPositionEncoding(headDim, config, theta)

proc forward*(layer: YarnRotaryPositionEncoding; x: Tensor;
    offset: int = 0): Tensor =
  ## Applies YaRN-extended RoPE using NTK-scaled frequencies.
  if x.shape.len < 2:
    raise newException(TensorError,
      "YarnRotaryPositionEncoding.forward: input rank must be at least 2")
  let seqLen = x.shape[^2]
  let headDim = x.shape[^1]
  if headDim != layer.headDim:
    raise newException(TensorError,
      "YarnRotaryPositionEncoding.forward: headDim mismatch")
  let totalLen = offset + seqLen
  if totalLen > layer.yarnConfig.extendedMaxSeqLen:
    raise newException(TensorError,
      "YarnRotaryPositionEncoding.forward: position exceeds maxSeqLen " &
        $layer.yarnConfig.extendedMaxSeqLen)
  let rank = x.shape.len
  let cosSlice = slice(layer.cosExt,
    [offset, 0], [totalLen, headDim], [1, 1])
  let sinSlice = slice(layer.sinExt,
    [offset, 0], [totalLen, headDim], [1, 1])
  let cosB = broadcastTo(cosSlice, x.shape, [rank - 2, rank - 1])
  let sinB = broadcastTo(sinSlice, x.shape, [rank - 2, rank - 1])
  applyRotate(x, cosB, sinB)

# ---- ALiBi -----------------------------------------------------------------

proc initAlibiBias*(numHeads, maxSeqLen: int): AlibiBias =
  ## Constructs ALiBi additive biases for causal attention.
  ## Shape: `[numHeads, maxSeqLen, maxSeqLen]`.
  if numHeads <= 0 or maxSeqLen <= 0:
    raise newException(TensorError,
      "initAlibiBias: numHeads and maxSeqLen must be positive")
  var biasData = newSeq[float32](numHeads * maxSeqLen * maxSeqLen)
  for h in 0 ..< numHeads:
    let power = 2.0'f64 + 8.0'f64 * float64(h) / float64(numHeads)
    let slope = 1.0'f32 / pow(2.0'f64, power).float32
    for i in 0 ..< maxSeqLen:
      for j in 0 ..< maxSeqLen:
        let idx = h * maxSeqLen * maxSeqLen + i * maxSeqLen + j
        if j > i:
          biasData[idx] = -1e9'f32
        else:
          biasData[idx] = slope * float32(i - j)
  AlibiBias(
    biases: buffer(constantF32([numHeads, maxSeqLen, maxSeqLen], biasData)),
    numHeads: numHeads,
    maxSeqLen: maxSeqLen,
  )

proc forward*(layer: AlibiBias; seqLen: int; kvLen: int = -1): Tensor =
  ## Returns the additive ALiBi bias for causal attention,
  ## shape `[numHeads, seqLen, kvLen]`. When `kvLen` < 0, uses `seqLen`.
  let kv = if kvLen < 0: seqLen else: kvLen
  if seqLen > layer.maxSeqLen or kv > layer.maxSeqLen:
    raise newException(TensorError,
      "AlibiBias.forward: sequence length exceeds maxSeqLen " &
        $layer.maxSeqLen)
  slice(layer.biases, [0, 0, 0], [layer.numHeads, seqLen, kv], [1, 1, 1])
