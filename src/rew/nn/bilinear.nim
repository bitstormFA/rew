## Bilinear layer — bilinear transformation of two inputs.
##
## Computes `x1^T @ A @ x2 + b` where `A` is a learnable weight tensor.

import std/math
import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ../ops/shape
import ./init

type
  Bilinear* = object
    ## Bilinear layer: `x1^T A x2 + b`.
    ## `weight` has shape `[outFeatures, in1Features, in2Features]`.
    ## `bias` has shape `[outFeatures]`.
    weight*: Param[Tensor]
    bias*: Param[Tensor]

proc initBilinear*(key: Key; in1Features, in2Features, outFeatures: int): Bilinear =
  ## Constructs a Bilinear layer with Kaiming uniform init.
  if in1Features <= 0 or in2Features <= 0 or outFeatures <= 0:
    raise newException(TensorError,
      "initBilinear: feature counts must be positive")
  let keys = split(key, 2)
  let bound = sqrt(1.0f32 / float32(in1Features * in2Features))
  let wCount = outFeatures * in1Features * in2Features
  let wData = uniformF32(keys[0], wCount, -bound, bound)
  let bData = newSeq[float32](outFeatures)
  Bilinear(
    weight: param(constantF32([outFeatures, in1Features, in2Features], wData)),
    bias: param(constantF32([outFeatures], bData)),
  )

proc forward*(layer: Bilinear; x1, x2: Tensor): Tensor =
  ## Computes `x1^T A x2 + b` for each batch element.
  ## `x1` is `[batch, in1Features]`, `x2` is `[batch, in2Features]`.
  ## Returns `[batch, outFeatures]`.
  if x1.shape.len != 2 or x2.shape.len != 2:
    raise newException(TensorError,
      "Bilinear.forward: inputs must be rank-2 [batch, features]")
  if x1.shape[0] != x2.shape[0]:
    raise newException(TensorError,
      "Bilinear.forward: batch dims must match")
  let batch = x1.shape[0]
  let outF = layer.weight.shape[0]
  let in1F = layer.weight.shape[1]
  let in2F = layer.weight.shape[2]
  if x1.shape[1] != in1F or x2.shape[1] != in2F:
    raise newException(TensorError,
      "Bilinear.forward: feature dims mismatch")
  # For each output feature k:
  #   out[k] = x1^T @ W[k] @ x2 + bias[k]
  #         = sum_{i,j} W[k,i,j] * x1[i] * x2[j] + bias[k]
  # Efficient: x1_expanded = [batch, in1F, 1], x2_expanded = [batch, 1, in2F]
  # outer = [batch, in1F, in2F]
  let x1e = unsqueeze(x1, 2)
  let x2e = unsqueeze(x2, 1)
  let outer = mul(
    broadcastTo(x1e, [batch, in1F, in2F], @[0, 1]),
    broadcastTo(x2e, [batch, in1F, in2F], @[0, 2]))
  # W has shape [outF, in1F, in2F]
  # We need: out[k, b] = sum_{i,j} W[k,i,j] * outer[b,i,j]
  # = einsum('kij,bij->bk') but we can use reshape + matmul:
  # Reshape W to [outF, in1F*in2F], outer to [batch, in1F*in2F]
  let wMat = reshape(layer.weight, [outF, in1F * in2F])
  let outerMat = reshape(outer, [batch, in1F * in2F])
  let logits = matmul(outerMat, transpose(wMat, [1, 0]))
  add(logits, broadcastTo(layer.bias, [batch, outF], @[0]))
