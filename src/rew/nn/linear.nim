## `Linear` \u2014 fully connected layer `y = x @ W + b`.
##
## Pure value type: `weight` and `bias` are explicit `Param[Tensor]`
## trainable leaves.
##
## `initLinear` materialises weights through the public literal path, so
## it works in both trace mode and eager mode after an eager backend is
## installed. PRNG is explicit \u2014 the caller passes a `Key` and is
## responsible for splitting it.

import std/math
import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/linalg
import ./init

type
  Linear* = object
    ## Affine layer carrying its weight matrix and bias vector. `forward`
    ## treats the leading axis of `x` as the batch dimension.
    weight*: Param[Tensor]   ## shape `[inFeatures, outFeatures]`, float32
    bias*: Param[Tensor]     ## shape `[outFeatures]`, float32

proc initLinear*(key: Key; inFeatures, outFeatures: int): Linear =
  ## Constructs a `Linear` with He-style uniform initialization
  ## (`bound = sqrt(1 / inFeatures)`) for the weight and a zero bias.
  if inFeatures <= 0 or outFeatures <= 0:
    raise newException(TensorError,
      "initLinear: inFeatures and outFeatures must be positive (got " &
        $inFeatures & ", " & $outFeatures & ")")
  let keys = split(key, 2)
  let bound = sqrt(1.0f32 / float32(inFeatures))
  let wData = uniformF32(keys[0], inFeatures * outFeatures, -bound, bound)
  let bData = newSeq[float32](outFeatures)
  result = Linear(
    weight: param(constantF32([inFeatures, outFeatures], wData)),
    bias: param(constantF32([outFeatures], bData)),
  )

proc forward*(layer: Linear; x: Tensor): Tensor =
  ## Computes `x @ layer.weight + layer.bias`. `x` must be rank-2 with
  ## `x.shape[1] == layer.weight.shape[0]`. The bias is broadcast over
  ## the batch dimension.
  let xw = matmul(x, layer.weight)
  let biasB = broadcastTo(layer.bias, xw.shape, [1])
  add(xw, biasB)
