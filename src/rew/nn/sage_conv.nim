## `SAGEConv` — GraphSAGE convolution (Hamilton et al., 2017).
##
## Pure value type following rew's functional nn invariant.

import std/math
import ../tensor
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/reduce
import ../ops/linalg
import ./linear
import ./init
import ./graph_util

type
  SAGEConv* = object
    neighborLinear*: Linear   ## [inChannels, outChannels] for neighbors
    selfLinear*: Linear       ## [inChannels, outChannels] for self
    aggr*: AggregationType
    normalize*: bool          ## If true, l2-normalize output
    addBias*: bool
    bias*: Tensor             ## [outChannels]

proc initSAGEConv*(key: Key; inChannels, outChannels: int;
    aggr: AggregationType = pAggrMean; normalize: bool = false;
    bias: bool = true): SAGEConv =
  if inChannels <= 0 or outChannels <= 0:
    raise newException(TensorError,
      "initSAGEConv: inChannels and outChannels must be positive")
  let keys = split(key, 2)
  result = SAGEConv(
    neighborLinear: initLinear(keys[0], inChannels, outChannels),
    selfLinear: initLinear(keys[1], inChannels, outChannels),
    aggr: aggr,
    normalize: normalize,
    addBias: bias,
  )
  if bias:
    let bData = zerosF32(outChannels)
    result.bias = constantF32(@[outChannels], bData)

proc forward*(layer: SAGEConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of SAGEConv.
  ##
  ## Computes: `W_self * x_i + W_neigh * aggregate(x_j for j in N(i))`.
  ##
  ## `x` shape `[N, inChannels]`, `edgeIndex` shape `[2, E]`.
  ## Returns `[N, outChannels]`.
  let n = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  # Gather neighbor features and aggregate.
  let xSrc = indexSelect(x, src)                    # [E, inCh]
  let neighborAggr = segmentAggregate(xSrc, dst, n, layer.aggr)  # [N, inCh]
  # Apply per-side linear transforms.
  let selfOut = layer.selfLinear.forward(x)          # [N, outCh]
  let neighOut = layer.neighborLinear.forward(neighborAggr)  # [N, outCh]
  result = add(selfOut, neighOut)
  if layer.addBias:
    result = add(result, broadcastTo(layer.bias, result.shape, [1]))
  if layer.normalize:
    # L2 normalize per node (over feature dimension).
    let sq = mul(result, result)
    let normSq = reduceSum(sq, [1])                   # [N]
    let normVal = sqrt(add(normSq,
      broadcastTo(scalarF32(1e-12'f32), normSq.shape, @[])))
    let normB = unsqueeze(normVal, 1)                  # [N, 1]
    result = divide(result, broadcastTo(normB, result.shape, [0, 1]))
