## `GraphConv` — general graph convolution layer (Morris et al., 2019).
##
## Simple convolution: aggregate neighbor features, add self-features,
## apply linear transformation.
##
## Pure value type following rew's functional nn invariant.

import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/linalg
import ./linear
import ./init
import ./graph_util

type
  GraphConv* = object
    linear*: Linear             ## [inChannels, outChannels]
    aggr*: AggregationType
    addSelf*: bool              ## If true, add self features to aggregation
    addBias*: bool
    bias*: Param[Tensor]        ## [outChannels]

proc initGraphConv*(key: Key; inChannels, outChannels: int;
    aggr: AggregationType = pAggrSum; addSelf: bool = true;
    bias: bool = true): GraphConv =
  if inChannels <= 0 or outChannels <= 0:
    raise newException(TensorError,
      "initGraphConv: inChannels and outChannels must be positive")
  result = GraphConv(
    linear: initLinear(key, inChannels, outChannels),
    aggr: aggr,
    addSelf: addSelf,
    addBias: bias,
  )
  if bias:
    let bData = zerosF32(outChannels)
    result.bias = param(constantF32(@[outChannels], bData))

proc forward*(layer: GraphConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of GraphConv.
  ##
  ## Computes: `W * (x_i + aggregate(x_j for j in N(i)))` if addSelf,
  ## otherwise `W * aggregate(x_j)`.
  ##
  ## `x` shape `[N, inChannels]`, `edgeIndex` shape `[2, E]`.
  ## Returns `[N, outChannels]`.
  let n = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  # Gather and aggregate neighbor features.
  let xSrc = indexSelect(x, src)                         # [E, inCh]
  let neighborAggr = segmentAggregate(xSrc, dst, n, layer.aggr)
  # Combine self and neighbor features, then apply linear.
  var combined: Tensor
  if layer.addSelf:
    combined = add(x, neighborAggr)                       # [N, inCh]
  else:
    combined = neighborAggr
  result = layer.linear.forward(combined)                  # [N, outCh]
  if layer.addBias:
    result = add(result, broadcastTo(layer.bias, result.shape, [1]))
