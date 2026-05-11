## `GATConv` — Graph Attention Network layer (Veličković et al., 2018).
##
## Pure value type following rew's functional nn invariant.

import std/math
import ../tensor
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/segment
import ../ops/reduce
import ../ops/linalg
import ./linear
import ./activation
import ./init
import ./graph_util

type
  GATConv* = object
    linear*: Linear         ## [inChannels, outChannels * numHeads]
    attSrc*: Linear         ## [outChannels * numHeads, numHeads] source attention
    attDst*: Linear         ## [outChannels * numHeads, numHeads] dest attention
    numHeads*: int
    concat*: bool           ## If true, concat heads; else average
    negativeSlope*: float32
    addBias*: bool
    bias*: Tensor            ## shape [outChannels * numHeads] or [outChannels]

proc initGATConv*(key: Key; inChannels, outChannels, numHeads: int;
    concat: bool = true; negativeSlope: float32 = 0.2'f32;
    bias: bool = true): GATConv =
  ## Construct a GATConv with `numHeads` attention heads.
  if inChannels <= 0 or outChannels <= 0 or numHeads <= 0:
    raise newException(TensorError,
      "initGATConv: all dimension params must be positive")
  let keys = split(key, 3)
  let outTotal = outChannels * numHeads
  result = GATConv(
    linear: initLinear(keys[0], inChannels, outTotal),
    attSrc: initLinear(keys[1], outTotal, numHeads),
    attDst: initLinear(keys[2], outTotal, numHeads),
    numHeads: numHeads,
    concat: concat,
    negativeSlope: negativeSlope,
    addBias: bias,
  )
  if bias:
    let biasSize = if concat: outTotal else: outChannels
    let bData = zerosF32(biasSize)
    result.bias = constantF32(@[biasSize], bData)

proc forward*(layer: GATConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of GATConv.
  ##
  ## `x` shape `[N, inChannels]`, `edgeIndex` shape `[2, E]`.
  ## Returns `[N, outChannels * numHeads]` (concat) or `[N, outChannels]`
  ## (averaged).
  let n = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  let outCh = layer.linear.weight.shape[1] div layer.numHeads
  # Linear transform on node features.
  let xLin = layer.linear.forward(x)     # [N, outCh * heads]
  # Extract source and destination indices.
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  # Gather source and destination features.
  let xSrc = indexSelect(xLin, src)      # [E, outCh * heads]
  let xDst = indexSelect(xLin, dst)      # [E, outCh * heads]
  # Per-head attention scores.
  let scoreSrc = layer.attSrc.forward(xSrc)  # [E, heads]
  let scoreDst = layer.attDst.forward(xDst)  # [E, heads]
  var alpha = add(scoreSrc, scoreDst)        # [E, heads]
  alpha = leakyRelu(alpha, layer.negativeSlope)
  # Per-destination softmax (normalize over incoming edges).
  alpha = softmaxPerSegment(alpha, dst, n)    # [E, heads]
  alpha = unsqueeze(alpha, 2)                 # [E, heads, 1]
  # Reshape source features for weighted aggregation.
  let xSrc3d = reshape(xSrc, [numEdges, layer.numHeads, outCh])
  let alphaB = broadcastTo(alpha, xSrc3d.shape, [0, 1, 2])
  let weighted = mul(xSrc3d, alphaB)          # [E, heads, outCh]
  let aggr = segmentSum(weighted, dst, n)     # [N, heads, outCh]
  if layer.concat:
    result = reshape(aggr, [n, outCh * layer.numHeads])
    if layer.addBias:
      result = add(result, broadcastTo(layer.bias, result.shape, [1]))
  else:
    result = reduceMean(aggr, [1])            # [N, outCh]
    if layer.addBias:
      result = add(result, broadcastTo(layer.bias, result.shape, [1]))
