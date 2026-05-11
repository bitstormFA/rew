## `GCNConv` — Graph Convolutional Network layer (Kipf & Welling, 2017).
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
import ../ops/linalg
import ./linear
import ./init
import ./graph_util

type
  GCNConv* = object
    linear*: Linear
    improved*: bool       ## If true, self-loop weight is 2 instead of 1
    addBias*: bool
    bias*: Tensor          ## shape [outChannels]

proc initGCNConv*(key: Key; inChannels, outChannels: int;
    improved: bool = false; bias: bool = true): GCNConv =
  if inChannels <= 0 or outChannels <= 0:
    raise newException(TensorError,
      "initGCNConv: inChannels and outChannels must be positive")
  let keys = split(key, 2)
  result = GCNConv(
    linear: initLinear(keys[0], inChannels, outChannels),
    improved: improved,
    addBias: bias,
  )
  if bias:
    let bData = zerosF32(outChannels)
    result.bias = constantF32(@[outChannels], bData)

proc forward*(layer: GCNConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of GCNConv.
  ##
  ## `x` shape `[N, inChannels]`, `edgeIndex` shape `[2, E]` int32.
  ## Returns `[N, outChannels]`.
  ##
  ## Computes `H = D^{-0.5} A D^{-0.5} X W + b`.
  let n = x.shape[0]
  # Linear transform on node features.
  let xTransformed = layer.linear.forward(x)   # [N, outCh]
  # Normalized edge index with self-loops.
  let (ei, norm) = normalizeEdgeIndex(edgeIndex, n)
  let numEdges = ei.shape[1]
  # Gather source node features.
  let src = squeeze(slice(ei, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(ei, [1, 0], [2, numEdges], [1, 1]), 0)
  let xSrc = indexSelect(xTransformed, src)     # [E, outCh]
  # Apply normalization weights (broadcast norm from [E] to [E, 1]).
  let normB = unsqueeze(norm, 1)                # [E, 1]
  let messages = mul(xSrc, broadcastTo(normB, xSrc.shape, [0, 1]))
  # Aggregate by destination node.
  result = segmentSum(messages, dst, n)              # [N, outCh]
  if layer.addBias:
    result = add(result, broadcastTo(layer.bias, result.shape, [1]))
