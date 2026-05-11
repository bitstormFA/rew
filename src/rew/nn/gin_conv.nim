## `GINConv` — Graph Isomorphism Network convolution (Xu et al., 2019).
##
## Pure value type following rew's functional nn invariant. The MLP is
## a pre-built `Sequential` passed at construction time.

import ../tensor
import ../ops/literal
import ../ops/arith
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/segment
import ../ops/linalg
import ./sequential

type
  GINConv* = object
    mlp*: Sequential          ## MLP applied to (1+eps)*self + sum(neighbors)
    eps*: float32             ## Epsilon for self-weight

proc initGINConv*(mlp: Sequential; eps: float32 = 0.0'f32): GINConv =
  GINConv(mlp: mlp, eps: eps)

proc forward*(layer: GINConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of GINConv.
  ##
  ## Computes: `h_i = MLP((1 + eps) * x_i + sum_{j in N(i)} x_j)`.
  ##
  ## `x` shape `[N, inChannels]`, `edgeIndex` shape `[2, E]`.
  ## Returns `[N, outChannels]` where outChannels is the MLP output dim.
  let n = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  # Gather source node features for all edges.
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  let xSrc = indexSelect(x, src)                      # [E, F]
  # Sum neighbor features.
  let neighborSum = segmentSum(xSrc, dst, n)           # [N, F]
  # Combine self and neighbor features.
  let onePlusEps = scalarF32(1.0'f32 + layer.eps)
  let selfWeighted = mul(broadcastTo(onePlusEps, x.shape, @[]), x)
  let combined = add(selfWeighted, neighborSum)
  # Apply MLP.
  layer.mlp.forward(combined)
