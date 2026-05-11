## `EdgeConv` — Dynamic Edge Convolution (Wang et al., 2019 / DGCNN).
##
## Used for point cloud and graph learning where edge features are
## computed as `MLP(concat(x_i, x_j - x_i))` and aggregated via max.
##
## Pure value type following rew's functional nn invariant.

import ../tensor
import ../ops/arith
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/segment
import ./sequential

type
  EdgeConv* = object
    mlp*: Sequential              ## MLP applied to per-edge features

proc initEdgeConv*(mlp: Sequential): EdgeConv =
  EdgeConv(mlp: mlp)

proc forward*(layer: EdgeConv; x, edgeIndex: Tensor): Tensor =
  ## Forward pass of EdgeConv.
  ##
  ## Computes per-edge features `concat(x_src, x_dst - x_src)`, applies
  ## an MLP, and aggregates via max per destination node.
  ##
  ## `x` shape `[N, F]`, `edgeIndex` shape `[2, E]`.
  ## Returns `[N, outChannels]`.
  let n = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  # Gather source and destination features.
  let xSrc = indexSelect(x, src)                        # [E, F]
  let xDst = indexSelect(x, dst)                        # [E, F]
  # Per-edge features: [x_src, x_dst - x_src].
  let diff = sub(xSrc, xDst)
  let edgeFeat = concat(@[xSrc, diff], 1)               # [E, 2F]
  # MLP on each edge.
  let messages = layer.mlp.forward(edgeFeat)             # [E, F_out]
  # Max aggregation per destination node.
  segmentMax(messages, dst, n)                           # [N, F_out]
