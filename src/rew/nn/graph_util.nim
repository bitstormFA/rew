## Graph utility functions — message-passing building blocks.
##
## These are the reusable helpers that GNN convolution layers call. They
## operate on raw tensors in trace/jit mode and encode the common
## message-passing decomposition: gather → transform → scatter.

import ../tensor
import ../dtype
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/shape
import ../ops/concat
import ../ops/gather
import ../ops/segment
import ../ops/linalg

type
  AggregationType* = enum
    pAggrSum
    pAggrMean
    pAggrMax

# ---- edge-index helpers ---------------------------------------------------

proc degree*(index: Tensor; numNodes: int; dtype: DType = dtFloat32): Tensor =
  ## Compute per-node degree from a 1-D edge-index component (src or dst).
  ##
  ## `index` is `[E]` int32 (one row from edgeIndex). Returns `[N]` float32.
  let e = index.shape[0]
  var onesData = newSeq[float32](e)
  for i in 0 ..< e:
    onesData[i] = 1'f32
  let ones = constantF32(@[e], onesData)
  segmentSum(ones, index, numNodes)

proc addSelfLoops*(edgeIndex: Tensor; numNodes: int): Tensor =
  ## Append self-loop edges `[i, i]` for `i` in `0..<numNodes`.
  ## `edgeIndex` is `[2, E]` int32; result is `[2, E + numNodes]` int32.
  let selfSrc = iota(dtInt32, @[numNodes], 0, edgeIndex.device)
  let selfDst = iota(dtInt32, @[numNodes], 0, edgeIndex.device)
  let selfLoops = stack(@[selfSrc, selfDst], 0)
  concat(@[edgeIndex, selfLoops], 1)

proc normalizeEdgeIndex*(edgeIndex: Tensor; numNodes: int): (Tensor, Tensor) =
  ## GCN edge index normalization (symmetric).
  ##
  ## Returns `(edgeIndexSelfLoops, norm)` where `norm[e]` is
  ## `1 / sqrt(deg[src[e]]) * 1 / sqrt(deg[dst[e]])` for each edge `e`.
  ##
  ## Uses `exp(-0.5 * log(deg))` since StableHLO has no `pow` with
  ## fractional exponents.
  let ei = addSelfLoops(edgeIndex, numNodes)
  let numEdges = ei.shape[1]
  # Extract source/dest rows.
  let src = squeeze(slice(ei, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(ei, [1, 0], [2, numEdges], [1, 1]), 0)
  # Degree of source and destination nodes.
  let degSrc = degree(src, numNodes)
  let degDst = degree(dst, numNodes)
  # Gather degree at edge endpoints.
  let degSrcEdge = indexSelect(degSrc, src)
  let degDstEdge = indexSelect(degDst, dst)
  # norm = exp(-0.5 * log(degSrcEdge)) * exp(-0.5 * log(degDstEdge))
  let negHalf = scalarF32(-0.5'f32)
  let negHalfB = broadcastTo(negHalf, degSrcEdge.shape, @[])
  let normSrc = exp(mul(negHalfB, log(degSrcEdge)))
  let normDst = exp(mul(negHalfB, log(degDstEdge)))
  let norm = mul(normSrc, normDst)
  (ei, norm)

# ---- aggregation helpers --------------------------------------------------

proc segmentAggregate*(src: Tensor; indices: Tensor; numSegments: int;
    aggr: AggregationType): Tensor =
  ## Dispatch to `segmentSum` / `segmentMean` / `segmentMax`.
  case aggr
  of pAggrSum: segmentSum(src, indices, numSegments)
  of pAggrMean: segmentMean(src, indices, numSegments)
  of pAggrMax: segmentMax(src, indices, numSegments)

proc softmaxPerSegment*(src: Tensor; indices: Tensor;
    numSegments: int): Tensor =
  ## Per-segment softmax: `softmax` of `src[e, :]` over all edges `e` that
  ## share the same `indices[e]`.
  ##
  ## `src` shape `[E, F]`, `indices` shape `[E]` int32,
  ## result shape `[E, F]`.
  ##
  ## Used by GAT and other attention-based conv layers.
  let maxPerNode = segmentMax(src, indices, numSegments)   # [numSegments, F]
  let maxPerEdge = indexSelect(maxPerNode, indices)         # [E, F]
  let shifted = sub(src, maxPerEdge)
  let expShifted = exp(shifted)
  let sumExp = segmentSum(expShifted, indices, numSegments) # [numSegments, F]
  let sumPerEdge = indexSelect(sumExp, indices)             # [E, F]
  divide(expShifted, sumPerEdge)

# ---- message-passing skeleton ---------------------------------------------

proc messageAndAggregate*(x: Tensor; edgeIndex: Tensor;
    aggr: AggregationType;
    messageFn: proc(xSrc, xDst: Tensor): Tensor): Tensor =
  ## High-level message-passing primitive.
  ##
  ## 1. Extract source/destination indices from `edgeIndex` `[2, E]`.
  ## 2. Gather node features at source and destination positions.
  ## 3. Apply `messageFn(xSrc, xDst)` producing per-edge messages.
  ## 4. Aggregate messages by destination node index.
  ##
  ## Returns `[N, F_out]` where `N = x.shape[0]`.
  let numNodes = x.shape[0]
  let numEdges = edgeIndex.shape[1]
  # Extract source and destination node indices.
  let src = squeeze(slice(edgeIndex, [0, 0], [1, numEdges], [1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, [1, 0], [2, numEdges], [1, 1]), 0)
  # Gather node features.
  let xSrc = indexSelect(x, src)
  let xDst = indexSelect(x, dst)
  # Per-edge message computation.
  let messages = messageFn(xSrc, xDst)
  # Aggregate by destination.
  segmentAggregate(messages, dst, numNodes, aggr)
