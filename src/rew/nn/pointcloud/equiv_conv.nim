## SE(3)-equivariant graph convolution layer.
##
## Implements an EGNN-style (Satorras et al., 2021) equivariant
## message-passing convolution. The core idea:
##
##   1. Compute edge messages from node features and relative positions:
##      m_ij = φ_e(||r_ij||^2, x_i, x_j)
##   2. Update node features:  x'_i = φ_x(messages aggregated over j)
##   3. Update positions (optional):  r'_i = r_i + Σ_j (r_i - r_j) * φ_r(m_ij)
##
## All operations are SE(3)-equivariant and compose from existing
## primitives (Linear, silu, segmentSum, etc.).
##
## Pure value type following rew's functional nn invariant.

import ../../tensor
import ../../rng
import ../../ops/arith
import ../../ops/shape
import ../../ops/reduce
import ../../ops/gather
import ../../ops/segment
import ../../ops/concat
import ../linear
import ../activation

type
  EquivGraphConv* = object
    ## EGNN-style equivariant convolution.
    ##
    ## Updates both scalar features and 3D positions in an SE(3)-equivariant
    ## manner. The position update is optional (controlled by `updatePositions`).
    edgeMLP1*: Linear     ## [2*inCh + 1, hiddenCh]  — maps [x_i, x_j, ||r_ij||^2] → hidden
    edgeMLP2*: Linear     ## [hiddenCh, hiddenCh]
    nodeMLP1*: Linear     ## [inCh + hiddenCh, hiddenCh]  — maps [x_i, agg_messages] → hidden
    nodeMLP2*: Linear     ## [hiddenCh, outCh]
    coordMLP*: Linear     ## [hiddenCh, 1]  — position update weight
    updatePositions*: bool
    inChannels*: int
    outChannels*: int
    hiddenChannels*: int

proc initEquivGraphConv*(key: Key; inChannels, outChannels: int;
    hiddenChannels: int = 64; updatePositions: bool = true): EquivGraphConv =
  ## Construct an EGNN-style equivariant graph convolution.
  if inChannels <= 0 or outChannels <= 0 or hiddenChannels <= 0:
    raise newException(TensorError,
      "initEquivGraphConv: channel counts must be positive")
  let keys = split(key, 5)
  EquivGraphConv(
    edgeMLP1: initLinear(keys[0], 2 * inChannels + 1, hiddenChannels),
    edgeMLP2: initLinear(keys[1], hiddenChannels, hiddenChannels),
    nodeMLP1: initLinear(keys[2], inChannels + hiddenChannels, hiddenChannels),
    nodeMLP2: initLinear(keys[3], hiddenChannels, outChannels),
    coordMLP: initLinear(keys[4], hiddenChannels, 1),
    updatePositions: updatePositions,
    inChannels: inChannels,
    outChannels: outChannels,
    hiddenChannels: hiddenChannels,
  )

proc forward*(layer: EquivGraphConv; features, positions: Tensor;
    edgeIndex: Tensor): (Tensor, Tensor) =
  ## Forward pass.
  ##
  ## `features`: `[N, inCh]` float32.
  ## `positions`: `[N, 3]` float32.
  ## `edgeIndex`: `[2, E]` int32.
  ##
  ## Returns `(newFeatures, newPositions)`:
  ##   `newFeatures`: `[N, outCh]`
  ##   `newPositions`: `[N, 3]` (updated only if `updatePositions`)
  let n = features.shape[0]
  let numEdges = edgeIndex.shape[1]
  # Extract src/dst node indices.
  let src = squeeze(slice(edgeIndex, @[0, 0], @[1, numEdges], @[1, 1]), 0)
  let dst = squeeze(slice(edgeIndex, @[1, 0], @[2, numEdges], @[1, 1]), 0)
  # Gather source and destination features + positions.
  let fSrc = indexSelect(features, src)
  let fDst = indexSelect(features, dst)
  let pSrc = indexSelect(positions, src)
  let pDst = indexSelect(positions, dst)
  # Compute relative positions and squared distances.
  let relPos = sub(pSrc, pDst)  # [E, 3]
  let dist2 = reduceSum(mul(relPos, relPos), @[1])  # [E]
  # Build edge features: [fSrc, fDst, dist2].
  let dist2u = unsqueeze(dist2, 1)  # [E, 1]
  let edgeFeat = concat(@[fSrc, fDst, dist2u], 1)  # [E, 2*inCh + 1]
  # Edge MLP: messages m_ij.
  var messages = forward(layer.edgeMLP1, edgeFeat)
  messages = silu(messages)
  messages = forward(layer.edgeMLP2, messages)  # [E, hiddenCh]
  messages = silu(messages)
  # Aggregate messages per destination node.
  let aggMessages = segmentSum(messages, dst, n)  # [N, hiddenCh]
  # Node update: concat node features with aggregated messages.
  let nodeFeat = concat(@[features, aggMessages], 1)  # [N, inCh + hiddenCh]
  var newFeatures = forward(layer.nodeMLP1, nodeFeat)
  newFeatures = silu(newFeatures)
  newFeatures = forward(layer.nodeMLP2, newFeatures)  # [N, outCh]
  # Position update (equivariant).
  var newPositions = positions
  if layer.updatePositions:
    # φ_r(m_ij): map messages to scalar coordinate weight.
    let coordWeights = forward(layer.coordMLP, messages)  # [E, 1]
    # Δr_i = Σ_j (r_i - r_j) * φ_r(m_ij)
    let weighted = mul(relPos, coordWeights)  # [E, 3]
    let deltaPos = segmentSum(weighted, dst, n)  # [N, 3]
    newPositions = add(positions, deltaPos)
  (newFeatures, newPositions)
