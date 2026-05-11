## Built-in graph datasets — benchmark data for testing GNN layers.
##
## Karate Club is small enough to hardcode. Planetoid (Cora, CiteSeer,
## PubMed) expects data files in `root/Planetoid/<name>/`.

import std/options
import ../dtype
import ../ops/literal
import ./graph_data

# ---- Karate Club (hardcoded) ---------------------------------------------
# Zachary's karate club: 34 nodes, 78 undirected edges, 2 communities.

const karateEdges = [
  (0'i32, 1'i32), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
  (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
  (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
  (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
  (3, 7), (3, 12), (3, 13), (4, 6), (4, 10), (5, 6), (5, 10), (5, 16),
  (6, 16), (7, 12), (7, 13), (8, 30), (8, 32), (8, 33), (9, 33),
  (10, 16), (13, 33), (14, 32), (14, 33), (15, 32), (15, 33),
  (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
  (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27),
  (24, 31), (25, 31), (26, 29), (26, 33), (27, 33), (28, 31), (28, 33),
  (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33), (32, 33),
]

const karateLabels = [
  0'i32, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]

proc loadKarateClub*(): GraphData =
  ## Zachary's karate club graph. 34 nodes, 78 undirected edges, 2 classes.
  ## Node features are one-hot identity (34-dim).
  ##
  ## This dataset fits entirely in host memory. Call inside a trace block
  ## to materialise the tensors.
  const numNodes = 34
  const numEdges = len(karateEdges)
  # Build edge index (directed, both directions).
  var src = newSeq[int32](numEdges * 2)
  var dst = newSeq[int32](numEdges * 2)
  for i, (s, d) in karateEdges:
    src[i] = s
    dst[i] = d
    src[i + numEdges] = d
    dst[i + numEdges] = s
  # Create node features as one-hot identity.
  var featData = newSeq[float32](numNodes * numNodes)
  for i in 0 ..< numNodes:
    featData[i * numNodes + i] = 1'f32
  # Create tensors (trace-mode only).
  let x = constantF32(@[numNodes, numNodes], featData)
  var eiData = newSeq[int32](2 * numEdges * 2)
  for i in 0 ..< numEdges * 2:
    eiData[i] = src[i]
    eiData[i + numEdges * 2] = dst[i]
  let eiBytes = i32Bytes(eiData)
  let edgeIndex = constant(dtInt32, @[2, numEdges * 2], eiBytes)
  var yData = newSeq[float32](numNodes)
  for i in 0 ..< numNodes:
    yData[i] = float32(karateLabels[i])
  let y = constantF32(@[numNodes], yData)
  # Train/val/test masks for semi-supervised node classification.
  # Default: train on {0, 1}, val on {2..11}, test on {12..33}.
  var trainData = newSeq[float32](numNodes)
  var valData = newSeq[float32](numNodes)
  var testData = newSeq[float32](numNodes)
  for i in 0 ..< numNodes:
    if i < 2: trainData[i] = 1'f32
    elif i < 12: valData[i] = 1'f32
    else: testData[i] = 1'f32
  let trainMask = constantF32(@[numNodes], trainData)
  let valMask = constantF32(@[numNodes], valData)
  let testMask = constantF32(@[numNodes], testData)
  initGraphData(edgeIndex, x = some(x), y = some(y),
    trainMask = some(trainMask), valMask = some(valMask),
    testMask = some(testMask), numClasses = 2)
