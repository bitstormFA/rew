## Graph data container — homogeneous graph representation and batching.
##
## `GraphData` is a lightweight metadata container. Optional tensor fields
## use `Option[Tensor]`. Convolution layers take raw tensors directly;
## `GraphData` bundles them for dataset iteration.

import std/options
import ../tensor

type
  GraphData* = object
    ## Homogeneous graph data (analogous to PyG's `Data`).
    x*: Option[Tensor]        ## Node features [N, F] float32
    edgeIndex*: Tensor        ## Edge list [2, E] int32
    edgeAttr*: Option[Tensor] ## Edge attributes [E, D] float32
    y*: Option[Tensor]        ## Labels (graph-level or node-level)
    pos*: Option[Tensor]      ## Node positions [N, 3]
    batch*: Option[Tensor]    ## Batch assignment [N_total] int32
    trainMask*: Option[Tensor] ## Training node mask [N]
    valMask*: Option[Tensor]  ## Validation node mask [N]
    testMask*: Option[Tensor] ## Test node mask [N]
    numNodes*: int
    numEdges*: int
    numNodeFeatures*: int
    numEdgeFeatures*: int
    numClasses*: int

proc initGraphData*(edgeIndex: Tensor; x: Option[Tensor] = none(Tensor);
    edgeAttr: Option[Tensor] = none(Tensor);
    y: Option[Tensor] = none(Tensor); pos: Option[Tensor] = none(Tensor);
    trainMask: Option[Tensor] = none(Tensor);
    valMask: Option[Tensor] = none(Tensor);
    testMask: Option[Tensor] = none(Tensor);
    numClasses: int = 0): GraphData =
  if edgeIndex.shape.len != 2:
    raise newException(TensorError,
      "GraphData: edgeIndex must be rank 2, got rank " &
        $edgeIndex.shape.len)
  if edgeIndex.shape[0] != 2:
    raise newException(TensorError,
      "GraphData: edgeIndex dim 0 must be 2 (source, target), got " &
        $edgeIndex.shape[0])
  result = GraphData(
    edgeIndex: edgeIndex,
    numEdges: edgeIndex.shape[1],
    numClasses: numClasses,
  )
  if x.isSome:
    let xv = x.get
    if xv.shape.len < 1:
      raise newException(TensorError,
        "GraphData: x must have rank >= 1")
    result.x = x
    result.numNodes = xv.shape[0]
    result.numNodeFeatures = if xv.shape.len >= 2: xv.shape[1] else: 0
  else:
    result.x = none(Tensor)

  if edgeAttr.isSome:
    let ea = edgeAttr.get
    if ea.shape.len < 1:
      raise newException(TensorError,
        "GraphData: edgeAttr must have rank >= 1")
    if ea.shape[0] != result.numEdges:
      raise newException(TensorError,
        "GraphData: edgeAttr dim 0 " & $ea.shape[0] &
          " != numEdges " & $result.numEdges)
    result.edgeAttr = edgeAttr
    result.numEdgeFeatures = if ea.shape.len >= 2: ea.shape[1] else: 0

  if y.isSome:
    result.y = y
  if pos.isSome:
    let pv = pos.get
    if pv.shape.len != 2 or pv.shape[1] != 3:
      raise newException(TensorError,
        "GraphData: pos must be [N, 3], got " & $pv.shape)
    result.pos = pos
  if trainMask.isSome:
    result.trainMask = trainMask
  if valMask.isSome:
    result.valMask = valMask
  if testMask.isSome:
    result.testMask = testMask
