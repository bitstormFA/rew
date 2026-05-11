## Point cloud graph construction utilities.
##
## Host-side algorithms for building edge indices from point positions.
## These compute graphs (kNN, radius, FPS) outside the model trace,
## then feed the resulting edge_index tensors as constants into the
## traced computation.

import std/[algorithm, math, random]
import ../../tensor
import ../../dtype
import ../../ops/literal

type
  DistanceMatrix* = seq[seq[float32]]
    ## Dense N×N or N×M matrix of squared Euclidean distances.

proc hostPairwiseDist2*(positions: openArray[float32];
    n, dim: int): DistanceMatrix =
  ## Compute all-pairs squared Euclidean distances on the host.
  ## `positions` is a flat `[N * D]` array.
  result = newSeq[seq[float32]](n)
  for i in 0 ..< n:
    result[i] = newSeq[float32](n)
    for j in 0 ..< n:
      var d = 0'f32
      for k in 0 ..< dim:
        let dx = positions[i * dim + k] - positions[j * dim + k]
        d += dx * dx
      result[i][j] = d

proc hostPairwiseDist2*(a: openArray[float32];
    aN, aDim: int; b: openArray[float32]; bN, bDim: int): DistanceMatrix =
  ## Cross-set squared distances.
  if aDim != bDim:
    raise newException(ValueError,
      "hostPairwiseDist2: dimension mismatch " & $aDim & " vs " & $bDim)
  result = newSeq[seq[float32]](aN)
  for i in 0 ..< aN:
    result[i] = newSeq[float32](bN)
    for j in 0 ..< bN:
      var d = 0'f32
      for k in 0 ..< aDim:
        let dx = a[i * aDim + k] - b[j * bDim + k]
        d += dx * dx
      result[i][j] = d

proc radiusGraph*(positions: openArray[float32];
    n, dim: int; radius: float32; maxNeighbors: int = -1;
    selfLoops: bool = false): (seq[int32], seq[int32]) =
  ## Build edge_index (src, dst) for a radius graph.
  ##
  ## Connects all pairs (i, j) where `||p_i - p_j||^2 <= radius^2`.
  ## If `maxNeighbors > 0`, keeps only the closest `maxNeighbors` per node.
  ## If `selfLoops` is false, excludes i == j.
  ## Returns `(src, dst)` as flat seqs of edge endpoints.
  let r2 = radius * radius
  let dists = hostPairwiseDist2(positions, n, dim)
  type Edge = tuple[dst: int32, dist: float32]
  var perNodeEdges = newSeq[seq[Edge]](n)
  for i in 0 ..< n:
    perNodeEdges[i] = @[]
  for i in 0 ..< n:
    for j in 0 ..< n:
      if not selfLoops and i == j:
        continue
      if dists[i][j] <= r2:
        perNodeEdges[i].add (int32(j), dists[i][j])
    if maxNeighbors > 0 and perNodeEdges[i].len > maxNeighbors:
      perNodeEdges[i].sort(proc(a, b: Edge): int =
        cmp(a.dist, b.dist))
      perNodeEdges[i].setLen(maxNeighbors)
  var src, dst: seq[int32]
  for i in 0 ..< n:
    for edge in perNodeEdges[i]:
      src.add int32(i)
      dst.add edge.dst
  (src, dst)

proc knnGraph*(positions: openArray[float32]; n, dim, k: int):
    (seq[int32], seq[int32]) =
  ## Build k-nearest neighbor edge_index.
  ##
  ## For each node i, connects to its k nearest neighbors.
  ## Returns `(src, dst)` as flat seqs.
  if k <= 0 or k >= n:
    raise newException(ValueError,
      "knnGraph: k must be in [1, n-1], got " & $k)
  let dists = hostPairwiseDist2(positions, n, dim)
  type Edge = tuple[dst: int32, dist: float32]
  var src, dst: seq[int32] = @[]
  for i in 0 ..< n:
    var edges = newSeq[Edge](n - 1)
    var idx = 0
    for j in 0 ..< n:
      if i != j:
        edges[idx] = (int32(j), dists[i][j])
        inc idx
    edges.sort(proc(a, b: Edge): int = cmp(a.dist, b.dist))
    for j in 0 ..< k:
      src.add int32(i)
      dst.add edges[j].dst
  (src, dst)

type
  FPSSampler* = object
    ## Farthest Point Sampling state.
    ##
    ## Use `initFPS` to construct, then call `sample` to extract points.
    numPoints*: int
    dim*: int

proc initFPS*(positions: openArray[float32]; numPoints, dim: int): FPSSampler =
  FPSSampler(numPoints: numPoints, dim: dim)

proc fps*(positions: openArray[float32]; n, dim, numSamples: int;
    startIdx: int = -1): seq[int32] =
  ## Farthest Point Sampling.
  ##
  ## Returns indices of `numSamples` points that are farthest from
  ## each other. `startIdx` specifies the first point (random if < 0).
  ##
  ## Algorithm:
  ##   1. Pick a random starting point (or use startIdx).
  ##   2. For each remaining point, compute distance to the nearest
  ##      selected point.
  ##   3. Pick the point farthest from any selected point.
  ##   4. Repeat until numSamples points are selected.
  if numSamples > n:
    raise newException(ValueError,
      "fps: numSamples " & $numSamples & " > n " & $n)
  var selected = newSeq[int32](numSamples)
  var minDists = newSeq[float32](n)
  for i in 0 ..< n:
    minDists[i] = Inf.float32
  # Pick first point.
  let first = if startIdx >= 0 and startIdx < n: startIdx else:
    int(rand(1.0) * float(n - 1)) mod n
  selected[0] = int32(first)
  for i in 1 ..< numSamples:
    let lastIdx = int(selected[i - 1])
    var farthestIdx = -1
    var farthestDist = 0'f32
    for j in 0 ..< n:
      var d = 0'f32
      for k in 0 ..< dim:
        let dx = positions[lastIdx * dim + k] - positions[j * dim + k]
        d += dx * dx
      if d < minDists[j]:
        minDists[j] = d
      if minDists[j] > farthestDist:
        farthestDist = minDists[j]
        farthestIdx = j
    selected[i] = int32(farthestIdx)
  selected

proc edgeIndexTensor*(src, dst: openArray[int32]): Tensor =
  ## Convert `(src, dst)` arrays to an edge_index tensor `[2, E]` int32.
  ## The result is a trace-mode constant tensor.
  let e = src.len
  var data = newSeq[int32](2 * e)
  for i in 0 ..< e:
    data[i] = src[i]
    data[e + i] = dst[i]
  constant(dtInt32, @[2, e], i32Bytes(data))

proc toFlat*(positions: openArray[seq[float32]]): seq[float32] =
  ## Flatten `N × D` positions into a `[N * D]` array.
  result = newSeq[float32]()
  for row in positions:
    for v in row:
      result.add v
