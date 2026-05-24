## Tensor-oriented numeric helpers used by ML estimators.
##
## The public entry points accept Tensor inputs where practical and use
## explicit host adapters when an algorithm has host-side control flow.

import std/[algorithm, math]
import ../tensor
import ./core

proc pairwiseDistances*(x, y: Matrix): Matrix =
  ## Squared Euclidean distance matrix `[x.rows, y.rows]`.
  requireMatrix(x, "pairwiseDistances")
  requireMatrix(y, "pairwiseDistances")
  if x[0].len != y[0].len:
    raise newException(MlError, "pairwiseDistances: feature width mismatch")
  result = newSeq[seq[float64]](x.len)
  for i in 0 ..< x.len:
    result[i] = newSeq[float64](y.len)
    for j in 0 ..< y.len:
      result[i][j] = squaredDistance(x[i], y[j])

proc pairwiseDistances*(x, y: Tensor): Matrix =
  ## Tensor adapter for `pairwiseDistances`.
  pairwiseDistances(matrixFromTensor(x), matrixFromTensor(y))

proc linearKernel*(x, y: Matrix): Matrix =
  ## Linear kernel matrix.
  requireMatrix(x, "linearKernel")
  requireMatrix(y, "linearKernel")
  if x[0].len != y[0].len:
    raise newException(MlError, "linearKernel: feature width mismatch")
  result = newSeq[seq[float64]](x.len)
  for i in 0 ..< x.len:
    result[i] = newSeq[float64](y.len)
    for j in 0 ..< y.len:
      result[i][j] = dotVec(x[i], y[j])

proc rbfKernel*(x, y: Matrix; gamma = 1.0): Matrix =
  ## Radial-basis-function kernel matrix.
  let dist = pairwiseDistances(x, y)
  result = dist
  for i in 0 ..< result.len:
    for j in 0 ..< result[i].len:
      result[i][j] = exp(-gamma * result[i][j])

proc covariance*(x: Matrix): Matrix =
  ## Column covariance matrix.
  requireMatrix(x, "covariance")
  let means = columnMeans(x)
  var centered = x
  for i in 0 ..< centered.len:
    for j in 0 ..< centered[i].len:
      centered[i][j] -= means[j]
  let xt = transpose(centered)
  result = matmul(xt, centered)
  let denom = max(1.0, float64(x.len - 1))
  for i in 0 ..< result.len:
    for j in 0 ..< result[i].len:
      result[i][j] /= denom

proc nearestIndices*(distances: openArray[float64]; k: int): seq[int] =
  ## Returns indices of the k smallest distances.
  if k <= 0:
    raise newException(MlError, "nearestIndices: k must be positive")
  var pairs: seq[tuple[idx: int, value: float64]]
  for i, value in distances:
    pairs.add (idx: i, value: value)
  pairs.sort(proc(a, b: tuple[idx: int, value: float64]): int =
    let byValue = cmp(a.value, b.value)
    if byValue != 0: byValue else: cmp(a.idx, b.idx))
  let count = min(k, pairs.len)
  for i in 0 ..< count:
    result.add pairs[i].idx

proc topKIndices*(values: openArray[float64]; k: int;
    largest = true): seq[int] =
  ## Returns indices of the top-k values.
  if k <= 0:
    raise newException(MlError, "topKIndices: k must be positive")
  var pairs: seq[tuple[idx: int, value: float64]]
  for i, value in values:
    pairs.add (idx: i, value: value)
  pairs.sort(proc(a, b: tuple[idx: int, value: float64]): int =
    let byValue = if largest: cmp(b.value, a.value) else: cmp(a.value, b.value)
    if byValue != 0: byValue else: cmp(a.idx, b.idx))
  let count = min(k, pairs.len)
  for i in 0 ..< count:
    result.add pairs[i].idx

proc labelCounts*(labels: openArray[int]): seq[tuple[label: int, count: int]] =
  ## Counts labels in ascending label order.
  let labelsUnique = uniqueLabels(labels)
  for label in labelsUnique:
    var count = 0
    for value in labels:
      if value == label:
        inc count
    result.add (label: label, count: count)

proc centroids*(x: Matrix; labels: openArray[int]): Matrix =
  ## Computes one centroid per sorted unique label.
  requireLabels(x, labels, "centroids")
  let classes = uniqueLabels(labels)
  result = newSeq[seq[float64]](classes.len)
  var counts = newSeq[int](classes.len)
  for c in 0 ..< classes.len:
    result[c] = newSeq[float64](x[0].len)
  for i, row in x:
    var c = -1
    for idx, label in classes:
      if label == labels[i]:
        c = idx
        break
    if c < 0:
      raise newException(MlError, "centroids: label lookup failed")
    inc counts[c]
    for j, value in row:
      result[c][j] += value
  for c in 0 ..< result.len:
    if counts[c] == 0:
      continue
    for j in 0 ..< result[c].len:
      result[c][j] /= float64(counts[c])

proc oneVsRestTargets*(y: openArray[int]; positiveLabel: int): seq[float64] =
  ## Builds binary one-vs-rest targets.
  for label in y:
    result.add if label == positiveLabel: 1.0 else: 0.0

proc thresholdLabels*(scores: openArray[float64]; threshold: float64;
    highIsOutlier = true): seq[int] =
  ## Converts anomaly scores into `InlierLabel`/`OutlierLabel`.
  for score in scores:
    let outlier =
      if highIsOutlier: score >= threshold else: score <= threshold
    result.add if outlier: OutlierLabel else: InlierLabel

proc quantile*(values: openArray[float64]; q: float64): float64 =
  ## Deterministic nearest-rank quantile.
  if values.len == 0:
    raise newException(MlError, "quantile: values must not be empty")
  if q < 0 or q > 1:
    raise newException(MlError, "quantile: q must be in [0, 1]")
  var sorted = @values
  sorted.sort()
  let idx = min(sorted.high, max(0, int(round(q * float64(sorted.len - 1)))))
  sorted[idx]
