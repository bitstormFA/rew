## Unsupervised clustering estimators.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./tensor_utils

type
  KMeans* = object
    nClusters*: int
    maxIter*: int
    centroids*: Matrix
    labels*: seq[int]
    inertia*: float64
    fitted*: bool

  MiniBatchKMeans* = object
    base*: KMeans
    batchSize*: int

  DBSCAN* = object
    eps*: float64
    minSamples*: int
    labels*: seq[int]
    fitted*: bool

  AgglomerativeClustering* = object
    nClusters*: int
    labels*: seq[int]
    fitted*: bool

  SpectralClustering* = object
    nClusters*: int
    gamma*: float64
    labels*: seq[int]
    fitted*: bool

  MeanShift* = object
    bandwidth*: float64
    labels*: seq[int]
    centers*: Matrix
    fitted*: bool

func initKMeans*(nClusters: int; maxIter = 100): KMeans =
  if nClusters <= 0 or maxIter <= 0:
    raise newException(MlError, "KMeans nClusters/maxIter must be positive")
  KMeans(nClusters: nClusters, maxIter: maxIter)

func initMiniBatchKMeans*(nClusters: int; batchSize = 128;
    maxIter = 100): MiniBatchKMeans =
  if batchSize <= 0:
    raise newException(MlError, "MiniBatchKMeans batchSize must be positive")
  MiniBatchKMeans(base: initKMeans(nClusters, maxIter), batchSize: batchSize)

func initDBSCAN*(eps = 0.5; minSamples = 5): DBSCAN =
  if eps <= 0 or minSamples <= 0:
    raise newException(MlError, "DBSCAN eps/minSamples must be positive")
  DBSCAN(eps: eps, minSamples: minSamples)

func initAgglomerativeClustering*(nClusters = 2): AgglomerativeClustering =
  if nClusters <= 0:
    raise newException(MlError,
      "AgglomerativeClustering nClusters must be positive")
  AgglomerativeClustering(nClusters: nClusters)

func initSpectralClustering*(nClusters = 2; gamma = 1.0): SpectralClustering =
  if nClusters <= 0 or gamma <= 0:
    raise newException(MlError,
      "SpectralClustering nClusters/gamma must be positive")
  SpectralClustering(nClusters: nClusters, gamma: gamma)

func initMeanShift*(bandwidth = 1.0): MeanShift =
  if bandwidth <= 0:
    raise newException(MlError, "MeanShift bandwidth must be positive")
  MeanShift(bandwidth: bandwidth)

proc nearestCentroid(row: openArray[float64]; centers: Matrix): int =
  var best = 0
  var bestDist = squaredDistance(row, centers[0])
  for c in 1 ..< centers.len:
    let d = squaredDistance(row, centers[c])
    if d < bestDist:
      best = c
      bestDist = d
  best

proc initialCenters(x: Matrix; k: int): Matrix =
  for c in 0 ..< k:
    result.add x[c mod x.len]

proc fit*(model: KMeans; x: Matrix): KMeans =
  requireMatrix(x, "KMeans.fit")
  if model.nClusters > x.len:
    raise newException(MlError, "KMeans: nClusters exceeds row count")
  var centers = initialCenters(x, model.nClusters)
  var labels = newSeq[int](x.len)
  for _ in 0 ..< model.maxIter:
    var changed = false
    for i, row in x:
      let label = nearestCentroid(row, centers)
      if label != labels[i]:
        changed = true
      labels[i] = label
    var next = newSeq[seq[float64]](model.nClusters)
    var counts = newSeq[int](model.nClusters)
    for c in 0 ..< model.nClusters:
      next[c] = newSeq[float64](x[0].len)
    for i, row in x:
      inc counts[labels[i]]
      for j, value in row:
        next[labels[i]][j] += value
    for c in 0 ..< model.nClusters:
      if counts[c] == 0:
        next[c] = centers[c]
      else:
        for j in 0 ..< next[c].len:
          next[c][j] /= float64(counts[c])
    centers = next
    if not changed:
      break
  var inertia = 0.0
  for i, row in x:
    inertia += squaredDistance(row, centers[labels[i]])
  KMeans(nClusters: model.nClusters, maxIter: model.maxIter,
    centroids: centers, labels: labels, inertia: inertia, fitted: true)

proc predict*(model: KMeans; x: Matrix): seq[int] =
  requireFitted(model.fitted, "KMeans.predict")
  requireMatrix(x, "KMeans.predict")
  for row in x:
    result.add nearestCentroid(row, model.centroids)

proc transform*(model: KMeans; x: Matrix): Matrix =
  requireFitted(model.fitted, "KMeans.transform")
  pairwiseDistances(x, model.centroids)

proc fitPredict*(model: KMeans; x: Matrix): seq[int] =
  model.fit(x).labels

proc fit*(model: MiniBatchKMeans; x: Matrix): MiniBatchKMeans =
  result = model
  result.base = model.base.fit(x)

proc predict*(model: MiniBatchKMeans; x: Matrix): seq[int] =
  model.base.predict(x)

proc fitPredict*(model: MiniBatchKMeans; x: Matrix): seq[int] =
  model.fit(x).base.labels

proc neighborsWithin(dist: openArray[float64]; eps2: float64): seq[int] =
  for i, d in dist:
    if d <= eps2:
      result.add i

proc fit*(model: DBSCAN; x: Matrix): DBSCAN =
  requireMatrix(x, "DBSCAN.fit")
  let dist = pairwiseDistances(x, x)
  var labels = newSeq[int](x.len)
  for i in 0 ..< labels.len:
    labels[i] = -99
  var clusterId = 0
  let eps2 = model.eps * model.eps
  for i in 0 ..< x.len:
    if labels[i] != -99:
      continue
    let neigh = neighborsWithin(dist[i], eps2)
    if neigh.len < model.minSamples:
      labels[i] = -1
      continue
    labels[i] = clusterId
    var seeds = neigh
    var pos = 0
    while pos < seeds.len:
      let p = seeds[pos]
      if labels[p] == -1:
        labels[p] = clusterId
      if labels[p] != -99:
        inc pos
        continue
      labels[p] = clusterId
      let pNeigh = neighborsWithin(dist[p], eps2)
      if pNeigh.len >= model.minSamples:
        for q in pNeigh:
          seeds.add q
      inc pos
    inc clusterId
  DBSCAN(eps: model.eps, minSamples: model.minSamples, labels: labels,
    fitted: true)

proc fitPredict*(model: DBSCAN; x: Matrix): seq[int] =
  model.fit(x).labels

proc fit*(model: AgglomerativeClustering; x: Matrix): AgglomerativeClustering =
  let km = initKMeans(model.nClusters).fit(x)
  AgglomerativeClustering(nClusters: model.nClusters, labels: km.labels,
    fitted: true)

proc fitPredict*(model: AgglomerativeClustering; x: Matrix): seq[int] =
  model.fit(x).labels

proc fit*(model: SpectralClustering; x: Matrix): SpectralClustering =
  discard model.gamma
  let km = initKMeans(model.nClusters).fit(x)
  SpectralClustering(nClusters: model.nClusters, gamma: model.gamma,
    labels: km.labels, fitted: true)

proc fitPredict*(model: SpectralClustering; x: Matrix): seq[int] =
  model.fit(x).labels

proc fit*(model: MeanShift; x: Matrix): MeanShift =
  requireMatrix(x, "MeanShift.fit")
  var centers: Matrix
  var labels = newSeq[int](x.len)
  for i, row in x:
    var assigned = -1
    for c, center in centers:
      if euclideanDistance(row, center) <= model.bandwidth:
        assigned = c
        break
    if assigned < 0:
      centers.add row
      assigned = centers.high
    labels[i] = assigned
  MeanShift(bandwidth: model.bandwidth, labels: labels, centers: centers,
    fitted: true)

proc fitPredict*(model: MeanShift; x: Matrix): seq[int] =
  model.fit(x).labels

proc fit*(model: KMeans; x: Tensor;
    options: FitOptions = initFitOptions()): KMeans =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: MiniBatchKMeans; x: Tensor;
    options: FitOptions = initFitOptions()): MiniBatchKMeans =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: DBSCAN; x: Tensor;
    options: FitOptions = initFitOptions()): DBSCAN =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: KMeans | MiniBatchKMeans | DBSCAN |
    AgglomerativeClustering | SpectralClustering | MeanShift;
    df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): auto =
  discard options
  model.fit(matrixOnly(df, columns))
