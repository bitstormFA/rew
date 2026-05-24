## Mixture models.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./cluster

type
  GaussianMixture* = object
    nComponents*: int
    maxIter*: int
    weights*: seq[float64]
    means*: Matrix
    variances*: Matrix
    fitted*: bool

func initGaussianMixture*(nComponents: int; maxIter = 50): GaussianMixture =
  if nComponents <= 0 or maxIter <= 0:
    raise newException(MlError,
      "GaussianMixture nComponents/maxIter must be positive")
  GaussianMixture(nComponents: nComponents, maxIter: maxIter)

proc gaussianDiag(row, mean, variance: openArray[float64]): float64 =
  result = 1.0
  for j, value in row:
    let varj = max(variance[j], 1e-9)
    let diff = value - mean[j]
    result *= exp(-diff * diff / (2.0 * varj)) / sqrt(2.0 * PI * varj)

proc fit*(model: GaussianMixture; x: Matrix): GaussianMixture =
  requireMatrix(x, "GaussianMixture.fit")
  let init = initKMeans(model.nComponents).fit(x)
  var means = init.centroids
  var variances = newSeq[seq[float64]](model.nComponents)
  var weights = newSeq[float64](model.nComponents)
  for c in 0 ..< model.nComponents:
    variances[c] = newSeq[float64](x[0].len)
    weights[c] = 1.0 / float64(model.nComponents)
    for j in 0 ..< x[0].len:
      variances[c][j] = 1.0
  var resp = newSeq[seq[float64]](x.len)
  for i in 0 ..< x.len:
    resp[i] = newSeq[float64](model.nComponents)
  for _ in 0 ..< model.maxIter:
    for i, row in x:
      var total = 0.0
      for c in 0 ..< model.nComponents:
        resp[i][c] = weights[c] * gaussianDiag(row, means[c], variances[c])
        total += resp[i][c]
      if total == 0:
        for c in 0 ..< model.nComponents:
          resp[i][c] = 1.0 / float64(model.nComponents)
      else:
        for c in 0 ..< model.nComponents:
          resp[i][c] /= total
    for c in 0 ..< model.nComponents:
      var nk = 0.0
      for i in 0 ..< x.len:
        nk += resp[i][c]
      weights[c] = nk / float64(x.len)
      for j in 0 ..< x[0].len:
        var mu = 0.0
        for i in 0 ..< x.len:
          mu += resp[i][c] * x[i][j]
        mu /= max(1e-9, nk)
        means[c][j] = mu
        var v = 0.0
        for i in 0 ..< x.len:
          let d = x[i][j] - mu
          v += resp[i][c] * d * d
        variances[c][j] = v / max(1e-9, nk) + 1e-9
  GaussianMixture(nComponents: model.nComponents, maxIter: model.maxIter,
    weights: weights, means: means, variances: variances, fitted: true)

proc predictProba*(model: GaussianMixture; x: Matrix): Matrix =
  requireFitted(model.fitted, "GaussianMixture.predictProba")
  requireMatrix(x, "GaussianMixture.predictProba")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.nComponents)
    var total = 0.0
    for c in 0 ..< model.nComponents:
      result[i][c] = model.weights[c] *
        gaussianDiag(row, model.means[c], model.variances[c])
      total += result[i][c]
    if total > 0:
      for c in 0 ..< model.nComponents:
        result[i][c] /= total

proc predict*(model: GaussianMixture; x: Matrix): seq[int] =
  let probs = model.predictProba(x)
  for row in probs:
    var best = 0
    for c in 1 ..< row.len:
      if row[c] > row[best]:
        best = c
    result.add best

proc fitPredict*(model: GaussianMixture; x: Matrix): seq[int] =
  model.fit(x).predict(x)

proc fit*(model: GaussianMixture; x: Tensor;
    options: FitOptions = initFitOptions()): GaussianMixture =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: GaussianMixture; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): GaussianMixture =
  discard options
  model.fit(matrixOnly(df, columns))
