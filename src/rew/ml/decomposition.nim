## Matrix decomposition and representation-learning transformers.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./tensor_utils

type
  PCA* = object
    components*: Matrix
    mean*: seq[float64]
    explainedVariance*: seq[float64]
    nComponents*: int
    maxIter*: int
    fitted*: bool

  IncrementalPCA* = object
    pca*: PCA
    batchSize*: int
    fitted*: bool

  TruncatedSVD* = object
    components*: Matrix
    explainedVariance*: seq[float64]
    nComponents*: int
    maxIter*: int
    fitted*: bool

  NMF* = object
    nComponents*: int
    maxIter*: int
    w*: Matrix
    h*: Matrix
    fitted*: bool

func initPCA*(nComponents: int; maxIter = 100): PCA =
  if nComponents <= 0 or maxIter <= 0:
    raise newException(MlError, "PCA nComponents/maxIter must be positive")
  PCA(nComponents: nComponents, maxIter: maxIter)

func initIncrementalPCA*(nComponents: int; batchSize = 128;
    maxIter = 100): IncrementalPCA =
  if batchSize <= 0:
    raise newException(MlError, "IncrementalPCA batchSize must be positive")
  IncrementalPCA(pca: initPCA(nComponents, maxIter), batchSize: batchSize)

func initTruncatedSVD*(nComponents: int; maxIter = 100): TruncatedSVD =
  if nComponents <= 0 or maxIter <= 0:
    raise newException(MlError,
      "TruncatedSVD nComponents/maxIter must be positive")
  TruncatedSVD(nComponents: nComponents, maxIter: maxIter)

func initNMF*(nComponents: int; maxIter = 100): NMF =
  if nComponents <= 0 or maxIter <= 0:
    raise newException(MlError, "NMF nComponents/maxIter must be positive")
  NMF(nComponents: nComponents, maxIter: maxIter)

proc centerMatrix(x: Matrix; meanOut: var seq[float64]): Matrix =
  requireMatrix(x, "centerMatrix")
  meanOut = columnMeans(x)
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](row.len)
    for j in 0 ..< row.len:
      result[i][j] = row[j] - meanOut[j]

func matVec(a: Matrix; v: openArray[float64]): seq[float64] =
  result = newSeq[float64](a.len)
  for i in 0 ..< a.len:
    result[i] = dotVec(a[i], v)

func normVec(v: openArray[float64]): float64 =
  sqrt(dotVec(v, v))

proc principalComponents(cov: Matrix; nComponents, maxIter: int):
    tuple[components: Matrix, explained: seq[float64]] =
  var work = cov
  let p = cov.len
  let comps = min(nComponents, p)
  result.components = newSeq[seq[float64]](comps)
  result.explained = newSeq[float64](comps)
  for c in 0 ..< comps:
    var v = newSeq[float64](p)
    v[c mod p] = 1.0
    for _ in 0 ..< maxIter:
      var next = matVec(work, v)
      let n = normVec(next)
      if n == 0:
        break
      for i in 0 ..< next.len:
        next[i] /= n
      v = next
    let av = matVec(work, v)
    let lambda = dotVec(v, av)
    result.components[c] = v
    result.explained[c] = lambda
    for i in 0 ..< p:
      for j in 0 ..< p:
        work[i][j] -= lambda * v[i] * v[j]

proc fit*(model: PCA; x: Matrix): PCA =
  requireMatrix(x, "PCA.fit")
  var meanVals: seq[float64]
  let centered = centerMatrix(x, meanVals)
  let solved = principalComponents(covariance(centered),
    model.nComponents, model.maxIter)
  PCA(components: solved.components, mean: meanVals,
    explainedVariance: solved.explained, nComponents: model.nComponents,
    maxIter: model.maxIter, fitted: true)

proc transform*(model: PCA; x: Matrix): Matrix =
  requireFitted(model.fitted, "PCA.transform")
  requireMatrix(x, "PCA.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.components.len)
    var centered = newSeq[float64](row.len)
    for j in 0 ..< row.len:
      centered[j] = row[j] - model.mean[j]
    for c, comp in model.components:
      result[i][c] = dotVec(centered, comp)

proc inverseTransform*(model: PCA; z: Matrix): Matrix =
  requireFitted(model.fitted, "PCA.inverseTransform")
  requireMatrix(z, "PCA.inverseTransform")
  result = newSeq[seq[float64]](z.len)
  for i, row in z:
    result[i] = model.mean
    for c, value in row:
      for j in 0 ..< result[i].len:
        result[i][j] += value * model.components[c][j]

proc fitTransform*(model: PCA; x: Matrix): Matrix =
  model.fit(x).transform(x)

proc fit*(model: IncrementalPCA; x: Matrix): IncrementalPCA =
  result = model
  result.pca = model.pca.fit(x)
  result.fitted = true

proc transform*(model: IncrementalPCA; x: Matrix): Matrix =
  requireFitted(model.fitted, "IncrementalPCA.transform")
  model.pca.transform(x)

proc partialFit*(model: IncrementalPCA; x: Matrix;
    options: FitOptions = initFitOptions()): IncrementalPCA =
  discard options
  model.fit(x)

proc fit*(model: TruncatedSVD; x: Matrix): TruncatedSVD =
  requireMatrix(x, "TruncatedSVD.fit")
  let solved = principalComponents(covariance(x),
    model.nComponents, model.maxIter)
  TruncatedSVD(components: solved.components,
    explainedVariance: solved.explained, nComponents: model.nComponents,
    maxIter: model.maxIter, fitted: true)

proc transform*(model: TruncatedSVD; x: Matrix): Matrix =
  requireFitted(model.fitted, "TruncatedSVD.transform")
  requireMatrix(x, "TruncatedSVD.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.components.len)
    for c, comp in model.components:
      result[i][c] = dotVec(row, comp)

proc fitTransform*(model: TruncatedSVD; x: Matrix): Matrix =
  model.fit(x).transform(x)

proc fit*(model: NMF; x: Matrix): NMF =
  requireMatrix(x, "NMF.fit")
  let n = x.len
  let p = x[0].len
  let k = min(model.nComponents, p)
  var w = newSeq[seq[float64]](n)
  var h = newSeq[seq[float64]](k)
  for i in 0 ..< n:
    w[i] = newSeq[float64](k)
    for c in 0 ..< k:
      w[i][c] = 0.5 + float64((i + c) mod 3) / 10.0
  for c in 0 ..< k:
    h[c] = newSeq[float64](p)
    for j in 0 ..< p:
      h[c][j] = 0.5 + float64((c + j) mod 3) / 10.0

  const eps = 1e-9
  for _ in 0 ..< model.maxIter:
    let wh = matmul(w, h)
    for c in 0 ..< k:
      for j in 0 ..< p:
        var num, den = 0.0
        for i in 0 ..< n:
          num += w[i][c] * x[i][j]
          den += w[i][c] * wh[i][j]
        h[c][j] *= num / max(eps, den)
    let wh2 = matmul(w, h)
    for i in 0 ..< n:
      for c in 0 ..< k:
        var num, den = 0.0
        for j in 0 ..< p:
          num += x[i][j] * h[c][j]
          den += wh2[i][j] * h[c][j]
        w[i][c] *= num / max(eps, den)
  NMF(nComponents: model.nComponents, maxIter: model.maxIter, w: w, h: h,
    fitted: true)

proc transform*(model: NMF; x: Matrix): Matrix =
  requireFitted(model.fitted, "NMF.transform")
  requireMatrix(x, "NMF.transform")
  let k = model.h.len
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](k)
    for c in 0 ..< k:
      let denom = max(1e-9, dotVec(model.h[c], model.h[c]))
      result[i][c] = max(0.0, dotVec(row, model.h[c]) / denom)

proc fitTransform*(model: NMF; x: Matrix): Matrix =
  model.fit(x).transform(x)

proc fit*(model: PCA; x: Tensor;
    options: FitOptions = initFitOptions()): PCA =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: IncrementalPCA; x: Tensor;
    options: FitOptions = initFitOptions()): IncrementalPCA =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: TruncatedSVD; x: Tensor;
    options: FitOptions = initFitOptions()): TruncatedSVD =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: NMF; x: Tensor;
    options: FitOptions = initFitOptions()): NMF =
  discard options
  model.fit(matrixFromTensor(x))

proc transform*(model: PCA | IncrementalPCA | TruncatedSVD | NMF;
    x: Tensor): Matrix =
  model.transform(matrixFromTensor(x))

proc fit*(model: PCA; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): PCA =
  discard options
  model.fit(matrixOnly(df, columns))

proc fit*(model: IncrementalPCA; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): IncrementalPCA =
  discard options
  model.fit(matrixOnly(df, columns))

proc fit*(model: TruncatedSVD; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): TruncatedSVD =
  discard options
  model.fit(matrixOnly(df, columns))

proc fit*(model: NMF; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): NMF =
  discard options
  model.fit(matrixOnly(df, columns))
