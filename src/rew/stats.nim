## Classical statistics and lightweight ML estimators.

import std/[algorithm, math, sequtils]
import ./tensor
import ./dtype
import ./eager
import ./data/dataset
import ./dataframe

type
  StatsError* = object of CatchableError
    ## Raised by stats estimators and metric helpers.

  Matrix* = seq[seq[float64]]

  LinearRegression* = object
    coef*: seq[float64]
    intercept*: float64
    fitted*: bool

  Ridge* = object
    alpha*: float64
    coef*: seq[float64]
    intercept*: float64
    fitted*: bool

  Lasso* = object
    alpha*: float64
    maxIter*: int
    tol*: float64
    coef*: seq[float64]
    intercept*: float64
    fitted*: bool

  LogisticRegression* = object
    lr*: float64
    maxIter*: int
    l2*: float64
    coef*: seq[float64]
    intercept*: float64
    fitted*: bool

  StandardScaler* = object
    mean*: seq[float64]
    scale*: seq[float64]
    fitted*: bool

  PCA* = object
    components*: Matrix
    mean*: seq[float64]
    explainedVariance*: seq[float64]
    nComponents*: int
    maxIter*: int
    fitted*: bool

  Pipeline*[Steps] = object
    steps*: Steps
    fitted*: bool

  DataSplit*[T] = object
    train*: seq[T]
    test*: seq[T]

  ScoredEstimator*[E] = object
    estimator*: E
    score*: float64

func initLinearRegression*(): LinearRegression =
  LinearRegression()

func initRidge*(alpha: float64 = 1.0): Ridge =
  if alpha < 0:
    raise newException(StatsError, "Ridge alpha must be non-negative")
  Ridge(alpha: alpha)

func initLasso*(alpha: float64 = 1.0; maxIter = 1000;
    tol = 1e-6): Lasso =
  if alpha < 0:
    raise newException(StatsError, "Lasso alpha must be non-negative")
  if maxIter <= 0:
    raise newException(StatsError, "Lasso maxIter must be positive")
  Lasso(alpha: alpha, maxIter: maxIter, tol: tol)

func initLogisticRegression*(lr: float64 = 0.1; maxIter = 500;
    l2: float64 = 0.0): LogisticRegression =
  if lr <= 0:
    raise newException(StatsError, "LogisticRegression lr must be positive")
  if maxIter <= 0:
    raise newException(StatsError, "LogisticRegression maxIter must be positive")
  if l2 < 0:
    raise newException(StatsError, "LogisticRegression l2 must be non-negative")
  LogisticRegression(lr: lr, maxIter: maxIter, l2: l2)

func initStandardScaler*(): StandardScaler =
  StandardScaler()

func initPCA*(nComponents: int; maxIter = 100): PCA =
  if nComponents <= 0:
    raise newException(StatsError, "PCA nComponents must be positive")
  if maxIter <= 0:
    raise newException(StatsError, "PCA maxIter must be positive")
  PCA(nComponents: nComponents, maxIter: maxIter)

func initPipeline*[Steps](steps: Steps): Pipeline[Steps] =
  Pipeline[Steps](steps: steps)

func pipeline*[Steps](steps: Steps): Pipeline[Steps] =
  initPipeline(steps)

func pipeline*[A, B](first: A; second: B): Pipeline[(A, B)] =
  initPipeline((first, second))

proc requireMatrix(x: Matrix; opName: string) =
  if x.len == 0:
    raise newException(StatsError, opName & ": matrix must not be empty")
  let width = x[0].len
  if width == 0:
    raise newException(StatsError, opName & ": matrix must have columns")
  for i, row in x:
    if row.len != width:
      raise newException(StatsError,
        opName & ": row " & $i & " width differs from row 0")

proc requireXY(x: Matrix; y: seq[float64]; opName: string) =
  requireMatrix(x, opName)
  if y.len != x.len:
    raise newException(StatsError,
      opName & ": target length " & $y.len & " does not match rows " &
        $x.len)

func mean(values: seq[float64]): float64 =
  for v in values:
    result += v
  result / float64(values.len)

func dot(a, b: seq[float64]): float64 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]

proc transpose(x: Matrix): Matrix =
  requireMatrix(x, "transpose")
  result = newSeq[seq[float64]](x[0].len)
  for j in 0 ..< x[0].len:
    result[j] = newSeq[float64](x.len)
    for i in 0 ..< x.len:
      result[j][i] = x[i][j]

proc matmul(a, b: Matrix): Matrix =
  requireMatrix(a, "matmul")
  requireMatrix(b, "matmul")
  if a[0].len != b.len:
    raise newException(StatsError, "matmul: inner dimension mismatch")
  result = newSeq[seq[float64]](a.len)
  for i in 0 ..< a.len:
    result[i] = newSeq[float64](b[0].len)
    for j in 0 ..< b[0].len:
      for k in 0 ..< b.len:
        result[i][j] += a[i][k] * b[k][j]

proc solveLinear(a: Matrix; b: seq[float64]): seq[float64] =
  requireMatrix(a, "solveLinear")
  let n = a.len
  if a[0].len != n or b.len != n:
    raise newException(StatsError, "solveLinear: expected square system")
  var aug = newSeq[seq[float64]](n)
  for i in 0 ..< n:
    aug[i] = a[i] & @[b[i]]

  for col in 0 ..< n:
    var pivot = col
    for r in col + 1 ..< n:
      if abs(aug[r][col]) > abs(aug[pivot][col]):
        pivot = r
    if abs(aug[pivot][col]) < 1e-12:
      raise newException(StatsError, "solveLinear: singular matrix")
    if pivot != col:
      swap aug[pivot], aug[col]
    let denom = aug[col][col]
    for c in col .. n:
      aug[col][c] /= denom
    for r in 0 ..< n:
      if r == col:
        continue
      let factor = aug[r][col]
      for c in col .. n:
        aug[r][c] -= factor * aug[col][c]

  result = newSeq[float64](n)
  for i in 0 ..< n:
    result[i] = aug[i][n]

proc designMatrix(x: Matrix): Matrix =
  requireMatrix(x, "designMatrix")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = @[1.0] & row

proc fitLeastSquares(x: Matrix; y: seq[float64]; l2: float64): tuple[
    coef: seq[float64], intercept: float64] =
  requireXY(x, y, "fitLeastSquares")
  let d = designMatrix(x)
  let xt = transpose(d)
  var xtx = matmul(xt, d)
  for i in 1 ..< xtx.len:
    xtx[i][i] += l2
  var xty = newSeq[float64](xt.len)
  for i in 0 ..< xt.len:
    xty[i] = dot(xt[i], y)
  let beta = solveLinear(xtx, xty)
  result.intercept = beta[0]
  result.coef = beta[1 .. ^1]

proc fit*(model: LinearRegression; x: Matrix; y: seq[float64]):
    LinearRegression =
  discard model
  let solved = fitLeastSquares(x, y, 0.0)
  LinearRegression(coef: solved.coef, intercept: solved.intercept,
    fitted: true)

proc fit*(model: Ridge; x: Matrix; y: seq[float64]): Ridge =
  let solved = fitLeastSquares(x, y, model.alpha)
  Ridge(alpha: model.alpha, coef: solved.coef, intercept: solved.intercept,
    fitted: true)

func softThreshold(value, penalty: float64): float64 =
  if value > penalty:
    value - penalty
  elif value < -penalty:
    value + penalty
  else:
    0.0

proc fit*(model: Lasso; x: Matrix; y: seq[float64]): Lasso =
  requireXY(x, y, "Lasso.fit")
  let n = x.len
  let p = x[0].len
  var coef = newSeq[float64](p)
  let yMean = y.mean()
  var intercept = yMean

  for _ in 0 ..< model.maxIter:
    let old = coef
    for j in 0 ..< p:
      var residualDot = 0.0
      var norm = 0.0
      for i in 0 ..< n:
        var pred = intercept
        for k in 0 ..< p:
          if k != j:
            pred += x[i][k] * coef[k]
        let r = y[i] - pred
        residualDot += x[i][j] * r
        norm += x[i][j] * x[i][j]
      if norm > 0:
        coef[j] = softThreshold(residualDot / float64(n), model.alpha) /
          (norm / float64(n))
    var residualMean = 0.0
    for i in 0 ..< n:
      residualMean += y[i] - dot(x[i], coef)
    intercept = residualMean / float64(n)
    var delta = 0.0
    for j in 0 ..< p:
      delta = max(delta, abs(coef[j] - old[j]))
    if delta <= model.tol:
      break
  Lasso(alpha: model.alpha, maxIter: model.maxIter, tol: model.tol,
    coef: coef, intercept: intercept, fitted: true)

func sigmoid(x: float64): float64 =
  if x >= 0:
    1.0 / (1.0 + exp(-x))
  else:
    let z = exp(x)
    z / (1.0 + z)

proc fit*(model: LogisticRegression; x: Matrix; y: seq[float64]):
    LogisticRegression =
  requireXY(x, y, "LogisticRegression.fit")
  let n = x.len
  let p = x[0].len
  var coef = newSeq[float64](p)
  var intercept = 0.0
  for _ in 0 ..< model.maxIter:
    var grad = newSeq[float64](p)
    var gradIntercept = 0.0
    for i in 0 ..< n:
      let pred = sigmoid(intercept + dot(x[i], coef))
      let err = pred - y[i]
      gradIntercept += err
      for j in 0 ..< p:
        grad[j] += err * x[i][j]
    intercept -= model.lr * gradIntercept / float64(n)
    for j in 0 ..< p:
      let penalty = model.l2 * coef[j]
      coef[j] -= model.lr * (grad[j] / float64(n) + penalty)
  LogisticRegression(lr: model.lr, maxIter: model.maxIter, l2: model.l2,
    coef: coef, intercept: intercept, fitted: true)

proc predict*(model: LinearRegression | Ridge | Lasso; x: Matrix): seq[float64] =
  if not model.fitted:
    raise newException(StatsError, "predict: estimator is not fitted")
  requireMatrix(x, "predict")
  for row in x:
    result.add model.intercept + dot(row, model.coef)

proc predictProba*(model: LogisticRegression; x: Matrix): seq[float64] =
  if not model.fitted:
    raise newException(StatsError, "predictProba: estimator is not fitted")
  requireMatrix(x, "predictProba")
  for row in x:
    result.add sigmoid(model.intercept + dot(row, model.coef))

proc predict*(model: LogisticRegression; x: Matrix): seq[int] =
  for p in model.predictProba(x):
    result.add if p >= 0.5: 1 else: 0

proc fit*(scaler: StandardScaler; x: Matrix): StandardScaler =
  discard scaler
  requireMatrix(x, "StandardScaler.fit")
  let p = x[0].len
  result.mean = newSeq[float64](p)
  result.scale = newSeq[float64](p)
  for j in 0 ..< p:
    var col: seq[float64]
    for row in x:
      col.add row[j]
    result.mean[j] = col.mean()
    for v in col:
      result.scale[j] += (v - result.mean[j]) * (v - result.mean[j])
    result.scale[j] = sqrt(result.scale[j] / float64(x.len))
    if result.scale[j] == 0:
      result.scale[j] = 1
  result.fitted = true

proc transform*(scaler: StandardScaler; x: Matrix): Matrix =
  if not scaler.fitted:
    raise newException(StatsError, "StandardScaler.transform: scaler is not fitted")
  requireMatrix(x, "StandardScaler.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](row.len)
    for j, value in row:
      result[i][j] = (value - scaler.mean[j]) / scaler.scale[j]

proc fitTransform*(scaler: StandardScaler; x: Matrix): Matrix =
  scaler.fit(x).transform(x)

proc center(x: Matrix; meanOut: var seq[float64]): Matrix =
  requireMatrix(x, "center")
  let p = x[0].len
  meanOut = newSeq[float64](p)
  for j in 0 ..< p:
    for row in x:
      meanOut[j] += row[j]
    meanOut[j] /= float64(x.len)
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](p)
    for j in 0 ..< p:
      result[i][j] = row[j] - meanOut[j]

proc covariance(x: Matrix): Matrix =
  let xt = transpose(x)
  result = matmul(xt, x)
  let denom = max(1.0, float64(x.len - 1))
  for i in 0 ..< result.len:
    for j in 0 ..< result[i].len:
      result[i][j] /= denom

func matVec(a: Matrix; v: seq[float64]): seq[float64] =
  result = newSeq[float64](a.len)
  for i in 0 ..< a.len:
    result[i] = dot(a[i], v)

func norm(v: seq[float64]): float64 =
  sqrt(dot(v, v))

proc fit*(model: PCA; x: Matrix): PCA =
  requireMatrix(x, "PCA.fit")
  var meanVals: seq[float64]
  let centered = center(x, meanVals)
  var cov = covariance(centered)
  let p = cov.len
  let comps = min(model.nComponents, p)
  result = model
  result.mean = meanVals
  result.components = newSeq[seq[float64]](comps)
  result.explainedVariance = newSeq[float64](comps)
  for c in 0 ..< comps:
    var v = newSeq[float64](p)
    v[c mod p] = 1.0
    for _ in 0 ..< model.maxIter:
      var next = matVec(cov, v)
      let n = next.norm()
      if n == 0:
        break
      for i in 0 ..< next.len:
        next[i] /= n
      v = next
    let av = matVec(cov, v)
    let lambda = dot(v, av)
    result.components[c] = v
    result.explainedVariance[c] = lambda
    for i in 0 ..< p:
      for j in 0 ..< p:
        cov[i][j] -= lambda * v[i] * v[j]
  result.fitted = true

proc transform*(model: PCA; x: Matrix): Matrix =
  if not model.fitted:
    raise newException(StatsError, "PCA.transform: PCA is not fitted")
  requireMatrix(x, "PCA.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.components.len)
    for c, comp in model.components:
      var centered = newSeq[float64](row.len)
      for j in 0 ..< row.len:
        centered[j] = row[j] - model.mean[j]
      result[i][c] = dot(centered, comp)

proc fitTransform*(model: PCA; x: Matrix): Matrix =
  model.fit(x).transform(x)

proc fit*[A, E](pipe: Pipeline[(A, E)]; x: Matrix; y: seq[float64]): auto =
  ## Fits a transformer followed by a supervised estimator.
  let fittedFirst = pipe.steps[0].fit(x)
  let transformed = fittedFirst.transform(x)
  let fittedSecond = pipe.steps[1].fit(transformed, y)
  Pipeline[(typeof(fittedFirst), typeof(fittedSecond))](
    steps: (fittedFirst, fittedSecond),
    fitted: true,
  )

proc fit*[A, B](pipe: Pipeline[(A, B)]; x: Matrix): auto =
  ## Fits two transformer steps in sequence.
  let fittedFirst = pipe.steps[0].fit(x)
  let transformed = fittedFirst.transform(x)
  let fittedSecond = pipe.steps[1].fit(transformed)
  Pipeline[(typeof(fittedFirst), typeof(fittedSecond))](
    steps: (fittedFirst, fittedSecond),
    fitted: true,
  )

proc transform*[A, B](pipe: Pipeline[(A, B)]; x: Matrix): Matrix =
  ## Applies a fitted two-transformer pipeline.
  if not pipe.fitted:
    raise newException(StatsError, "Pipeline.transform: pipeline is not fitted")
  pipe.steps[1].transform(pipe.steps[0].transform(x))

proc predict*[A, E](pipe: Pipeline[(A, E)]; x: Matrix): auto =
  ## Applies a fitted transformer plus estimator pipeline.
  if not pipe.fitted:
    raise newException(StatsError, "Pipeline.predict: pipeline is not fitted")
  pipe.steps[1].predict(pipe.steps[0].transform(x))

proc predictProba*[A, E](pipe: Pipeline[(A, E)]; x: Matrix): auto =
  ## Applies a fitted transformer plus probabilistic estimator pipeline.
  if not pipe.fitted:
    raise newException(StatsError,
      "Pipeline.predictProba: pipeline is not fitted")
  pipe.steps[1].predictProba(pipe.steps[0].transform(x))

func asFloat(value: DataValue; column: string): float64 =
  case value.kind
  of dfvInt:
    float64(value.intVal)
  of dfvFloat:
    value.floatVal
  else:
    raise newException(StatsError,
      "expected numeric DataFrame value in column: " & column)

proc matrixAndTarget(df: DataFrame; features: openArray[string];
    target: string): tuple[x: Matrix, y: seq[float64]] =
  let rows = df.collect()
  if features.len == 0:
    raise newException(StatsError, "features must not be empty")
  result.x = newSeq[seq[float64]](rows.rowCount)
  for i in 0 ..< rows.rowCount:
    result.x[i] = newSeq[float64](features.len)
  for j, name in features:
    let col = rows.columns[rows.requireColumn(name)]
    for i in 0 ..< rows.rowCount:
      result.x[i][j] = asFloat(col.values[i], name)
  let yCol = rows.columns[rows.requireColumn(target)]
  result.y = newSeq[float64](rows.rowCount)
  for i in 0 ..< rows.rowCount:
    result.y[i] = asFloat(yCol.values[i], target)

proc matrixOnly(df: DataFrame; columns: openArray[string]): Matrix =
  let rows = df.collect()
  if columns.len == 0:
    raise newException(StatsError, "columns must not be empty")
  result = newSeq[seq[float64]](rows.rowCount)
  for i in 0 ..< rows.rowCount:
    result[i] = newSeq[float64](columns.len)
  for j, name in columns:
    let col = rows.columns[rows.requireColumn(name)]
    for i in 0 ..< rows.rowCount:
      result[i][j] = asFloat(col.values[i], name)

proc fit*(model: LinearRegression; df: DataFrame;
    features: openArray[string]; target: string): LinearRegression =
  let data = matrixAndTarget(df, features, target)
  model.fit(data.x, data.y)

proc fit*(model: Ridge; df: DataFrame; features: openArray[string];
    target: string): Ridge =
  let data = matrixAndTarget(df, features, target)
  model.fit(data.x, data.y)

proc fit*(model: Lasso; df: DataFrame; features: openArray[string];
    target: string): Lasso =
  let data = matrixAndTarget(df, features, target)
  model.fit(data.x, data.y)

proc fit*(model: LogisticRegression; df: DataFrame;
    features: openArray[string]; target: string): LogisticRegression =
  let data = matrixAndTarget(df, features, target)
  model.fit(data.x, data.y)

proc fit*(scaler: StandardScaler; df: DataFrame;
    columns: openArray[string]): StandardScaler =
  scaler.fit(matrixOnly(df, columns))

proc fit*(model: PCA; df: DataFrame; columns: openArray[string]): PCA =
  model.fit(matrixOnly(df, columns))

proc fit*[B, E](model: E; ds: Dataset[B];
    extractor: proc(batch: B): tuple[x: Matrix, y: seq[float64]]): E =
  var allX: Matrix
  var allY: seq[float64]
  for batch in ds:
    let part = extractor(batch)
    allX.add part.x
    allY.add part.y
  model.fit(allX, allY)

proc tensorToFloat64Seq(t: Tensor): seq[float64] =
  if t.dtype == dtFloat32:
    for v in t.toHost(float32):
      result.add float64(v)
  elif t.dtype == dtFloat64:
    result = t.toHost(float64)
  elif t.dtype == dtInt64:
    for v in t.toHost(int64):
      result.add float64(v)
  else:
    raise newException(StatsError,
      "unsupported tensor dtype for stats: " & t.dtype.name)

proc matrixFromTensor*(x: Tensor): Matrix =
  if x.shape.len != 2:
    raise newException(StatsError,
      "matrixFromTensor: expected rank-2 tensor, got " & $x.shape)
  let data = tensorToFloat64Seq(x)
  result = newSeq[seq[float64]](x.shape[0])
  for i in 0 ..< x.shape[0]:
    result[i] = newSeq[float64](x.shape[1])
    for j in 0 ..< x.shape[1]:
      result[i][j] = data[i * x.shape[1] + j]

proc vectorFromTensor*(y: Tensor): seq[float64] =
  if y.shape.len != 1:
    raise newException(StatsError,
      "vectorFromTensor: expected rank-1 tensor, got " & $y.shape)
  tensorToFloat64Seq(y)

proc fit*(model: LinearRegression; x, y: Tensor): LinearRegression =
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: Ridge; x, y: Tensor): Ridge =
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: Lasso; x, y: Tensor): Lasso =
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: LogisticRegression; x, y: Tensor): LogisticRegression =
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc trainTestSplit*[T](data: seq[T]; testSize: float64 = 0.25;
    seed: int = 0): DataSplit[T] =
  if testSize <= 0 or testSize >= 1:
    raise newException(StatsError, "testSize must be in (0, 1)")
  var idx = toSeq(0 ..< data.len)
  idx.sort(proc(a, b: int): int =
    cmp((a * 1103515245 + seed) mod 2147483647,
        (b * 1103515245 + seed) mod 2147483647))
  let testCount = int(round(float64(data.len) * testSize))
  for pos, original in idx:
    if pos < testCount:
      result.test.add data[original]
    else:
      result.train.add data[original]

func meanSquaredError*(yTrue, yPred: seq[float64]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(StatsError, "meanSquaredError: invalid lengths")
  for i in 0 ..< yTrue.len:
    let d = yTrue[i] - yPred[i]
    result += d * d
  result / float64(yTrue.len)

func accuracy*(yTrue: seq[int]; yPred: seq[int]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(StatsError, "accuracy: invalid lengths")
  var hits = 0
  for i in 0 ..< yTrue.len:
    if yTrue[i] == yPred[i]:
      inc hits
  float64(hits) / float64(yTrue.len)

func r2Score*(yTrue, yPred: seq[float64]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(StatsError, "r2Score: invalid lengths")
  let avg = yTrue.mean()
  var ssRes = 0.0
  var ssTot = 0.0
  for i in 0 ..< yTrue.len:
    ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i])
    ssTot += (yTrue[i] - avg) * (yTrue[i] - avg)
  if ssTot == 0:
    return if ssRes == 0: 1.0 else: 0.0
  1.0 - ssRes / ssTot

proc score*(model: LinearRegression | Ridge | Lasso; x: Matrix;
    y: seq[float64]): float64 =
  r2Score(y, model.predict(x))

proc score*(model: LogisticRegression; x: Matrix; y: seq[int]): float64 =
  accuracy(y, model.predict(x))

proc score*(model: LogisticRegression; x: Matrix; y: seq[float64]): float64 =
  var labels = newSeq[int](y.len)
  for i, value in y:
    labels[i] = if value >= 0.5: 1 else: 0
  model.score(x, labels)

proc score*[A, E](pipe: Pipeline[(A, E)]; x: Matrix;
    y: seq[float64]): float64 =
  if not pipe.fitted:
    raise newException(StatsError, "Pipeline.score: pipeline is not fitted")
  pipe.steps[1].score(pipe.steps[0].transform(x), y)

proc score*[A, E](pipe: Pipeline[(A, E)]; x: Matrix; y: seq[int]): float64 =
  if not pipe.fitted:
    raise newException(StatsError, "Pipeline.score: pipeline is not fitted")
  pipe.steps[1].score(pipe.steps[0].transform(x), y)

func rocAuc*(yTrue: seq[int]; scores: seq[float64]): float64 =
  if yTrue.len != scores.len or yTrue.len == 0:
    raise newException(StatsError, "rocAuc: invalid lengths")
  var pairs: seq[tuple[label: int, score: float64]]
  for i in 0 ..< yTrue.len:
    pairs.add (label: yTrue[i], score: scores[i])
  pairs.sort(proc(a, b: tuple[label: int, score: float64]): int =
    cmp(a.score, b.score))
  var rankSum = 0.0
  var positives = 0
  var negatives = 0
  for i, p in pairs:
    if p.label == 1:
      rankSum += float64(i + 1)
      inc positives
    else:
      inc negatives
  if positives == 0 or negatives == 0:
    raise newException(StatsError, "rocAuc: both classes are required")
  (rankSum - float64(positives * (positives + 1)) / 2.0) /
    float64(positives * negatives)

proc crossValScore*[E](model: E; x: Matrix; y: seq[float64]; folds = 5):
    seq[float64] =
  requireXY(x, y, "crossValScore")
  if folds <= 1 or folds > x.len:
    raise newException(StatsError, "crossValScore: invalid fold count")
  for fold in 0 ..< folds:
    var trainX, testX: Matrix
    var trainY, testY: seq[float64]
    for i in 0 ..< x.len:
      if i mod folds == fold:
        testX.add x[i]
        testY.add y[i]
      else:
        trainX.add x[i]
        trainY.add y[i]
    let fitted = model.fit(trainX, trainY)
    result.add meanSquaredError(testY, fitted.predict(testX))
