## Tensor-first linear supervised estimators.

import std/[math]
import ../tensor
import ../dataframe
import ../data/dataset
import ./core
import ./tensor_utils
import ./metrics

type
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

  ElasticNet* = object
    alpha*: float64
    l1Ratio*: float64
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

  SoftmaxRegression* = object
    lr*: float64
    maxIter*: int
    l2*: float64
    classes*: seq[int]
    coef*: Matrix
    intercept*: seq[float64]
    fitted*: bool

  LinearSVM* = object
    lr*: float64
    maxIter*: int
    c*: float64
    classes*: seq[int]
    coef*: Matrix
    intercept*: seq[float64]
    fitted*: bool

func initLinearRegression*(): LinearRegression =
  LinearRegression()

func initRidge*(alpha: float64 = 1.0): Ridge =
  if alpha < 0:
    raise newException(MlError, "Ridge alpha must be non-negative")
  Ridge(alpha: alpha)

func initLasso*(alpha: float64 = 1.0; maxIter = 1000;
    tol = 1e-6): Lasso =
  if alpha < 0:
    raise newException(MlError, "Lasso alpha must be non-negative")
  if maxIter <= 0:
    raise newException(MlError, "Lasso maxIter must be positive")
  Lasso(alpha: alpha, maxIter: maxIter, tol: tol)

func initElasticNet*(alpha: float64 = 1.0; l1Ratio: float64 = 0.5;
    maxIter = 1000; tol = 1e-6): ElasticNet =
  if alpha < 0:
    raise newException(MlError, "ElasticNet alpha must be non-negative")
  if l1Ratio < 0 or l1Ratio > 1:
    raise newException(MlError, "ElasticNet l1Ratio must be in [0, 1]")
  if maxIter <= 0:
    raise newException(MlError, "ElasticNet maxIter must be positive")
  ElasticNet(alpha: alpha, l1Ratio: l1Ratio, maxIter: maxIter, tol: tol)

func initLogisticRegression*(lr: float64 = 0.1; maxIter = 500;
    l2: float64 = 0.0): LogisticRegression =
  if lr <= 0:
    raise newException(MlError, "LogisticRegression lr must be positive")
  if maxIter <= 0:
    raise newException(MlError, "LogisticRegression maxIter must be positive")
  if l2 < 0:
    raise newException(MlError, "LogisticRegression l2 must be non-negative")
  LogisticRegression(lr: lr, maxIter: maxIter, l2: l2)

func initSoftmaxRegression*(lr: float64 = 0.1; maxIter = 500;
    l2: float64 = 0.0): SoftmaxRegression =
  if lr <= 0 or maxIter <= 0 or l2 < 0:
    raise newException(MlError,
      "SoftmaxRegression requires positive lr/maxIter and non-negative l2")
  SoftmaxRegression(lr: lr, maxIter: maxIter, l2: l2)

func initLinearSVM*(lr: float64 = 0.1; maxIter = 500;
    c: float64 = 1.0): LinearSVM =
  if lr <= 0 or maxIter <= 0 or c <= 0:
    raise newException(MlError,
      "LinearSVM requires positive lr, maxIter, and c")
  LinearSVM(lr: lr, maxIter: maxIter, c: c)

proc fit*(model: LinearRegression; x: Matrix; y: openArray[float64]):
    LinearRegression =
  discard model
  let solved = fitLeastSquares(x, y, 0.0)
  LinearRegression(coef: solved.coef, intercept: solved.intercept,
    fitted: true)

proc fit*(model: Ridge; x: Matrix; y: openArray[float64]): Ridge =
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

proc fitCoordinateDescent(x: Matrix; y: openArray[float64]; alpha, l1Ratio: float64;
    maxIter: int; tol: float64): tuple[coef: seq[float64], intercept: float64] =
  requireXY(x, y, "coordinateDescent")
  let n = x.len
  let p = x[0].len
  var coef = newSeq[float64](p)
  var intercept = meanValue(y)
  let l1 = alpha * l1Ratio
  let l2 = alpha * (1.0 - l1Ratio)

  for _ in 0 ..< maxIter:
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
        coef[j] = softThreshold(residualDot / float64(n), l1) /
          (norm / float64(n) + l2)
    var residualMean = 0.0
    for i in 0 ..< n:
      residualMean += y[i] - dotVec(x[i], coef)
    intercept = residualMean / float64(n)
    var delta = 0.0
    for j in 0 ..< p:
      delta = max(delta, abs(coef[j] - old[j]))
    if delta <= tol:
      break
  (coef: coef, intercept: intercept)

proc fit*(model: Lasso; x: Matrix; y: openArray[float64]): Lasso =
  let solved = fitCoordinateDescent(x, y, model.alpha, 1.0,
    model.maxIter, model.tol)
  Lasso(alpha: model.alpha, maxIter: model.maxIter, tol: model.tol,
    coef: solved.coef, intercept: solved.intercept, fitted: true)

proc fit*(model: ElasticNet; x: Matrix; y: openArray[float64]): ElasticNet =
  let solved = fitCoordinateDescent(x, y, model.alpha, model.l1Ratio,
    model.maxIter, model.tol)
  ElasticNet(alpha: model.alpha, l1Ratio: model.l1Ratio,
    maxIter: model.maxIter, tol: model.tol, coef: solved.coef,
    intercept: solved.intercept, fitted: true)

proc fit*(model: LogisticRegression; x: Matrix; y: openArray[float64]):
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
      let pred = sigmoidValue(intercept + dotVec(x[i], coef))
      let err = pred - y[i]
      gradIntercept += err
      for j in 0 ..< p:
        grad[j] += err * x[i][j]
    intercept -= model.lr * gradIntercept / float64(n)
    for j in 0 ..< p:
      coef[j] -= model.lr * (grad[j] / float64(n) + model.l2 * coef[j])
  LogisticRegression(lr: model.lr, maxIter: model.maxIter, l2: model.l2,
    coef: coef, intercept: intercept, fitted: true)

proc fit*(model: LogisticRegression; x: Matrix; y: openArray[int]):
    LogisticRegression =
  var yf: seq[float64]
  for label in y:
    yf.add float64(label)
  model.fit(x, yf)

proc predict*(model: LinearRegression | Ridge | Lasso | ElasticNet;
    x: Matrix): seq[float64] =
  requireFitted(model.fitted, "predict")
  requireMatrix(x, "predict")
  for row in x:
    result.add model.intercept + dotVec(row, model.coef)

proc predictProba*(model: LogisticRegression; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "predictProba")
  requireMatrix(x, "predictProba")
  for row in x:
    result.add sigmoidValue(model.intercept + dotVec(row, model.coef))

proc predict*(model: LogisticRegression; x: Matrix): seq[int] =
  for p in model.predictProba(x):
    result.add if p >= 0.5: 1 else: 0

proc fit*(model: SoftmaxRegression; x: Matrix; y: openArray[int]):
    SoftmaxRegression =
  requireLabels(x, y, "SoftmaxRegression.fit")
  let classes = uniqueLabels(y)
  if classes.len < 2:
    raise newException(MlError,
      "SoftmaxRegression.fit: at least two classes required")
  result = model
  result.classes = classes
  result.coef = newSeq[seq[float64]](classes.len)
  result.intercept = newSeq[float64](classes.len)
  for c, label in classes:
    let fitted = initLogisticRegression(model.lr, model.maxIter, model.l2).
      fit(x, oneVsRestTargets(y, label))
    result.coef[c] = fitted.coef
    result.intercept[c] = fitted.intercept
  result.fitted = true

proc predictProba*(model: SoftmaxRegression; x: Matrix): Matrix =
  requireFitted(model.fitted, "SoftmaxRegression.predictProba")
  requireMatrix(x, "SoftmaxRegression.predictProba")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.classes.len)
    var sum = 0.0
    for c in 0 ..< model.classes.len:
      let p = sigmoidValue(model.intercept[c] + dotVec(row, model.coef[c]))
      result[i][c] = p
      sum += p
    if sum == 0:
      let uniform = 1.0 / float64(model.classes.len)
      for c in 0 ..< model.classes.len:
        result[i][c] = uniform
    else:
      for c in 0 ..< model.classes.len:
        result[i][c] /= sum

proc predict*(model: SoftmaxRegression; x: Matrix): seq[int] =
  let probs = model.predictProba(x)
  for row in probs:
    var best = 0
    for c in 1 ..< row.len:
      if row[c] > row[best]:
        best = c
    result.add model.classes[best]

proc fit*(model: LinearSVM; x: Matrix; y: openArray[int]): LinearSVM =
  requireLabels(x, y, "LinearSVM.fit")
  let classes = uniqueLabels(y)
  result = model
  result.classes = classes
  result.coef = newSeq[seq[float64]](classes.len)
  result.intercept = newSeq[float64](classes.len)
  for c, label in classes:
    var w = newSeq[float64](x[0].len)
    var b = 0.0
    for _ in 0 ..< model.maxIter:
      for i, row in x:
        let target = if y[i] == label: 1.0 else: -1.0
        let margin = target * (dotVec(row, w) + b)
        for j in 0 ..< w.len:
          w[j] *= (1.0 - model.lr)
        if margin < 1.0:
          for j, value in row:
            w[j] += model.lr * model.c * target * value
          b += model.lr * model.c * target
    result.coef[c] = w
    result.intercept[c] = b
  result.fitted = true

proc decisionFunction*(model: LinearSVM; x: Matrix): Matrix =
  requireFitted(model.fitted, "LinearSVM.decisionFunction")
  requireMatrix(x, "LinearSVM.decisionFunction")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](model.classes.len)
    for c in 0 ..< model.classes.len:
      result[i][c] = model.intercept[c] + dotVec(row, model.coef[c])

proc predict*(model: LinearSVM; x: Matrix): seq[int] =
  let scores = model.decisionFunction(x)
  for row in scores:
    var best = 0
    for c in 1 ..< row.len:
      if row[c] > row[best]:
        best = c
    result.add model.classes[best]

proc score*(model: LinearRegression | Ridge | Lasso | ElasticNet; x: Matrix;
    y: openArray[float64]): float64 =
  r2Score(y, model.predict(x))

proc score*(model: LogisticRegression | SoftmaxRegression | LinearSVM;
    x: Matrix; y: openArray[int]): float64 =
  accuracy(y, model.predict(x))

proc score*(model: LogisticRegression; x: Matrix;
    y: openArray[float64]): float64 =
  model.score(x, labelsFromFloat(y))

proc fit*(model: LinearRegression; x, y: Tensor;
    options: FitOptions = initFitOptions()): LinearRegression =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: Ridge; x, y: Tensor;
    options: FitOptions = initFitOptions()): Ridge =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: Lasso; x, y: Tensor;
    options: FitOptions = initFitOptions()): Lasso =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: ElasticNet; x, y: Tensor;
    options: FitOptions = initFitOptions()): ElasticNet =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: LogisticRegression; x, y: Tensor;
    options: FitOptions = initFitOptions()): LogisticRegression =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: SoftmaxRegression; x, y: Tensor;
    options: FitOptions = initFitOptions()): SoftmaxRegression =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: LinearSVM; x, y: Tensor;
    options: FitOptions = initFitOptions()): LinearSVM =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: LinearRegression; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): LinearRegression =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: LinearRegression; df: DataFrame;
    features: openArray[string]; target: string;
    options: FitOptions = initFitOptions()): LinearRegression =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: Ridge; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): Ridge =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: Ridge; df: DataFrame; features: openArray[string];
    target: string; options: FitOptions = initFitOptions()): Ridge =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: Lasso; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): Lasso =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: Lasso; df: DataFrame; features: openArray[string];
    target: string; options: FitOptions = initFitOptions()): Lasso =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: ElasticNet; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): ElasticNet =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: ElasticNet; df: DataFrame; features: openArray[string];
    target: string; options: FitOptions = initFitOptions()): ElasticNet =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: LogisticRegression; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): LogisticRegression =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: LogisticRegression; df: DataFrame;
    features: openArray[string]; target: string;
    options: FitOptions = initFitOptions()): LogisticRegression =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: SoftmaxRegression; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): SoftmaxRegression =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, labelsFromFloat(data.y))

proc fit*(model: SoftmaxRegression; df: DataFrame;
    features: openArray[string]; target: string;
    options: FitOptions = initFitOptions()): SoftmaxRegression =
  model.fit(df, featureSpec(features, target), options)

proc fit*(model: LinearSVM; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): LinearSVM =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, labelsFromFloat(data.y))

proc fit*(model: LinearSVM; df: DataFrame; features: openArray[string];
    target: string; options: FitOptions = initFitOptions()): LinearSVM =
  model.fit(df, featureSpec(features, target), options)

proc fit*[B, E](model: E; ds: Dataset[B];
    extractor: proc(batch: B): tuple[x: Matrix, y: seq[float64]];
    options: FitOptions = initFitOptions()): E =
  discard options
  let data = xyFromDataset(ds, extractor)
  model.fit(data.x, data.y)

proc partialFit*(model: LinearRegression | Ridge | Lasso | ElasticNet |
    LogisticRegression; x: Matrix; y: openArray[float64];
    options: FitOptions = initFitOptions()): auto =
  discard options
  model.fit(x, y)
