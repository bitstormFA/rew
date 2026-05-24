## Shared public types and host/Tensor adapters for rew ML estimators.

import std/[algorithm, math]
import ../tensor
import ../dtype
import ../eager
import ../rng
import ../train/runtime
import ../data/dataset
import ../dataframe

type
  MlError* = object of CatchableError
    ## Raised by high-level ML estimators, metrics, and adapters.

  NotFittedError* = object of MlError
    ## Raised when prediction or transformation is requested before fitting.

  Matrix* = seq[seq[float64]]
    ## Host matrix used by explicit adapters and tests.

  FitOptions* = object
    ## Execution policy for ML estimator entry points.
    runtime*: Runtime
    key*: Key
    compile*: bool
    batchSize*: int

  FeatureSpec* = object
    ## Explicit DataFrame column extraction for estimators.
    features*: seq[string]
    target*: string
    weight*: string
    hasWeight*: bool

  SemiLabels* = object
    ## Semi-supervised labels with a sentinel for unlabeled samples.
    y*: seq[int]
    unlabeled*: int

const
  InlierLabel* = 1
    ## Conventional inlier label returned by outlier detectors.
  OutlierLabel* = -1
    ## Conventional outlier label returned by outlier detectors.

func initFitOptions*(runtime: Runtime = Runtime(); key: Key = initKey(0);
    compile = true; batchSize = 0): FitOptions =
  ## Creates explicit fit options. A default Runtime value avoids implicit
  ## device/plugin initialization for host-only estimators.
  FitOptions(runtime: runtime, key: key, compile: compile,
    batchSize: batchSize)

func featureSpec*(features: openArray[string]; target = "";
    weight = ""): FeatureSpec =
  ## Creates an explicit DataFrame feature/target mapping.
  if features.len == 0:
    raise newException(MlError, "featureSpec: features must not be empty")
  for name in features:
    if name.len == 0:
      raise newException(MlError,
        "featureSpec: feature names must not be empty")
  FeatureSpec(features: @features, target: target, weight: weight,
    hasWeight: weight.len > 0)

func initSemiLabels*(y: openArray[int]; unlabeled = -1): SemiLabels =
  ## Creates semi-supervised labels.
  if y.len == 0:
    raise newException(MlError, "SemiLabels: labels must not be empty")
  SemiLabels(y: @y, unlabeled: unlabeled)

proc requireMatrix*(x: Matrix; opName: string) =
  ## Validates a rectangular non-empty matrix.
  if x.len == 0:
    raise newException(MlError, opName & ": matrix must not be empty")
  let width = x[0].len
  if width == 0:
    raise newException(MlError, opName & ": matrix must have columns")
  for i, row in x:
    if row.len != width:
      raise newException(MlError,
        opName & ": row " & $i & " width differs from row 0")

proc requireXY*(x: Matrix; y: openArray[float64]; opName: string) =
  ## Validates supervised matrix/vector inputs.
  requireMatrix(x, opName)
  if y.len != x.len:
    raise newException(MlError,
      opName & ": target length " & $y.len & " does not match rows " &
        $x.len)

proc requireLabels*(x: Matrix; y: openArray[int]; opName: string) =
  ## Validates supervised matrix/int-label inputs.
  requireMatrix(x, opName)
  if y.len != x.len:
    raise newException(MlError,
      opName & ": label length " & $y.len & " does not match rows " &
        $x.len)

proc requireFitted*(fitted: bool; opName: string) =
  ## Raises `NotFittedError` when an estimator has not been fitted.
  if not fitted:
    raise newException(NotFittedError, opName & ": estimator is not fitted")

func meanValue*(values: openArray[float64]): float64 =
  if values.len == 0:
    raise newException(MlError, "meanValue: values must not be empty")
  for v in values:
    result += v
  result / float64(values.len)

func varianceValue*(values: openArray[float64]; mean: float64): float64 =
  if values.len == 0:
    raise newException(MlError, "varianceValue: values must not be empty")
  for v in values:
    let d = v - mean
    result += d * d
  result / float64(values.len)

func dotVec*(a, b: openArray[float64]): float64 =
  if a.len != b.len:
    raise newException(MlError, "dotVec: vector length mismatch")
  for i in 0 ..< a.len:
    result += a[i] * b[i]

func squaredDistance*(a, b: openArray[float64]): float64 =
  if a.len != b.len:
    raise newException(MlError, "squaredDistance: vector length mismatch")
  for i in 0 ..< a.len:
    let d = a[i] - b[i]
    result += d * d

func euclideanDistance*(a, b: openArray[float64]): float64 =
  sqrt(squaredDistance(a, b))

proc transpose*(x: Matrix): Matrix =
  requireMatrix(x, "transpose")
  result = newSeq[seq[float64]](x[0].len)
  for j in 0 ..< x[0].len:
    result[j] = newSeq[float64](x.len)
    for i in 0 ..< x.len:
      result[j][i] = x[i][j]

proc matmul*(a, b: Matrix): Matrix =
  requireMatrix(a, "matmul")
  requireMatrix(b, "matmul")
  if a[0].len != b.len:
    raise newException(MlError, "matmul: inner dimension mismatch")
  result = newSeq[seq[float64]](a.len)
  for i in 0 ..< a.len:
    result[i] = newSeq[float64](b[0].len)
    for j in 0 ..< b[0].len:
      for k in 0 ..< b.len:
        result[i][j] += a[i][k] * b[k][j]

proc solveLinear*(a: Matrix; b: openArray[float64]): seq[float64] =
  ## Solves a dense square linear system with Gaussian elimination.
  requireMatrix(a, "solveLinear")
  let n = a.len
  if a[0].len != n or b.len != n:
    raise newException(MlError, "solveLinear: expected square system")
  var aug = newSeq[seq[float64]](n)
  for i in 0 ..< n:
    aug[i] = a[i] & @[b[i]]

  for col in 0 ..< n:
    var pivot = col
    for r in col + 1 ..< n:
      if abs(aug[r][col]) > abs(aug[pivot][col]):
        pivot = r
    if abs(aug[pivot][col]) < 1e-12:
      raise newException(MlError, "solveLinear: singular matrix")
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

proc designMatrix*(x: Matrix): Matrix =
  ## Adds an intercept column to `x`.
  requireMatrix(x, "designMatrix")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = @[1.0] & row

proc fitLeastSquares*(x: Matrix; y: openArray[float64];
    l2: float64): tuple[coef: seq[float64], intercept: float64] =
  ## Fits a dense linear model with optional L2 penalty on coefficients.
  requireXY(x, y, "fitLeastSquares")
  let d = designMatrix(x)
  let xt = transpose(d)
  var xtx = matmul(xt, d)
  for i in 1 ..< xtx.len:
    xtx[i][i] += l2
  var xty = newSeq[float64](xt.len)
  for i in 0 ..< xt.len:
    xty[i] = dotVec(xt[i], y)
  let beta = solveLinear(xtx, xty)
  result.intercept = beta[0]
  result.coef = beta[1 .. ^1]

func sigmoidValue*(x: float64): float64 =
  if x >= 0:
    1.0 / (1.0 + exp(-x))
  else:
    let z = exp(x)
    z / (1.0 + z)

func labelsFromFloat*(y: openArray[float64]; threshold = 0.5): seq[int] =
  for value in y:
    result.add if value >= threshold: 1 else: 0

proc uniqueLabels*(y: openArray[int]): seq[int] =
  result = @y
  result.sort()
  var write = 0
  for value in result:
    if write == 0 or value != result[write - 1]:
      result[write] = value
      inc write
  result.setLen(write)

func majorityLabel*(labels: openArray[int]): int =
  if labels.len == 0:
    raise newException(MlError, "majorityLabel: labels must not be empty")
  var best = labels[0]
  var bestCount = -1
  for label in labels:
    var count = 0
    for other in labels:
      if other == label:
        inc count
    if count > bestCount or (count == bestCount and label < best):
      best = label
      bestCount = count
  best

func asFloat*(value: DataValue; column: string): float64 =
  ## Converts a materialized DataFrame scalar to float64.
  case value.kind
  of dfvInt:
    float64(value.intVal)
  of dfvFloat:
    value.floatVal
  else:
    raise newException(MlError,
      "expected numeric DataFrame value in column: " & column)

proc matrixAndTarget*(df: DataFrame; features: openArray[string];
    target: string): tuple[x: Matrix, y: seq[float64]] =
  ## Collects DataFrame feature and target columns explicitly.
  let rows = df.collect()
  if features.len == 0:
    raise newException(MlError, "features must not be empty")
  if target.len == 0:
    raise newException(MlError, "target must not be empty")
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

proc matrixOnly*(df: DataFrame; columns: openArray[string]): Matrix =
  ## Collects numeric DataFrame columns into a host matrix.
  let rows = df.collect()
  if columns.len == 0:
    raise newException(MlError, "columns must not be empty")
  result = newSeq[seq[float64]](rows.rowCount)
  for i in 0 ..< rows.rowCount:
    result[i] = newSeq[float64](columns.len)
  for j, name in columns:
    let col = rows.columns[rows.requireColumn(name)]
    for i in 0 ..< rows.rowCount:
      result[i][j] = asFloat(col.values[i], name)

proc tensorToFloat64Seq*(t: Tensor): seq[float64] =
  ## Transfers a numeric eager tensor to host float64 values.
  if t.dtype == dtFloat32:
    for v in t.toHost(float32):
      result.add float64(v)
  elif t.dtype == dtFloat64:
    result = t.toHost(float64)
  elif t.dtype == dtInt64:
    for v in t.toHost(int64):
      result.add float64(v)
  elif t.dtype == dtInt32:
    for v in t.toHost(int32):
      result.add float64(v)
  else:
    raise newException(MlError,
      "unsupported tensor dtype for ML adapter: " & t.dtype.name)

proc matrixFromTensor*(x: Tensor): Matrix =
  ## Transfers a rank-2 Tensor to a host matrix.
  if x.shape.len != 2:
    raise newException(MlError,
      "matrixFromTensor: expected rank-2 tensor, got " & $x.shape)
  let data = tensorToFloat64Seq(x)
  result = newSeq[seq[float64]](x.shape[0])
  for i in 0 ..< x.shape[0]:
    result[i] = newSeq[float64](x.shape[1])
    for j in 0 ..< x.shape[1]:
      result[i][j] = data[i * x.shape[1] + j]

proc vectorFromTensor*(y: Tensor): seq[float64] =
  ## Transfers a rank-1 Tensor to a host vector.
  if y.shape.len != 1:
    raise newException(MlError,
      "vectorFromTensor: expected rank-1 tensor, got " & $y.shape)
  tensorToFloat64Seq(y)

proc intVectorFromTensor*(y: Tensor): seq[int] =
  ## Transfers a rank-1 Tensor to host integer labels.
  for value in vectorFromTensor(y):
    result.add int(round(value))

proc matrixFromDataset*[B](ds: Dataset[B];
    extractor: proc(batch: B): Matrix): Matrix =
  ## Collects Dataset batches into one host matrix through an explicit adapter.
  for batch in ds:
    result.add extractor(batch)

proc xyFromDataset*[B](ds: Dataset[B];
    extractor: proc(batch: B): tuple[x: Matrix, y: seq[float64]]):
    tuple[x: Matrix, y: seq[float64]] =
  ## Collects Dataset batches into host supervised arrays.
  for batch in ds:
    let part = extractor(batch)
    result.x.add part.x
    result.y.add part.y

proc columnMeans*(x: Matrix): seq[float64] =
  requireMatrix(x, "columnMeans")
  result = newSeq[float64](x[0].len)
  for row in x:
    for j, value in row:
      result[j] += value
  for j in 0 ..< result.len:
    result[j] /= float64(x.len)

proc columnStd*(x: Matrix; means: openArray[float64]): seq[float64] =
  requireMatrix(x, "columnStd")
  result = newSeq[float64](x[0].len)
  for row in x:
    for j, value in row:
      let d = value - means[j]
      result[j] += d * d
  for j in 0 ..< result.len:
    result[j] = sqrt(result[j] / float64(x.len))
    if result[j] == 0:
      result[j] = 1.0

proc selectRows*(x: Matrix; indices: openArray[int]): Matrix =
  requireMatrix(x, "selectRows")
  for idx in indices:
    if idx < 0 or idx >= x.len:
      raise newException(MlError, "selectRows: index out of bounds")
    result.add x[idx]

proc selectValues*[T](y: openArray[T]; indices: openArray[int]): seq[T] =
  for idx in indices:
    if idx < 0 or idx >= y.len:
      raise newException(MlError, "selectValues: index out of bounds")
    result.add y[idx]
