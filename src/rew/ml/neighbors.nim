## Nearest-neighbor estimators.

import ../tensor
import ../dataframe
import ./core
import ./tensor_utils
import ./metrics

type
  KNeighborsClassifier* = object
    k*: int
    x*: Matrix
    y*: seq[int]
    fitted*: bool

  KNeighborsRegressor* = object
    k*: int
    x*: Matrix
    y*: seq[float64]
    fitted*: bool

func initKNeighborsClassifier*(k = 5): KNeighborsClassifier =
  if k <= 0:
    raise newException(MlError, "KNeighborsClassifier k must be positive")
  KNeighborsClassifier(k: k)

func initKNeighborsRegressor*(k = 5): KNeighborsRegressor =
  if k <= 0:
    raise newException(MlError, "KNeighborsRegressor k must be positive")
  KNeighborsRegressor(k: k)

proc fit*(model: KNeighborsClassifier; x: Matrix; y: openArray[int]):
    KNeighborsClassifier =
  requireLabels(x, y, "KNeighborsClassifier.fit")
  KNeighborsClassifier(k: model.k, x: x, y: @y, fitted: true)

proc fit*(model: KNeighborsClassifier; x: Matrix; y: openArray[float64]):
    KNeighborsClassifier =
  model.fit(x, labelsFromFloat(y))

proc fit*(model: KNeighborsRegressor; x: Matrix; y: openArray[float64]):
    KNeighborsRegressor =
  requireXY(x, y, "KNeighborsRegressor.fit")
  KNeighborsRegressor(k: model.k, x: x, y: @y, fitted: true)

proc predict*(model: KNeighborsClassifier; x: Matrix): seq[int] =
  requireFitted(model.fitted, "KNeighborsClassifier.predict")
  requireMatrix(x, "KNeighborsClassifier.predict")
  let dist = pairwiseDistances(x, model.x)
  for row in dist:
    let idx = nearestIndices(row, model.k)
    var labels: seq[int]
    for i in idx:
      labels.add model.y[i]
    result.add majorityLabel(labels)

proc predictProba*(model: KNeighborsClassifier; x: Matrix): Matrix =
  requireFitted(model.fitted, "KNeighborsClassifier.predictProba")
  let classes = uniqueLabels(model.y)
  let dist = pairwiseDistances(x, model.x)
  result = newSeq[seq[float64]](x.len)
  for i, row in dist:
    result[i] = newSeq[float64](classes.len)
    let idx = nearestIndices(row, model.k)
    for neighbor in idx:
      for c, cls in classes:
        if model.y[neighbor] == cls:
          result[i][c] += 1.0
    for c in 0 ..< classes.len:
      result[i][c] /= float64(idx.len)

proc predict*(model: KNeighborsRegressor; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "KNeighborsRegressor.predict")
  requireMatrix(x, "KNeighborsRegressor.predict")
  let dist = pairwiseDistances(x, model.x)
  for row in dist:
    let idx = nearestIndices(row, model.k)
    var total = 0.0
    for i in idx:
      total += model.y[i]
    result.add total / float64(idx.len)

proc score*(model: KNeighborsClassifier; x: Matrix; y: openArray[int]):
    float64 =
  accuracy(y, model.predict(x))

proc score*(model: KNeighborsRegressor; x: Matrix; y: openArray[float64]):
    float64 =
  r2Score(y, model.predict(x))

proc fit*(model: KNeighborsClassifier; x, y: Tensor;
    options: FitOptions = initFitOptions()): KNeighborsClassifier =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: KNeighborsRegressor; x, y: Tensor;
    options: FitOptions = initFitOptions()): KNeighborsRegressor =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: KNeighborsClassifier; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): KNeighborsClassifier =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: KNeighborsRegressor; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): KNeighborsRegressor =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)
