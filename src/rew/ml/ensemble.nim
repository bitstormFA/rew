## Ensemble estimators built from value-type tree learners.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./metrics
import ./tree

type
  RandomForestClassifier* = object
    nEstimators*: int
    maxDepth*: int
    trees*: seq[DecisionTreeClassifier]
    fitted*: bool

  RandomForestRegressor* = object
    nEstimators*: int
    maxDepth*: int
    trees*: seq[DecisionTreeRegressor]
    fitted*: bool

  ExtraTreesClassifier* = object
    nEstimators*: int
    maxDepth*: int
    trees*: seq[DecisionTreeClassifier]
    fitted*: bool

  GradientBoostingRegressor* = object
    nEstimators*: int
    learningRate*: float64
    maxDepth*: int
    initValue*: float64
    trees*: seq[DecisionTreeRegressor]
    fitted*: bool

  AdaBoostClassifier* = object
    nEstimators*: int
    learningRate*: float64
    stumps*: seq[DecisionTreeClassifier]
    weights*: seq[float64]
    classes*: seq[int]
    fitted*: bool

  RandomForest* = RandomForestClassifier
  ExtraTrees* = ExtraTreesClassifier
  GradientBoosting* = GradientBoostingRegressor
  AdaBoost* = AdaBoostClassifier

func initRandomForestClassifier*(nEstimators = 10; maxDepth = 4):
    RandomForestClassifier =
  if nEstimators <= 0 or maxDepth <= 0:
    raise newException(MlError,
      "RandomForestClassifier nEstimators/maxDepth must be positive")
  RandomForestClassifier(nEstimators: nEstimators, maxDepth: maxDepth)

func initRandomForestRegressor*(nEstimators = 10; maxDepth = 4):
    RandomForestRegressor =
  if nEstimators <= 0 or maxDepth <= 0:
    raise newException(MlError,
      "RandomForestRegressor nEstimators/maxDepth must be positive")
  RandomForestRegressor(nEstimators: nEstimators, maxDepth: maxDepth)

func initRandomForest*(nEstimators = 10; maxDepth = 4): RandomForest =
  initRandomForestClassifier(nEstimators, maxDepth)

func initExtraTreesClassifier*(nEstimators = 10; maxDepth = 4):
    ExtraTreesClassifier =
  if nEstimators <= 0 or maxDepth <= 0:
    raise newException(MlError,
      "ExtraTreesClassifier nEstimators/maxDepth must be positive")
  ExtraTreesClassifier(nEstimators: nEstimators, maxDepth: maxDepth)

func initExtraTrees*(nEstimators = 10; maxDepth = 4): ExtraTrees =
  initExtraTreesClassifier(nEstimators, maxDepth)

func initGradientBoostingRegressor*(nEstimators = 20;
    learningRate = 0.1; maxDepth = 2): GradientBoostingRegressor =
  if nEstimators <= 0 or learningRate <= 0 or maxDepth <= 0:
    raise newException(MlError,
      "GradientBoostingRegressor parameters must be positive")
  GradientBoostingRegressor(nEstimators: nEstimators,
    learningRate: learningRate, maxDepth: maxDepth)

func initGradientBoosting*(nEstimators = 20; learningRate = 0.1;
    maxDepth = 2): GradientBoosting =
  initGradientBoostingRegressor(nEstimators, learningRate, maxDepth)

func initAdaBoostClassifier*(nEstimators = 20; learningRate = 1.0):
    AdaBoostClassifier =
  if nEstimators <= 0 or learningRate <= 0:
    raise newException(MlError,
      "AdaBoostClassifier nEstimators/learningRate must be positive")
  AdaBoostClassifier(nEstimators: nEstimators, learningRate: learningRate)

func initAdaBoost*(nEstimators = 20; learningRate = 1.0): AdaBoost =
  initAdaBoostClassifier(nEstimators, learningRate)

proc deterministicRows(x: Matrix; y: openArray[int]; offset: int):
    tuple[x: Matrix, y: seq[int]] =
  for i in 0 ..< x.len:
    if ((i + offset) mod 3) != 0 or x.len < 4:
      result.x.add x[i]
      result.y.add y[i]

proc deterministicRowsReg(x: Matrix; y: openArray[float64]; offset: int):
    tuple[x: Matrix, y: seq[float64]] =
  for i in 0 ..< x.len:
    if ((i + offset) mod 3) != 0 or x.len < 4:
      result.x.add x[i]
      result.y.add y[i]

proc fit*(model: RandomForestClassifier; x: Matrix; y: openArray[int]):
    RandomForestClassifier =
  requireLabels(x, y, "RandomForestClassifier.fit")
  result = model
  for t in 0 ..< model.nEstimators:
    let sample = deterministicRows(x, y, t)
    result.trees.add initDecisionTreeClassifier(model.maxDepth).fit(
      sample.x, sample.y)
  result.fitted = true

proc fit*(model: RandomForestClassifier; x: Matrix; y: openArray[float64]):
    RandomForestClassifier =
  model.fit(x, labelsFromFloat(y))

proc fit*(model: RandomForestRegressor; x: Matrix; y: openArray[float64]):
    RandomForestRegressor =
  requireXY(x, y, "RandomForestRegressor.fit")
  result = model
  for t in 0 ..< model.nEstimators:
    let sample = deterministicRowsReg(x, y, t)
    result.trees.add initDecisionTreeRegressor(model.maxDepth).fit(
      sample.x, sample.y)
  result.fitted = true

proc predict*(model: RandomForestClassifier; x: Matrix): seq[int] =
  requireFitted(model.fitted, "RandomForestClassifier.predict")
  var votes = newSeq[seq[int]](x.len)
  for tree in model.trees:
    let pred = tree.predict(x)
    for i, label in pred:
      votes[i].add label
  for rowVotes in votes:
    result.add majorityLabel(rowVotes)

proc predict*(model: RandomForestRegressor; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "RandomForestRegressor.predict")
  result = newSeq[float64](x.len)
  for tree in model.trees:
    let pred = tree.predict(x)
    for i, value in pred:
      result[i] += value
  for i in 0 ..< result.len:
    result[i] /= float64(model.trees.len)

proc fit*(model: ExtraTreesClassifier; x: Matrix; y: openArray[int]):
    ExtraTreesClassifier =
  requireLabels(x, y, "ExtraTreesClassifier.fit")
  result = model
  for _ in 0 ..< model.nEstimators:
    result.trees.add initDecisionTreeClassifier(model.maxDepth).fit(x, y)
  result.fitted = true

proc fit*(model: ExtraTreesClassifier; x: Matrix; y: openArray[float64]):
    ExtraTreesClassifier =
  model.fit(x, labelsFromFloat(y))

proc predict*(model: ExtraTreesClassifier; x: Matrix): seq[int] =
  requireFitted(model.fitted, "ExtraTreesClassifier.predict")
  var votes = newSeq[seq[int]](x.len)
  for tree in model.trees:
    let pred = tree.predict(x)
    for i, label in pred:
      votes[i].add label
  for rowVotes in votes:
    result.add majorityLabel(rowVotes)

proc fit*(model: GradientBoostingRegressor; x: Matrix;
    y: openArray[float64]): GradientBoostingRegressor =
  requireXY(x, y, "GradientBoostingRegressor.fit")
  result = model
  result.initValue = meanValue(y)
  var current = newSeq[float64](y.len)
  for i in 0 ..< current.len:
    current[i] = result.initValue
  for _ in 0 ..< model.nEstimators:
    var residual = newSeq[float64](y.len)
    for i in 0 ..< y.len:
      residual[i] = y[i] - current[i]
    let tree = initDecisionTreeRegressor(model.maxDepth).fit(x, residual)
    let update = tree.predict(x)
    for i, value in update:
      current[i] += model.learningRate * value
    result.trees.add tree
  result.fitted = true

proc predict*(model: GradientBoostingRegressor; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "GradientBoostingRegressor.predict")
  result = newSeq[float64](x.len)
  for i in 0 ..< result.len:
    result[i] = model.initValue
  for tree in model.trees:
    let update = tree.predict(x)
    for i, value in update:
      result[i] += model.learningRate * value

proc fit*(model: AdaBoostClassifier; x: Matrix; y: openArray[int]):
    AdaBoostClassifier =
  requireLabels(x, y, "AdaBoostClassifier.fit")
  result = model
  result.classes = uniqueLabels(y)
  var sampleWeight = newSeq[float64](x.len)
  for i in 0 ..< sampleWeight.len:
    sampleWeight[i] = 1.0 / float64(x.len)
  for _ in 0 ..< model.nEstimators:
    let stump = initDecisionTreeClassifier(maxDepth = 1).fit(x, y)
    let pred = stump.predict(x)
    var err = 0.0
    for i in 0 ..< y.len:
      if pred[i] != y[i]:
        err += sampleWeight[i]
    if err <= 0 or err >= 0.5:
      break
    let alpha = model.learningRate * 0.5 * ln((1.0 - err) / err)
    for i in 0 ..< y.len:
      if pred[i] == y[i]:
        sampleWeight[i] *= exp(-alpha)
      else:
        sampleWeight[i] *= exp(alpha)
    var total = 0.0
    for w in sampleWeight: total += w
    for i in 0 ..< sampleWeight.len: sampleWeight[i] /= total
    result.stumps.add stump
    result.weights.add alpha
  if result.stumps.len == 0:
    result.stumps.add initDecisionTreeClassifier(maxDepth = 1).fit(x, y)
    result.weights.add 1.0
  result.fitted = true

proc fit*(model: AdaBoostClassifier; x: Matrix; y: openArray[float64]):
    AdaBoostClassifier =
  model.fit(x, labelsFromFloat(y))

proc predict*(model: AdaBoostClassifier; x: Matrix): seq[int] =
  requireFitted(model.fitted, "AdaBoostClassifier.predict")
  let classes = model.classes
  var scores = newSeq[seq[float64]](x.len)
  for i in 0 ..< x.len: scores[i] = newSeq[float64](classes.len)
  for t, stump in model.stumps:
    let pred = stump.predict(x)
    for i, label in pred:
      for c, cls in classes:
        if label == cls:
          scores[i][c] += model.weights[t]
  for row in scores:
    var best = 0
    for c in 1 ..< row.len:
      if row[c] > row[best]:
        best = c
    result.add classes[best]

proc score*(model: RandomForestClassifier | ExtraTreesClassifier |
    AdaBoostClassifier; x: Matrix; y: openArray[int]): float64 =
  accuracy(y, model.predict(x))

proc score*(model: RandomForestRegressor | GradientBoostingRegressor;
    x: Matrix; y: openArray[float64]): float64 =
  r2Score(y, model.predict(x))

proc fit*(model: RandomForestClassifier; x, y: Tensor;
    options: FitOptions = initFitOptions()): RandomForestClassifier =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: ExtraTreesClassifier; x, y: Tensor;
    options: FitOptions = initFitOptions()): ExtraTreesClassifier =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: AdaBoostClassifier; x, y: Tensor;
    options: FitOptions = initFitOptions()): AdaBoostClassifier =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: GradientBoostingRegressor; x, y: Tensor;
    options: FitOptions = initFitOptions()): GradientBoostingRegressor =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: RandomForestClassifier | ExtraTreesClassifier |
    AdaBoostClassifier | GradientBoostingRegressor; df: DataFrame;
    spec: FeatureSpec; options: FitOptions = initFitOptions()): auto =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)
