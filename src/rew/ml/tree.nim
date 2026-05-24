## Lightweight decision-tree estimators.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./metrics

type
  DecisionNode* = object
    feature*: int
    threshold*: float64
    left*: int
    right*: int
    predictionClass*: int
    predictionValue*: float64
    isLeaf*: bool

  DecisionTreeClassifier* = object
    maxDepth*: int
    minSamplesSplit*: int
    nodes*: seq[DecisionNode]
    fitted*: bool

  DecisionTreeRegressor* = object
    maxDepth*: int
    minSamplesSplit*: int
    nodes*: seq[DecisionNode]
    fitted*: bool

  DecisionTree* = DecisionTreeClassifier

func initDecisionTreeClassifier*(maxDepth = 3;
    minSamplesSplit = 2): DecisionTreeClassifier =
  if maxDepth <= 0 or minSamplesSplit <= 0:
    raise newException(MlError,
      "DecisionTreeClassifier maxDepth/minSamplesSplit must be positive")
  DecisionTreeClassifier(maxDepth: maxDepth,
    minSamplesSplit: minSamplesSplit)

func initDecisionTreeRegressor*(maxDepth = 3;
    minSamplesSplit = 2): DecisionTreeRegressor =
  if maxDepth <= 0 or minSamplesSplit <= 0:
    raise newException(MlError,
      "DecisionTreeRegressor maxDepth/minSamplesSplit must be positive")
  DecisionTreeRegressor(maxDepth: maxDepth,
    minSamplesSplit: minSamplesSplit)

func initDecisionTree*(maxDepth = 3; minSamplesSplit = 2): DecisionTree =
  initDecisionTreeClassifier(maxDepth, minSamplesSplit)

func gini(labels: openArray[int]): float64 =
  if labels.len == 0:
    return 0
  result = 1.0
  for cls in uniqueLabels(labels):
    var count = 0
    for label in labels:
      if label == cls:
        inc count
    let p = float64(count) / float64(labels.len)
    result -= p * p

func mse(values: openArray[float64]): float64 =
  if values.len == 0:
    return 0
  let avg = meanValue(values)
  for value in values:
    let d = value - avg
    result += d * d
  result / float64(values.len)

proc splitRows(x: Matrix; indices: openArray[int]; feature: int;
    threshold: float64): tuple[left, right: seq[int]] =
  for idx in indices:
    if x[idx][feature] <= threshold:
      result.left.add idx
    else:
      result.right.add idx

proc featureMean(x: Matrix; indices: openArray[int]; feature: int): float64 =
  for idx in indices:
    result += x[idx][feature]
  result / float64(indices.len)

proc buildClassifier(tree: var DecisionTreeClassifier; x: Matrix;
    y: openArray[int]; indices: seq[int]; depth: int): int =
  var labels: seq[int]
  for idx in indices:
    labels.add y[idx]
  let prediction = majorityLabel(labels)
  let nodeIndex = tree.nodes.len
  tree.nodes.add DecisionNode(isLeaf: true, predictionClass: prediction,
    left: -1, right: -1, feature: -1)
  if depth >= tree.maxDepth or indices.len < tree.minSamplesSplit or
      uniqueLabels(labels).len <= 1:
    return nodeIndex

  var bestFeature = -1
  var bestThreshold = 0.0
  var bestImpurity = Inf
  for feature in 0 ..< x[0].len:
    let threshold = featureMean(x, indices, feature)
    let parts = splitRows(x, indices, feature, threshold)
    if parts.left.len == 0 or parts.right.len == 0:
      continue
    var leftLabels, rightLabels: seq[int]
    for idx in parts.left: leftLabels.add y[idx]
    for idx in parts.right: rightLabels.add y[idx]
    let impurity = (float64(leftLabels.len) * gini(leftLabels) +
      float64(rightLabels.len) * gini(rightLabels)) / float64(indices.len)
    if impurity < bestImpurity:
      bestImpurity = impurity
      bestFeature = feature
      bestThreshold = threshold
  if bestFeature < 0:
    return nodeIndex

  let parts = splitRows(x, indices, bestFeature, bestThreshold)
  let left = buildClassifier(tree, x, y, parts.left, depth + 1)
  let right = buildClassifier(tree, x, y, parts.right, depth + 1)
  tree.nodes[nodeIndex].isLeaf = false
  tree.nodes[nodeIndex].feature = bestFeature
  tree.nodes[nodeIndex].threshold = bestThreshold
  tree.nodes[nodeIndex].left = left
  tree.nodes[nodeIndex].right = right
  nodeIndex

proc buildRegressor(tree: var DecisionTreeRegressor; x: Matrix;
    y: openArray[float64]; indices: seq[int]; depth: int): int =
  var values: seq[float64]
  for idx in indices:
    values.add y[idx]
  let prediction = meanValue(values)
  let nodeIndex = tree.nodes.len
  tree.nodes.add DecisionNode(isLeaf: true, predictionValue: prediction,
    left: -1, right: -1, feature: -1)
  if depth >= tree.maxDepth or indices.len < tree.minSamplesSplit:
    return nodeIndex

  var bestFeature = -1
  var bestThreshold = 0.0
  var bestLoss = Inf
  for feature in 0 ..< x[0].len:
    let threshold = featureMean(x, indices, feature)
    let parts = splitRows(x, indices, feature, threshold)
    if parts.left.len == 0 or parts.right.len == 0:
      continue
    var leftValues, rightValues: seq[float64]
    for idx in parts.left: leftValues.add y[idx]
    for idx in parts.right: rightValues.add y[idx]
    let loss = (float64(leftValues.len) * mse(leftValues) +
      float64(rightValues.len) * mse(rightValues)) / float64(indices.len)
    if loss < bestLoss:
      bestLoss = loss
      bestFeature = feature
      bestThreshold = threshold
  if bestFeature < 0:
    return nodeIndex

  let parts = splitRows(x, indices, bestFeature, bestThreshold)
  let left = buildRegressor(tree, x, y, parts.left, depth + 1)
  let right = buildRegressor(tree, x, y, parts.right, depth + 1)
  tree.nodes[nodeIndex].isLeaf = false
  tree.nodes[nodeIndex].feature = bestFeature
  tree.nodes[nodeIndex].threshold = bestThreshold
  tree.nodes[nodeIndex].left = left
  tree.nodes[nodeIndex].right = right
  nodeIndex

proc fit*(model: DecisionTreeClassifier; x: Matrix; y: openArray[int]):
    DecisionTreeClassifier =
  requireLabels(x, y, "DecisionTreeClassifier.fit")
  result = model
  result.nodes = @[]
  var indices: seq[int]
  for i in 0 ..< x.len: indices.add i
  discard buildClassifier(result, x, y, indices, 0)
  result.fitted = true

proc fit*(model: DecisionTreeClassifier; x: Matrix; y: openArray[float64]):
    DecisionTreeClassifier =
  model.fit(x, labelsFromFloat(y))

proc fit*(model: DecisionTreeRegressor; x: Matrix; y: openArray[float64]):
    DecisionTreeRegressor =
  requireXY(x, y, "DecisionTreeRegressor.fit")
  result = model
  result.nodes = @[]
  var indices: seq[int]
  for i in 0 ..< x.len: indices.add i
  discard buildRegressor(result, x, y, indices, 0)
  result.fitted = true

proc traverse(nodes: openArray[DecisionNode]; row: openArray[float64]): int =
  result = 0
  while not nodes[result].isLeaf:
    let node = nodes[result]
    if row[node.feature] <= node.threshold:
      result = node.left
    else:
      result = node.right

proc predict*(model: DecisionTreeClassifier; x: Matrix): seq[int] =
  requireFitted(model.fitted, "DecisionTreeClassifier.predict")
  requireMatrix(x, "DecisionTreeClassifier.predict")
  for row in x:
    result.add model.nodes[traverse(model.nodes, row)].predictionClass

proc predict*(model: DecisionTreeRegressor; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "DecisionTreeRegressor.predict")
  requireMatrix(x, "DecisionTreeRegressor.predict")
  for row in x:
    result.add model.nodes[traverse(model.nodes, row)].predictionValue

proc score*(model: DecisionTreeClassifier; x: Matrix; y: openArray[int]):
    float64 =
  accuracy(y, model.predict(x))

proc score*(model: DecisionTreeRegressor; x: Matrix; y: openArray[float64]):
    float64 =
  r2Score(y, model.predict(x))

proc fit*(model: DecisionTreeClassifier; x, y: Tensor;
    options: FitOptions = initFitOptions()): DecisionTreeClassifier =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: DecisionTreeRegressor; x, y: Tensor;
    options: FitOptions = initFitOptions()): DecisionTreeRegressor =
  discard options
  model.fit(matrixFromTensor(x), vectorFromTensor(y))

proc fit*(model: DecisionTreeClassifier; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): DecisionTreeClassifier =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: DecisionTreeRegressor; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): DecisionTreeRegressor =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)
