## Naive Bayes classifiers.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./metrics

type
  GaussianNB* = object
    classes*: seq[int]
    theta*: Matrix
    variance*: Matrix
    classPrior*: seq[float64]
    fitted*: bool

  MultinomialNB* = object
    alpha*: float64
    classes*: seq[int]
    featureLogProb*: Matrix
    classLogPrior*: seq[float64]
    fitted*: bool

  BernoulliNB* = object
    alpha*: float64
    classes*: seq[int]
    featureProb*: Matrix
    classLogPrior*: seq[float64]
    fitted*: bool

func initGaussianNB*(): GaussianNB =
  GaussianNB()

func initMultinomialNB*(alpha = 1.0): MultinomialNB =
  if alpha < 0:
    raise newException(MlError, "MultinomialNB alpha must be non-negative")
  MultinomialNB(alpha: alpha)

func initBernoulliNB*(alpha = 1.0): BernoulliNB =
  if alpha < 0:
    raise newException(MlError, "BernoulliNB alpha must be non-negative")
  BernoulliNB(alpha: alpha)

proc fit*(model: GaussianNB; x: Matrix; y: openArray[int]): GaussianNB =
  requireLabels(x, y, "GaussianNB.fit")
  let classes = uniqueLabels(y)
  result.classes = classes
  result.theta = newSeq[seq[float64]](classes.len)
  result.variance = newSeq[seq[float64]](classes.len)
  result.classPrior = newSeq[float64](classes.len)
  for c, cls in classes:
    var rows: Matrix
    for i, row in x:
      if y[i] == cls:
        rows.add row
    result.theta[c] = columnMeans(rows)
    result.variance[c] = columnStd(rows, result.theta[c])
    for j in 0 ..< result.variance[c].len:
      result.variance[c][j] = result.variance[c][j] * result.variance[c][j] +
        1e-9
    result.classPrior[c] = float64(rows.len) / float64(x.len)
  result.fitted = true

proc fit*(model: GaussianNB; x: Matrix; y: openArray[float64]): GaussianNB =
  model.fit(x, labelsFromFloat(y))

proc gaussianJoint(model: GaussianNB; row: openArray[float64]; c: int): float64 =
  result = ln(model.classPrior[c])
  for j, value in row:
    let varj = model.variance[c][j]
    let diff = value - model.theta[c][j]
    result += -0.5 * ln(2.0 * PI * varj) - diff * diff / (2.0 * varj)

proc predict*(model: GaussianNB; x: Matrix): seq[int] =
  requireFitted(model.fitted, "GaussianNB.predict")
  requireMatrix(x, "GaussianNB.predict")
  for row in x:
    var best = 0
    var bestScore = gaussianJoint(model, row, 0)
    for c in 1 ..< model.classes.len:
      let score = gaussianJoint(model, row, c)
      if score > bestScore:
        best = c
        bestScore = score
    result.add model.classes[best]

proc fit*(model: MultinomialNB; x: Matrix; y: openArray[int]): MultinomialNB =
  requireLabels(x, y, "MultinomialNB.fit")
  let classes = uniqueLabels(y)
  result = model
  result.classes = classes
  result.featureLogProb = newSeq[seq[float64]](classes.len)
  result.classLogPrior = newSeq[float64](classes.len)
  for c, cls in classes:
    var sums = newSeq[float64](x[0].len)
    var count = 0
    for i, row in x:
      if y[i] == cls:
        inc count
        for j, value in row:
          sums[j] += max(0.0, value)
    var total = model.alpha * float64(sums.len)
    for value in sums: total += value
    result.featureLogProb[c] = newSeq[float64](sums.len)
    for j in 0 ..< sums.len:
      result.featureLogProb[c][j] = ln((sums[j] + model.alpha) / total)
    result.classLogPrior[c] = ln(float64(count) / float64(x.len))
  result.fitted = true

proc fit*(model: MultinomialNB; x: Matrix; y: openArray[float64]):
    MultinomialNB =
  model.fit(x, labelsFromFloat(y))

proc multinomialJoint(model: MultinomialNB; row: openArray[float64];
    c: int): float64 =
  result = model.classLogPrior[c]
  for j, value in row:
    result += max(0.0, value) * model.featureLogProb[c][j]

proc predict*(model: MultinomialNB; x: Matrix): seq[int] =
  requireFitted(model.fitted, "MultinomialNB.predict")
  requireMatrix(x, "MultinomialNB.predict")
  for row in x:
    var best = 0
    var bestScore = multinomialJoint(model, row, 0)
    for c in 1 ..< model.classes.len:
      let score = multinomialJoint(model, row, c)
      if score > bestScore:
        best = c
        bestScore = score
    result.add model.classes[best]

proc fit*(model: BernoulliNB; x: Matrix; y: openArray[int]): BernoulliNB =
  requireLabels(x, y, "BernoulliNB.fit")
  let classes = uniqueLabels(y)
  result = model
  result.classes = classes
  result.featureProb = newSeq[seq[float64]](classes.len)
  result.classLogPrior = newSeq[float64](classes.len)
  for c, cls in classes:
    var counts = newSeq[float64](x[0].len)
    var rows = 0
    for i, row in x:
      if y[i] == cls:
        inc rows
        for j, value in row:
          if value > 0:
            counts[j] += 1.0
    result.featureProb[c] = newSeq[float64](counts.len)
    for j in 0 ..< counts.len:
      result.featureProb[c][j] = (counts[j] + model.alpha) /
        (float64(rows) + 2.0 * model.alpha)
    result.classLogPrior[c] = ln(float64(rows) / float64(x.len))
  result.fitted = true

proc fit*(model: BernoulliNB; x: Matrix; y: openArray[float64]): BernoulliNB =
  model.fit(x, labelsFromFloat(y))

proc bernoulliJoint(model: BernoulliNB; row: openArray[float64]; c: int):
    float64 =
  result = model.classLogPrior[c]
  for j, value in row:
    let p = model.featureProb[c][j]
    if value > 0:
      result += ln(p)
    else:
      result += ln(1.0 - p)

proc predict*(model: BernoulliNB; x: Matrix): seq[int] =
  requireFitted(model.fitted, "BernoulliNB.predict")
  requireMatrix(x, "BernoulliNB.predict")
  for row in x:
    var best = 0
    var bestScore = bernoulliJoint(model, row, 0)
    for c in 1 ..< model.classes.len:
      let score = bernoulliJoint(model, row, c)
      if score > bestScore:
        best = c
        bestScore = score
    result.add model.classes[best]

proc score*(model: GaussianNB | MultinomialNB | BernoulliNB; x: Matrix;
    y: openArray[int]): float64 =
  accuracy(y, model.predict(x))

proc fit*(model: GaussianNB; x, y: Tensor;
    options: FitOptions = initFitOptions()): GaussianNB =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: MultinomialNB; x, y: Tensor;
    options: FitOptions = initFitOptions()): MultinomialNB =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: BernoulliNB; x, y: Tensor;
    options: FitOptions = initFitOptions()): BernoulliNB =
  discard options
  model.fit(matrixFromTensor(x), intVectorFromTensor(y))

proc fit*(model: GaussianNB; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): GaussianNB =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: MultinomialNB; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): MultinomialNB =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)

proc fit*(model: BernoulliNB; df: DataFrame; spec: FeatureSpec;
    options: FitOptions = initFitOptions()): BernoulliNB =
  discard options
  let data = matrixAndTarget(df, spec.features, spec.target)
  model.fit(data.x, data.y)
