## Semi-supervised estimators and helpers.

import ../tensor
import ../dataframe
import ./core
import ./tensor_utils
import ./metrics

type
  LabelPropagation* = object
    k*: int
    maxIter*: int
    x*: Matrix
    labels*: seq[int]
    unlabeled*: int
    fitted*: bool

  LabelSpreading* = object
    k*: int
    alpha*: float64
    maxIter*: int
    propagated*: LabelPropagation
    fitted*: bool

  SelfTrainingClassifier*[E] = object
    baseEstimator*: E
    threshold*: float64
    maxIter*: int
    estimator*: E
    fitted*: bool

  CoTrainingClassifier*[E] = object
    baseEstimator*: E
    maxIter*: int
    estimator*: E
    fitted*: bool

  FixMatch*[E] = object
    baseEstimator*: E
    threshold*: float64
    estimator*: E
    fitted*: bool

  MeanTeacher*[E] = object
    student*: E
    teacher*: E
    momentum*: float64
    fitted*: bool

func initLabelPropagation*(k = 5; maxIter = 100): LabelPropagation =
  if k <= 0 or maxIter <= 0:
    raise newException(MlError,
      "LabelPropagation k/maxIter must be positive")
  LabelPropagation(k: k, maxIter: maxIter)

func initLabelSpreading*(k = 5; alpha = 0.2; maxIter = 100): LabelSpreading =
  if k <= 0 or maxIter <= 0 or alpha < 0 or alpha > 1:
    raise newException(MlError,
      "LabelSpreading requires positive k/maxIter and alpha in [0, 1]")
  LabelSpreading(k: k, alpha: alpha, maxIter: maxIter)

func initSelfTrainingClassifier*[E](baseEstimator: E; threshold = 0.8;
    maxIter = 10): SelfTrainingClassifier[E] =
  if threshold <= 0 or threshold > 1 or maxIter <= 0:
    raise newException(MlError,
      "SelfTrainingClassifier threshold must be in (0, 1] and maxIter positive")
  SelfTrainingClassifier[E](baseEstimator: baseEstimator,
    threshold: threshold, maxIter: maxIter)

func initCoTrainingClassifier*[E](baseEstimator: E; maxIter = 10):
    CoTrainingClassifier[E] =
  if maxIter <= 0:
    raise newException(MlError, "CoTrainingClassifier maxIter must be positive")
  CoTrainingClassifier[E](baseEstimator: baseEstimator, maxIter: maxIter)

func initFixMatch*[E](baseEstimator: E; threshold = 0.95): FixMatch[E] =
  if threshold <= 0 or threshold > 1:
    raise newException(MlError, "FixMatch threshold must be in (0, 1]")
  FixMatch[E](baseEstimator: baseEstimator, threshold: threshold)

func initMeanTeacher*[E](student: E; teacher: E; momentum = 0.99):
    MeanTeacher[E] =
  if momentum < 0 or momentum > 1:
    raise newException(MlError, "MeanTeacher momentum must be in [0, 1]")
  MeanTeacher[E](student: student, teacher: teacher, momentum: momentum)

proc semiLabelsFromMask*(y: openArray[int]; trainMask: openArray[bool];
    unlabeled = -1): SemiLabels =
  ## Builds semi-supervised labels from a graph-style train mask.
  if y.len != trainMask.len:
    raise newException(MlError,
      "semiLabelsFromMask: labels and mask lengths differ")
  var values: seq[int]
  for i, label in y:
    values.add if trainMask[i]: label else: unlabeled
  initSemiLabels(values, unlabeled)

proc labeledSubset(x: Matrix; labels: SemiLabels): tuple[x: Matrix, y: seq[int]] =
  requireMatrix(x, "labeledSubset")
  if labels.y.len != x.len:
    raise newException(MlError, "labeledSubset: label length mismatch")
  for i, label in labels.y:
    if label != labels.unlabeled:
      result.x.add x[i]
      result.y.add label
  if result.y.len == 0:
    raise newException(MlError, "semi-supervised fit requires labels")

proc fit*(model: LabelPropagation; x: Matrix; labels: SemiLabels):
    LabelPropagation =
  requireMatrix(x, "LabelPropagation.fit")
  if labels.y.len != x.len:
    raise newException(MlError, "LabelPropagation.fit: label length mismatch")
  var current = labels.y
  for _ in 0 ..< model.maxIter:
    var changed = false
    let dist = pairwiseDistances(x, x)
    for i in 0 ..< x.len:
      if labels.y[i] != labels.unlabeled:
        continue
      let idx = nearestIndices(dist[i], min(model.k + 1, x.len))
      var votes: seq[int]
      for j in idx:
        if j != i and current[j] != labels.unlabeled:
          votes.add current[j]
      if votes.len > 0:
        let next = majorityLabel(votes)
        if current[i] != next:
          current[i] = next
          changed = true
    if not changed:
      break
  LabelPropagation(k: model.k, maxIter: model.maxIter, x: x,
    labels: current, unlabeled: labels.unlabeled, fitted: true)

proc predict*(model: LabelPropagation; x: Matrix): seq[int] =
  requireFitted(model.fitted, "LabelPropagation.predict")
  let dist = pairwiseDistances(x, model.x)
  for row in dist:
    let idx = nearestIndices(row, min(model.k, row.len))
    var votes: seq[int]
    for j in idx:
      if model.labels[j] != model.unlabeled:
        votes.add model.labels[j]
    if votes.len == 0:
      result.add model.unlabeled
    else:
      result.add majorityLabel(votes)

proc fit*(model: LabelSpreading; x: Matrix; labels: SemiLabels):
    LabelSpreading =
  let propagated = initLabelPropagation(model.k, model.maxIter).fit(x, labels)
  LabelSpreading(k: model.k, alpha: model.alpha, maxIter: model.maxIter,
    propagated: propagated, fitted: true)

proc predict*(model: LabelSpreading; x: Matrix): seq[int] =
  requireFitted(model.fitted, "LabelSpreading.predict")
  model.propagated.predict(x)

proc score*(model: LabelPropagation | LabelSpreading; x: Matrix;
    y: openArray[int]): float64 =
  accuracy(y, model.predict(x))

proc fit*[E](model: SelfTrainingClassifier[E]; x: Matrix; labels: SemiLabels):
    SelfTrainingClassifier[E] =
  let subset = labeledSubset(x, labels)
  let estimator = model.baseEstimator.fit(subset.x, subset.y)
  SelfTrainingClassifier[E](baseEstimator: model.baseEstimator,
    threshold: model.threshold, maxIter: model.maxIter, estimator: estimator,
    fitted: true)

proc predict*[E](model: SelfTrainingClassifier[E]; x: Matrix): auto =
  requireFitted(model.fitted, "SelfTrainingClassifier.predict")
  model.estimator.predict(x)

proc fit*[E](model: CoTrainingClassifier[E]; x: Matrix; labels: SemiLabels):
    CoTrainingClassifier[E] =
  let subset = labeledSubset(x, labels)
  let estimator = model.baseEstimator.fit(subset.x, subset.y)
  CoTrainingClassifier[E](baseEstimator: model.baseEstimator,
    maxIter: model.maxIter, estimator: estimator, fitted: true)

proc predict*[E](model: CoTrainingClassifier[E]; x: Matrix): auto =
  requireFitted(model.fitted, "CoTrainingClassifier.predict")
  model.estimator.predict(x)

proc fit*[E](model: FixMatch[E]; x: Matrix; labels: SemiLabels): FixMatch[E] =
  let subset = labeledSubset(x, labels)
  let estimator = model.baseEstimator.fit(subset.x, subset.y)
  FixMatch[E](baseEstimator: model.baseEstimator, threshold: model.threshold,
    estimator: estimator, fitted: true)

proc predict*[E](model: FixMatch[E]; x: Matrix): auto =
  requireFitted(model.fitted, "FixMatch.predict")
  model.estimator.predict(x)

proc fit*[E](model: MeanTeacher[E]; x: Matrix; labels: SemiLabels):
    MeanTeacher[E] =
  let subset = labeledSubset(x, labels)
  let student = model.student.fit(subset.x, subset.y)
  let teacher = model.teacher.fit(subset.x, subset.y)
  MeanTeacher[E](student: student, teacher: teacher,
    momentum: model.momentum, fitted: true)

proc predict*[E](model: MeanTeacher[E]; x: Matrix): auto =
  requireFitted(model.fitted, "MeanTeacher.predict")
  model.teacher.predict(x)

proc fit*(model: LabelPropagation; x: Tensor; labels: SemiLabels;
    options: FitOptions = initFitOptions()): LabelPropagation =
  discard options
  model.fit(matrixFromTensor(x), labels)

proc fit*(model: LabelSpreading; x: Tensor; labels: SemiLabels;
    options: FitOptions = initFitOptions()): LabelSpreading =
  discard options
  model.fit(matrixFromTensor(x), labels)

proc fit*(model: LabelPropagation | LabelSpreading; df: DataFrame;
    columns: openArray[string]; labels: SemiLabels;
    options: FitOptions = initFitOptions()): auto =
  discard options
  model.fit(matrixOnly(df, columns), labels)
