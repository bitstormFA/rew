## Tensor-first ML module smoke and behavior tests.

import rew/ml

proc close(a, b: float64; eps = 1e-6): bool =
  abs(a - b) <= eps

let x: Matrix = @[
  @[1.0],
  @[2.0],
  @[3.0],
  @[4.0],
]
let y = @[2.0, 4.0, 6.0, 8.0]

block linear_and_pipeline:
  let model = initLinearRegression().fit(x, y)
  doAssert model.fitted
  doAssert model.predict(@[@[5.0]])[0].close(10.0)

  let pipe = pipeline(initStandardScaler(), initLinearRegression()).fit(x, y)
  doAssert pipe.fitted
  doAssert pipe.score(x, y) > 0.99

block supervised_classifiers:
  let cx: Matrix = @[
    @[-2.0],
    @[-1.0],
    @[1.0],
    @[2.0],
  ]
  let cy = @[0, 0, 1, 1]

  doAssert initLogisticRegression(lr = 0.5, maxIter = 300).
    fit(cx, cy).predict(cx) == cy
  doAssert initSoftmaxRegression(lr = 0.5, maxIter = 300).
    fit(cx, cy).predict(cx) == cy
  doAssert initLinearSVM(lr = 0.1, maxIter = 50).fit(cx, cy).predict(cx) == cy
  doAssert initKNeighborsClassifier(k = 1).fit(cx, cy).predict(cx) == cy
  doAssert initGaussianNB().fit(cx, cy).predict(cx) == cy
  doAssert initDecisionTreeClassifier(maxDepth = 2).fit(cx, cy).predict(cx) == cy
  doAssert initRandomForestClassifier(nEstimators = 3, maxDepth = 2).
    fit(cx, cy).predict(cx) == cy

block unsupervised_estimators:
  let points: Matrix = @[
    @[0.0, 0.0],
    @[0.1, 0.0],
    @[5.0, 5.0],
    @[5.1, 5.0],
  ]
  let scaler = initMinMaxScaler().fit(points)
  let scaled = scaler.transform(points)
  doAssert scaled[0][0].close(0.0)
  doAssert scaled[^1][0].close(1.0)

  let pca = initPCA(1).fit(points)
  doAssert pca.transform(points).len == points.len

  let km = initKMeans(2).fit(points)
  doAssert km.labels.len == points.len

  let gmm = initGaussianMixture(2, maxIter = 3).fit(points)
  doAssert gmm.predict(points).len == points.len

  let db = initDBSCAN(eps = 0.3, minSamples = 1).fit(points)
  doAssert db.labels.len == points.len

block semi_supervised_and_outliers:
  let points: Matrix = @[
    @[0.0, 0.0],
    @[0.1, 0.0],
    @[5.0, 5.0],
    @[5.1, 5.0],
  ]
  let labels = initSemiLabels([0, -1, 1, -1], unlabeled = -1)
  let propagated = initLabelPropagation(k = 1).fit(points, labels)
  doAssert propagated.labels == @[0, 0, 1, 1]

  let detector = initKNNOutlierDetector(k = 1, contamination = 0.25).fit(
    points & @[@[100.0, 100.0]])
  let pred = detector.predict(points & @[@[100.0, 100.0]])
  doAssert pred[^1] == OutlierLabel

block model_selection_metrics:
  doAssert meanSquaredError(@[1.0, 2.0], @[1.0, 4.0]).close(2.0)
  doAssert accuracy(@[0, 1], @[0, 1]).close(1.0)
  doAssert rocAuc(@[0, 1, 0, 1], @[0.1, 0.9, 0.2, 0.8]).close(1.0)
  let split = trainTestSplit(@[1, 2, 3, 4], testSize = 0.5, seed = 7)
  doAssert split.train.len == 2
  doAssert split.test.len == 2
  doAssert crossValScore(initLinearRegression(), x, y, folds = 2).len == 2
