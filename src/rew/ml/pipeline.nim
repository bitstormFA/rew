## Tuple-based transformer/estimator pipelines.

import ./core
import ./[preprocessing, decomposition, linear_model, neighbors, naive_bayes,
  svm, tree, ensemble, cluster, mixture, outlier, semi_supervised]

type
  Pipeline*[Steps] = object
    steps*: Steps
    fitted*: bool

func initPipeline*[Steps](steps: Steps): Pipeline[Steps] =
  Pipeline[Steps](steps: steps)

func pipeline*[Steps](steps: Steps): Pipeline[Steps] =
  initPipeline(steps)

func pipeline*[A, B](first: A; second: B): Pipeline[(A, B)] =
  initPipeline((first, second))

func pipeline*[A, B, C](first: A; second: B; third: C): Pipeline[(A, B, C)] =
  initPipeline((first, second, third))

proc fit*[A, E, Y](pipe: Pipeline[(A, E)]; x: Matrix;
    y: openArray[Y]): auto =
  ## Fits one transformer followed by one supervised estimator.
  let fittedFirst = pipe.steps[0].fit(x)
  let transformed = fittedFirst.transform(x)
  let fittedSecond = pipe.steps[1].fit(transformed, y)
  Pipeline[(typeof(fittedFirst), typeof(fittedSecond))](
    steps: (fittedFirst, fittedSecond), fitted: true)

proc fit*[A, B, E, Y](pipe: Pipeline[(A, B, E)]; x: Matrix;
    y: openArray[Y]): auto =
  ## Fits two transformers followed by one supervised estimator.
  let fittedA = pipe.steps[0].fit(x)
  let xA = fittedA.transform(x)
  let fittedB = pipe.steps[1].fit(xA)
  let xB = fittedB.transform(xA)
  let fittedE = pipe.steps[2].fit(xB, y)
  Pipeline[(typeof(fittedA), typeof(fittedB), typeof(fittedE))](
    steps: (fittedA, fittedB, fittedE), fitted: true)

proc fit*[A, B](pipe: Pipeline[(A, B)]; x: Matrix): auto =
  ## Fits two transformer steps in sequence.
  let fittedFirst = pipe.steps[0].fit(x)
  let transformed = fittedFirst.transform(x)
  let fittedSecond = pipe.steps[1].fit(transformed)
  Pipeline[(typeof(fittedFirst), typeof(fittedSecond))](
    steps: (fittedFirst, fittedSecond), fitted: true)

proc fit*[A, B, C](pipe: Pipeline[(A, B, C)]; x: Matrix): auto =
  ## Fits three transformer steps in sequence.
  let fittedA = pipe.steps[0].fit(x)
  let xA = fittedA.transform(x)
  let fittedB = pipe.steps[1].fit(xA)
  let xB = fittedB.transform(xA)
  let fittedC = pipe.steps[2].fit(xB)
  Pipeline[(typeof(fittedA), typeof(fittedB), typeof(fittedC))](
    steps: (fittedA, fittedB, fittedC), fitted: true)

proc transform*[A, B](pipe: Pipeline[(A, B)]; x: Matrix): Matrix =
  requireFitted(pipe.fitted, "Pipeline.transform")
  pipe.steps[1].transform(pipe.steps[0].transform(x))

proc transform*[A, B, C](pipe: Pipeline[(A, B, C)]; x: Matrix): Matrix =
  requireFitted(pipe.fitted, "Pipeline.transform")
  pipe.steps[2].transform(pipe.steps[1].transform(pipe.steps[0].transform(x)))

proc predict*[A, E](pipe: Pipeline[(A, E)]; x: Matrix): auto =
  requireFitted(pipe.fitted, "Pipeline.predict")
  pipe.steps[1].predict(pipe.steps[0].transform(x))

proc predict*[A, B, E](pipe: Pipeline[(A, B, E)]; x: Matrix): auto =
  requireFitted(pipe.fitted, "Pipeline.predict")
  pipe.steps[2].predict(pipe.steps[1].transform(pipe.steps[0].transform(x)))

proc predictProba*[A, E](pipe: Pipeline[(A, E)]; x: Matrix): auto =
  requireFitted(pipe.fitted, "Pipeline.predictProba")
  pipe.steps[1].predictProba(pipe.steps[0].transform(x))

proc predictProba*[A, B, E](pipe: Pipeline[(A, B, E)]; x: Matrix): auto =
  requireFitted(pipe.fitted, "Pipeline.predictProba")
  pipe.steps[2].predictProba(pipe.steps[1].transform(pipe.steps[0].transform(x)))

proc score*[A, E, Y](pipe: Pipeline[(A, E)]; x: Matrix;
    y: openArray[Y]): float64 =
  requireFitted(pipe.fitted, "Pipeline.score")
  pipe.steps[1].score(pipe.steps[0].transform(x), y)

proc score*[A, B, E, Y](pipe: Pipeline[(A, B, E)]; x: Matrix;
    y: openArray[Y]): float64 =
  requireFitted(pipe.fitted, "Pipeline.score")
  pipe.steps[2].score(pipe.steps[1].transform(pipe.steps[0].transform(x)), y)
