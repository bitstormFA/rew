## Model-selection helpers.

import std/[algorithm, math, sequtils]
import ./core
import ./metrics

type
  DataSplit*[T] = object
    train*: seq[T]
    test*: seq[T]

  KFold* = object
    nSplits*: int
    shuffle*: bool
    seed*: int

func initKFold*(nSplits = 5; shuffle = false; seed = 0): KFold =
  if nSplits <= 1:
    raise newException(MlError, "KFold nSplits must be greater than 1")
  KFold(nSplits: nSplits, shuffle: shuffle, seed: seed)

proc shuffledIndices(n: int; seed: int): seq[int] =
  result = toSeq(0 ..< n)
  result.sort(proc(a, b: int): int =
    cmp((a * 1103515245 + seed) mod 2147483647,
        (b * 1103515245 + seed) mod 2147483647))

proc trainTestSplit*[T](data: seq[T]; testSize: float64 = 0.25;
    seed: int = 0): DataSplit[T] =
  if testSize <= 0 or testSize >= 1:
    raise newException(MlError, "testSize must be in (0, 1)")
  let idx = shuffledIndices(data.len, seed)
  let testCount = int(round(float64(data.len) * testSize))
  for pos, original in idx:
    if pos < testCount:
      result.test.add data[original]
    else:
      result.train.add data[original]

proc split*(kf: KFold; nSamples: int): seq[tuple[train, test: seq[int]]] =
  if kf.nSplits > nSamples:
    raise newException(MlError, "KFold nSplits exceeds sample count")
  let idx =
    if kf.shuffle: shuffledIndices(nSamples, kf.seed)
    else: toSeq(0 ..< nSamples)
  for fold in 0 ..< kf.nSplits:
    var train, test: seq[int]
    for pos, original in idx:
      if pos mod kf.nSplits == fold:
        test.add original
      else:
        train.add original
    result.add (train: train, test: test)

proc crossValScore*[E](model: E; x: Matrix; y: openArray[float64];
    folds = 5): seq[float64] =
  requireXY(x, y, "crossValScore")
  let kf = initKFold(folds)
  for part in kf.split(x.len):
    let trainX = selectRows(x, part.train)
    let trainY = selectValues(y, part.train)
    let testX = selectRows(x, part.test)
    let testY = selectValues(y, part.test)
    let fitted = model.fit(trainX, trainY)
    result.add meanSquaredError(testY, fitted.predict(testX))

proc crossValAccuracy*[E](model: E; x: Matrix; y: openArray[int];
    folds = 5): seq[float64] =
  requireLabels(x, y, "crossValAccuracy")
  let kf = initKFold(folds)
  for part in kf.split(x.len):
    let fitted = model.fit(selectRows(x, part.train),
      selectValues(y, part.train))
    result.add accuracy(selectValues(y, part.test),
      fitted.predict(selectRows(x, part.test)))
