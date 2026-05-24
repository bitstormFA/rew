## Metrics for supervised, clustering, and outlier workflows.

import std/[algorithm, math]
import ./core
import ./tensor_utils

func meanSquaredError*(yTrue, yPred: openArray[float64]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "meanSquaredError: invalid lengths")
  for i in 0 ..< yTrue.len:
    let d = yTrue[i] - yPred[i]
    result += d * d
  result / float64(yTrue.len)

func meanAbsoluteError*(yTrue, yPred: openArray[float64]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "meanAbsoluteError: invalid lengths")
  for i in 0 ..< yTrue.len:
    result += abs(yTrue[i] - yPred[i])
  result / float64(yTrue.len)

func r2Score*(yTrue, yPred: openArray[float64]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "r2Score: invalid lengths")
  let avg = meanValue(yTrue)
  var ssRes = 0.0
  var ssTot = 0.0
  for i in 0 ..< yTrue.len:
    ssRes += (yTrue[i] - yPred[i]) * (yTrue[i] - yPred[i])
    ssTot += (yTrue[i] - avg) * (yTrue[i] - avg)
  if ssTot == 0:
    return if ssRes == 0: 1.0 else: 0.0
  1.0 - ssRes / ssTot

func accuracy*(yTrue, yPred: openArray[int]): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "accuracy: invalid lengths")
  var hits = 0
  for i in 0 ..< yTrue.len:
    if yTrue[i] == yPred[i]:
      inc hits
  float64(hits) / float64(yTrue.len)

func precisionScore*(yTrue, yPred: openArray[int]; positiveLabel = 1): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "precisionScore: invalid lengths")
  var tp, fp = 0
  for i in 0 ..< yTrue.len:
    if yPred[i] == positiveLabel:
      if yTrue[i] == positiveLabel: inc tp else: inc fp
  if tp + fp == 0: 0.0 else: float64(tp) / float64(tp + fp)

func recallScore*(yTrue, yPred: openArray[int]; positiveLabel = 1): float64 =
  if yTrue.len != yPred.len or yTrue.len == 0:
    raise newException(MlError, "recallScore: invalid lengths")
  var tp, fn = 0
  for i in 0 ..< yTrue.len:
    if yTrue[i] == positiveLabel:
      if yPred[i] == positiveLabel: inc tp else: inc fn
  if tp + fn == 0: 0.0 else: float64(tp) / float64(tp + fn)

func f1Score*(yTrue, yPred: openArray[int]; positiveLabel = 1): float64 =
  let p = precisionScore(yTrue, yPred, positiveLabel)
  let r = recallScore(yTrue, yPred, positiveLabel)
  if p + r == 0: 0.0 else: 2.0 * p * r / (p + r)

func rocAuc*(yTrue: openArray[int]; scores: openArray[float64]): float64 =
  if yTrue.len != scores.len or yTrue.len == 0:
    raise newException(MlError, "rocAuc: invalid lengths")
  var pairs: seq[tuple[label: int, score: float64]]
  for i in 0 ..< yTrue.len:
    pairs.add (label: yTrue[i], score: scores[i])
  pairs.sort(proc(a, b: tuple[label: int, score: float64]): int =
    cmp(a.score, b.score))
  var rankSum = 0.0
  var positives = 0
  var negatives = 0
  for i, p in pairs:
    if p.label == 1:
      rankSum += float64(i + 1)
      inc positives
    else:
      inc negatives
  if positives == 0 or negatives == 0:
    raise newException(MlError, "rocAuc: both classes are required")
  (rankSum - float64(positives * (positives + 1)) / 2.0) /
    float64(positives * negatives)

proc silhouetteScore*(x: Matrix; labels: openArray[int]): float64 =
  ## Mean silhouette coefficient for clustering labels.
  requireLabels(x, labels, "silhouetteScore")
  let classes = uniqueLabels(labels)
  if classes.len < 2:
    raise newException(MlError, "silhouetteScore: at least two labels required")
  let dist = pairwiseDistances(x, x)
  for i in 0 ..< x.len:
    var aSum = 0.0
    var aCount = 0
    var b = Inf
    for cls in classes:
      var sum = 0.0
      var count = 0
      for j in 0 ..< x.len:
        if i != j and labels[j] == cls:
          sum += sqrt(dist[i][j])
          inc count
      if count == 0:
        continue
      let avg = sum / float64(count)
      if cls == labels[i]:
        aSum = sum
        aCount = count
      else:
        b = min(b, avg)
    let a = if aCount == 0: 0.0 else: aSum / float64(aCount)
    let denom = max(a, b)
    if denom > 0:
      result += (b - a) / denom
  result / float64(x.len)
