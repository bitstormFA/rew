## Feature preprocessing transformers.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./tensor_utils

type
  StandardScaler* = object
    mean*: seq[float64]
    scale*: seq[float64]
    fitted*: bool

  MinMaxScaler* = object
    dataMin*: seq[float64]
    dataMax*: seq[float64]
    featureMin*: float64
    featureMax*: float64
    fitted*: bool

  RobustScaler* = object
    center*: seq[float64]
    scale*: seq[float64]
    fitted*: bool

  Normalizer* = object
    norm*: string
    fitted*: bool

func initStandardScaler*(): StandardScaler =
  StandardScaler()

func initMinMaxScaler*(featureMin = 0.0; featureMax = 1.0): MinMaxScaler =
  if featureMin >= featureMax:
    raise newException(MlError, "MinMaxScaler: featureMin must be < featureMax")
  MinMaxScaler(featureMin: featureMin, featureMax: featureMax)

func initRobustScaler*(): RobustScaler =
  RobustScaler()

func initNormalizer*(norm = "l2"): Normalizer =
  if norm notin ["l1", "l2", "max"]:
    raise newException(MlError, "Normalizer norm must be l1, l2, or max")
  Normalizer(norm: norm, fitted: true)

proc fit*(scaler: StandardScaler; x: Matrix): StandardScaler =
  discard scaler
  requireMatrix(x, "StandardScaler.fit")
  result.mean = columnMeans(x)
  result.scale = columnStd(x, result.mean)
  result.fitted = true

proc transform*(scaler: StandardScaler; x: Matrix): Matrix =
  requireFitted(scaler.fitted, "StandardScaler.transform")
  requireMatrix(x, "StandardScaler.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](row.len)
    for j, value in row:
      result[i][j] = (value - scaler.mean[j]) / scaler.scale[j]

proc fitTransform*(scaler: StandardScaler; x: Matrix): Matrix =
  scaler.fit(x).transform(x)

proc fit*(scaler: MinMaxScaler; x: Matrix): MinMaxScaler =
  requireMatrix(x, "MinMaxScaler.fit")
  result = scaler
  result.dataMin = newSeq[float64](x[0].len)
  result.dataMax = newSeq[float64](x[0].len)
  for j in 0 ..< x[0].len:
    result.dataMin[j] = Inf
    result.dataMax[j] = NegInf
  for row in x:
    for j, value in row:
      result.dataMin[j] = min(result.dataMin[j], value)
      result.dataMax[j] = max(result.dataMax[j], value)
  result.fitted = true

proc transform*(scaler: MinMaxScaler; x: Matrix): Matrix =
  requireFitted(scaler.fitted, "MinMaxScaler.transform")
  requireMatrix(x, "MinMaxScaler.transform")
  result = newSeq[seq[float64]](x.len)
  let outSpan = scaler.featureMax - scaler.featureMin
  for i, row in x:
    result[i] = newSeq[float64](row.len)
    for j, value in row:
      let span = scaler.dataMax[j] - scaler.dataMin[j]
      if span == 0:
        result[i][j] = scaler.featureMin
      else:
        result[i][j] = scaler.featureMin +
          (value - scaler.dataMin[j]) / span * outSpan

proc fitTransform*(scaler: MinMaxScaler; x: Matrix): Matrix =
  scaler.fit(x).transform(x)

proc columnQuantile(x: Matrix; column: int; q: float64): float64 =
  var values: seq[float64]
  for row in x:
    values.add row[column]
  quantile(values, q)

proc fit*(scaler: RobustScaler; x: Matrix): RobustScaler =
  discard scaler
  requireMatrix(x, "RobustScaler.fit")
  result.center = newSeq[float64](x[0].len)
  result.scale = newSeq[float64](x[0].len)
  for j in 0 ..< x[0].len:
    result.center[j] = columnQuantile(x, j, 0.5)
    let q1 = columnQuantile(x, j, 0.25)
    let q3 = columnQuantile(x, j, 0.75)
    result.scale[j] = q3 - q1
    if result.scale[j] == 0:
      result.scale[j] = 1.0
  result.fitted = true

proc transform*(scaler: RobustScaler; x: Matrix): Matrix =
  requireFitted(scaler.fitted, "RobustScaler.transform")
  requireMatrix(x, "RobustScaler.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    result[i] = newSeq[float64](row.len)
    for j, value in row:
      result[i][j] = (value - scaler.center[j]) / scaler.scale[j]

proc fitTransform*(scaler: RobustScaler; x: Matrix): Matrix =
  scaler.fit(x).transform(x)

proc fit*(normalizer: Normalizer; x: Matrix): Normalizer =
  requireMatrix(x, "Normalizer.fit")
  normalizer

proc transform*(normalizer: Normalizer; x: Matrix): Matrix =
  requireMatrix(x, "Normalizer.transform")
  result = newSeq[seq[float64]](x.len)
  for i, row in x:
    var denom = 0.0
    case normalizer.norm
    of "l1":
      for value in row: denom += abs(value)
    of "l2":
      for value in row: denom += value * value
      denom = sqrt(denom)
    of "max":
      for value in row: denom = max(denom, abs(value))
    else:
      raise newException(MlError, "Normalizer.transform: unknown norm")
    if denom == 0:
      denom = 1.0
    result[i] = newSeq[float64](row.len)
    for j, value in row:
      result[i][j] = value / denom

proc fitTransform*(normalizer: Normalizer; x: Matrix): Matrix =
  normalizer.fit(x).transform(x)

proc fit*(scaler: StandardScaler; x: Tensor;
    options: FitOptions = initFitOptions()): StandardScaler =
  discard options
  scaler.fit(matrixFromTensor(x))

proc fit*(scaler: MinMaxScaler; x: Tensor;
    options: FitOptions = initFitOptions()): MinMaxScaler =
  discard options
  scaler.fit(matrixFromTensor(x))

proc fit*(scaler: RobustScaler; x: Tensor;
    options: FitOptions = initFitOptions()): RobustScaler =
  discard options
  scaler.fit(matrixFromTensor(x))

proc transform*(scaler: StandardScaler | MinMaxScaler | RobustScaler |
    Normalizer; x: Tensor): Matrix =
  scaler.transform(matrixFromTensor(x))

proc fit*(scaler: StandardScaler; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): StandardScaler =
  discard options
  scaler.fit(matrixOnly(df, columns))

proc fit*(scaler: MinMaxScaler; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): MinMaxScaler =
  discard options
  scaler.fit(matrixOnly(df, columns))

proc fit*(scaler: RobustScaler; df: DataFrame; columns: openArray[string];
    options: FitOptions = initFitOptions()): RobustScaler =
  discard options
  scaler.fit(matrixOnly(df, columns))
