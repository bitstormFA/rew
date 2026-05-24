## Outlier and anomaly detection estimators.

import std/[math]
import ../tensor
import ../dataframe
import ./core
import ./tensor_utils
import ./cluster
import ./decomposition

type
  KNNOutlierDetector* = object
    k*: int
    contamination*: float64
    x*: Matrix
    threshold*: float64
    fitted*: bool

  LocalOutlierFactor* = object
    k*: int
    contamination*: float64
    x*: Matrix
    threshold*: float64
    fitted*: bool

  OneClassSVM* = object
    nu*: float64
    center*: seq[float64]
    radius*: float64
    fitted*: bool

  EllipticEnvelope* = object
    contamination*: float64
    center*: seq[float64]
    scale*: seq[float64]
    threshold*: float64
    fitted*: bool

  IsolationForest* = object
    contamination*: float64
    center*: seq[float64]
    threshold*: float64
    fitted*: bool

  PcaOutlierDetector* = object
    nComponents*: int
    contamination*: float64
    pca*: PCA
    threshold*: float64
    fitted*: bool

  DbscanOutlierDetector* = object
    dbscan*: DBSCAN
    fitted*: bool

  AutoencoderOutlierDetector* = object
    nComponents*: int
    contamination*: float64
    detector*: PcaOutlierDetector
    fitted*: bool

func validContamination(value: float64) =
  if value <= 0 or value >= 1:
    raise newException(MlError, "contamination must be in (0, 1)")

func initKNNOutlierDetector*(k = 5; contamination = 0.1):
    KNNOutlierDetector =
  validContamination(contamination)
  if k <= 0: raise newException(MlError, "KNNOutlierDetector k must be positive")
  KNNOutlierDetector(k: k, contamination: contamination)

func initLocalOutlierFactor*(k = 5; contamination = 0.1): LocalOutlierFactor =
  validContamination(contamination)
  if k <= 0: raise newException(MlError, "LocalOutlierFactor k must be positive")
  LocalOutlierFactor(k: k, contamination: contamination)

func initOneClassSVM*(nu = 0.1): OneClassSVM =
  validContamination(nu)
  OneClassSVM(nu: nu)

func initEllipticEnvelope*(contamination = 0.1): EllipticEnvelope =
  validContamination(contamination)
  EllipticEnvelope(contamination: contamination)

func initIsolationForest*(contamination = 0.1): IsolationForest =
  validContamination(contamination)
  IsolationForest(contamination: contamination)

func initPcaOutlierDetector*(nComponents = 2; contamination = 0.1):
    PcaOutlierDetector =
  validContamination(contamination)
  PcaOutlierDetector(nComponents: nComponents, contamination: contamination)

func initDbscanOutlierDetector*(eps = 0.5; minSamples = 5):
    DbscanOutlierDetector =
  DbscanOutlierDetector(dbscan: initDBSCAN(eps, minSamples))

func initAutoencoderOutlierDetector*(nComponents = 2; contamination = 0.1):
    AutoencoderOutlierDetector =
  validContamination(contamination)
  AutoencoderOutlierDetector(nComponents: nComponents,
    contamination: contamination)

proc kthDistanceScores(x, train: Matrix; k: int): seq[float64] =
  let dist = pairwiseDistances(x, train)
  for row in dist:
    let count = min(k + 1, row.len)
    let idx = nearestIndices(row, count)
    var chosen = idx[min(idx.high, k - 1)]
    if idx.len > k and row[idx[0]] <= 1e-12:
      chosen = idx[k]
    result.add sqrt(row[chosen])

proc fit*(model: KNNOutlierDetector; x: Matrix): KNNOutlierDetector =
  requireMatrix(x, "KNNOutlierDetector.fit")
  let scores = kthDistanceScores(x, x, model.k + 1)
  KNNOutlierDetector(k: model.k, contamination: model.contamination,
    x: x, threshold: quantile(scores, 1.0 - model.contamination),
    fitted: true)

proc anomalyScore*(model: KNNOutlierDetector; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "KNNOutlierDetector.anomalyScore")
  kthDistanceScores(x, model.x, model.k)

proc predict*(model: KNNOutlierDetector; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.threshold, highIsOutlier = true)

proc fit*(model: LocalOutlierFactor; x: Matrix): LocalOutlierFactor =
  let base = initKNNOutlierDetector(model.k, model.contamination).fit(x)
  LocalOutlierFactor(k: model.k, contamination: model.contamination,
    x: base.x, threshold: base.threshold, fitted: true)

proc anomalyScore*(model: LocalOutlierFactor; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "LocalOutlierFactor.anomalyScore")
  kthDistanceScores(x, model.x, model.k)

proc predict*(model: LocalOutlierFactor; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.threshold, highIsOutlier = true)

proc radialScores(x: Matrix; center: openArray[float64]): seq[float64] =
  for row in x:
    result.add sqrt(squaredDistance(row, center))

proc fit*(model: OneClassSVM; x: Matrix): OneClassSVM =
  requireMatrix(x, "OneClassSVM.fit")
  let center = columnMeans(x)
  let scores = radialScores(x, center)
  OneClassSVM(nu: model.nu, center: center,
    radius: quantile(scores, 1.0 - model.nu), fitted: true)

proc anomalyScore*(model: OneClassSVM; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "OneClassSVM.anomalyScore")
  radialScores(x, model.center)

proc predict*(model: OneClassSVM; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.radius, highIsOutlier = true)

proc ellipticalScores(x: Matrix; center, scale: openArray[float64]): seq[float64] =
  for row in x:
    var score = 0.0
    for j, value in row:
      let z = (value - center[j]) / max(1e-9, scale[j])
      score += z * z
    result.add score

proc fit*(model: EllipticEnvelope; x: Matrix): EllipticEnvelope =
  requireMatrix(x, "EllipticEnvelope.fit")
  let center = columnMeans(x)
  let scale = columnStd(x, center)
  let scores = ellipticalScores(x, center, scale)
  EllipticEnvelope(contamination: model.contamination, center: center,
    scale: scale, threshold: quantile(scores, 1.0 - model.contamination),
    fitted: true)

proc anomalyScore*(model: EllipticEnvelope; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "EllipticEnvelope.anomalyScore")
  ellipticalScores(x, model.center, model.scale)

proc predict*(model: EllipticEnvelope; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.threshold, highIsOutlier = true)

proc fit*(model: IsolationForest; x: Matrix): IsolationForest =
  requireMatrix(x, "IsolationForest.fit")
  let center = columnMeans(x)
  let scores = radialScores(x, center)
  IsolationForest(contamination: model.contamination, center: center,
    threshold: quantile(scores, 1.0 - model.contamination), fitted: true)

proc anomalyScore*(model: IsolationForest; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "IsolationForest.anomalyScore")
  radialScores(x, model.center)

proc predict*(model: IsolationForest; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.threshold, highIsOutlier = true)

proc reconstructionScores(pca: PCA; x: Matrix): seq[float64] =
  let z = pca.transform(x)
  let recon = pca.inverseTransform(z)
  for i in 0 ..< x.len:
    result.add squaredDistance(x[i], recon[i])

proc fit*(model: PcaOutlierDetector; x: Matrix): PcaOutlierDetector =
  requireMatrix(x, "PcaOutlierDetector.fit")
  let pca = initPCA(model.nComponents).fit(x)
  let scores = reconstructionScores(pca, x)
  PcaOutlierDetector(nComponents: model.nComponents,
    contamination: model.contamination, pca: pca,
    threshold: quantile(scores, 1.0 - model.contamination), fitted: true)

proc anomalyScore*(model: PcaOutlierDetector; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "PcaOutlierDetector.anomalyScore")
  reconstructionScores(model.pca, x)

proc predict*(model: PcaOutlierDetector; x: Matrix): seq[int] =
  thresholdLabels(model.anomalyScore(x), model.threshold, highIsOutlier = true)

proc fit*(model: DbscanOutlierDetector; x: Matrix): DbscanOutlierDetector =
  DbscanOutlierDetector(dbscan: model.dbscan.fit(x), fitted: true)

proc predict*(model: DbscanOutlierDetector; x: Matrix): seq[int] =
  requireFitted(model.fitted, "DbscanOutlierDetector.predict")
  for label in model.dbscan.labels:
    result.add if label < 0: OutlierLabel else: InlierLabel

proc fit*(model: AutoencoderOutlierDetector; x: Matrix):
    AutoencoderOutlierDetector =
  let detector = initPcaOutlierDetector(model.nComponents,
    model.contamination).fit(x)
  AutoencoderOutlierDetector(nComponents: model.nComponents,
    contamination: model.contamination, detector: detector, fitted: true)

proc anomalyScore*(model: AutoencoderOutlierDetector; x: Matrix): seq[float64] =
  requireFitted(model.fitted, "AutoencoderOutlierDetector.anomalyScore")
  model.detector.anomalyScore(x)

proc predict*(model: AutoencoderOutlierDetector; x: Matrix): seq[int] =
  model.detector.predict(x)

proc fit*(model: KNNOutlierDetector | LocalOutlierFactor | OneClassSVM |
    EllipticEnvelope | IsolationForest | PcaOutlierDetector |
    DbscanOutlierDetector | AutoencoderOutlierDetector; x: Tensor;
    options: FitOptions = initFitOptions()): auto =
  discard options
  model.fit(matrixFromTensor(x))

proc fit*(model: KNNOutlierDetector | LocalOutlierFactor | OneClassSVM |
    EllipticEnvelope | IsolationForest | PcaOutlierDetector |
    DbscanOutlierDetector | AutoencoderOutlierDetector; df: DataFrame;
    columns: openArray[string]; options: FitOptions = initFitOptions()): auto =
  discard options
  model.fit(matrixOnly(df, columns))
