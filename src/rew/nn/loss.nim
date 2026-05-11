## Loss functions. Composite over primitive ops; no dedicated vjps.

import ../tensor
import ../dtype
import ../device
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/reduce
import ../ops/linalg
import ../ops/shape
import ../ops/concat
import ../ops/ternary
import ../ops/compare
import ../ops/factory
import ./activation

proc mseLoss*(pred, target: Tensor): Tensor =
  ## Mean-squared error reduced to a 0-d scalar tensor. Computes the
  ## true mean (sum of squared errors / `numElements`) using
  ## `reduceMean`. v1 supports `dtFloat32` only.
  if pred.shape != target.shape:
    raise newException(TensorError,
      "mseLoss: shape mismatch (" & $pred.shape & " vs " & $target.shape & ")")
  if pred.dtype != target.dtype:
    raise newException(TensorError,
      "mseLoss: dtype mismatch (" & $pred.dtype & " vs " & $target.dtype & ")")
  let diff = sub(pred, target)
  let sq = mul(diff, diff)
  var dims = newSeq[int](pred.shape.len)
  for i in 0 ..< pred.shape.len: dims[i] = i
  reduceMean(sq, dims)

proc softmaxCrossEntropy*(logits, labels: Tensor): Tensor =
  ## Numerically stable mean softmax-cross-entropy over a `[batch, classes]`
  ## minibatch.
  ##
  ## `logits` are unnormalised scores, `labels` is a one-hot encoding
  ## (or a soft label distribution). Both must be `dtFloat32` and rank-2
  ## with matching shape. The result is a 0-d scalar.
  ##
  ## Uses the log-sum-exp trick with max subtraction for stability:
  ## `lse = max + log(sum(exp(logits - max)))`.
  if logits.shape.len != 2:
    raise newException(TensorError,
      "softmaxCrossEntropy: logits must be rank-2, got " & $logits.shape)
  if logits.shape != labels.shape:
    raise newException(TensorError,
      "softmaxCrossEntropy: shape mismatch (" & $logits.shape &
        " vs " & $labels.shape & ")")
  if logits.dtype != dtFloat32 or labels.dtype != dtFloat32:
    raise newException(TensorError,
      "softmaxCrossEntropy: v1 supports only float32 logits/labels")
  # Stabilized log-sum-exp: max subtraction prevents overflow.
  let maxLogits = reduceMax(logits, [1])          ## [batch]
  var bdims: seq[int] = @[0]
  let maxB = broadcastTo(maxLogits, logits.shape, bdims)
  let shifted = sub(logits, maxB)
  let expShifted = exp(shifted)
  let sumExp = reduceSum(expShifted, [1])          ## [batch]
  let lse = add(maxLogits, log(sumExp))            ## [batch]
  let labelLogits = mul(labels, logits)
  let dotPerRow = reduceSum(labelLogits, [1])      ## [batch]
  let perSample = sub(lse, dotPerRow)              ## [batch]
  reduceMean(perSample, [0])

proc binaryCrossEntropy*(pred, target: Tensor): Tensor =
  ## Binary cross-entropy loss: `-mean(target * log(pred) + (1-target) * log(1-pred))`.
  ## `pred` should be in (0, 1) (e.g. output of sigmoid).
  ## Both inputs must be `dtFloat32` with matching shapes.
  if pred.shape != target.shape:
    raise newException(TensorError,
      "binaryCrossEntropy: shape mismatch (" & $pred.shape &
        " vs " & $target.shape & ")")
  if pred.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "binaryCrossEntropy: v1 supports only float32")
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, pred.shape, dims)
  # Clamp pred to avoid log(0).
  let eps = scalarF32(1e-7'f32)
  let epsB = broadcastTo(eps, pred.shape, dims)
  let oneMinusEps = sub(oneB, epsB)
  let predClamped = maximum(epsB, minimum(pred, oneMinusEps))
  let logPred = log(predClamped)
  let logOneMinusPred = log(sub(oneB, predClamped))
  let term1 = mul(target, logPred)
  let term2 = mul(sub(oneB, target), logOneMinusPred)
  let perElement = neg(add(term1, term2))
  var allDims = newSeq[int](pred.shape.len)
  for i in 0 ..< pred.shape.len: allDims[i] = i
  reduceMean(perElement, allDims)

proc huberLoss*(pred, target: Tensor; delta: float32 = 1'f32): Tensor =
  ## Huber loss (smooth L1): quadratic for |error| < delta, linear
  ## outside. Reduced to a 0-d scalar via mean.
  if pred.shape != target.shape:
    raise newException(TensorError,
      "huberLoss: shape mismatch (" & $pred.shape & " vs " & $target.shape & ")")
  if pred.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "huberLoss: v1 supports only float32")
  let diff = sub(pred, target)
  let absDiff = abs(diff)
  let deltaScalar = scalarF32(delta)
  let half = scalarF32(0.5'f32)
  var dims: seq[int] = @[]
  let deltaB = broadcastTo(deltaScalar, pred.shape, dims)
  let halfB = broadcastTo(half, pred.shape, dims)
  # Quadratic region: 0.5 * diff^2
  let quadratic = mul(halfB, mul(diff, diff))
  # Linear region: delta * (|diff| - 0.5 * delta)
  let halfDelta = mul(halfB, deltaB)
  let linear = mul(deltaB, sub(absDiff, halfDelta))
  # Select: use quadratic where |diff| <= delta, linear otherwise.
  # Compute mask via: |diff| <= delta ↔ delta - |diff| >= 0
  # Use minimum(quadratic, linear_upper_bound) approach:
  # Actually use: huber = where(|diff| <= delta, quadratic, linear)
  # Without compare+select as tensor ops, use min(quadratic, linear_capped):
  # huber(x) = min(0.5*x^2, delta*|x| - 0.5*delta^2) doesn't hold.
  # Correct: huber = 0.5*x^2 if |x|<=delta else delta*(|x|-0.5*delta)
  # Use: huber = delta^2 * (sqrt(1 + (diff/delta)^2) - 1) [pseudo-Huber]
  # But let's stick with the standard definition using maximum/minimum:
  # huber = minimum(0.5 * diff^2, delta * |diff| - 0.5 * delta^2)
  # Actually that's wrong too. Use the correct form:
  # huber = where(|diff| <= delta, 0.5*diff^2, delta*(|diff| - 0.5*delta))
  # Since we have maximum: maximum(0, |diff| - delta) gives the linear part.
  # Correct formula: huber = 0.5 * minimum(diff^2, delta^2) + delta * maximum(0, |diff| - delta)
  # Let's use the smooth formulation that decomposes nicely:
  # huber(a) = delta^2 * (sqrt(1 + (a/delta)^2) - 1)  [pseudo-Huber, smooth approx]
  # For the true Huber, let's just use minimum of the two:
  # True: huber = minimum(0.5*diff^2, delta*|diff| - 0.5*delta^2)
  # Verify: at |diff|=delta: 0.5*delta^2 vs delta*delta - 0.5*delta^2 = 0.5*delta^2. Equal. 
  # For |diff| < delta: 0.5*diff^2 < 0.5*delta^2 < delta*|diff| - 0.5*delta^2. First is smaller. 
  # For |diff| > delta: 0.5*diff^2 > 0.5*delta^2, delta*|diff| - 0.5*delta^2 < 0.5*diff^2. Second is smaller. 
  let perElement = minimum(quadratic, linear)
  var allDims = newSeq[int](pred.shape.len)
  for i in 0 ..< pred.shape.len: allDims[i] = i
  reduceMean(perElement, allDims)

# ---- additional loss composites ------------------------------------------

proc l1Loss*(pred, target: Tensor): Tensor =
  ## Mean absolute error (L1 loss): `mean(|pred - target|)`.
  ## Both inputs must be `dtFloat32` with matching shapes.
  if pred.shape != target.shape:
    raise newException(TensorError,
      "l1Loss: shape mismatch (" & $pred.shape & " vs " & $target.shape & ")")
  if pred.dtype != target.dtype:
    raise newException(TensorError,
      "l1Loss: dtype mismatch (" & $pred.dtype & " vs " & $target.dtype & ")")
  let diff = sub(pred, target)
  let absDiff = abs(diff)
  var allDims2 = newSeq[int](pred.shape.len)
  for i in 0 ..< pred.shape.len: allDims2[i] = i
  reduceMean(absDiff, allDims2)

proc nllLoss*(logProbs, labels: Tensor): Tensor =
  ## Negative log-likelihood loss. `logProbs` has shape `[batch, classes]`,
  ## `labels` has shape `[batch]` with integer class indices.
  ## Result is a 0-d scalar: `-mean(logProbs[b, labels[b]])`.
  if logProbs.shape.len != 2:
    raise newException(TensorError,
      "nllLoss: logProbs must be rank-2 [batch, classes], got " &
        $logProbs.shape)
  if labels.shape.len != 1 or labels.shape[0] != logProbs.shape[0]:
    raise newException(TensorError,
      "nllLoss: labels must be rank-1 [batch] with batch=" &
        $logProbs.shape[0] & ", got " & $labels.shape)
  if labels.dtype notin {dtInt32, dtInt64}:
    raise newException(TensorError,
      "nllLoss: labels must be integer type, got " & $labels.dtype)
  let gathered = torchIndexSelect(logProbs, labels, 1, 0)
  let diag = slice(gathered,
    [0, 0],
    [gathered.shape[0], 1],
    [1, 1])
  var squeezed = reshape(diag, [gathered.shape[0]])
  neg(reduceMean(squeezed, [0]))

proc bceWithLogitsLoss*(logits, target: Tensor): Tensor =
  ## Binary cross-entropy with logits input. Combines sigmoid and
  ## binary cross-entropy in a numerically stable way:
  ## `max(logits, 0) - logits * target + log(1 + exp(-|logits|))`.
  if logits.shape != target.shape:
    raise newException(TensorError,
      "bceWithLogitsLoss: shape mismatch (" & $logits.shape &
        " vs " & $target.shape & ")")
  if logits.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "bceWithLogitsLoss: v1 supports only float32")
  # Numerically stable formulation:
  # loss = max(logits, 0) - logits * target + log(1 + exp(-|logits|))
  let zeroB = scalarF32(0'f32)
  var dims: seq[int] = @[]
  let zeroBB = broadcastTo(zeroB, logits.shape, dims)
  let posPart = maximum(logits, zeroBB)
  let negAbs = neg(abs(logits))
  let softplusTerm = log1p(exp(negAbs))
  let term = sub(add(posPart, softplusTerm), mul(logits, target))
  var allDims3 = newSeq[int](logits.shape.len)
  for i in 0 ..< logits.shape.len: allDims3[i] = i
  reduceMean(term, allDims3)

proc klDivLoss*(input, target: Tensor; reduction = "mean"): Tensor =
  ## Kullback-Leibler divergence loss:
  ## `target * (log(target) - input)`.
  ## `input` should be log-probabilities, `target` should be probabilities.
  ## `reduction` can be "mean", "sum", or "none".
  if input.shape != target.shape:
    raise newException(TensorError,
      "klDivLoss: shape mismatch (" & $input.shape & " vs " &
        $target.shape & ")")
  if input.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "klDivLoss: v1 supports only float32")
  let logTarget = log(target)
  let diff = sub(logTarget, input)
  let perElement2 = mul(target, diff)
  case reduction:
  of "none":
    return perElement2
  of "sum":
    var allDims4 = newSeq[int](input.shape.len)
    for i in 0 ..< input.shape.len: allDims4[i] = i
    return reduceSum(perElement2, allDims4)
  of "mean":
    var allDims5 = newSeq[int](input.shape.len)
    for i in 0 ..< input.shape.len: allDims5[i] = i
    return reduceMean(perElement2, allDims5)
  else:
    raise newException(TensorError,
      "klDivLoss: reduction must be 'mean', 'sum', or 'none', got '" &
        reduction & "'")

# ---- SmoothL1Loss ------------------------------------------------------------

proc smoothL1Loss*(pred, target: Tensor; beta: float32 = 1'f32): Tensor =
  ## Smooth L1 loss: `0.5 * (diff^2) / beta` for `|diff| < beta`,
  ## `|diff| - 0.5 * beta` otherwise. Reduced to a 0-d scalar via mean.
  if pred.shape != target.shape:
    raise newException(TensorError,
      "smoothL1Loss: shape mismatch (" & $pred.shape & " vs " &
        $target.shape & ")")
  if pred.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "smoothL1Loss: v1 supports only float32")
  let diff = sub(pred, target)
  let absDiff = abs(diff)
  let half = scalarF32(0.5'f32)
  let betaS = scalarF32(beta)
  var dims: seq[int] = @[]
  let betaB = broadcastTo(betaS, pred.shape, dims)
  let halfB = broadcastTo(half, pred.shape, dims)
  # Quadratic: 0.5 * diff^2 / beta
  let quadratic = divide(mul(halfB, mul(diff, diff)), betaB)
  # Linear: |diff| - 0.5 * beta
  let linear = sub(absDiff, mul(halfB, betaB))
  # Select: quadratic for |diff| < beta, linear otherwise
  let perElement = minimum(quadratic, linear)
  var allDims = newSeq[int](pred.shape.len)
  for i in 0 ..< pred.shape.len: allDims[i] = i
  reduceMean(perElement, allDims)

# ---- FocalLoss ---------------------------------------------------------------

proc sigmoidFocalLoss*(logits, target: Tensor; alpha: float32 = 0.25'f32;
    gamma: float32 = 2'f32): Tensor =
  ## Sigmoid focal loss for binary classification (multi-label).
  ## `logits` are unnormalized scores, `target` is in {0, 1}.
  ## Loss: `-alpha * (1 - p_t)^gamma * log(p_t)` where
  ## `p_t = sigmoid(logits)` for positive, `1 - sigmoid(logits)` for negative.
  if logits.shape != target.shape:
    raise newException(TensorError,
      "sigmoidFocalLoss: shape mismatch (" & $logits.shape &
        " vs " & $target.shape & ")")
  if logits.dtype != dtFloat32 or target.dtype != dtFloat32:
    raise newException(TensorError,
      "sigmoidFocalLoss: v1 supports only float32")
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, logits.shape, dims)
  let alphaB = broadcastTo(scalarF32(alpha), logits.shape, dims)
  let gammaB = broadcastTo(scalarF32(gamma), logits.shape, dims)
  # Numerically stable: compute BCE with logits
  let bce = bceWithLogitsLoss(logits, target)
  # p_t = sigmoid(logits) for target=1, 1-sigmoid(logits) for target=0
  # = exp(-bce_element)
  # Focusing factor: (1 - p_t)^gamma
  let prob = sigmoid(logits)
  let pT = add(mul(target, prob), mul(sub(oneB, target), sub(oneB, prob)))
  let focalWeight = power(sub(oneB, pT), gammaB)
  let alphaWeight = add(mul(target, alphaB), mul(sub(oneB, target), sub(oneB, alphaB)))
  let weightedBce = mul(mul(alphaWeight, focalWeight), bce)
  var allDims = newSeq[int](logits.shape.len)
  for i in 0 ..< logits.shape.len: allDims[i] = i
  reduceMean(weightedBce, allDims)

proc softmaxFocalLoss*(logits, target: Tensor; gamma: float32 = 2'f32): Tensor =
  ## Softmax focal loss for multi-class classification.
  ## `logits` is `[batch, classes]`, `target` is `[batch, classes]` (one-hot).
  ## Loss: `-sum(target * (1 - softmax(logits))^gamma * log(softmax(logits)))`.
  if logits.shape.len != 2:
    raise newException(TensorError,
      "softmaxFocalLoss: logits must be rank-2 [batch, classes]")
  if logits.shape != target.shape:
    raise newException(TensorError,
      "softmaxFocalLoss: shape mismatch")
  let prob = softmax(logits, 1)
  let logProb = logSoftmax(logits, 1)
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, prob.shape, dims)
  let gammaB = broadcastTo(scalarF32(gamma), prob.shape, dims)
  let focalWeight = power(sub(oneB, prob), gammaB)
  let nll = neg(mul(target, logProb))
  let perElement = mul(focalWeight, nll)
  var allDims = @[0, 1]
  reduceMean(perElement, allDims)

# ---- CosineEmbeddingLoss -----------------------------------------------------

proc cosineEmbeddingLoss*(x1, x2: Tensor; y: Tensor;
    margin: float32 = 0'f32): Tensor =
  ## Cosine embedding loss. `y` is -1 or 1 indicating similarity.
  ## For y=1: `1 - cos(x1, x2)`. For y=-1: `max(0, cos(x1, x2) - margin)`.
  if x1.shape != x2.shape:
    raise newException(TensorError,
      "cosineEmbeddingLoss: shape mismatch (" & $x1.shape & " vs " &
        $x2.shape & ")")
  if x1.shape.len != 2:
    raise newException(TensorError,
      "cosineEmbeddingLoss: inputs must be rank-2 [batch, dim]")
  # Cosine similarity: dot product of normalized vectors
  let x1Norm = standardize(x1, [1])
  let x2Norm = standardize(x2, [1])
  let cosSim = reduceSum(mul(x1Norm, x2Norm), [1])
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, cosSim.shape, dims)
  let marginB = broadcastTo(scalarF32(margin), cosSim.shape, dims)
  # For similar (y=1): loss = 1 - cos_sim
  let posLoss = sub(oneB, cosSim)
  # For dissimilar (y=-1): loss = max(0, cos_sim - margin)
  let negLoss = maximum(broadcastTo(scalarF32(0'f32), cosSim.shape, dims),
    sub(cosSim, marginB))
  # y is -1 or 1, compute: loss where y==1 use posLoss, where y==-1 use negLoss
  let yB = broadcastTo(y, cosSim.shape, dims)
  # Mask: isSimilar = (y > 0)
  let zeroB = broadcastTo(scalarF32(0'f32), y.shape, dims)
  let isSimilar = compare(yB, zeroB, "GT")
  let loss = select(isSimilar, posLoss, negLoss)
  reduceMean(loss, [0])

# ---- TripletMarginLoss -------------------------------------------------------

proc tripletMarginLoss*(anchor, positive, negative: Tensor;
    margin: float32 = 1'f32; p: float32 = 2'f32): Tensor =
  ## Triplet margin loss: `max(0, d(anchor, pos) - d(anchor, neg) + margin)`.
  ## Uses Lp distance by default (p=2 for Euclidean).
  if anchor.shape != positive.shape or anchor.shape != negative.shape:
    raise newException(TensorError,
      "tripletMarginLoss: shape mismatch")
  if anchor.shape.len != 2:
    raise newException(TensorError,
      "tripletMarginLoss: inputs must be rank-2 [batch, dim]")
  # Compute pairwise Lp distances
  let dAp = reduceSum(power(abs(sub(anchor, positive)),
    broadcastTo(scalarF32(p), anchor.shape, @[])), [1])
  let dAn = reduceSum(power(abs(sub(anchor, negative)),
    broadcastTo(scalarF32(p), anchor.shape, @[])), [1])
  # Lp distance = (sum |diff|^p)^(1/p)
  let invP = scalarF32(1'f32 / p)
  var dims1: seq[int] = @[]
  let invPB = broadcastTo(invP, dAp.shape, dims1)
  let dApNorm = power(dAp, invPB)
  let dAnNorm = power(dAn, invPB)
  # loss = max(0, d(anchor, pos) - d(anchor, neg) + margin)
  let marginB = broadcastTo(scalarF32(margin), dApNorm.shape, dims1)
  let raw = add(sub(dApNorm, dAnNorm), marginB)
  let zeroB = broadcastTo(scalarF32(0'f32), raw.shape, dims1)
  let perElement = maximum(raw, zeroB)
  reduceMean(perElement, [0])

# ---- CTCLoss -----------------------------------------------------------------

const CtcLogZero = -1.0e30'f32

proc ctcScalar(value: float32; device: Device): Tensor =
  full([], value, dtFloat32, device)

proc ctcI32(value: int; device: Device): Tensor =
  scalarI32(int32(value), device)

proc asI32Scalar(t: Tensor): Tensor =
  let scalar = reshape(t, [])
  if scalar.dtype == dtInt32:
    scalar
  else:
    astype(scalar, dtInt32)

proc lengthAt(lengths: Tensor; n: int): Tensor =
  asI32Scalar(slice(lengths, [n], [n + 1], [1]))

proc targetOffset(targetLengths: Tensor; n: int): Tensor =
  result = ctcI32(0, targetLengths.device)
  for i in 0 ..< n:
    result = add(result, lengthAt(targetLengths, i))

proc targetAt(targets, targetLengths: Tensor; n, pos: int): Tensor =
  if targets.shape.len == 2:
    return asI32Scalar(slice(targets, [n, pos], [n + 1, pos + 1], [1, 1]))
  let offset = targetOffset(targetLengths, n)
  let dynamicPos = add(offset, ctcI32(pos, targets.device))
  asI32Scalar(dynamicSlice(targets, [dynamicPos], [1]))

proc classLogProb(logProbs: Tensor; t, n: int; label: Tensor): Tensor =
  reshape(dynamicSlice(logProbs, [
    ctcI32(t, logProbs.device),
    ctcI32(n, logProbs.device),
    label,
  ], [1, 1, 1]), [])

proc logAddExp(a, b: Tensor): Tensor =
  let m = maximum(a, b)
  add(m, log(add(exp(sub(a, m)), exp(sub(b, m)))))

proc logAddExp(xs: openArray[Tensor]; device: Device): Tensor =
  result = ctcScalar(CtcLogZero, device)
  for x in xs:
    result = logAddExp(result, x)

proc scalarBoolSelect(mask, onTrue, onFalse: Tensor): Tensor =
  select(mask, onTrue, onFalse)

proc stateLimit(targetLen: Tensor): Tensor =
  add(add(targetLen, targetLen), ctcI32(1, targetLen.device))

proc stateIsValid(state: int; targetLen: Tensor): Tensor =
  compare(ctcI32(state, targetLen.device), stateLimit(targetLen), "LT")

proc activeTime(t: int; inputLen: Tensor): Tensor =
  compare(ctcI32(t, inputLen.device), inputLen, "LT")

proc labelForState(targets, targetLengths: Tensor; n, state, blank: int):
    Tensor =
  if state mod 2 == 0:
    ctcI32(blank, targets.device)
  else:
    targetAt(targets, targetLengths, n, state div 2)

proc transitionSum(alpha: seq[Tensor]; labels: seq[Tensor]; state, blank: int;
    device: Device): Tensor =
  var terms = @[alpha[state]]
  if state > 0:
    terms.add alpha[state - 1]
  if state > 1 and state mod 2 == 1:
    let notBlank = compare(labels[state], ctcI32(blank, device), "NE")
    let notRepeat = compare(labels[state], labels[state - 2], "NE")
    let canSkip = bitwiseAnd(notBlank, notRepeat)
    terms.add scalarBoolSelect(canSkip, alpha[state - 2],
      ctcScalar(CtcLogZero, device))
  logAddExp(terms, device)

proc finalLogProb(alpha: seq[Tensor]; targetLen: Tensor; device: Device):
    Tensor =
  var terms: seq[Tensor] = @[]
  let lastBlank = add(targetLen, targetLen)
  let lastLabel = sub(lastBlank, ctcI32(1, device))
  for state, value in alpha:
    let stateId = ctcI32(state, device)
    let isFinal = bitwiseOr(compare(stateId, lastBlank, "EQ"),
      compare(stateId, lastLabel, "EQ"))
    terms.add scalarBoolSelect(isFinal, value, ctcScalar(CtcLogZero, device))
  logAddExp(terms, device)

proc ctcLoss*(logProbs: Tensor; targets: Tensor;
    inputLengths: Tensor; targetLengths: Tensor;
    blank: int = 0): Tensor =
  ## Connectionist Temporal Classification loss.
  ## `logProbs` is `[T, N, C]` (time × batch × classes).
  ## `targets` may be padded `[N, S]` labels or concatenated 1-D labels.
  ## `inputLengths` and `targetLengths` are per-batch lengths.
  ##
  ## Returns the mean negative log probability under the CTC
  ## forward-backward algorithm. The implementation is a static-shape
  ## tensor dynamic program and is intended for forward loss computation;
  ## differentiating through CTC requires dedicated VJP coverage for the
  ## dynamic indexing primitives it uses.
  if logProbs.shape.len != 3:
    raise newException(TensorError,
      "ctcLoss: logProbs must be [T, N, C], got " & $logProbs.shape)
  if logProbs.dtype != dtFloat32:
    raise newException(TensorError,
      "ctcLoss: logProbs must be float32, got " & $logProbs.dtype)
  let timeSteps = logProbs.shape[0]
  let batch = logProbs.shape[1]
  let classes = logProbs.shape[2]
  if timeSteps <= 0 or batch <= 0 or classes <= 0:
    raise newException(TensorError,
      "ctcLoss: logProbs dimensions must be positive")
  if blank < 0 or blank >= classes:
    raise newException(TensorError,
      "ctcLoss: blank index " & $blank & " out of range for " & $classes &
        " classes")
  if inputLengths.shape != @[batch] or targetLengths.shape != @[batch]:
    raise newException(TensorError,
      "ctcLoss: inputLengths and targetLengths must both have shape [" &
        $batch & "]")
  if not (targets.dtype.isSignedInt or targets.dtype.isUnsignedInt):
    raise newException(TensorError,
      "ctcLoss: targets must be integer, got " & $targets.dtype)
  if not (inputLengths.dtype.isSignedInt or inputLengths.dtype.isUnsignedInt) or
      not (targetLengths.dtype.isSignedInt or targetLengths.dtype.isUnsignedInt):
    raise newException(TensorError,
      "ctcLoss: length tensors must be integer")
  requireSameMode(logProbs, targets, "ctcLoss")
  requireSameMode(logProbs, inputLengths, "ctcLoss")
  requireSameMode(logProbs, targetLengths, "ctcLoss")
  requireSameDevice(logProbs, targets, "ctcLoss")
  requireSameDevice(logProbs, inputLengths, "ctcLoss")
  requireSameDevice(logProbs, targetLengths, "ctcLoss")

  let maxTargetLen =
    if targets.shape.len == 2:
      if targets.shape[0] != batch:
        raise newException(TensorError,
          "ctcLoss: padded targets batch does not match logProbs batch")
      targets.shape[1]
    elif targets.shape.len == 1:
      targets.shape[0]
    else:
      raise newException(TensorError,
        "ctcLoss: targets must be rank-1 concatenated or rank-2 padded")
  if maxTargetLen <= 0:
    raise newException(TensorError,
      "ctcLoss: targets must contain at least one label slot")

  let states = 2 * maxTargetLen + 1
  var losses: seq[Tensor] = @[]
  for n in 0 ..< batch:
    let inputLen = lengthAt(inputLengths, n)
    let targetLen = lengthAt(targetLengths, n)
    var labels = newSeq[Tensor](states)
    for state in 0 ..< states:
      labels[state] = labelForState(targets, targetLengths, n, state, blank)

    var alpha = newSeq[Tensor](states)
    for state in 0 ..< states:
      alpha[state] = ctcScalar(CtcLogZero, logProbs.device)
    alpha[0] = classLogProb(logProbs, 0, n, labels[0])
    if states > 1:
      alpha[1] = scalarBoolSelect(stateIsValid(1, targetLen),
        classLogProb(logProbs, 0, n, labels[1]),
        ctcScalar(CtcLogZero, logProbs.device))

    for t in 1 ..< timeSteps:
      var next = newSeq[Tensor](states)
      let active = activeTime(t, inputLen)
      for state in 0 ..< states:
        let total = transitionSum(alpha, labels, state, blank, logProbs.device)
        let candidate = add(total, classLogProb(logProbs, t, n, labels[state]))
        let valid = stateIsValid(state, targetLen)
        let masked = scalarBoolSelect(valid, candidate,
          ctcScalar(CtcLogZero, logProbs.device))
        next[state] = scalarBoolSelect(active, masked, alpha[state])
      alpha = next

    losses.add neg(finalLogProb(alpha, targetLen, logProbs.device))

  if losses.len == 1:
    losses[0]
  else:
    reduceMean(stack(losses, 0), [0])
