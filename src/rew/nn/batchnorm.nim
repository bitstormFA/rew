## BatchNorm layers — stateful normalization with running mean/variance.
##
## Pure value types following rew's functional nn invariant.
## The running mean and running variance are updated during training
## and used for inference.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/normalization
import ../ops/arith
import ../ops/linalg
import ./init

type
  BatchNorm1d* = object
    ## 1-D batch normalization over the feature axis (axis 1 for
    ## `[N, C]` input, axis 2 for `[N, L, C]` input).
    gamma*: Param[Tensor]
    beta*: Param[Tensor]
    runningMean*: Buffer[Tensor]
    runningVar*: Buffer[Tensor]
    momentum*: float32
    eps*: float32
    numFeatures*: int

  BatchNorm2d* = object
    ## 2-D batch normalization over the channel axis for NHWC input
    ## (featureIndex = 3).
    gamma*: Param[Tensor]
    beta*: Param[Tensor]
    runningMean*: Buffer[Tensor]
    runningVar*: Buffer[Tensor]
    momentum*: float32
    eps*: float32
    numFeatures*: int

proc initBatchNorm1d*(numFeatures: int; momentum: float32 = 0.1'f32;
    eps: float32 = 1e-5'f32): BatchNorm1d =
  if numFeatures <= 0:
    raise newException(TensorError,
      "initBatchNorm1d: numFeatures must be positive")
  let ones = onesF32(numFeatures)
  let zeros = zerosF32(numFeatures)
  BatchNorm1d(
    gamma: constantF32([numFeatures], ones),
    beta: constantF32([numFeatures], zeros),
    runningMean: buffer(constantF32([numFeatures], zeros)),
    runningVar: buffer(constantF32([numFeatures], ones)),
    momentum: momentum,
    eps: eps,
    numFeatures: numFeatures,
  )

proc initBatchNorm2d*(numFeatures: int; momentum: float32 = 0.1'f32;
    eps: float32 = 1e-5'f32): BatchNorm2d =
  if numFeatures <= 0:
    raise newException(TensorError,
      "initBatchNorm2d: numFeatures must be positive")
  let ones = onesF32(numFeatures)
  let zeros = zerosF32(numFeatures)
  BatchNorm2d(
    gamma: constantF32([numFeatures], ones),
    beta: constantF32([numFeatures], zeros),
    runningMean: buffer(constantF32([numFeatures], zeros)),
    runningVar: buffer(constantF32([numFeatures], ones)),
    momentum: momentum,
    eps: eps,
    numFeatures: numFeatures,
  )

proc forward*(layer: var BatchNorm1d; x: Tensor; training: bool = true): Tensor =
  ## Applies 1-D batch normalization.
  ## `x` shape: `[N, C]` (featureIndex = 1) or `[N, L, C]`
  ## (featureIndex = 2). The feature index is inferred from the rank.
  let featureIndex = if x.shape.len == 2: 1 else: 2
  if x.shape[featureIndex] != layer.numFeatures:
    raise newException(TensorError,
      "BatchNorm1d.forward: feature dim " & $x.shape[featureIndex] &
        " != numFeatures " & $layer.numFeatures)
  if training:
    let (normOut, batchMean, batchVar) = batchNormTraining(
      x, layer.gamma, layer.beta, layer.eps, featureIndex)
    let mom = scalarF32(layer.momentum)
    let oneMinusMom = scalarF32(1'f32 - layer.momentum)
    var bdims: seq[int] = @[]
    let momB = broadcastTo(mom, layer.runningMean.value.shape, bdims)
    let oneMinusMomB = broadcastTo(oneMinusMom,
      layer.runningMean.value.shape, bdims)
    layer.runningMean = add(mul(oneMinusMomB, layer.runningMean),
                             mul(momB, batchMean))
    layer.runningVar = add(mul(oneMinusMomB, layer.runningVar),
                            mul(momB, batchVar))
    normOut
  else:
    batchNormInference(x, layer.gamma, layer.beta,
      layer.runningMean, layer.runningVar, layer.eps, featureIndex)

proc forward*(layer: var BatchNorm2d; x: Tensor; training: bool = true): Tensor =
  ## Applies 2-D batch normalization to an NHWC input
  ## `[N, H, W, C]` (featureIndex = 3).
  if x.shape.len != 4:
    raise newException(TensorError,
      "BatchNorm2d.forward: expected NHWC rank-4, got " & $x.shape)
  let featureIndex = 3
  if x.shape[featureIndex] != layer.numFeatures:
    raise newException(TensorError,
      "BatchNorm2d.forward: channel dim " & $x.shape[featureIndex] &
        " != numFeatures " & $layer.numFeatures)
  if training:
    let (normOut, batchMean, batchVar) = batchNormTraining(
      x, layer.gamma, layer.beta, layer.eps, featureIndex)
    let mom = scalarF32(layer.momentum)
    let oneMinusMom = scalarF32(1'f32 - layer.momentum)
    var bdims: seq[int] = @[]
    let momB = broadcastTo(mom, layer.runningMean.value.shape, bdims)
    let oneMinusMomB = broadcastTo(oneMinusMom,
      layer.runningMean.value.shape, bdims)
    layer.runningMean = add(mul(oneMinusMomB, layer.runningMean),
                             mul(momB, batchMean))
    layer.runningVar = add(mul(oneMinusMomB, layer.runningVar),
                            mul(momB, batchVar))
    normOut
  else:
    batchNormInference(x, layer.gamma, layer.beta,
      layer.runningMean, layer.runningVar, layer.eps, featureIndex)
