## Normalization primitive ops.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

proc requireBatchNormInputs(operand, scale, offset, mean, variance: Tensor;
    featureIndex: int; opName: string) =
  requireSameMode(operand, scale, opName)
  requireSameMode(operand, offset, opName)
  requireSameMode(operand, mean, opName)
  requireSameMode(operand, variance, opName)
  requireSameDevice(operand, scale, opName)
  requireSameDevice(operand, offset, opName)
  requireSameDevice(operand, mean, opName)
  requireSameDevice(operand, variance, opName)
  if not operand.dtype.isFloat:
    raise newException(TensorError,
      opName & ": operand must be floating point, got " & $operand.dtype)
  if featureIndex < 0 or featureIndex >= operand.shape.len:
    raise newException(TensorError,
      opName & ": featureIndex " & $featureIndex &
        " out of range for rank " & $operand.shape.len)
  let featureShape = @[operand.shape[featureIndex]]
  for input in [("scale", scale), ("offset", offset),
                ("mean", mean), ("variance", variance)]:
    let inputName = input[0]
    let inputTensor = input[1]
    if inputTensor.dtype != operand.dtype:
      raise newException(TensorError,
        opName & ": " & inputName & " dtype " & $inputTensor.dtype &
          " differs from operand dtype " & $operand.dtype)
    if inputTensor.shape != featureShape:
      raise newException(TensorError,
        opName & ": " & inputName & " shape " & $inputTensor.shape &
          " must be " & $featureShape)

proc requireFeatureTensor(operand, input: Tensor; inputName: string;
    featureShape: openArray[int]; opName: string) =
  requireSameMode(operand, input, opName)
  requireSameDevice(operand, input, opName)
  if input.dtype != operand.dtype:
    raise newException(TensorError,
      opName & ": " & inputName & " dtype " & $input.dtype &
        " differs from operand dtype " & $operand.dtype)
  if input.shape != @featureShape:
    raise newException(TensorError,
      opName & ": " & inputName & " shape " & $input.shape &
        " must be " & $(@featureShape))

proc requireBatchNormFeatureIndex(operand: Tensor; featureIndex: int;
    opName: string): seq[int] =
  if not operand.dtype.isFloat:
    raise newException(TensorError,
      opName & ": operand must be floating point, got " & $operand.dtype)
  if featureIndex < 0 or featureIndex >= operand.shape.len:
    raise newException(TensorError,
      opName & ": featureIndex " & $featureIndex &
        " out of range for rank " & $operand.shape.len)
  @[operand.shape[featureIndex]]

proc requireBatchNormTrainingInputs(operand, scale, offset: Tensor;
    featureIndex: int; opName: string) =
  let featureShape = requireBatchNormFeatureIndex(operand, featureIndex, opName)
  requireFeatureTensor(operand, scale, "scale", featureShape, opName)
  requireFeatureTensor(operand, offset, "offset", featureShape, opName)

proc requireBatchNormGradInputs(operand, scale, mean, variance,
    gradOutput: Tensor; featureIndex: int; opName: string) =
  let featureShape = requireBatchNormFeatureIndex(operand, featureIndex, opName)
  requireSameMode(operand, gradOutput, opName)
  requireSameDevice(operand, gradOutput, opName)
  if gradOutput.dtype != operand.dtype or gradOutput.shape != operand.shape:
    raise newException(TensorError,
      opName & ": gradOutput type " & $gradOutput.dtype & $gradOutput.shape &
        " must match operand type " & $operand.dtype & $operand.shape)
  requireFeatureTensor(operand, scale, "scale", featureShape, opName)
  requireFeatureTensor(operand, mean, "mean", featureShape, opName)
  requireFeatureTensor(operand, variance, "variance", featureShape, opName)

proc batchNormInference*(operand, scale, offset, mean, variance: Tensor;
    epsilon: float32; featureIndex: int): Tensor {.rewOp.} =
  ## Applies StableHLO batch-normalization inference.
  ## `scale`, `offset`, `mean`, and `variance` are rank-1 tensors whose
  ## length matches `operand.shape[featureIndex]`.
  requireBatchNormInputs(operand, scale, offset, mean, variance,
    featureIndex, "batchNormInference")
  case currentMode()
  of dmTrace:
    requireTrace(operand, "batchNormInference")
    requireTrace(scale, "batchNormInference")
    requireTrace(offset, "batchNormInference")
    requireTrace(mean, "batchNormInference")
    requireTrace(variance, "batchNormInference")
    let ctx = currentTraceContext()
    let id = shops.batchNormInference(ctx.builder, operand.traceId,
      scale.traceId, offset.traceId, mean.traceId, variance.traceId,
      epsilon, featureIndex)
    result = initTraceTensor(id, operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("batchNormInference",
      [operand, scale, offset, mean, variance], result)
  of dmEager:
    requireEager(operand, "batchNormInference")
    requireEager(scale, "batchNormInference")
    requireEager(offset, "batchNormInference")
    requireEager(mean, "batchNormInference")
    requireEager(variance, "batchNormInference")
    let outs = dispatchEager("batchNormInference",
      [operand, scale, offset, mean, variance], [
        ("epsilon", $epsilon),
        ("feature_index", $featureIndex),
      ])
    doAssert outs.len == 1,
      "batchNormInference: eager backend returned wrong arity"
    result = outs[0]

proc batchNormTraining*(operand, scale, offset: Tensor;
    epsilon: float32; featureIndex: int):
    tuple[output, batchMean, batchVar: Tensor] {.rewOp.} =
  ## Applies StableHLO batch-normalization training.
  ## Returns the normalized output plus the computed per-feature mean and
  ## variance.
  requireBatchNormTrainingInputs(operand, scale, offset,
    featureIndex, "batchNormTraining")
  case currentMode()
  of dmTrace:
    requireTrace(operand, "batchNormTraining")
    requireTrace(scale, "batchNormTraining")
    requireTrace(offset, "batchNormTraining")
    let ctx = currentTraceContext()
    let ids = shops.batchNormTraining(ctx.builder, operand.traceId,
      scale.traceId, offset.traceId, epsilon, featureIndex)
    let featureShape = @[operand.shape[featureIndex]]
    result.output = initTraceTensor(ids[0], operand.dtype, operand.shape,
      operand.device, operand.sharding)
    result.batchMean = initTraceTensor(ids[1], operand.dtype, featureShape,
      operand.device, operand.sharding)
    result.batchVar = initTraceTensor(ids[2], operand.dtype, featureShape,
      operand.device, operand.sharding)
    recordTraceOp("batchNormTraining", [operand, scale, offset],
      result.output)
    recordTraceOp("batchNormTraining", [operand, scale, offset],
      result.batchMean)
    recordTraceOp("batchNormTraining", [operand, scale, offset],
      result.batchVar)
  of dmEager:
    requireEager(operand, "batchNormTraining")
    requireEager(scale, "batchNormTraining")
    requireEager(offset, "batchNormTraining")
    let outs = dispatchEager("batchNormTraining",
      [operand, scale, offset], [
        ("epsilon", $epsilon),
        ("feature_index", $featureIndex),
      ])
    doAssert outs.len == 3,
      "batchNormTraining: eager backend returned wrong arity"
    result = (outs[0], outs[1], outs[2])

proc batchNormGrad*(operand, scale, mean, variance, gradOutput: Tensor;
    epsilon: float32; featureIndex: int):
    tuple[gradOperand, gradScale, gradOffset: Tensor] {.rewOp.} =
  ## Applies StableHLO batch-normalization gradient.
  requireBatchNormGradInputs(operand, scale, mean, variance, gradOutput,
    featureIndex, "batchNormGrad")
  case currentMode()
  of dmTrace:
    requireTrace(operand, "batchNormGrad")
    requireTrace(scale, "batchNormGrad")
    requireTrace(mean, "batchNormGrad")
    requireTrace(variance, "batchNormGrad")
    requireTrace(gradOutput, "batchNormGrad")
    let ctx = currentTraceContext()
    let ids = shops.batchNormGrad(ctx.builder, operand.traceId,
      scale.traceId, mean.traceId, variance.traceId, gradOutput.traceId,
      epsilon, featureIndex)
    let featureShape = @[operand.shape[featureIndex]]
    result.gradOperand = initTraceTensor(ids[0], operand.dtype,
      operand.shape, operand.device, operand.sharding)
    result.gradScale = initTraceTensor(ids[1], operand.dtype, featureShape,
      operand.device, operand.sharding)
    result.gradOffset = initTraceTensor(ids[2], operand.dtype, featureShape,
      operand.device, operand.sharding)
    recordTraceOp("batchNormGrad",
      [operand, scale, mean, variance, gradOutput], result.gradOperand)
    recordTraceOp("batchNormGrad",
      [operand, scale, mean, variance, gradOutput], result.gradScale)
    recordTraceOp("batchNormGrad",
      [operand, scale, mean, variance, gradOutput], result.gradOffset)
  of dmEager:
    requireEager(operand, "batchNormGrad")
    requireEager(scale, "batchNormGrad")
    requireEager(mean, "batchNormGrad")
    requireEager(variance, "batchNormGrad")
    requireEager(gradOutput, "batchNormGrad")
    let outs = dispatchEager("batchNormGrad",
      [operand, scale, mean, variance, gradOutput], [
        ("epsilon", $epsilon),
        ("feature_index", $featureIndex),
      ])
    doAssert outs.len == 3,
      "batchNormGrad: eager backend returned wrong arity"
    result = (outs[0], outs[1], outs[2])
