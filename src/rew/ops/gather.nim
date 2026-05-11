## Gather ops: `gather`, `dynamic_gather`, `indexSelect`.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker

proc gather*(operand, startIndices: Tensor;
    dims: GatherDimensionNumbers; sliceSizes: openArray[int];
    resultShape: openArray[int];
    indicesAreSorted = false): Tensor {.rewOp.} =
  ## Gathers slices from `operand` at `startIndices` using
  ## `GatherDimensionNumbers`. `resultShape` must be supplied explicitly
  ## and determines the output shape.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "gather")
    requireTrace(startIndices, "gather")
    let ctx = currentTraceContext()
    let id = shops.gather(ctx.builder, operand.traceId,
      startIndices.traceId, dims, sliceSizes, resultShape, indicesAreSorted)
    result = initTraceTensor(id, operand.dtype, resultShape, operand.device,
      operand.sharding)
    recordTraceOp("gather", [operand, startIndices], result)
  of dmEager:
    raise newException(TensorError,
      "gather: only supported in trace/jit mode")

proc dynamicGather*(operand, startIndices, sliceSizes: Tensor;
    dims: GatherDimensionNumbers; resultShape: openArray[int];
    indicesAreSorted = false): Tensor {.rewOp.} =
  ## Dynamic variant of `gather` where `sliceSizes` is a runtime tensor.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "dynamic_gather")
    requireTrace(startIndices, "dynamic_gather")
    requireTrace(sliceSizes, "dynamic_gather")
    let ctx = currentTraceContext()
    let id = shops.dynamicGather(ctx.builder, operand.traceId,
      startIndices.traceId, sliceSizes.traceId, dims, resultShape,
      indicesAreSorted)
    result = initTraceTensor(id, operand.dtype, resultShape, operand.device,
      operand.sharding)
    recordTraceOp("dynamicGather", [operand, startIndices, sliceSizes],
      result)
  of dmEager:
    raise newException(TensorError,
      "dynamic_gather: only supported in trace/jit mode")

proc indexSelect*(operand, index: Tensor; dim: int = 0;
    batchDims: int = 0): Tensor {.rewOp.} =
  ## Gathers slices from `operand` at positions given by `index` along `dim`.
  ## Equivalent to PyTorch `torch.index_select`. `index` must be 1-D integer.
  ##
  ## Example: `indexSelect(x, edgeIndexSrc, dim=0)` gathers source node
  ## features for message passing — `x` is `[N, F]`, index is `[E]`,
  ## result is `[E, F]`.
  if dim < 0 or dim >= operand.shape.len:
    raise newException(TensorError,
      "indexSelect: dim " & $dim &
        " out of range for operand rank " & $operand.shape.len)
  if not (index.dtype.isSignedInt or index.dtype.isUnsignedInt):
    raise newException(TensorError,
      "indexSelect: index must be integer, got " & $index.dtype)
  case currentMode()
  of dmTrace:
    requireTrace(operand, "indexSelect")
    requireTrace(index, "indexSelect")
    let ctx = currentTraceContext()
    let outShape = shops.torchIndexSelectOutputShape(
      operand.shape, index.shape, dim, batchDims)
    let id = shops.torchIndexSelect(ctx.builder, operand.traceId,
      index.traceId, dim, batchDims)
    result = initTraceTensor(id, operand.dtype, outShape, operand.device,
      operand.sharding)
    recordTraceOp("indexSelect", [operand, index], result)
  of dmEager:
    raise newException(TensorError,
      "indexSelect: only supported in trace/jit mode")
