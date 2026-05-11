## Sort op: `sort` along a dimension with a comparator.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker
import ./shape
import ./concat

proc sort*(operand: Tensor; dimension = -1; isStable = false;
    comparator: proc (b: var ShBuilder; lhs, rhs: ShValueId): ShValueId
      {.closure.}): Tensor {.rewOp.} =
  ## Sorts `operand` along `dimension` (default: last axis). The
  ## `comparator` receives two scalar SSA values and must return a scalar
  ## `i1` (bool) SSA value indicating whether `lhs` comes before `rhs`.
  case currentMode()
  of dmTrace:
    requireTrace(operand, "sort")
    let ctx = currentTraceContext()
    let rank = operand.shape.len
    let dim = if dimension < 0: rank + dimension else: dimension
    let ids = shops.sort(ctx.builder, [operand.traceId], dim, isStable,
      proc(b: var ShBuilder; args: openArray[ShValueId]): seq[ShValueId] =
        result = @[comparator(b, args[0], args[1])])
    result = initTraceTensor(ids[0], operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("sort", [operand], result,
      @[("dimension", @[dim]), ("isStable", @[if isStable: 1 else: 0])])
  of dmEager:
    raise newException(TensorError,
      "sort: only supported in trace/jit mode")

proc topK*(operand: Tensor; k: int; dimension = -1;
    largest = true): (Tensor, Tensor) =
  ## Returns the `k` largest (or smallest) elements along `dimension`
  ## and their indices. Trace/jit mode only.
  ##
  ## Returns `(values, indices)` where both have rank =
  ## `operand.shape.len` and dim `dimension` size `k`.
  if k <= 0:
    raise newException(TensorError, "topK: k must be positive, got " & $k)
  let rank = operand.shape.len
  let dim = if dimension < 0: rank + dimension else: dimension
  if k > operand.shape[dim]:
    raise newException(TensorError,
      "topK: k " & $k & " > dim size " & $operand.shape[dim])
  case currentMode()
  of dmTrace:
    requireTrace(operand, "topK")
    let ctx = currentTraceContext()
    let ids = shops.sort(ctx.builder, [operand.traceId], dim, true,
      proc(b: var ShBuilder; args: openArray[ShValueId]): seq[ShValueId] =
        if largest:
          result = @[shops.compare(b, args[0], args[1], "GE")]
        else:
          result = @[shops.compare(b, args[0], args[1], "LE")])
    let sorted = initTraceTensor(ids[0], operand.dtype, operand.shape,
      operand.device, operand.sharding)
    recordTraceOp("topK_sort", [operand], sorted)
    # Slice the first k elements along dim.
    var starts = newSeq[int](rank)
    var limits = newSeq[int](rank)
    var strides = newSeq[int](rank)
    for i in 0 ..< rank:
      starts[i] = 0
      limits[i] = (if i == dim: k else: operand.shape[i])
      strides[i] = 1
    let values = slice(sorted, starts, limits, strides)
    # Build sorted indices: create an iota, sort by the same comparator.
    let indicesUnsorted = iota(dtInt32, operand.shape, dim, operand.device)
    let idxIds = shops.sort(ctx.builder, [operand.traceId, indicesUnsorted.traceId],
      dim, true,
      proc(b: var ShBuilder; args: openArray[ShValueId]): seq[ShValueId] =
        if largest:
          result = @[shops.compare(b, args[0], args[1], "GE")]
        else:
          result = @[shops.compare(b, args[0], args[1], "LE")])
    let sortedIdx = initTraceTensor(idxIds[1], dtInt32, operand.shape,
      operand.device, operand.sharding)
    let indices = slice(sortedIdx, starts, limits, strides)
    recordTraceOp("topK", [operand], values, @[
      ("k", @[k]), ("dimension", @[dim]),
      ("largest", @[if largest: 1 else: 0])])
    (values, indices)
  of dmEager:
    raise newException(TensorError,
      "topK: only supported in trace/jit mode")
