## Concatenation and slicing ops — `concat`, `slice`.

import ../tensor
import ../dispatch
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker

proc concat*(tensors: openArray[Tensor]; dimension: int): Tensor {.rewOp.} =
  ## Concatenates tensors along `dimension`. All must share dtype, device,
  ## and identical shape except along the concatenation axis.
  if tensors.len == 0:
    raise newException(TensorError, "concat: empty input list")
  let first = tensors[0]
  if dimension < 0 or dimension >= first.shape.len:
    raise newException(TensorError,
      "concat: dimension " & $dimension & " out of range for rank " &
        $first.shape.len)
  var totalDim = 0
  for i, t in tensors:
    if t.dtype != first.dtype:
      raise newException(TensorError,
        "concat: dtype mismatch at operand #" & $i)
    if t.shape.len != first.shape.len:
      raise newException(TensorError,
        "concat: rank mismatch at operand #" & $i)
    for j in 0 ..< t.shape.len:
      if j != dimension and t.shape[j] != first.shape[j]:
        raise newException(TensorError,
          "concat: shape mismatch at operand #" & $i & " dim " & $j)
    totalDim += t.shape[dimension]
  var outShape = first.shape
  outShape[dimension] = totalDim
  case currentMode()
  of dmTrace:
    for i, t in tensors:
      requireTrace(t, "concat")
    let ctx = currentTraceContext()
    var ids: seq[ShValueId] = @[]
    for t in tensors: ids.add t.traceId
    let id = shops.concatenate(ctx.builder, ids, dimension)
    result = initTraceTensor(id, first.dtype, outShape, first.device, first.sharding)
    recordTraceOp("concat", tensors, result,
      @[("dimension", @[dimension])])
  of dmEager:
    for t in tensors:
      requireEager(t, "concat")
    let outs = dispatchEager("concat", tensors,
      [("dimension", $dimension)])
    doAssert outs.len == 1, "concat: eager backend returned wrong arity"
    result = outs[0]

proc slice*(t: Tensor; startIndices, limitIndices: openArray[int];
    strides: openArray[int] = []): Tensor {.rewOp.} =
  ## Static slice of `t` with per-dimension start/limit/strides.
  ## If `strides` is empty, all-ones strides are used.
  let rank = t.shape.len
  if startIndices.len != rank or limitIndices.len != rank:
    raise newException(TensorError,
      "slice: index arrays must match operand rank " & $rank)
  var actualStrides = newSeq[int](rank)
  if strides.len == 0:
    for i in 0 ..< rank: actualStrides[i] = 1
  elif strides.len == rank:
    actualStrides = @strides
  else:
    raise newException(TensorError,
      "slice: strides length must be 0 or match rank " & $rank)
  var outShape = newSeq[int](rank)
  for i in 0 ..< rank:
    if startIndices[i] < 0 or limitIndices[i] > t.shape[i] or
       startIndices[i] >= limitIndices[i] or actualStrides[i] <= 0:
      raise newException(TensorError,
        "slice: invalid bounds at dim " & $i & " [" &
          $startIndices[i] & ":" & $limitIndices[i] & ":" &
          $actualStrides[i] & "] for size " & $t.shape[i])
    outShape[i] = (limitIndices[i] - startIndices[i] +
      actualStrides[i] - 1) div actualStrides[i]
  case currentMode()
  of dmTrace:
    requireTrace(t, "slice")
    let ctx = currentTraceContext()
    let id = shops.slice(ctx.builder, t.traceId,
      startIndices, limitIndices, actualStrides)
    result = initTraceTensor(id, t.dtype, outShape, t.device, t.sharding)
    recordTraceOp("slice", [t], result, @[
      ("startIndices", @startIndices),
      ("limitIndices", @limitIndices),
      ("strides", @actualStrides),
    ])
  of dmEager:
    requireEager(t, "slice")
    let outs = dispatchEager("slice", [t], [
      ("start_indices", $(@startIndices)),
      ("limit_indices", $(@limitIndices)),
      ("strides", $(@actualStrides)),
    ])
    doAssert outs.len == 1, "slice: eager backend returned wrong arity"
    result = outs[0]
