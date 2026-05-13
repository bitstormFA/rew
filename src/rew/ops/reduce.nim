## Reduction ops — `reduceSum`, `reduceMax`, `reduceMin`.
##
## ## v1 limitation
## Only `dtFloat32` is supported; the init constant for the reduction
## must be representable for that dtype. Other dtypes are accepted by
## the dispatcher but raise `TensorError` here. Broaden as needed when
## later layers (cross-entropy, integer reductions) require it.

import std/math
import ../tensor
import ../dispatch
import ../dtype
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker
import ./literal
import ./arith
import ./linalg
import ./unary
import ./compare
import ./ternary
import ./shape

proc reducedShape(shape: openArray[int]; dims: openArray[int]): seq[int] =
  result = @[]
  var dropped = newSeq[bool](shape.len)
  for d in dims: dropped[d] = true
  for i, s in shape:
    if not dropped[i]: result.add s

proc normalizeDims(opName: string; rank: int;
    dims: openArray[int]): seq[int] =
  if dims.len == 0:
    raise newException(TensorError, opName & ": dims must be non-empty")
  var seen = newSeq[bool](rank)
  for raw in dims:
    let d = if raw < 0: rank + raw else: raw
    if d < 0 or d >= rank:
      raise newException(TensorError,
        opName & ": dim " & $raw & " out of range for rank " & $rank)
    if seen[d]:
      raise newException(TensorError,
        opName & ": dim " & $raw & " repeated")
    seen[d] = true
    result.add d

proc allDims(rank: int): seq[int] =
  for i in 0 ..< rank:
    result.add i

proc normalizeDimsOrAll(opName: string; rank: int;
    dims: openArray[int]): seq[int] =
  if dims.len == 0:
    result = allDims(rank)
  else:
    result = normalizeDims(opName, rank, dims)

proc survivingDims(rank: int; dims: openArray[int]): seq[int] =
  var dropped = newSeq[bool](rank)
  for d in dims:
    dropped[d] = true
  for i in 0 ..< rank:
    if not dropped[i]:
      result.add i

proc requireFloat32(opName: string; dtype: DType) =
  if dtype != dtFloat32:
    raise newException(TensorError,
      opName & ": v1 supports only float32 (got " & $dtype & ")")

proc requireBool(opName: string; dtype: DType) =
  if dtype != dtBool:
    raise newException(TensorError,
      opName & ": expected bool tensor (got " & $dtype & ")")

proc float32Bytes(v: float32): seq[byte] =
  ## Little-endian raw bytes of `v`. Used for stablehlo.constant init
  ## values; the consumer parses by dtype.
  let bits = cast[uint32](v)
  result = newSeq[byte](4)
  result[0] = byte(bits and 0xFF'u32)
  result[1] = byte((bits shr 8) and 0xFF'u32)
  result[2] = byte((bits shr 16) and 0xFF'u32)
  result[3] = byte((bits shr 24) and 0xFF'u32)

proc dimsToString(dims: openArray[int]): string =
  result = "["
  for i, d in dims:
    if i > 0: result.add ','
    result.add $d
  result.add ']'

template reductionImpl(opName, reducerCall: untyped;
    initFloat32: float32): untyped {.dirty.} =
  ## Internal: validates, then either traces the reduction or routes to
  ## the eager backend. `reducerCall(b, x, y)` must emit an op into the
  ## reducer region and return its single SSA result.
  let reduceDims = normalizeDims(opName, t.shape.len, dims)
  requireFloat32(opName, t.dtype)
  let outShape = reducedShape(t.shape, reduceDims)
  case currentMode()
  of dmTrace:
    requireTrace(t, opName)
    let ctx = currentTraceContext()
    let initBytes = float32Bytes(initFloat32)
    let initId = ctx.builder.constant(t.dtype, [], initBytes)
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      reducerCall(b, lhs, rhs)
    let outId = ctx.builder.reduce(t.traceId, initId, reduceDims, body)
    result = initTraceTensor(outId, t.dtype, outShape, t.device, t.sharding)
    recordTraceOp(opName, [t], result, @[("dims", reduceDims)])
  of dmEager:
    requireEager(t, opName)
    let outs = dispatchEager(opName, [t],
      [("dims", dimsToString(reduceDims))])
    doAssert outs.len == 1, opName & ": eager backend returned wrong arity"
    result = outs[0]

proc reduceSum*(t: Tensor; dims: openArray[int]): Tensor {.rewOp.} =
  ## Sum-reduce `t` over the listed `dims`. Init value is `0.0f`.
  reductionImpl("reduceSum", shops.add, 0.0f32)

proc reduceMax*(t: Tensor; dims: openArray[int]): Tensor {.rewOp.} =
  ## Max-reduce `t` over the listed `dims`. Init value is `-Inf` so the
  ## reducer correctly picks up the largest finite value.
  reductionImpl("reduceMax", shops.maximum, NegInf.float32)

proc reduceMin*(t: Tensor; dims: openArray[int]): Tensor {.rewOp.} =
  ## Min-reduce `t` over the listed `dims`. Init value is `+Inf`.
  reductionImpl("reduceMin", shops.minimum, Inf.float32)

proc reduceMean*(t: Tensor; dims: openArray[int]): Tensor =
  ## Mean-reduce `t` over the listed `dims`. Composite over
  ## `reduceSum + divide`; no dedicated vjp (autograd composes through
  ## the primitive rules). v1 supports `dtFloat32` only \u2014 the
  ## divisor is materialised as a scalar `float32` constant and
  ## broadcast to the reduced shape.
  let reduceDims = normalizeDims("reduceMean", t.shape.len, dims)
  let summed = reduceSum(t, reduceDims)
  var n = 1
  for d in reduceDims: n *= t.shape[d]
  if n <= 0:
    raise newException(TensorError,
      "reduceMean: empty reduction (n = " & $n & ")")
  let divisor = scalarF32(float32(n), t.device)
  if summed.shape.len == 0:
    return divide(summed, divisor)
  var bdims: seq[int] = @[]  # 0-d operand: no surviving axes
  let bcast = broadcastTo(divisor, summed.shape, bdims)
  divide(summed, bcast)

proc reduceWindow*(input, initValue: Tensor;
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    baseDilations: openArray[int]; windowDilations: openArray[int];
    body: proc (b: var ShBuilder; lhs, rhs: ShValueId): ShValueId
      {.closure.}): Tensor {.rewOp.} =
  ## Generic window reduction. Slides a window over `input`, applying
  ## `body` as a binary element-wise reducer at each position. `initValue`
  ## must be a 0-rank tensor with the same dtype as `input`.
  case currentMode()
  of dmTrace:
    requireTrace(input, "reduceWindow")
    requireTrace(initValue, "reduceWindow")
    let ctx = currentTraceContext()
    let outShape = shops.reduceWindowOutputShape(input.shape, windowDimensions,
      windowStrides, padding, baseDilations, windowDilations)
    let id = shops.reduceWindow(ctx.builder, input.traceId,
      initValue.traceId, windowDimensions, windowStrides, padding,
      baseDilations, windowDilations, body)
    result = initTraceTensor(id, input.dtype, outShape, input.device,
      input.sharding)
    recordTraceOp("reduceWindow", [input, initValue], result)
  of dmEager:
    raise newException(TensorError,
      "reduceWindow: only supported in trace/jit mode")

# ---- reduction composites -------------------------------------------------

proc reduceProd*(t: Tensor; dims: openArray[int]): Tensor {.rewOp.} =
  ## Product-reduce `t` over the listed `dims`. Init value is `1.0`.
  let reduceDims = normalizeDims("reduceProd", t.shape.len, dims)
  requireFloat32("reduceProd", t.dtype)
  let outShape = reducedShape(t.shape, reduceDims)
  case currentMode()
  of dmTrace:
    requireTrace(t, "reduceProd")
    let ctx = currentTraceContext()
    let initId = ctx.builder.constant(t.dtype, [], float32Bytes(1'f32))
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      shops.mul(b, lhs, rhs)
    let outId = ctx.builder.reduce(t.traceId, initId, reduceDims, body)
    result = initTraceTensor(outId, t.dtype, outShape, t.device, t.sharding)
    recordTraceOp("reduceProd", [t], result, @[("dims", reduceDims)])
  of dmEager:
    requireEager(t, "reduceProd")
    let outs = dispatchEager("reduceProd", [t],
      [("dims", dimsToString(reduceDims))])
    doAssert outs.len == 1, "reduceProd: eager backend returned wrong arity"
    result = outs[0]

proc argmax*(t: Tensor; dim: int; keepdims = false): Tensor =
  ## Return indices of the maximum values along `dim`. Ties go to the
  ## smallest index.
  let pos = if dim < 0: t.shape.len + dim else: dim
  if pos < 0 or pos >= t.shape.len:
    raise newException(TensorError,
      "argmax: dim " & $dim & " out of range for rank " & $t.shape.len)
  let maxVals = reduceMax(t, [pos])
  # Expand max back for comparison.
  var bdims: seq[int] = @[]
  for i in 0 ..< t.shape.len:
    if i != pos: bdims.add i
  let maxB = broadcastTo(maxVals, t.shape, bdims)
  let mask = compare(t, maxB, "EQ")
  # Create index grid and select where max achieved, then take min index.
  let indices = iota(dtInt32, t.shape, pos, t.device)
  let largeVal = scalarI32(int32.high, t.device)
  var zeroDims: seq[int] = @[]
  let largeBroad = broadcastTo(largeVal, t.shape, zeroDims)
  let maskedIndices = select(mask, indices, largeBroad)
  let resultMin = reduceMin(maskedIndices, [pos])
  if keepdims:
    return unsqueeze(resultMin, pos)
  return resultMin

proc argmin*(t: Tensor; dim: int; keepdims = false): Tensor =
  ## Return indices of the minimum values along `dim`. Ties go to the
  ## smallest index.
  let pos = if dim < 0: t.shape.len + dim else: dim
  if pos < 0 or pos >= t.shape.len:
    raise newException(TensorError,
      "argmin: dim " & $dim & " out of range for rank " & $t.shape.len)
  let minVals = reduceMin(t, [pos])
  var bdims: seq[int] = @[]
  for i in 0 ..< t.shape.len:
    if i != pos: bdims.add i
  let minB = broadcastTo(minVals, t.shape, bdims)
  let mask = compare(t, minB, "EQ")
  let indices = iota(dtInt32, t.shape, pos, t.device)
  let largeVal = scalarI32(int32.high, t.device)
  var zeroDims: seq[int] = @[]
  let largeBroad = broadcastTo(largeVal, t.shape, zeroDims)
  let maskedIndices = select(mask, indices, largeBroad)
  let resultMin = reduceMin(maskedIndices, [pos])
  if keepdims:
    return unsqueeze(resultMin, pos)
  return resultMin

proc restoreReducedDims(t: Tensor; rank: int; dims: openArray[int]): Tensor =
  result = t
  var dropped = newSeq[bool](rank)
  for d in dims:
    dropped[d] = true
  for dim in 0 ..< rank:
    if dropped[dim]:
      result = unsqueeze(result, dim)

template boolReductionImpl(opName, reducerCall: untyped;
    initValue: bool): untyped {.dirty.} =
  requireBool(opName, t.dtype)
  let reduceDims = normalizeDims(opName, t.shape.len, dims)
  let outShape = reducedShape(t.shape, reduceDims)
  case currentMode()
  of dmTrace:
    requireTrace(t, opName)
    let ctx = currentTraceContext()
    let initBytes = @[byte(if initValue: 1'u8 else: 0'u8)]
    let initId = ctx.builder.constant(dtBool, [], initBytes)
    let body = proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      reducerCall(b, lhs, rhs)
    let outId = ctx.builder.reduce(t.traceId, initId, reduceDims, body)
    result = initTraceTensor(outId, dtBool, outShape, t.device, t.sharding)
    recordTraceOp(opName, [t], result, @[("dims", reduceDims)])
  of dmEager:
    requireEager(t, opName)
    let outs = dispatchEager(opName, [t],
      [("dims", dimsToString(reduceDims))])
    doAssert outs.len == 1, opName & ": eager backend returned wrong arity"
    result = outs[0]
  if keepdims:
    result = restoreReducedDims(result, t.shape.len, reduceDims)

proc all*(t: Tensor; dims: openArray[int]; keepdims = false): Tensor
    {.rewOp.} =
  ## Test whether all elements along `dims` are true.
  boolReductionImpl("all", shops.andOp, true)

proc all*(t: Tensor; dim: int; keepdims = false): Tensor =
  ## Test whether all elements along `dim` are true.
  all(t, [dim], keepdims)

proc any*(t: Tensor; dims: openArray[int]; keepdims = false): Tensor
    {.rewOp.} =
  ## Test whether any element along `dims` is true.
  boolReductionImpl("any", shops.orOp, false)

proc any*(t: Tensor; dim: int; keepdims = false): Tensor =
  ## Test whether any element along `dim` is true.
  any(t, [dim], keepdims)

proc logSumExp*(t: Tensor; dims: openArray[int]): Tensor =
  ## Numerically stable log-sum-exp over `dims`:
  ## `max + log(sum(exp(x - max)))`.
  let reduceDims = normalizeDims("logSumExp", t.shape.len, dims)
  let maxVals = reduceMax(t, reduceDims)
  let bdims = survivingDims(t.shape.len, reduceDims)
  let maxB = broadcastTo(maxVals, t.shape, bdims)
  let shifted = sub(t, maxB)
  let expShifted = exp(shifted)
  let sumExp = reduceSum(expShifted, reduceDims)
  let logSum = log(sumExp)
  add(maxVals, logSum)

proc variance*(t: Tensor; dims: openArray[int]; unbiased = true): Tensor =
  ## Variance over `dims`. Uses Bessel's correction when `unbiased`.
  let reduceDims = normalizeDims("variance", t.shape.len, dims)
  let meanVals = reduceMean(t, reduceDims)
  let bdims = survivingDims(t.shape.len, reduceDims)
  let meanB = broadcastTo(meanVals, t.shape, bdims)
  let diff = sub(t, meanB)
  let sq = mul(diff, diff)
  let sumSq = reduceSum(sq, reduceDims)
  var n = 1
  for d in reduceDims: n *= t.shape[d]
  if unbiased and n > 1:
    let denom = scalarF32(float32(n - 1), t.device)
    var zeroDims: seq[int] = @[]
    let denomB = broadcastTo(denom, sumSq.shape, zeroDims)
    divide(sumSq, denomB)
  else:
    let denom = scalarF32(float32(n), t.device)
    var zeroDims: seq[int] = @[]
    let denomB = broadcastTo(denom, sumSq.shape, zeroDims)
    divide(sumSq, denomB)

proc std*(t: Tensor; dims: openArray[int]; unbiased = true): Tensor =
  ## Standard deviation over `dims`.
  let v = variance(t, dims, unbiased)
  sqrt(v)

proc norm*(t: Tensor; p: float32 = 2'f32; dims: openArray[int] = []): Tensor =
  ## Vector norm over `dims` (or all dims if empty). p=2 is the Euclidean
  ## norm.
  if p <= 0'f32:
    raise newException(TensorError,
      "norm: p must be positive, got " & $p)
  let dimsToReduce = normalizeDimsOrAll("norm", t.shape.len, dims)
  if p == 2'f32:
    return sqrt(reduceSum(mul(t, t), dimsToReduce))
  if p == 1'f32:
    return reduceSum(abs(t), dimsToReduce)
  let pScalar = scalarF32(p, t.device)
  var zeroDims: seq[int] = @[]
  let pB = broadcastTo(pScalar, t.shape, zeroDims)
  let absPow = power(abs(t), pB)
  let summed = reduceSum(absPow, dimsToReduce)
  let invP = scalarF32(1'f32 / p, t.device)
  let invPB = broadcastTo(invP, summed.shape, zeroDims)
  power(summed, invPB)

proc countNonzero*(t: Tensor; dims: openArray[int] = []): Tensor =
  ## Count non-zero elements along `dims`.
  let dimsToReduce = normalizeDimsOrAll("countNonzero", t.shape.len, dims)
  let zeroB = scalarF32(0'f32, t.device)
  var zeroDims: seq[int] = @[]
  let zeroBB = broadcastTo(zeroB, t.shape, zeroDims)
  let mask = compare(t, zeroBB, "NE")
  let ones = astype(mask, dtFloat32)
  reduceSum(ones, dimsToReduce)

proc norm*(t: Tensor; ord: string; dims: openArray[int] = [];
    keepdims: bool = false): Tensor =
  ## Matrix or vector norm. `ord` is one of `"fro"`, `"nuc"`, `"1"`,
  ## `"2"`, `"inf"`. When `dims` has two entries a matrix norm is
  ## computed, otherwise `dims` (or all dims if empty) are treated
  ## as vector dimensions.
  ##
  ## Only `"fro"`, `"1"` and `"2"` are supported in v1; `"nuc"`
  ## needs SVD and `"inf"` is unreachable through a single reduce.
  let dimsToReduce = normalizeDimsOrAll("norm", t.shape.len, dims)
  var res: Tensor
  case ord
  of "fro", "2":
    res = sqrt(reduceSum(mul(t, t), dimsToReduce))
  of "1":
    res = reduceSum(abs(t), dimsToReduce)
  else:
    raise newException(TensorError,
      "norm: unsupported ord '" & ord & "' in v1 (use \"fro\", \"1\", or \"2\")")
  if keepdims:
    var sd = dimsToReduce
    for i in 0 ..< sd.len:
      for j in i + 1 ..< sd.len:
        if sd[i] < sd[j]: swap(sd[i], sd[j])
    for d in sd:
      res = unsqueeze(res, d)
  res

proc trace*(a: Tensor; offset: int = 0; axis1: int = 0;
    axis2: int = 1): Tensor =
  ## Sum of elements along the diagonal defined by `axis1` and
  ## `axis2`, optionally shifted by `offset`. A positive `offset`
  ## selects the super-diagonal, a negative `offset` the
  ## sub-diagonal. The result has rank `rank(a) - 2`.
  let rank = a.shape.len
  let ax1 = if axis1 < 0: rank + axis1 else: axis1
  let ax2 = if axis2 < 0: rank + axis2 else: axis2
  if ax1 < 0 or ax1 >= rank or ax2 < 0 or ax2 >= rank:
    raise newException(TensorError,
      "trace: axes out of range for rank " & $rank)
  if ax1 == ax2:
    raise newException(TensorError,
      "trace: axis1 and axis2 must be distinct")
  # Build iota along axis1 and axis2, broadcast to full shape.
  var iotaShape1 = newSeq[int](rank)
  var iotaShape2 = newSeq[int](rank)
  for i in 0 ..< rank:
    iotaShape1[i] = 1; iotaShape2[i] = 1
  iotaShape1[ax1] = a.shape[ax1]
  iotaShape2[ax2] = a.shape[ax2]
  let i1 = broadcastTo(
    iota(dtInt32, iotaShape1, ax1, a.device), a.shape, @[ax1])
  let i2 = broadcastTo(
    iota(dtInt32, iotaShape2, ax2, a.device), a.shape, @[ax2])
  # Build shifted i1: i1 + offset == i2 picks the diagonal.
  var i1Shifted = i1
  if offset != 0:
    let off = scalarI32(int32(offset), a.device)
    var offB = broadcastTo(off, i1.shape, [])
    i1Shifted = add(i1Shifted, astype(offB, dtInt32))
  else:
    i1Shifted = i1
  let mask = compare(i1Shifted, i2, "EQ")
  let masked = mul(a, astype(mask, a.dtype))
  var reduceDims = @[ax1, ax2]
  if reduceDims[0] < reduceDims[1]:
    swap(reduceDims[0], reduceDims[1])
  reduceSum(masked, reduceDims)

proc det*(a: Tensor; lower = true): Tensor =
  ## Determinant of symmetric positive-definite matrices via
  ## Cholesky. `a` has shape `[..., N, N]`. Returns `[...]`
  ## (one scalar per matrix).
  ##
  ## Uses the identity `det(A) = prod(diag(L))^2` where
  ## `L` is the Cholesky factor. Requires `dtFloat32` in v1.
  if not a.dtype.isFloat:
    raise newException(TensorError,
      "det: operand must be floating point, got " & $a.dtype)
  if a.shape.len < 2:
    raise newException(TensorError,
      "det: rank must be at least 2, got " & $a.shape.len)
  if a.shape[^1] != a.shape[^2]:
    raise newException(TensorError,
      "det: innermost dims must be square, got " & $a.shape)
  let L = cholesky(a, lower)
  let n = a.shape[^1]
  let rank = L.shape.len
  # Diagonal mask for innermost [n, n].
  let rows = iota(dtInt32, [n, n], 0, a.device)
  let cols = iota(dtInt32, [n, n], 1, a.device)
  var diagMask = compare(rows, cols, "EQ")
  if rank > 2:
    diagMask = broadcastTo(diagMask, L.shape,
      [rank - 2, rank - 1])
  # Replace non-diagonal elements with 1 so log(diag) survives
  # and log(1) = 0 contributes nothing to the sum.
  let oneScalar = scalarF32(1'f32, a.device)
  var one = broadcastTo(oneScalar, L.shape, [])
  if a.dtype != dtFloat32:
    one = astype(one, a.dtype)
  let Lsafe = select(diagMask, L, one)
  let sumLogDiag = reduceSum(log(Lsafe), [rank - 2, rank - 1])
  let two = broadcastTo(scalarF32(2'f32, a.device), sumLogDiag.shape, [])
  exp(mul(two, sumLogDiag))

proc cumsum*(t: Tensor; dim: int): Tensor =
  ## Cumulative sum along `dim`. Computed via a lower-triangular
  ## matrix multiply. Works for any floating-point dtype.
  let pos = normalizeDims("cumsum", t.shape.len, [dim])[0]
  let axSize = t.shape[pos]
  # Build lower-triangular matrix [axSize, axSize].
  let trilMask = compare(
    iota(dtInt32, [axSize, axSize], 0, t.device),
    iota(dtInt32, [axSize, axSize], 1, t.device), "GE")
  let tril = astype(trilMask, t.dtype)
  # Permute target dim to last position.
  var perm: seq[int] = @[]
  for i in 0 ..< t.shape.len:
    if i != pos: perm.add i
  perm.add pos
  let tT = transpose(t, perm)
  # Flatten leading dims into a single batch dimension.
  var batchSize = 1
  for i in 0 ..< tT.shape.len - 1:
    batchSize *= tT.shape[i]
  let tFlat = reshape(tT, @[batchSize, axSize])
  let tCol = unsqueeze(tFlat, 2)   # [batchSize, axSize, 1]
  let tril3d = unsqueeze(tril, 0)   # [1, axSize, axSize]
  # Batched matmul: contract axSize, batch over batchSize/1.
  let res = dotGeneral(tCol, tril3d, [0], [0], [1], [1])
  let resSq = squeeze(res, 1)       # [batchSize, axSize]
  let resT = reshape(resSq, tT.shape)
  # Inverse permute.
  var invPerm = newSeq[int](perm.len)
  for i, p in perm: invPerm[p] = i
  transpose(resT, invPerm)

proc cumprod*(t: Tensor; dim: int): Tensor =
  ## Cumulative product along `dim`. Computed via `exp(cumsum(log(t)))`.
  ## Requires all elements of `t` to be strictly positive.
  exp(cumsum(log(t), dim))

proc diag*(a: Tensor): Tensor =
  ## Extract the main diagonal of the innermost two dimensions.
  ## For a `[..., M, N]` tensor returns `[..., min(M, N)]`.
  if a.shape.len < 2:
    raise newException(TensorError,
      "diag: rank must be at least 2, got " & $a.shape.len)
  let rank = a.shape.len
  let m = a.shape[rank - 2]
  let n = a.shape[rank - 1]
  let rows = iota(dtInt32, [m, n], 0, a.device)
  let cols = iota(dtInt32, [m, n], 1, a.device)
  var mask2d = compare(rows, cols, "EQ")
  if rank > 2:
    mask2d = broadcastTo(mask2d, a.shape, [rank - 2, rank - 1])
  let masked = mul(a, astype(mask2d, a.dtype))
  reduceSum(masked, @[rank - 1])

proc average*(t: Tensor; dims: openArray[int] = []): Tensor =
  ## Weighted average over `dims`. When called with no dims
  ## this reduces over every dimension.
  if dims.len == 0 and t.shape.len == 0:
    return t
  reduceMean(t, normalizeDimsOrAll("average", t.shape.len, dims))

proc cov*(x: Tensor; rowvar: bool = true): Tensor =
  ## Covariance matrix. Rows are variables by default (`rowvar=true`);
  ## set `rowvar=false` when columns are variables.
  if x.shape.len != 2:
    raise newException(TensorError,
      "cov: expected a rank-2 tensor, got shape " & $x.shape)
  var data = x
  if not rowvar:
    data = transpose(x, [1, 0])
  # data shape: [nObs, nVars]
  let n = data.shape[0]
  if n <= 1:
    raise newException(TensorError,
      "cov: need at least two observations, got " & $n)
  let centered = sub(data, reduceMean(data, @[0]))
  # cov = centered^T @ centered / (n - 1)
  let covMat = matmul(transpose(centered, [1, 0]), centered)
  let denom = broadcastTo(scalarF32(float32(n - 1), x.device),
    covMat.shape, @[])
  divide(covMat, denom)

proc corrcoef*(x: Tensor; rowvar: bool = true): Tensor =
  ## Pearson correlation coefficient matrix. Same convention as
  ## `cov`: rows are variables by default.
  if x.shape.len != 2:
    raise newException(TensorError,
      "corrcoef: expected a rank-2 tensor, got shape " & $x.shape)
  var data = x
  if not rowvar:
    data = transpose(x, [1, 0])
  let n = data.shape[0]
  if n <= 1:
    raise newException(TensorError,
      "corrcoef: need at least two observations, got " & $n)
  let centered = sub(data, reduceMean(data, @[0]))
  let covMat = matmul(transpose(centered, [1, 0]), centered)
  let denom = scalarF32(float32(n - 1), x.device)
  var denomB = broadcastTo(denom, covMat.shape, @[])
  let covNorm = divide(covMat, denomB)
  # Correlation: cov / outer(std, std)
  let variance = reduceMean(mul(centered, centered), @[0])
  let std = sqrt(variance)
  let outerStd = mul(unsqueeze(std, 1), unsqueeze(std, 0))
  divide(covNorm, outerStd)
