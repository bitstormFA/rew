## Reverse-mode rules for the primitive ops.
##
## Each rule is a pure function from `(primals, output, cotangent,
## intAttrs)` to a `seq[Tensor]` of input cotangents. The transform
## layer pauses the tape before invoking a rule, so the primitive ops a
## rule emits do not pollute the tape.
##
## Composite ops (relu, mseLoss, Linear.forward, Sgd.step) deliberately
## have no rule \u2014 they decompose into primitives at trace time and
## the transform differentiates through that decomposition.
##
## Rules that need attributes (`reshape`, `transpose`, `reduceSum`,
## `dotGeneral`, `broadcastTo`) read them from `intAttrs` by key.

import ../tensor
import ../dispatch
import ../dtype
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/shape
import ../ops/reduce
import ../ops/linalg
import ../ops/concat
import ./registry
import ./tape

# ---- attr helpers ---------------------------------------------------------

proc lookup(attrs: IntAttrs; key: string): seq[int] =
  for (k, v) in attrs:
    if k == key: return v
  raise newException(VjpRegistryError,
    "vjp rule: missing required int-attr '" & key & "'")

proc scalarLike(t: Tensor; value: float32): Tensor =
  ## Builds a scalar constant broadcast to the shape of `t`. v1 supports
  ## float32 only; broaden when more dtypes ship.
  let scalar = scalarF32(value)
  if t.shape.len == 0:
    return scalar
  var dims: seq[int] = @[]  # 0-d operand: empty broadcast_dimensions
  broadcastTo(scalar, t.shape, dims)

proc onesLike(t: Tensor): Tensor =
  ## Builds a constant `1.0` tensor with the shape and dtype of `t`.
  scalarLike(t, 1'f32)

proc zerosLike(t: Tensor): Tensor =
  ## Constant `0.0` tensor matching `t`. Used by the maximum/minimum
  ## rules to mask out the loser branch.
  scalarLike(t, 0'f32)

proc broadcastScalarLike(value, target: Tensor): Tensor =
  if value.shape.len == 0 and target.shape.len != 0:
    var dims: seq[int] = @[]
    return broadcastTo(value, target.shape, dims)
  value

proc reduceToShape(value, target: Tensor): Tensor =
  if target.shape.len == 0 and value.shape.len != 0:
    var dims: seq[int] = @[]
    for i in 0 ..< value.shape.len:
      dims.add i
    return reduceSum(value, dims)
  value

proc zeroBlockLike(t: Tensor; dim, size: int): Tensor =
  var zShape = t.shape
  zShape[dim] = size
  let z = scalarF32(0'f32)
  var bdims: seq[int] = @[]
  broadcastTo(z, zShape, bdims)

proc appendZeroBlock(parts: var seq[Tensor]; t: Tensor; dim, size: int) =
  if size > 0:
    parts.add zeroBlockLike(t, dim, size)

proc unitSliceGradDim(t: Tensor; dim, origDim, start, limit: int): Tensor =
  var parts: seq[Tensor] = @[]
  appendZeroBlock(parts, t, dim, start)
  parts.add t
  appendZeroBlock(parts, t, dim, origDim - limit)
  if parts.len == 1:
    parts[0]
  else:
    concat(parts, dim)

proc sliceOneDim(t: Tensor; dim, index: int): Tensor =
  var starts = newSeq[int](t.shape.len)
  var limits = t.shape
  var strides = newSeq[int](t.shape.len)
  for i in 0 ..< t.shape.len:
    strides[i] = 1
  starts[dim] = index
  limits[dim] = index + 1
  slice(t, starts, limits, strides)

proc stridedSliceGradDim(t: Tensor; dim, origDim, start, limit,
    stride: int): Tensor =
  let selected = t.shape[dim]
  let expected = (limit - start + stride - 1) div stride
  if selected != expected:
    raise newException(VjpRegistryError,
      "slice vjp: cotangent dim " & $dim & " has size " & $selected &
        ", expected " & $expected)
  var parts: seq[Tensor] = @[]
  appendZeroBlock(parts, t, dim, start)
  for j in 0 ..< selected:
    parts.add sliceOneDim(t, dim, j)
    let pos = start + j * stride
    let gap =
      if j + 1 < selected: stride - 1
      else: origDim - pos - 1
    appendZeroBlock(parts, t, dim, gap)
  if parts.len == 1:
    parts[0]
  else:
    concat(parts, dim)

proc compareTrace(a, b: Tensor; direction: string): Tensor =
  ## Trace-mode `stablehlo.compare`, inlined here so this module does not
  ## need to import `transform/control` (which would create a layer
  ## cycle). Used only by the maximum/minimum vjps; both operands are
  ## guaranteed to be matching trace tensors at rule time.
  let ctx = currentTraceContext()
  let id = shops.compare(ctx.builder, a.traceId, b.traceId, direction)
  initTraceTensor(id, dtBool, a.shape, a.device, a.sharding)

proc selectTrace(pred, a, b: Tensor): Tensor =
  ## Trace-mode `stablehlo.select`. Same rationale as `compareTrace`.
  let ctx = currentTraceContext()
  let id = shops.select(ctx.builder, pred.traceId, a.traceId, b.traceId)
  initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)

proc reductionBroadcastDims(rank: int; dims: openArray[int]): seq[int] =
  var dropped = newSeq[bool](rank)
  for d in dims: dropped[d] = true
  for i in 0 ..< rank:
    if not dropped[i]: result.add i

proc matchShape(t: Tensor; targetShape: openArray[int]): Tensor =
  ## Crops and high-pads static convolution transpose results so VJP
  ## cotangents exactly match the primal shape when forward stride floors.
  result = t
  var needsSlice = false
  var starts = newSeq[int](result.shape.len)
  var limits = newSeq[int](result.shape.len)
  var strides = newSeq[int](result.shape.len)
  for i in 0 ..< result.shape.len:
    starts[i] = 0
    limits[i] = min(result.shape[i], targetShape[i])
    strides[i] = 1
    if result.shape[i] > targetShape[i]:
      needsSlice = true
  if needsSlice:
    result = slice(result, starts, limits, strides)

  var needsPad = false
  var lowPads = newSeq[int](result.shape.len)
  var highPads = newSeq[int](result.shape.len)
  var interiorPads = newSeq[int](result.shape.len)
  for i in 0 ..< result.shape.len:
    if result.shape[i] < targetShape[i]:
      highPads[i] = targetShape[i] - result.shape[i]
      needsPad = true
  if needsPad:
    result = pad(result, scalarF32(0'f32, result.device),
      lowPads, highPads, interiorPads)

proc maxSelectAndScatterTrace(operand, source: Tensor;
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]]): Tensor =
  ## Trace-mode max-pool transpose. This mirrors StableHLO's canonical
  ## max-pool gradient: select the maximum element in each window and
  ## accumulate the corresponding source cotangent into the operand shape.
  let initValue = scalarF32(0'f32, operand.device)
  let ctx = currentTraceContext()
  let id = shops.selectAndScatter(ctx.builder, operand.traceId,
    source.traceId, initValue.traceId, windowDimensions, windowStrides,
    padding,
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[shops.compare(b, xs[0], xs[1], "GE")],
    proc(b: var ShBuilder; xs: openArray[ShValueId]): seq[ShValueId] =
      @[shops.add(b, xs[0], xs[1])])
  initTraceTensor(id, operand.dtype, operand.shape, operand.device,
    operand.sharding)

# ---- rules ----------------------------------------------------------------

proc registerArithRules() =
  registerVjpRule("add", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[cotangent, cotangent])

  registerVjpRule("sub", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[cotangent, neg(cotangent)])

  registerVjpRule("mul", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[mul(cotangent, primals[1]), mul(cotangent, primals[0])])

  registerVjpRule("neg", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[neg(cotangent)])

  registerVjpRule("divide", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let a = primals[0]
    let b = primals[1]
    let dA = divide(cotangent, b)
    let bSq = mul(b, b)
    let dB = neg(divide(mul(cotangent, a), bSq))
    @[dA, dB])

  registerVjpRule("atan2", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let y = primals[0]
    let x = primals[1]
    let denom = add(mul(y, y), mul(x, x))
    @[divide(mul(cotangent, x), denom),
      neg(divide(mul(cotangent, y), denom))])

  registerVjpRule("power", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let base = primals[0]
    let exponent = primals[1]
    let one = onesLike(exponent)
    let dBase = mul(cotangent,
      mul(exponent, power(base, sub(exponent, one))))
    let dExponent = mul(cotangent, mul(output, log(base)))
    @[dBase, dExponent])

proc registerUnaryRules() =
  registerVjpRule("optimizationBarrier",
    proc(primals: openArray[Tensor]; output: Tensor;
        cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
      @[cotangent])

  registerVjpRule("stopGradient",
    proc(primals: openArray[Tensor]; output: Tensor;
        cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
      @[zerosLike(primals[0])])

  registerVjpRule("exp", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[mul(cotangent, output)])

  registerVjpRule("log", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    @[divide(cotangent, primals[0])])

  registerVjpRule("sqrt", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # d/dx sqrt(x) = 0.5 / sqrt(x); express as cotangent / (2 * out).
    let two = add(output, output)
    @[divide(cotangent, two)])

  registerVjpRule("tanh", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let outSq = mul(output, output)
    let one = onesLike(output)
    @[mul(cotangent, sub(one, outSq))])

  registerVjpRule("cbrt", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let outSq = mul(output, output)
    let three = scalarLike(output, 3'f32)
    @[divide(cotangent, mul(three, outSq))])

  registerVjpRule("expm1", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let one = onesLike(output)
    @[mul(cotangent, add(output, one))])

  registerVjpRule("log1p", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let one = onesLike(primals[0])
    @[divide(cotangent, add(one, primals[0]))])

  registerVjpRule("logistic", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let one = onesLike(output)
    @[mul(cotangent, mul(output, sub(one, output)))])

  registerVjpRule("tan", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let one = onesLike(output)
    @[mul(cotangent, add(one, mul(output, output)))])

proc registerMinMaxRules() =
  # `maximum` / `minimum` split the cotangent at the win boundary. Ties
  # go entirely to the first operand so the two rules sum to the
  # cotangent (matches JAX's convention).
  registerVjpRule("maximum", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let a = primals[0]
    let b = primals[1]
    let zero = zerosLike(a)
    let aWins = compareTrace(a, b, "GE")
    let dA = selectTrace(aWins, cotangent, zero)
    let dB = sub(cotangent, dA)
    @[dA, dB])

  registerVjpRule("minimum", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let a = primals[0]
    let b = primals[1]
    let zero = zerosLike(a)
    let aWins = compareTrace(a, b, "LE")
    let dA = selectTrace(aWins, cotangent, zero)
    let dB = sub(cotangent, dA)
    @[dA, dB])

proc invertPermutation(perm: openArray[int]): seq[int] =
  result = newSeq[int](perm.len)
  for i, p in perm:
    result[p] = i

proc registerShapeRules() =
  registerVjpRule("reshape", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    @[reshape(cotangent, primals[0].shape)])

  registerVjpRule("transpose", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let perm = lookup(intAttrs, "permutation")
    @[transpose(cotangent, invertPermutation(perm))])

  registerVjpRule("reverse", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    @[reverse(cotangent, lookup(intAttrs, "dimensions"))])

# ---- linalg / reduce ------------------------------------------------------

proc registerLinalgRules() =
  registerVjpRule("dot", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let a = primals[0]
    let b = primals[1]
    if a.shape.len == 2 and b.shape.len == 2:
      return @[dot(cotangent, transpose(b, [1, 0])),
        dot(transpose(a, [1, 0]), cotangent)]
    if a.shape.len == 1 and b.shape.len == 1:
      let cot = broadcastTo(cotangent, a.shape, [])
      return @[mul(cot, b), mul(cot, a)]
    if a.shape.len == 2 and b.shape.len == 1:
      let cotCol = reshape(cotangent, [a.shape[0], 1])
      let bRow = reshape(b, [1, b.shape[0]])
      let cotB = broadcastTo(cotCol, a.shape, [0, 1])
      let bB = broadcastTo(bRow, a.shape, [0, 1])
      return @[mul(cotB, bB), dot(transpose(a, [1, 0]), cotangent)]
    if a.shape.len == 1 and b.shape.len == 2:
      let aCol = reshape(a, [a.shape[0], 1])
      let cotRow = reshape(cotangent, [1, cotangent.shape[0]])
      let aB = broadcastTo(aCol, b.shape, [0, 1])
      let cotB = broadcastTo(cotRow, b.shape, [0, 1])
      return @[dot(cotangent, transpose(b, [1, 0])), mul(aB, cotB)]
    raise newException(VjpRegistryError,
      "dot vjp: expected rank-1/rank-2 operands"))

  registerVjpRule("matmul", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let a = primals[0]
    let b = primals[1]
    @[matmul(cotangent, transpose(b, [1, 0])),
      matmul(transpose(a, [1, 0]), cotangent)])

  registerVjpRule("broadcastTo", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let bdims = lookup(intAttrs, "broadcastDimensions")
    let inShape = primals[0].shape
    let outRank = output.shape.len
    # Reduce out the axes that aren't in `broadcastDimensions` (those
    # were created by the broadcast). Then reshape size-1 axes the
    # broadcast may have stretched.
    var reduceDims: seq[int] = @[]
    var inBdims = newSeq[bool](outRank)
    for d in bdims: inBdims[d] = true
    for d in 0 ..< outRank:
      if not inBdims[d]: reduceDims.add d
    var reduced = cotangent
    if reduceDims.len > 0:
      reduced = reduceSum(cotangent, reduceDims)
    # `reduced` now has the surviving axes in the same order as `bdims`
    # which equals the operand axis order, so reshape to `inShape` to
    # fold in any size-1 dims that were broadcast up.
    if reduced.shape != @inShape:
      reduced = reshape(reduced, inShape)
    @[reduced])

proc registerReduceRules() =
  registerVjpRule("reduceSum", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let dims = lookup(intAttrs, "dims")
    let inShape = primals[0].shape
    # broadcast_in_dim: surviving operand axes map to the corresponding
    # output positions; reduced axes don't appear in the cotangent and
    # are introduced by the broadcast.
    let bdims = reductionBroadcastDims(inShape.len, dims)
    @[broadcastTo(cotangent, inShape, bdims)])

  registerVjpRule("reduceProd", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let dims = lookup(intAttrs, "dims")
    let x = primals[0]
    let bdims = reductionBroadcastDims(x.shape.len, dims)
    let zero = zerosLike(x)
    let one = onesLike(x)
    let zeroMask = compareTrace(x, zero, "EQ")
    let zeroIndicators = selectTrace(zeroMask, one, zero)
    let zeroCount = reduceSum(zeroIndicators, dims)
    let zeroCountB = broadcastTo(zeroCount, x.shape, bdims)
    let noZeros = compareTrace(zeroCountB, zero, "EQ")
    let oneZero = compareTrace(zeroCountB, one, "EQ")

    let outputB = broadcastTo(output, x.shape, bdims)
    let cotB = broadcastTo(cotangent, x.shape, bdims)
    let safeX = selectTrace(zeroMask, one, x)
    let noZeroGrad = divide(mul(cotB, outputB), safeX)

    let nonzeroProd = reduceProd(safeX, dims)
    let nonzeroProdB = broadcastTo(nonzeroProd, x.shape, bdims)
    let onlyZeroGetsGrad = selectTrace(zeroMask,
      mul(cotB, nonzeroProdB), zero)
    let zeroOrOneZeroGrad = selectTrace(oneZero, onlyZeroGetsGrad, zero)
    @[selectTrace(noZeros, noZeroGrad, zeroOrOneZeroGrad)])

proc registerAbsRule() =
  registerVjpRule("abs", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # d/dx |x| = sign(x). sign(x) = 1 if x>0, -1 if x<0, 0 if x==0.
    let x = primals[0]
    let zero = zerosLike(x)
    let one = onesLike(x)
    let positiveMask = compareTrace(x, zero, "GT")
    let negativeMask = compareTrace(x, zero, "LT")
    let posContrib = selectTrace(positiveMask, one, zero)
    let negContrib = selectTrace(negativeMask, neg(one), zero)
    let sign = add(posContrib, negContrib)
    @[mul(cotangent, sign)])

proc registerReduceMaxMinRules() =
  registerVjpRule("reduceMax", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let dims = lookup(intAttrs, "dims")
    let inShape = primals[0].shape
    # Broadcast the reduced output back to the input shape, then mask
    # to positions that achieved the max.
    var dropped = newSeq[bool](inShape.len)
    for d in dims: dropped[d] = true
    var bdims: seq[int] = @[]
    for i in 0 ..< inShape.len:
      if not dropped[i]: bdims.add i
    let outBcast = broadcastTo(output, inShape, bdims)
    let mask = compareTrace(primals[0], outBcast, "EQ")
    # Count how many elements achieved the max (for ties, distribute evenly).
    let one = onesLike(primals[0])
    let zero = zerosLike(primals[0])
    let indicators = selectTrace(mask, one, zero)
    let count = reduceSum(indicators, dims)
    let countBcast = broadcastTo(count, inShape, bdims)
    let cotBcast = broadcastTo(cotangent, inShape, bdims)
    let scaled = divide(cotBcast, countBcast)
    @[selectTrace(mask, scaled, zero)])

  registerVjpRule("reduceMin", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let dims = lookup(intAttrs, "dims")
    let inShape = primals[0].shape
    var dropped = newSeq[bool](inShape.len)
    for d in dims: dropped[d] = true
    var bdims: seq[int] = @[]
    for i in 0 ..< inShape.len:
      if not dropped[i]: bdims.add i
    let outBcast = broadcastTo(output, inShape, bdims)
    let mask = compareTrace(primals[0], outBcast, "EQ")
    let one = onesLike(primals[0])
    let zero = zerosLike(primals[0])
    let indicators = selectTrace(mask, one, zero)
    let count = reduceSum(indicators, dims)
    let countBcast = broadcastTo(count, inShape, bdims)
    let cotBcast = broadcastTo(cotangent, inShape, bdims)
    let scaled = divide(cotBcast, countBcast)
    @[selectTrace(mask, scaled, zero)])

proc registerDotGeneralRule() =
  registerVjpRule("dotGeneral", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let lhsBatching = lookup(intAttrs, "lhsBatching")
    let rhsBatching = lookup(intAttrs, "rhsBatching")
    let lhsContracting = lookup(intAttrs, "lhsContracting")
    let rhsContracting = lookup(intAttrs, "rhsContracting")
    let lhs = primals[0]
    let rhs = primals[1]
    # Output layout: [batch_dims..., lhs_free_dims..., rhs_free_dims...]
    # Compute which dims in each operand are "free" (not batching/contracting).
    let lhsRank = lhs.shape.len
    let rhsRank = rhs.shape.len
    var lhsUsed = newSeq[bool](lhsRank)
    var rhsUsed = newSeq[bool](rhsRank)
    for d in lhsBatching: lhsUsed[d] = true
    for d in lhsContracting: lhsUsed[d] = true
    for d in rhsBatching: rhsUsed[d] = true
    for d in rhsContracting: rhsUsed[d] = true
    var lhsFreeDims: seq[int] = @[]
    var rhsFreeDims: seq[int] = @[]
    for i in 0 ..< lhsRank:
      if not lhsUsed[i]: lhsFreeDims.add i
    for i in 0 ..< rhsRank:
      if not rhsUsed[i]: rhsFreeDims.add i
    let numBatch = lhsBatching.len
    let numLhsFree = lhsFreeDims.len
    let numRhsFree = rhsFreeDims.len
    # grad_lhs = dotGeneral(cotangent, rhs) contracting over rhs_free_dims
    # Output index: batch dims of cotangent contract with batch dims of rhs,
    # rhs_free_dims of cotangent contract with rhs_free_dims of rhs.
    # Result free dims: lhs_free from cotangent, contracting from rhs.
    # cotangent shape: [batch..., lhsFree..., rhsFree...]
    # For grad_lhs: contract cotangent's rhs_free positions with rhs's free dims
    var cotRhsFreePositions: seq[int] = @[]
    for i in 0 ..< numRhsFree:
      cotRhsFreePositions.add(numBatch + numLhsFree + i)
    var cotBatchPositions: seq[int] = @[]
    for i in 0 ..< numBatch:
      cotBatchPositions.add(i)
    let gradLhs = dotGeneral(cotangent, rhs,
      cotBatchPositions, rhsBatching,
      cotRhsFreePositions, rhsFreeDims)
    # grad_lhs has shape [batch..., lhsFree..., rhsContracting...]
    # We need to transpose to match lhs layout: [batch..., free..., contracting...]
    # Build permutation to reorder into original lhs axis order.
    var lhsGradPerm = newSeq[int](lhsRank)
    var srcIdx = 0
    for i, d in lhsBatching:
      lhsGradPerm[d] = srcIdx
      srcIdx += 1
    for i, d in lhsFreeDims:
      lhsGradPerm[d] = srcIdx
      srcIdx += 1
    for i, d in lhsContracting:
      lhsGradPerm[d] = srcIdx
      srcIdx += 1
    let dLhs = transpose(gradLhs, lhsGradPerm)
    # For grad_rhs: contract cotangent's lhs_free positions with lhs's free dims
    var cotLhsFreePositions: seq[int] = @[]
    for i in 0 ..< numLhsFree:
      cotLhsFreePositions.add(numBatch + i)
    let gradRhs = dotGeneral(cotangent, lhs,
      cotBatchPositions, lhsBatching,
      cotLhsFreePositions, lhsFreeDims)
    # grad_rhs has shape [batch..., rhsFree..., lhsContracting...]
    # Transpose to match rhs layout.
    var rhsGradPerm = newSeq[int](rhsRank)
    srcIdx = 0
    for i, d in rhsBatching:
      rhsGradPerm[d] = srcIdx
      srcIdx += 1
    for i, d in rhsFreeDims:
      rhsGradPerm[d] = srcIdx
      srcIdx += 1
    for i, d in rhsContracting:
      rhsGradPerm[d] = srcIdx
      srcIdx += 1
    let dRhs = transpose(gradRhs, rhsGradPerm)
    @[dLhs, dRhs])

# ---- entry point ----------------------------------------------------------

proc registerNewUnaryRules() =
  registerVjpRule("sine", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # d/dx sin(x) = cos(x)
    @[mul(cotangent, cosine(primals[0]))])

  registerVjpRule("cosine", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # d/dx cos(x) = -sin(x)
    @[mul(cotangent, neg(sine(primals[0])))])

  registerVjpRule("rsqrt", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # d/dx x^{-1/2} = -0.5 * x^{-3/2} = -0.5 * rsqrt(x)^3
    let outCubed = mul(output, mul(output, output))
    let half = scalarF32(0.5f32)
    var bdims: seq[int] = @[]
    let halfB = broadcastTo(half, output.shape, bdims)
    @[neg(mul(cotangent, mul(halfB, outCubed)))])

  registerVjpRule("clamp", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # Gradient flows through only where minVal < x < maxVal (strictly interior).
    let minVal = primals[0]
    let x = primals[1]
    let maxVal = primals[2]
    let minForCompare = broadcastScalarLike(minVal, x)
    let maxForCompare = broadcastScalarLike(maxVal, x)
    let zero = zerosLike(x)
    let aboveMin = compareTrace(x, minForCompare, "GT")
    let belowMax = compareTrace(x, maxForCompare, "LT")
    let one = onesLike(x)
    let maskMin = selectTrace(aboveMin, one, zero)
    let maskMax = selectTrace(belowMax, one, zero)
    let mask = mul(maskMin, maskMax)
    let dx = mul(cotangent, mask)
    # Gradient for minVal: flows where x <= minVal
    let atMin = compareTrace(x, minForCompare, "LE")
    let dMin = reduceToShape(selectTrace(atMin, cotangent, zero), minVal)
    # Gradient for maxVal: flows where x >= maxVal
    let atMax = compareTrace(x, maxForCompare, "GE")
    let dMax = reduceToShape(selectTrace(atMax, cotangent, zero), maxVal)
    @[dMin, dx, dMax])

proc registerConcatSliceRules() =
  registerVjpRule("concat", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    let dims = lookup(intAttrs, "dimension")
    let dimension = dims[0]
    # Split cotangent back into slices matching each input's size along `dimension`.
    var grads = newSeq[Tensor](primals.len)
    var offset = 0
    for i, p in primals:
      let size = p.shape[dimension]
      var starts = newSeq[int](p.shape.len)
      var limits = newSeq[int](p.shape.len)
      var strides = newSeq[int](p.shape.len)
      for j in 0 ..< p.shape.len:
        starts[j] = 0
        limits[j] = cotangent.shape[j]
        strides[j] = 1
      starts[dimension] = offset
      limits[dimension] = offset + size
      grads[i] = slice(cotangent, starts, limits, strides)
      offset += size
    grads)

  registerVjpRule("slice", proc(primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] =
    # Pad cotangent back to the original shape with zeros.
    let startIndices = lookup(intAttrs, "startIndices")
    let limitIndices = lookup(intAttrs, "limitIndices")
    let strides = lookup(intAttrs, "strides")
    let origShape = primals[0].shape
    var padded = cotangent
    for d in 0 ..< origShape.len:
      if strides[d] == 1:
        padded = unitSliceGradDim(padded, d, origShape[d],
          startIndices[d], limitIndices[d])
      else:
        padded = stridedSliceGradDim(padded, d, origShape[d],
          startIndices[d], limitIndices[d], strides[d])
    @[padded])

proc convolutionTrace(lhs, rhs: Tensor;
    windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    lhsDilation, rhsDilation: openArray[int];
    dims: ConvDimensionNumbers;
    featureGroupCount, batchGroupCount: int;
    windowReversal: openArray[bool];
    outShape: openArray[int]): Tensor =
  ## Trace-mode `stablehlo.convolution` emitted directly from a vjp
  ## rule. Bypasses the public `conv2d` op so the rule can use custom
  ## dimension numbers and `window_reversal` (the public op only
  ## exposes the standard NHWC/OIHW layout).
  let ctx = currentTraceContext()
  let id = shops.convolution(ctx.builder, lhs.traceId, rhs.traceId,
    windowStrides, padding, lhsDilation, rhsDilation, dims,
    featureGroupCount, batchGroupCount, windowReversal)
  initTraceTensor(id, lhs.dtype, outShape, lhs.device, lhs.sharding)

proc registerConvRule() =
  registerVjpRule("conv2d", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let strides = lookup(intAttrs, "strides")
    let padFlat = lookup(intAttrs, "padding")
    let dilation = lookup(intAttrs, "dilation")
    let x = primals[0]   # NHWC [N, H, W, C_in]
    let k = primals[1]   # OIHW [C_out, C_in, kH, kW]
    let cot = cotangent  # NHWC [N, H', W', C_out]
    let kH = k.shape[2]
    let kW = k.shape[3]
    let dH = dilation[0]
    let dW = dilation[1]
    let ploH = padFlat[0]
    let phiH = padFlat[1]
    let ploW = padFlat[2]
    let phiW = padFlat[3]
    let effKH = (kH - 1) * dH + 1
    let effKW = (kW - 1) * dW + 1

    # ---- grad_x: conv(cot, k, window_reversal=true, swapped kernel I/O,
    #               lhs dilation restores forward stride, rhs dilation
    #               preserves forward kernel dilation.
    let gradXDims = ConvDimensionNumbers(
      inputBatch: 0, inputFeature: 3, inputSpatial: @[1, 2],
      kernelInputFeature: 0,   # was 1; swapped (cot's C_out contracts here)
      kernelOutputFeature: 1,  # was 0; produces C_in
      kernelSpatial: @[2, 3],
      outputBatch: 0, outputFeature: 3, outputSpatial: @[1, 2])
    let gradXPad: seq[array[2, int]] = @[
      [effKH - 1 - ploH, effKH - 1 - phiH],
      [effKW - 1 - ploW, effKW - 1 - phiW],
    ]
    let rawGradXShape = shops.convolutionOutputShape(cot.shape, k.shape,
      [1, 1], gradXPad, strides, dilation, gradXDims, 1, 1)
    let rawGradX = convolutionTrace(cot, k,
      [1, 1], gradXPad, strides, dilation, gradXDims, 1, 1,
      [true, true], rawGradXShape)
    let gradX = matchShape(rawGradX, x.shape)

    # ---- grad_k: conv(x_reinterpreted, cot_reinterpreted) with the
    #               output dim_numbers laid out as OIHW directly. Forward
    #               kernel dilation becomes the output stride; forward
    #               stride becomes rhs dilation over cotangent positions.
    let gradKDims = ConvDimensionNumbers(
      inputBatch: 3,           # x's C_in axis becomes batch (output: C_in)
      inputFeature: 0,         # x's N axis is contracted as input feature
      inputSpatial: @[1, 2],
      kernelInputFeature: 0,   # cot's N axis is contracted
      kernelOutputFeature: 3,  # cot's C_out becomes output feature
      kernelSpatial: @[1, 2],
      outputBatch: 1,          # C_in lands at axis 1 (OIHW)
      outputFeature: 0,        # C_out lands at axis 0
      outputSpatial: @[2, 3])  # kH, kW at axes 2, 3
    let gradKPad: seq[array[2, int]] = @[
      [ploH, phiH],
      [ploW, phiW],
    ]
    let rawGradKShape = shops.convolutionOutputShape(x.shape, cot.shape,
      dilation, gradKPad, [1, 1], strides, gradKDims, 1, 1)
    let rawGradK = convolutionTrace(x, cot,
      dilation, gradKPad, [1, 1], strides, gradKDims, 1, 1,
      [], rawGradKShape)
    let gradK = matchShape(rawGradK, k.shape)

    @[gradX, gradK])

proc registerMaxPool2dRule() =
  registerVjpRule("maxPool2d", proc(primals: openArray[Tensor];
      output: Tensor; cotangent: Tensor;
      intAttrs: IntAttrs): seq[Tensor] =
    let kernelSize = lookup(intAttrs, "kernelSize")
    let strides = lookup(intAttrs, "strides")
    let padFlat = lookup(intAttrs, "padding")
    let kH = kernelSize[0]
    let kW = kernelSize[1]
    let sH = strides[0]
    let sW = strides[1]
    let x = primals[0]
    let windowDims = [1, kH, kW, 1]
    let windowStrides = [1, sH, sW, 1]
    let padding: array[4, array[2, int]] = [
      [0, 0],
      [padFlat[0], padFlat[1]],
      [padFlat[2], padFlat[3]],
      [0, 0],
    ]
    @[maxSelectAndScatterTrace(x, cotangent, windowDims, windowStrides,
      padding)])

proc installAllVjpRules*() =
  ## Installs every primitive vjp closure. Idempotent. Called once at
  ## startup by the autograd umbrella.
  registerArithRules()
  registerUnaryRules()
  registerShapeRules()
  registerLinalgRules()
  registerReduceRules()
  registerMinMaxRules()
  registerAbsRule()
  registerReduceMaxMinRules()
  registerDotGeneralRule()
  registerNewUnaryRules()
  registerConcatSliceRules()
  registerConvRule()
  registerMaxPool2dRule()

# Auto-install at module load.
installAllVjpRules()
