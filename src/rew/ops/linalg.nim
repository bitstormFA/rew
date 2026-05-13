## Linear-algebra dispatch ops \u2014 `matmul`, `dotGeneral`, `broadcastTo`.
##
## All three sit on top of the StableHLO builder ops in
## `stablehlo/ops.nim` and follow the same dispatcher contract as the
## other op modules: trace-mode emits IR, eager-mode routes through the
## registered backend.

import ../tensor
import ../dtype
import ../dispatch
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../autograd/tape
import ./marker
import ./shape
import ./arith
import ./compare
import ./unary
import ./concat

proc joinInts(xs: openArray[int]): string =
  ## Comma-joined list, used to thread shape attributes through the
  ## eager-backend stringly-typed attr channel until that gets a richer
  ## API in Phase 7.
  result = "["
  for i, x in xs:
    if i > 0: result.add ','
    result.add $x
  result.add ']'

proc validateShape(opName: string; shape: openArray[int]) =
  for i, dim in shape:
    if dim < 0:
      raise newException(TensorError,
        opName & ": shape dimension #" & $i &
          " must be non-negative, got " & $dim)

proc normalizeDim(opName: string; rank, dim: int): int =
  result = if dim < 0: rank + dim else: dim
  if result < 0 or result >= rank:
    raise newException(TensorError,
      opName & ": dim " & $dim & " out of range for rank " & $rank)

proc collectDims(opName, argName: string; rank: int;
    dims: openArray[int]; used: var seq[bool]): seq[int] =
  for raw in dims:
    if raw < 0 or raw >= rank:
      raise newException(TensorError,
        opName & ": " & argName & " dim " & $raw &
          " out of range for rank " & $rank)
    if used[raw]:
      raise newException(TensorError,
        opName & ": " & argName & " dim " & $raw & " is repeated")
    used[raw] = true
    result.add raw

proc validateBroadcast(opName: string; operandShape, outputShape,
    broadcastDimensions: openArray[int]) =
  if broadcastDimensions.len != operandShape.len:
    raise newException(TensorError,
      opName & ": broadcastDimensions length " &
        $broadcastDimensions.len & " does not match operand rank " &
        $operandShape.len)
  validateShape(opName, outputShape)
  var seen = newSeq[bool](outputShape.len)
  for i, outDim in broadcastDimensions:
    if outDim < 0 or outDim >= outputShape.len:
      raise newException(TensorError,
        opName & ": broadcast dimension " & $outDim &
          " out of range for result rank " & $outputShape.len)
    if seen[outDim]:
      raise newException(TensorError,
        opName & ": broadcast dimension " & $outDim & " is repeated")
    seen[outDim] = true
    let inSize = operandShape[i]
    let outSize = outputShape[outDim]
    if inSize != 1 and inSize != outSize:
      raise newException(TensorError,
        opName & ": operand dim " & $i & " has size " & $inSize &
          " but result dim " & $outDim & " has size " & $outSize)

proc dotGeneralOutputShape(lShape, rShape: openArray[int];
    lhsBatching, rhsBatching, lhsContracting, rhsContracting: openArray[int]):
    seq[int] =
  if lhsBatching.len != rhsBatching.len:
    raise newException(TensorError,
      "dotGeneral: lhs/rhs batching dim count mismatch")
  if lhsContracting.len != rhsContracting.len:
    raise newException(TensorError,
      "dotGeneral: lhs/rhs contracting dim count mismatch")
  var lhsUsed = newSeq[bool](lShape.len)
  var rhsUsed = newSeq[bool](rShape.len)
  let lhsBatch = collectDims("dotGeneral", "lhs batching", lShape.len,
    lhsBatching, lhsUsed)
  let lhsContract = collectDims("dotGeneral", "lhs contracting", lShape.len,
    lhsContracting, lhsUsed)
  let rhsBatch = collectDims("dotGeneral", "rhs batching", rShape.len,
    rhsBatching, rhsUsed)
  let rhsContract = collectDims("dotGeneral", "rhs contracting", rShape.len,
    rhsContracting, rhsUsed)
  for i in 0 ..< lhsBatch.len:
    if lShape[lhsBatch[i]] != rShape[rhsBatch[i]]:
      raise newException(TensorError,
        "dotGeneral: batching dim size mismatch (" &
          $lShape[lhsBatch[i]] & " vs " & $rShape[rhsBatch[i]] & ")")
  for i in 0 ..< lhsContract.len:
    if lShape[lhsContract[i]] != rShape[rhsContract[i]]:
      raise newException(TensorError,
        "dotGeneral: contracting dim size mismatch (" &
          $lShape[lhsContract[i]] & " vs " & $rShape[rhsContract[i]] & ")")
  result = @[]
  for d in lhsBatch: result.add lShape[d]
  for i in 0 ..< lShape.len:
    if not lhsUsed[i]: result.add lShape[i]
  for i in 0 ..< rShape.len:
    if not rhsUsed[i]: result.add rShape[i]

proc dotOutputShape(lShape, rShape: openArray[int]): seq[int] =
  if lShape.len < 1 or lShape.len > 2:
    raise newException(TensorError,
      "dot: lhs must be rank 1 or 2, got rank " & $lShape.len)
  if rShape.len < 1 or rShape.len > 2:
    raise newException(TensorError,
      "dot: rhs must be rank 1 or 2, got rank " & $rShape.len)
  if lShape[^1] != rShape[0]:
    raise newException(TensorError,
      "dot: contracting dim mismatch (" & $lShape[^1] &
        " vs " & $rShape[0] & ")")
  if lShape.len == 2:
    result.add lShape[0]
  if rShape.len == 2:
    result.add rShape[1]

proc fftOutput(t: Tensor; fftType: FftType;
    fftLength: openArray[int]): ShTensorType =
  try:
    shops.fftOutputType(t.tensorTypeOf, fftType, fftLength)
  except ShBuilderError as e:
    raise newException(TensorError, e.msg)

proc triangularSolveShape(a, b: Tensor; leftSide: bool): seq[int] =
  try:
    shops.triangularSolveOutputShape(a.tensorTypeOf, b.tensorTypeOf,
      leftSide)
  except ShBuilderError as e:
    raise newException(TensorError, e.msg)

proc torchIndexSelectShape(operand, index: Tensor; dim, batchDims: int):
    seq[int] =
  try:
    shops.torchIndexSelectOutputShape(operand.shape, index.shape,
      dim, batchDims)
  except ShBuilderError as e:
    raise newException(TensorError, e.msg)

proc dot*(a, b: Tensor): Tensor {.rewOp.} =
  ## StableHLO `dot` for rank-1/rank-2 dot products. For new code,
  ## `matmul` and `dotGeneral` are more explicit, but this exposes the
  ## pinned StableHLO op directly.
  requireSameMode(a, b, "dot")
  requireSameDevice(a, b, "dot")
  if a.dtype != b.dtype:
    raise newException(TensorError,
      "dot: dtype mismatch (" & $a.dtype & " vs " & $b.dtype & ")")
  let outShape = dotOutputShape(a.shape, b.shape)
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.dot(ctx.builder, a.traceId, b.traceId)
    result = initTraceTensor(id, a.dtype, outShape, a.device, a.sharding)
    recordTraceOp("dot", [a, b], result)
  of dmEager:
    let outs = dispatchEager("dot", [a, b])
    doAssert outs.len == 1, "dot: eager backend returned wrong arity"
    result = outs[0]

proc cholesky*(a: Tensor; lower = true): Tensor {.rewOp.} =
  ## Cholesky factorization of the innermost square matrices in `a`.
  if not a.dtype.isFloat:
    raise newException(TensorError,
      "cholesky: operand must be floating point, got " & $a.dtype)
  if a.shape.len < 2:
    raise newException(TensorError,
      "cholesky: operand rank must be at least 2, got " & $a.shape.len)
  if a.shape[^1] != a.shape[^2]:
    raise newException(TensorError,
      "cholesky: innermost matrix dimensions must be square, got " &
        $a.shape)
  case currentMode()
  of dmTrace:
    requireTrace(a, "cholesky")
    let ctx = currentTraceContext()
    let id = shops.cholesky(ctx.builder, a.traceId, lower)
    result = initTraceTensor(id, a.dtype, a.shape, a.device, a.sharding)
    recordTraceOp("cholesky", [a], result)
  of dmEager:
    requireEager(a, "cholesky")
    let lowerAttr = if lower: "true" else: "false"
    let outs = dispatchEager("cholesky", [a],
      [("lower", lowerAttr)])
    doAssert outs.len == 1, "cholesky: eager backend returned wrong arity"
    result = outs[0]

proc dotGeneral*(a, b: Tensor;
    lhsBatching, rhsBatching, lhsContracting, rhsContracting: openArray[int]):
    Tensor {.rewOp.} =
  ## Generalised matrix product. The dim lists describe how the
  ## operand axes pair up; see `stablehlo.dot_general`.
  requireSameMode(a, b, "dotGeneral")
  requireSameDevice(a, b, "dotGeneral")
  if a.dtype != b.dtype:
    raise newException(TensorError,
      "dotGeneral: dtype mismatch (" & $a.dtype & " vs " & $b.dtype & ")")
  let outShape = dotGeneralOutputShape(a.shape, b.shape,
    lhsBatching, rhsBatching, lhsContracting, rhsContracting)
  case currentMode()
  of dmTrace:
    let ctx = currentTraceContext()
    let id = shops.dotGeneral(ctx.builder, a.traceId, b.traceId,
      lhsBatching, rhsBatching, lhsContracting, rhsContracting)
    result = initTraceTensor(id, a.dtype, outShape, a.device, a.sharding)
    recordTraceOp("dotGeneral", [a, b], result, @[
      ("lhsBatching", @lhsBatching),
      ("rhsBatching", @rhsBatching),
      ("lhsContracting", @lhsContracting),
      ("rhsContracting", @rhsContracting),
    ])
  of dmEager:
    let attrs = @[
      ("lhs_batching", joinInts(lhsBatching)),
      ("rhs_batching", joinInts(rhsBatching)),
      ("lhs_contracting", joinInts(lhsContracting)),
      ("rhs_contracting", joinInts(rhsContracting)),
    ]
    let outs = dispatchEager("dotGeneral", [a, b], attrs)
    doAssert outs.len == 1, "dotGeneral: eager backend returned wrong arity"
    result = outs[0]

proc matmul*(a, b: Tensor): Tensor {.rewOp.} =
  ## Rank-2 matrix multiplication \u2014 contracts `a`'s dim 1 against
  ## `b`'s dim 0. Convenience wrapper over `dotGeneral`.
  if a.shape.len != 2 or b.shape.len != 2:
    raise newException(TensorError,
      "matmul: expected rank-2 operands, got shapes " & $a.shape &
        " and " & $b.shape)
  if a.shape[1] != b.shape[0]:
    raise newException(TensorError,
      "matmul: inner dim mismatch \u2014 " & $a.shape & " * " & $b.shape)
  result = dotGeneral(a, b, [], [], [1], [0])
  # Replace the dotGeneral entry the inner call recorded with a
  # `matmul` entry so the vjp transform picks the matmul-specific rule.
  let tape = currentTape()
  if not tape.isNil and tape.entries.len > 0 and
      tape.entries[^1].opName == "dotGeneral":
    tape.entries[^1].opName = "matmul"
    tape.entries[^1].intAttrs = @[]

proc broadcastTo*(t: Tensor; outputShape: openArray[int];
    broadcastDimensions: openArray[int]): Tensor {.rewOp.} =
  ## Broadcast `t` to `outputShape`. Each operand axis `i` maps to
  ## output axis `broadcastDimensions[i]`. Operand axes must be size 1
  ## or already match.
  validateBroadcast("broadcastTo", t.shape, outputShape, broadcastDimensions)
  case currentMode()
  of dmTrace:
    requireTrace(t, "broadcastTo")
    let ctx = currentTraceContext()
    let id = shops.broadcastInDim(ctx.builder, t.traceId,
      outputShape, broadcastDimensions)
    result = initTraceTensor(id, t.dtype, outputShape, t.device, t.sharding)
    recordTraceOp("broadcastTo", [t], result, @[
      ("outputShape", @outputShape),
      ("broadcastDimensions", @broadcastDimensions),
    ])
  of dmEager:
    requireEager(t, "broadcastTo")
    let attrs = @[
      ("output_shape", joinInts(outputShape)),
      ("broadcast_dimensions", joinInts(broadcastDimensions)),
    ]
    let outs = dispatchEager("broadcastTo", [t], attrs)
    doAssert outs.len == 1,
      "broadcastTo: eager backend returned wrong arity"
    result = outs[0]

proc dynamicBroadcastInDim*(t, outputDimensions: Tensor;
    resultShape, broadcastDimensions: openArray[int];
    knownExpandingDimensions: openArray[int] = [];
    knownNonexpandingDimensions: openArray[int] = []): Tensor {.rewOp.} =
  ## StableHLO `dynamic_broadcast_in_dim`. `outputDimensions` carries the
  ## runtime sizes, while `resultShape` declares the ranked result type.
  requireSameMode(t, outputDimensions, "dynamicBroadcastInDim")
  requireSameDevice(t, outputDimensions, "dynamicBroadcastInDim")
  if not (outputDimensions.dtype.isSignedInt or
      outputDimensions.dtype.isUnsignedInt) or
      outputDimensions.shape != @[resultShape.len]:
    raise newException(TensorError,
      "dynamicBroadcastInDim: outputDimensions must be an integer vector " &
        "of length " & $resultShape.len)
  validateBroadcast("dynamicBroadcastInDim", t.shape, resultShape,
    broadcastDimensions)
  case currentMode()
  of dmTrace:
    requireTrace(t, "dynamicBroadcastInDim")
    requireTrace(outputDimensions, "dynamicBroadcastInDim")
    let ctx = currentTraceContext()
    let id = shops.dynamicBroadcastInDim(ctx.builder, t.traceId,
      outputDimensions.traceId, resultShape, broadcastDimensions,
      knownExpandingDimensions, knownNonexpandingDimensions)
    result = initTraceTensor(id, t.dtype, resultShape, t.device, t.sharding)
    recordTraceOp("dynamicBroadcastInDim", [t, outputDimensions], result)
  of dmEager:
    requireEager(t, "dynamicBroadcastInDim")
    requireEager(outputDimensions, "dynamicBroadcastInDim")
    let attrs = @[
      ("result_shape", joinInts(resultShape)),
      ("broadcast_dimensions", joinInts(broadcastDimensions)),
      ("known_expanding_dimensions", joinInts(knownExpandingDimensions)),
      ("known_nonexpanding_dimensions",
        joinInts(knownNonexpandingDimensions)),
    ]
    let outs = dispatchEager("dynamicBroadcastInDim",
      [t, outputDimensions], attrs)
    doAssert outs.len == 1,
      "dynamicBroadcastInDim: eager backend returned wrong arity"
    result = outs[0]

proc fft*(a: Tensor; fftType: FftType;
    fftLength: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO `fft`. Supports FFT/IFFT, RFFT, and IRFFT type/shape
  ## transitions.
  let outTy = fftOutput(a, fftType, fftLength)
  case currentMode()
  of dmTrace:
    requireTrace(a, "fft")
    let ctx = currentTraceContext()
    let id = shops.fft(ctx.builder, a.traceId, fftType, fftLength)
    result = initTraceTensor(id, outTy.dtype, outTy.shape,
      a.device, a.sharding)
    recordTraceOp("fft", [a], result)
  of dmEager:
    requireEager(a, "fft")
    let outs = dispatchEager("fft", [a], [
      ("fft_type", fftType.stablehloName),
      ("fft_length", joinInts(fftLength)),
    ])
    doAssert outs.len == 1, "fft: eager backend returned wrong arity"
    result = outs[0]

proc triangularSolve*(a, b: Tensor; leftSide = true; lower = true;
    unitDiagonal = false; transposeA = tkNoTranspose): Tensor {.rewOp.} =
  ## StableHLO `triangular_solve`. Solves triangular systems in the
  ## innermost matrix dimensions; the result shape matches `b`.
  requireSameMode(a, b, "triangularSolve")
  requireSameDevice(a, b, "triangularSolve")
  let outShape = triangularSolveShape(a, b, leftSide)
  case currentMode()
  of dmTrace:
    requireTrace(a, "triangularSolve")
    requireTrace(b, "triangularSolve")
    let ctx = currentTraceContext()
    let id = shops.triangularSolve(ctx.builder, a.traceId, b.traceId,
      leftSide, lower, unitDiagonal, transposeA)
    result = initTraceTensor(id, b.dtype, outShape, a.device, a.sharding)
    recordTraceOp("triangularSolve", [a, b], result)
  of dmEager:
    requireEager(a, "triangularSolve")
    requireEager(b, "triangularSolve")
    let attrs = @[
      ("left_side", if leftSide: "true" else: "false"),
      ("lower", if lower: "true" else: "false"),
      ("unit_diagonal", if unitDiagonal: "true" else: "false"),
      ("transpose_a", transposeA.stablehloName),
    ]
    let outs = dispatchEager("triangularSolve", [a, b], attrs)
    doAssert outs.len == 1,
      "triangularSolve: eager backend returned wrong arity"
    result = outs[0]

proc einsum*(a, b: Tensor; config: string;
    outputShape: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO `einsum` with explicit result shape. `config` uses the
  ## TensorFlow einsum grammar (for example `"ab,bc->ac"`).
  requireSameMode(a, b, "einsum")
  requireSameDevice(a, b, "einsum")
  if a.dtype != b.dtype:
    raise newException(TensorError,
      "einsum: dtype mismatch (" & $a.dtype & " vs " & $b.dtype & ")")
  if config.len == 0:
    raise newException(TensorError, "einsum: config must not be empty")
  case currentMode()
  of dmTrace:
    requireTrace(a, "einsum")
    requireTrace(b, "einsum")
    let ctx = currentTraceContext()
    let id = shops.einsum(ctx.builder, a.traceId, b.traceId,
      config, outputShape)
    result = initTraceTensor(id, a.dtype, outputShape, a.device, a.sharding)
    recordTraceOp("einsum", [a, b], result)
  of dmEager:
    requireEager(a, "einsum")
    requireEager(b, "einsum")
    let outs = dispatchEager("einsum", [a, b], [
      ("einsum_config", config),
      ("output_shape", joinInts(outputShape)),
    ])
    doAssert outs.len == 1, "einsum: eager backend returned wrong arity"
    result = outs[0]

proc unaryEinsum*(a: Tensor; config: string;
    outputShape: openArray[int]): Tensor {.rewOp.} =
  ## StableHLO `unary_einsum` with explicit result shape.
  if config.len == 0:
    raise newException(TensorError,
      "unaryEinsum: config must not be empty")
  case currentMode()
  of dmTrace:
    requireTrace(a, "unaryEinsum")
    let ctx = currentTraceContext()
    let id = shops.unaryEinsum(ctx.builder, a.traceId, config, outputShape)
    result = initTraceTensor(id, a.dtype, outputShape, a.device, a.sharding)
    recordTraceOp("unaryEinsum", [a], result)
  of dmEager:
    requireEager(a, "unaryEinsum")
    let outs = dispatchEager("unaryEinsum", [a], [
      ("einsum_config", config),
      ("output_shape", joinInts(outputShape)),
    ])
    doAssert outs.len == 1, "unaryEinsum: eager backend returned wrong arity"
    result = outs[0]

proc torchIndexSelect*(operand, index: Tensor; dim, batchDims: int):
    Tensor {.rewOp.} =
  ## StableHLO `torch_index_select`, including the StableHLO
  ## `batch_dims` extension.
  requireSameMode(operand, index, "torchIndexSelect")
  requireSameDevice(operand, index, "torchIndexSelect")
  if not (index.dtype.isSignedInt or index.dtype.isUnsignedInt):
    raise newException(TensorError,
      "torchIndexSelect: index must be an integer tensor")
  let outShape = torchIndexSelectShape(operand, index, dim, batchDims)
  case currentMode()
  of dmTrace:
    requireTrace(operand, "torchIndexSelect")
    requireTrace(index, "torchIndexSelect")
    let ctx = currentTraceContext()
    let id = shops.torchIndexSelect(ctx.builder, operand.traceId,
      index.traceId, dim, batchDims)
    result = initTraceTensor(id, operand.dtype, outShape,
      operand.device, operand.sharding)
    recordTraceOp("torchIndexSelect", [operand, index], result)
  of dmEager:
    requireEager(operand, "torchIndexSelect")
    requireEager(index, "torchIndexSelect")
    let outs = dispatchEager("torchIndexSelect", [operand, index], [
      ("dim", $dim),
      ("batch_dims", $batchDims),
    ])
    doAssert outs.len == 1,
      "torchIndexSelect: eager backend returned wrong arity"
    result = outs[0]

# ---- shape ops that need broadcastTo ------------------------------------

proc tile*(a: Tensor; multiples: openArray[int]): Tensor =
  ## Repeat `a` along each axis according to `multiples`.
  if multiples.len != a.shape.len:
    raise newException(TensorError,
      "tile: multiples length " & $multiples.len &
        " must match rank " & $a.shape.len)
  for i, multiple in multiples:
    if multiple <= 0:
      raise newException(TensorError,
        "tile: multiple #" & $i & " must be positive, got " & $multiple)
    if a.shape[i] != 0 and multiple > high(int) div a.shape[i]:
      raise newException(TensorError,
        "tile: output dimension #" & $i & " overflows int")
  var intermediate = a
  for i in 0 ..< a.shape.len:
    if multiples[i] > 1:
      var intermedShape = intermediate.shape
      intermedShape.insert(1, 2 * i + 1)
      intermediate = reshape(intermediate, intermedShape)
      var bcastShape = intermediate.shape
      bcastShape[2 * i + 1] = multiples[i]
      var bdims: seq[int] = @[]
      for j in 0 ..< bcastShape.len:
        if j != 2 * i + 1: bdims.add j
      intermediate = broadcastTo(intermediate, bcastShape, bdims)
  var finalShape: seq[int] = @[]
  for i in 0 ..< a.shape.len:
    finalShape.add a.shape[i] * multiples[i]
  reshape(intermediate, finalShape)

proc repeat*(a: Tensor; repeats: int; dim: int): Tensor =
  ## Repeat elements of `a` along `dim` `repeats` times.
  let pos = normalizeDim("repeat", a.shape.len, dim)
  if repeats <= 0:
    raise newException(TensorError,
      "repeat: repeats must be positive, got " & $repeats)
  var multiples = newSeq[int](a.shape.len)
  for i in 0 ..< a.shape.len: multiples[i] = 1
  multiples[pos] = repeats
  tile(a, multiples)

# ---- linalg composites ---------------------------------------------------

proc outer*(a, b: Tensor): Tensor =
  ## Outer product of two 1-D tensors. Result has shape `[a.len, b.len]`.
  if a.shape.len != 1 or b.shape.len != 1:
    raise newException(TensorError,
      "outer: operands must be 1-D, got " & $a.shape & " and " & $b.shape)
  let a2d = unsqueeze(a, 1)
  let b2d = unsqueeze(b, 0)
  mul(a2d, b2d)

proc cross*(a, b: Tensor; dim = -1): Tensor =
  ## Cross product of 3-element vectors along `dim`. Both operands must
  ## have size 3 on that dimension.
  let pos = if dim < 0: a.shape.len + dim else: dim
  if pos < 0 or pos >= a.shape.len:
    raise newException(TensorError,
      "cross: dim " & $dim & " out of range for rank " & $a.shape.len)
  if a.shape != b.shape:
    raise newException(TensorError,
      "cross: shape mismatch (" & $a.shape & " vs " & $b.shape & ")")
  if a.shape[pos] != 3:
    raise newException(TensorError,
      "cross: dim " & $pos & " must have size 3, got " & $a.shape[pos])
  # Extract slices along dim, all other dims keep full slices.
  let rank = a.shape.len
  var starts0 = newSeq[int](rank)
  var limits0 = newSeq[int](rank)
  var starts1 = newSeq[int](rank)
  var limits1 = newSeq[int](rank)
  var starts2 = newSeq[int](rank)
  var limits2 = newSeq[int](rank)
  var strides = newSeq[int](rank)
  for i in 0 ..< rank:
    starts0[i] = 0; limits0[i] = a.shape[i]; strides[i] = 1
    starts1[i] = 0; limits1[i] = a.shape[i]
    starts2[i] = 0; limits2[i] = a.shape[i]
  starts0[pos] = 0; limits0[pos] = 1
  starts1[pos] = 1; limits1[pos] = 2
  starts2[pos] = 2; limits2[pos] = 3
  let a0 = slice(a, starts0, limits0, strides)
  let a1 = slice(a, starts1, limits1, strides)
  let a2 = slice(a, starts2, limits2, strides)
  let b0 = slice(b, starts0, limits0, strides)
  let b1 = slice(b, starts1, limits1, strides)
  let b2 = slice(b, starts2, limits2, strides)
  # c0 = a1*b2 - a2*b1
  let c0 = sub(mul(a1, b2), mul(a2, b1))
  # c1 = a2*b0 - a0*b2
  let c1 = sub(mul(a2, b0), mul(a0, b2))
  # c2 = a0*b1 - a1*b0
  let c2 = sub(mul(a0, b1), mul(a1, b0))
  concat([c0, c1, c2], pos)

proc mv*(a, b: Tensor): Tensor =
  ## Matrix-vector product. `a` is `[M, N]`, `b` is `[N]`. Returns `[M]`.
  if a.shape.len != 2 or b.shape.len != 1:
    raise newException(TensorError,
      "mv: a must be 2-D, b must be 1-D, got " & $a.shape & " and " & $b.shape)
  if a.shape[1] != b.shape[0]:
    raise newException(TensorError,
      "mv: inner dim mismatch (" & $a.shape[1] & " vs " & $b.shape[0] & ")")
  let bCol = unsqueeze(b, 1)
  let ab = matmul(a, bCol)
  squeeze(ab, 1)

proc bmm*(a, b: Tensor): Tensor =
  ## Batch matrix-matrix product. `a` is `[B, M, N]`, `b` is `[B, N, P]`.
  ## Returns `[B, M, P]`. Uses `dotGeneral` with batch dim 0.
  if a.shape.len != 3 or b.shape.len != 3:
    raise newException(TensorError,
      "bmm: operands must be 3-D, got " & $a.shape & " and " & $b.shape)
  if a.shape[0] != b.shape[0]:
    raise newException(TensorError,
      "bmm: batch dim mismatch (" & $a.shape[0] & " vs " & $b.shape[0] & ")")
  if a.shape[2] != b.shape[1]:
    raise newException(TensorError,
      "bmm: inner dim mismatch (" & $a.shape[2] & " vs " & $b.shape[1] & ")")
  dotGeneral(a, b, [0], [0], [2], [1])

proc inv*(a: Tensor; lower = true): Tensor =
  ## Inverse of a symmetric positive-definite matrix via Cholesky
  ## factorization. Supports batched matrices: `a` has shape
  ## `[..., N, N]` where the innermost two dims are square.
  ##
  ## The result is A^{-1} computed by two triangular solves through
  ## the Cholesky factor.
  if a.shape.len < 2:
    raise newException(TensorError,
      "inv: operand rank must be at least 2, got " & $a.shape.len)
  if a.shape[^1] != a.shape[^2]:
    raise newException(TensorError,
      "inv: innermost dims must be square, got " & $a.shape)
  let L = cholesky(a, lower)
  let n = a.shape[^1]
  var eye2d = astype(compare(
    iota(dtInt32, [n, n], 0, a.device),
    iota(dtInt32, [n, n], 1, a.device), "EQ"), a.dtype)
  if a.shape.len > 2:
    eye2d = broadcastTo(eye2d, a.shape, [a.shape.len - 2, a.shape.len - 1])
  let invL = triangularSolve(L, eye2d, leftSide=true, lower=lower,
    unitDiagonal=false, transposeA=tkNoTranspose)
  if lower:
    triangularSolve(L, invL, leftSide=false, lower=true,
      unitDiagonal=false, transposeA=tkAdjoint)
  else:
    triangularSolve(L, invL, leftSide=false, lower=false,
      unitDiagonal=false, transposeA=tkNoTranspose)

proc invSymmetric*(a: Tensor; lower = true): Tensor =
  ## Inverse of a symmetric positive-definite matrix via Cholesky.
  ## Convenience alias for `inv` — kept for backwards
  ## compatibility.
  inv(a, lower)

proc solve*(a, b: Tensor; lower = true): Tensor =
  ## Solve `a @ X = b` for `X` where `a` is symmetric
  ## positive-definite. Uses Cholesky factorisation followed by two
  ## triangular solves.
  ##
  ## `a` has shape `[..., N, N]`, `b` has shape `[..., N, K]`
  ## (or `[..., N]`).  Returns `X` with the same shape as `b`.
  if a.shape.len < 2:
    raise newException(TensorError,
      "solve: a must have rank >= 2, got " & $a.shape.len)
  if a.shape[^1] != a.shape[^2]:
    raise newException(TensorError,
      "solve: innermost dims of a must be square, got " & $a.shape)
  if a.dtype != b.dtype:
    raise newException(TensorError,
      "solve: dtype mismatch (" & $a.dtype & " vs " & $b.dtype & ")")
  let L = cholesky(a, lower)
  let Y = triangularSolve(L, b, leftSide=true, lower=lower,
    unitDiagonal=false, transposeA=tkNoTranspose)
  triangularSolve(L, Y, leftSide=true, lower=lower,
    unitDiagonal=false, transposeA=tkTranspose)

proc pinv*(a: Tensor; rcond: float32 = 1e-15'f32; lower = true): Tensor =
  ## Moore-Penrose pseudoinverse of a symmetric positive-definite
  ## matrix via Cholesky. The `rcond` threshold is not applied in
  ## this path — Cholesky fails for singular matrices, so the
  ## result is always the true inverse for full-rank SPD inputs.
  ##
  ## For rank-deficient matrices this will raise from `cholesky`.
  inv(a, lower)

proc matrixPower*(a: Tensor; n: int): Tensor =
  ## Raise a square matrix to an integer power via exponentiation by
  ## squaring. Supports batched matrices: `a` has shape
  ## `[..., N, N]` where the innermost two dims are square.
  ##
  ## Negative powers compute the inverse first via Cholesky
  ## (requires `a` to be symmetric positive-definite). Zero power
  ## returns the identity matrix.
  if a.shape.len < 2:
    raise newException(TensorError,
      "matrixPower: rank must be at least 2, got " & $a.shape.len)
  if a.shape[^1] != a.shape[^2]:
    raise newException(TensorError,
      "matrixPower: innermost dims must be square, got " & $a.shape)
  if n == 0:
    let nMat = a.shape[^1]
    var eye2d = astype(compare(
      iota(dtInt32, [nMat, nMat], 0, a.device),
      iota(dtInt32, [nMat, nMat], 1, a.device), "EQ"), a.dtype)
    if a.shape.len > 2:
      eye2d = broadcastTo(eye2d, a.shape,
        [a.shape.len - 2, a.shape.len - 1])
    return eye2d
  let rank = a.shape.len
  var batchDims: seq[int] = @[]
  for i in 0 ..< rank - 2: batchDims.add i
  proc mm(x, y: Tensor): Tensor =
    dotGeneral(x, y, batchDims, batchDims, [rank - 1], [rank - 2])
  if n > 0:
    var remaining = n
    result = a
    var pwr = a
    remaining -= 1
    while remaining > 0:
      if (remaining mod 2) == 1:
        result = mm(result, pwr)
      pwr = mm(pwr, pwr)
      remaining = remaining div 2
  else:
    let invA = inv(a, true)
    if n == -1:
      return invA
    return matrixPower(invA, -n)

proc kron*(a, b: Tensor): Tensor =
  ## Kronecker product of `a` and `b`. Every element of `a` is multiplied
  ## by every element of `b`.
  let aRank = a.shape.len
  let bRank = b.shape.len
  if aRank != bRank:
    raise newException(TensorError,
      "kron: operands must have the same rank, got " & $aRank &
        " and " & $bRank)
  # Interleave: for each dim of a, broadcast b's matching dim after it.
  var intermed = a
  for i in 0 ..< aRank:
    intermed = unsqueeze(intermed, 2 * i + 1)
  for i in 0 ..< bRank:
    intermed = unsqueeze(intermed, 2 * aRank + i)
  # Build output shape.
  var outShape: seq[int] = @[]
  for i in 0 ..< aRank:
    outShape.add a.shape[i]
    outShape.add b.shape[i]
  # Broadcast a shape up.
  var aBroadShape: seq[int] = @[]
  for i in 0 ..< aRank:
    aBroadShape.add a.shape[i]
    aBroadShape.add 1
  for i in 0 ..< bRank:
    aBroadShape.add b.shape[i]
  var bBroadShape: seq[int] = @[]
  for i in 0 ..< aRank:
    bBroadShape.add 1
    bBroadShape.add b.shape[i]
  for i in 0 ..< bRank:
    bBroadShape.add b.shape[i]
  # Broadcast intermed (a with sized-1 gaps) to match output pattern.
  var aBdims: seq[int] = @[]
  for i in 0 ..< aRank:
    aBdims.add 2 * i
  for i in 0 ..< bRank:
    aBdims.add 2 * aRank + i
  let aB = broadcastTo(intermed, aBroadShape, aBdims)
  # Broadcast b to the interleaved shape.
  var bIntermed = b
  for i in 0 ..< aRank:
    bIntermed = unsqueeze(bIntermed, 2 * i)
  var bBdims: seq[int] = @[]
  for i in 0 ..< aRank:
    bBdims.add 2 * i + 1
  for i in 0 ..< bRank:
    bBdims.add 2 * aRank + i
  let bB = broadcastTo(bIntermed, bBroadShape, bBdims)
  # Multiply aB and bB element-wise.
  let resultMul = mul(aB, bB)
  # Reshape to output shape.
  reshape(resultMul, outShape)

proc tril*(a: Tensor): Tensor =
  ## Lower triangle of the innermost two dimensions (main diagonal
  ## and below). Elements above the diagonal are set to zero.
  if a.shape.len < 2:
    raise newException(TensorError,
      "tril: rank must be at least 2, got " & $a.shape.len)
  let rank = a.shape.len
  let m = a.shape[rank - 2]
  let n = a.shape[rank - 1]
  let rows = iota(dtInt32, [m, n], 0, a.device)
  let cols = iota(dtInt32, [m, n], 1, a.device)
  var mask2d = compare(rows, cols, "GE")
  if rank > 2:
    mask2d = broadcastTo(mask2d, a.shape, [rank - 2, rank - 1])
  mul(a, astype(mask2d, a.dtype))

proc triu*(a: Tensor): Tensor =
  ## Upper triangle of the innermost two dimensions (main diagonal
  ## and above). Elements below the diagonal are set to zero.
  if a.shape.len < 2:
    raise newException(TensorError,
      "triu: rank must be at least 2, got " & $a.shape.len)
  let rank = a.shape.len
  let m = a.shape[rank - 2]
  let n = a.shape[rank - 1]
  let rows = iota(dtInt32, [m, n], 0, a.device)
  let cols = iota(dtInt32, [m, n], 1, a.device)
  var mask2d = compare(rows, cols, "LE")
  if rank > 2:
    mask2d = broadcastTo(mask2d, a.shape, [rank - 2, rank - 1])
  mul(a, astype(mask2d, a.dtype))

# ---- FFT helpers ----------------------------------------------------------

proc fft2*(a: Tensor): Tensor =
  ## 2-D forward FFT over the innermost two dimensions.
  if a.shape.len < 2:
    raise newException(TensorError,
      "fft2: rank must be at least 2, got " & $a.shape.len)
  fft(a, ftFft, [a.shape[^2], a.shape[^1]])

proc ifft2*(a: Tensor): Tensor =
  ## 2-D inverse FFT over the innermost two dimensions.
  if a.shape.len < 2:
    raise newException(TensorError,
      "ifft2: rank must be at least 2, got " & $a.shape.len)
  fft(a, ftIfft, [a.shape[^2], a.shape[^1]])

proc rfft2*(a: Tensor): Tensor =
  ## 2-D real-to-complex FFT over the innermost two dimensions.
  if a.shape.len < 2:
    raise newException(TensorError,
      "rfft2: rank must be at least 2, got " & $a.shape.len)
  fft(a, ftRfft, [a.shape[^2], a.shape[^1]])

proc irfft2*(a: Tensor; outputSamples: int = -1): Tensor =
  ## 2-D complex-to-real inverse FFT over the innermost two
  ## dimensions. The last dimension of `a` should be
  ## `fftLength[1] // 2 + 1` from the forward RFFT.
  if a.shape.len < 2:
    raise newException(TensorError,
      "irfft2: rank must be at least 2, got " & $a.shape.len)
  let n = if outputSamples > 0: outputSamples else: (a.shape[^2] - 1) * 2
  fft(a, ftIrfft, [a.shape[^2], n])

proc fftshift*(a: Tensor; dims: openArray[int] = []): Tensor =
  ## Shift the zero-frequency component to the centre of the
  ## spectrum along `dims` (all dims by default).
  var shiftDims = @dims
  if shiftDims.len == 0:
    for i in 0 ..< a.shape.len: shiftDims.add i
  var t = a
  for d in shiftDims:
    let pos = normalizeDim("fftshift", t.shape.len, d)
    let halfLen = t.shape[pos] div 2
    t = roll(t, halfLen, pos)
  t

proc ifftshift*(a: Tensor; dims: openArray[int] = []): Tensor =
  ## Inverse of `fftshift` — shift the centre component back.
  var shiftDims = @dims
  if shiftDims.len == 0:
    for i in 0 ..< a.shape.len: shiftDims.add i
  var t = a
  for d in shiftDims:
    let pos = normalizeDim("ifftshift", t.shape.len, d)
    let halfLen = (t.shape[pos] + 1) div 2
    t = roll(t, halfLen, pos)
  t

proc fftn*(a: Tensor): Tensor =
  ## N-D forward FFT over all dimensions.
  var dims: seq[int] = @[]
  for d in a.shape: dims.add d
  fft(a, ftFft, dims)

proc ifftn*(a: Tensor): Tensor =
  ## N-D inverse FFT over all dimensions.
  var dims: seq[int] = @[]
  for d in a.shape: dims.add d
  fft(a, ftIfft, dims)

proc rfftn*(a: Tensor): Tensor =
  ## N-D real-to-complex FFT over all dimensions.
  var dims: seq[int] = @[]
  for d in a.shape: dims.add d
  fft(a, ftRfft, dims)

proc irfftn*(a: Tensor; outputSamples: openArray[int] = []): Tensor =
  ## N-D complex-to-real inverse FFT.
  if a.shape.len == 0:
    raise newException(TensorError,
      "irfftn: operand must have rank >= 1")
  var dims: seq[int]
  if outputSamples.len > 0:
    if outputSamples.len != a.shape.len:
      raise newException(TensorError,
        "irfftn: outputSamples length must match operand rank")
    for i, n in outputSamples:
      if n <= 0:
        raise newException(TensorError,
          "irfftn: outputSamples #" & $i & " must be positive")
    dims = @outputSamples
  else:
    for d in a.shape: dims.add d
    dims[^1] = (a.shape[^1] - 1) * 2
  fft(a, ftIrfft, dims)

proc broadcastArrays*(tensors: varargs[Tensor]): seq[Tensor] =
  ## Broadcast all tensors to a common shape following NumPy
  ## broadcasting rules.
  if tensors.len == 0:
    raise newException(TensorError,
      "broadcastArrays: at least one tensor required")
  # Compute common broadcast shape.
  var commonShape = tensors[0].shape
  for t in tensors[1 .. ^1]:
    let rank = max(commonShape.len, t.shape.len)
    var newShape: seq[int] = @[]
    for i in 0 ..< rank:
      let ci = i - (rank - commonShape.len)
      let ti = i - (rank - t.shape.len)
      let cs = if ci >= 0: commonShape[ci] else: 1
      let ts = if ti >= 0: t.shape[ti] else: 1
      if cs != ts and cs != 1 and ts != 1:
        raise newException(TensorError,
          "broadcastArrays: shapes not broadcastable: " &
            $commonShape & " vs " & $t.shape)
      newShape.add(max(cs, ts))
    commonShape = newShape
  # Broadcast each tensor.
  for t in tensors:
    if t.shape == commonShape:
      result.add t
    else:
      let offset = commonShape.len - t.shape.len
      var bdims: seq[int] = @[]
      for j in 0 ..< t.shape.len:
        bdims.add j + offset
      result.add broadcastTo(t, commonShape, bdims)

# ---- host-side FFT frequency helpers ------------------------------------

proc fftfreq*(n: int; d: float32 = 1.0'f32): seq[float32] =
  ## Return the discrete Fourier transform sample frequencies for
  ## a signal of length `n` with sample spacing `d`.
  if n <= 0:
    raise newException(TensorError, "fftfreq: n must be positive")
  if d == 0'f32:
    raise newException(TensorError, "fftfreq: d must be non-zero")
  result = newSeq[float32](n)
  let val = 1.0'f32 / (float32(n) * d)
  let half = (n + 1) div 2
  for i in 0 ..< n:
    if i < half:
      result[i] = float32(i) * val
    else:
      result[i] = float32(int(i) - n) * val

proc rfftfreq*(n: int; d: float32 = 1.0'f32): seq[float32] =
  ## Return the sample frequencies for `rfft` with `n` input points.
  if n <= 0:
    raise newException(TensorError, "rfftfreq: n must be positive")
  if d == 0'f32:
    raise newException(TensorError, "rfftfreq: d must be non-zero")
  result = newSeq[float32](n div 2 + 1)
  let val = 1.0'f32 / (float32(n) * d)
  for i in 0 ..< result.len:
    result[i] = float32(i) * val
