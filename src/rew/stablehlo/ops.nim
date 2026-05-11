## StableHLO builder — the single surface that emits IR.
##
## Both the eager dispatcher (per-op compile-and-cache) and the trace
## dispatcher (`jit`/`lazy`) call into this module. There is no parallel
## emit path elsewhere; that is the rule the layer-import lint enforces.
##
## The builder is a value `object` mutated through `var` parameters. It
## does not own buffers and is cheap to construct.

import std/sequtils
import ../dtype
import ./ir

type
  ShBuilder* = object
    ## In-progress module + a cursor into the function currently being
    ## built. Use `initBuilder`, then `beginFunc`/`endFunc`, then `build`.
    module: ShModule
    curFn: int  ## index into module.funcs, -1 when no function is open
    regionStack: seq[ShRegion]
      ## Innermost-on-top stack of partially-built regions. When non-empty,
      ## `emitOp` appends to the top region's `ops` instead of the
      ## function's body. Allows nested region emission for `reduce`,
      ## `scan`, `while`, and `if/else`.

  ShBuilderError* = object of CatchableError
    ## Misuse of the builder API (e.g. emitting an op outside a function,
    ## ending a function with no return). Distinct from `verify` errors.

  FftType* = enum
    ## StableHLO FFT algorithm selector.
    ftFft
    ftIfft
    ftRfft
    ftIrfft

  TransposeKind* = enum
    ## StableHLO transpose enum used by `triangular_solve`.
    tkNoTranspose
    tkTranspose
    tkAdjoint

  RngDistribution* = enum
    ## StableHLO random distribution selector.
    rdUniform
    rdNormal

  RngAlgorithm* = enum
    ## StableHLO random bit-generator algorithm selector.
    raDefault
    raThreeFry
    raPhilox

  ChannelHandle* = object
    ## StableHLO channel handle attribute. Use `NoChannelHandle` to omit
    ## optional channel metadata.
    handle*: int
    handleType*: int

  GatherDimensionNumbers* = object
    ## Dimension-number attribute for `stablehlo.gather` and
    ## `stablehlo.dynamic_gather`.
    offsetDims*: seq[int]
    collapsedSliceDims*: seq[int]
    operandBatchingDims*: seq[int]
    startIndicesBatchingDims*: seq[int]
    startIndexMap*: seq[int]
    indexVectorDim*: int

  ScatterDimensionNumbers* = object
    ## Dimension-number attribute for `stablehlo.scatter`.
    updateWindowDims*: seq[int]
    insertedWindowDims*: seq[int]
    inputBatchingDims*: seq[int]
    scatterIndicesBatchingDims*: seq[int]
    scatterDimsToOperandDims*: seq[int]
    indexVectorDim*: int

  ShRegionBuilder* = proc(b: var ShBuilder): seq[ShValueId] {.closure.}
    ## Builder callback for single-block region ops with no region args.

  ShArgRegionBuilder* = proc(b: var ShBuilder;
    args: openArray[ShValueId]): seq[ShValueId] {.closure.}
    ## Builder callback for single-block region ops with entry args.

const NoChannelHandle* = ChannelHandle(handle: -1, handleType: -1)
  ## Sentinel used by builder procs when `channel_handle` is absent.

func stablehloName*(kind: FftType): string =
  ## StableHLO enum spelling for `fft_type`.
  case kind
  of ftFft: "FFT"
  of ftIfft: "IFFT"
  of ftRfft: "RFFT"
  of ftIrfft: "IRFFT"

func stablehloName*(kind: TransposeKind): string =
  ## StableHLO enum spelling for `transpose_a`.
  case kind
  of tkNoTranspose: "NO_TRANSPOSE"
  of tkTranspose: "TRANSPOSE"
  of tkAdjoint: "ADJOINT"

func stablehloName*(kind: RngDistribution): string =
  ## StableHLO enum spelling for `rng_distribution`.
  case kind
  of rdUniform: "UNIFORM"
  of rdNormal: "NORMAL"

func stablehloName*(kind: RngAlgorithm): string =
  ## StableHLO enum spelling for `rng_algorithm`.
  case kind
  of raDefault: "DEFAULT"
  of raThreeFry: "THREE_FRY"
  of raPhilox: "PHILOX"

proc toI64(xs: openArray[int]): seq[int64] =
  result = newSeq[int64](xs.len)
  for i, x in xs:
    result[i] = int64(x)

proc i64Attr(name: string; value: int): ShAttrEntry =
  ShAttrEntry(name: name,
    value: ShAttr(kind: akI64, i64: int64(value)))

proc i64ArrayAttr(name: string; values: openArray[int]): ShAttrEntry =
  ShAttrEntry(name: name,
    value: ShAttr(kind: akI64Array, i64s: toI64(values)))

proc boolAttr(name: string; value: bool): ShAttrEntry =
  ShAttrEntry(name: name, value: ShAttr(kind: akBool, b: value))

proc stringAttr(name, value: string): ShAttrEntry =
  ShAttrEntry(name: name, value: ShAttr(kind: akString, str: value))

proc rawAttr(name, value: string): ShAttrEntry =
  ShAttrEntry(name: name, value: ShAttr(kind: akRawMlir, mlir: value))

func hasChannelHandle*(ch: ChannelHandle): bool =
  ## True when `ch` names a concrete StableHLO channel.
  ch.handle >= 0 and ch.handleType >= 0

proc initChannelHandle*(handle, handleType: int): ChannelHandle =
  ## Constructs a StableHLO channel handle.
  if handle < 0 or handleType < 0:
    raise newException(ValueError,
      "initChannelHandle: handle and type must be non-negative")
  ChannelHandle(handle: handle, handleType: handleType)

func channelHandleMlir(ch: ChannelHandle): string =
  "#stablehlo.channel_handle<handle = " & $ch.handle &
    ", type = " & $ch.handleType & ">"

proc matrixAttr(name: string; pairs: openArray[array[2, int]]): ShAttrEntry =
  var vals = newSeq[int64](pairs.len * 2)
  for i, pair in pairs:
    vals[i * 2] = int64(pair[0])
    vals[i * 2 + 1] = int64(pair[1])
  ShAttrEntry(name: name,
    value: ShAttr(kind: akI64Matrix, matRows: pairs.len, matCols: 2,
      matVals: vals))

func renderIntList(xs: openArray[int]): string =
  result = "["
  for i, x in xs:
    if i > 0: result.add ", "
    result.add $x
  result.add ']'

func gatherDimsMlir(dims: GatherDimensionNumbers): string =
  "#stablehlo.gather<offset_dims = " & renderIntList(dims.offsetDims) &
    ", collapsed_slice_dims = " & renderIntList(dims.collapsedSliceDims) &
    ", operand_batching_dims = " & renderIntList(dims.operandBatchingDims) &
    ", start_indices_batching_dims = " &
    renderIntList(dims.startIndicesBatchingDims) &
    ", start_index_map = " & renderIntList(dims.startIndexMap) &
    ", index_vector_dim = " & $dims.indexVectorDim & ">"

func scatterDimsMlir(dims: ScatterDimensionNumbers): string =
  "#stablehlo.scatter<update_window_dims = " &
    renderIntList(dims.updateWindowDims) &
    ", inserted_window_dims = " & renderIntList(dims.insertedWindowDims) &
    ", input_batching_dims = " & renderIntList(dims.inputBatchingDims) &
    ", scatter_indices_batching_dims = " &
    renderIntList(dims.scatterIndicesBatchingDims) &
    ", scatter_dims_to_operand_dims = " &
    renderIntList(dims.scatterDimsToOperandDims) &
    ", index_vector_dim = " & $dims.indexVectorDim & ">"

proc replicaGroupsMlir(groups: openArray[seq[int]]): string =
  let rows = groups.len
  let cols = if rows == 0: 0 else: groups[0].len
  result = "dense<["
  for r, group in groups:
    if group.len != cols:
      raise newException(ValueError,
        "replica_groups must be a rectangular int matrix")
    if r > 0: result.add ", "
    result.add '['
    for c, id in group:
      if c > 0: result.add ", "
      result.add $id
    result.add ']'
  result.add "]> : tensor<"
  result.add $rows
  result.add 'x'
  result.add $cols
  result.add "xi64>"

proc requireNonNegativeShape(opName: string; shape: openArray[int]) =
  for i, d in shape:
    if d < 0:
      raise newException(ShBuilderError,
        opName & ": shape dimension #" & $i & " must be non-negative")

proc requireUniqueSortedDims(opName, attrName: string; dims: openArray[int];
    rank: int; sorted = false) =
  var seen = newSeq[bool](rank)
  var prev = -1
  for d in dims:
    if d < 0 or d >= rank:
      raise newException(ShBuilderError,
        opName & ": " & attrName & " dimension " & $d &
          " out of range for rank " & $rank)
    if seen[d]:
      raise newException(ShBuilderError,
        opName & ": " & attrName & " dimension " & $d & " repeated")
    if sorted and d <= prev:
      raise newException(ShBuilderError,
        opName & ": " & attrName & " must be strictly sorted")
    seen[d] = true
    prev = d

func initBuilder*(name = "module"): ShBuilder =
  ## Start a fresh module-level builder. Pass `name` to set the module
  ## symbol; defaults to `"module"`.
  ShBuilder(
    module: ShModule(name: name, funcs: @[]),
    curFn: -1,
  )

func freshValue(fn: var ShFunction; ty: ShValueType): ShValue =
  ## Allocates the next SSA id inside `fn` and records its general type.
  let id = ShValueId(fn.types.len)
  fn.types.add ty.tensorTypeOrDefault
  fn.valueTypes.add ty
  initShValue(id, ty)

func freshId(fn: var ShFunction; ty: ShTensorType): ShValue =
  ## Allocates the next tensor SSA id inside `fn` and records its type.
  freshValue(fn, initValueType(ty))

proc beginValueFunc*(b: var ShBuilder; name: string;
    inputs: openArray[ShValueType]; outputs: openArray[ShValueType];
    visibility = svPublic): seq[ShValueId] =
  ## Open a new function with general StableHLO value descriptors.
  ## Returns the SSA ids for the entry-block arguments, in declaration
  ## order, so the caller can feed them into subsequent op calls.
  if b.curFn >= 0:
    raise newException(ShBuilderError,
      "beginFunc called while another function is open")
  var inputTypes = newSeq[ShTensorType](inputs.len)
  var outputTypes = newSeq[ShTensorType](outputs.len)
  for i, ty in inputs:
    inputTypes[i] = ty.tensorTypeOrDefault
  for i, ty in outputs:
    outputTypes[i] = ty.tensorTypeOrDefault
  var fn = ShFunction(
    name: name,
    visibility: visibility,
    inputTypes: inputTypes,
    inputValueTypes: @inputs,
    outputTypes: outputTypes,
    outputValueTypes: @outputs,
    args: @[],
    ops: @[],
    types: @[ShTensorType()],  # sentinel for InvalidShValueId
    valueTypes: @[initValueType(ShTensorType())],
  )
  result = newSeqOfCap[ShValueId](inputs.len)
  for ty in inputs:
    let v = freshValue(fn, ty)
    fn.args.add v
    result.add v.id
  b.module.funcs.add fn
  b.curFn = b.module.funcs.high

proc beginFunc*(b: var ShBuilder; name: string;
    inputs: openArray[ShTensorType]; outputs: openArray[ShTensorType];
    visibility = svPublic): seq[ShValueId] =
  ## Open a tensor-only function. Returns the SSA ids for the entry-block
  ## arguments, in declaration order, so the caller can feed them into
  ## subsequent op calls.
  beginValueFunc(b, name, tensorValueTypes(inputs), tensorValueTypes(outputs),
    visibility)

proc endFunc*(b: var ShBuilder) =
  ## Close the currently open function. The verifier (in `verify.nim`)
  ## checks that the body ends with `okReturn` whose value types match
  ## `outputTypes`; we don't repeat that check here.
  if b.curFn < 0:
    raise newException(ShBuilderError, "endFunc called with no open function")
  b.curFn = -1

proc getValueType*(b: ShBuilder; id: ShValueId): ShValueType
  {.raises: [ShBuilderError].}

proc beginValueRegion*(b: var ShBuilder;
    argTypes: openArray[ShValueType]): seq[ShValueId]
proc endRegion*(b: var ShBuilder): ShRegion
proc stablehloReturn*(b: var ShBuilder; values: openArray[ShValueId])

proc getType*(b: ShBuilder; id: ShValueId): ShTensorType
    {.raises: [ShBuilderError].} =
  ## Look up the type previously recorded for `id` in the current
  ## function. Raises if `id` is invalid or refers to another function.
  if b.curFn < 0:
    raise newException(ShBuilderError, "getType called with no open function")
  let ty = getValueType(b, id)
  if not ty.isTensor:
    raise newException(ShBuilderError,
      "value id " & $id & " has non-tensor type " & $ty)
  ty.tensor

proc getValueType*(b: ShBuilder; id: ShValueId): ShValueType
    {.raises: [ShBuilderError].} =
  ## Look up the general value type recorded for `id` in the current
  ## function. Falls back to the tensor table for hand-built legacy IR.
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "getValueType called with no open function")
  let fn = b.module.funcs[b.curFn]
  if id.int <= 0:
    raise newException(ShBuilderError, "value id " & $id & " is out of range")
  if fn.valueTypes.len > 0:
    if id.int >= fn.valueTypes.len:
      raise newException(ShBuilderError,
        "value id " & $id & " is out of range")
    fn.valueTypes[id.int]
  else:
    if id.int >= fn.types.len:
      raise newException(ShBuilderError,
        "value id " & $id & " is out of range")
    initValueType(fn.types[id.int])

# ---- ops ------------------------------------------------------------------

proc emitValueOp(b: var ShBuilder; kind: ShOpKind;
    operands: openArray[ShValueId]; resultTypes: openArray[ShValueType];
    attrs: openArray[ShAttrEntry];
    regions: openArray[ShRegion] = []): seq[ShValueId] =
  ## Append an op to the current emission target (innermost open region
  ## if any, else the function body) and allocate result ids. Internal
  ## helper; public op procs wrap it with type checks.
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "no function is currently being built; call beginFunc first")
  var fn = addr b.module.funcs[b.curFn]
  var op = ShOp(
    kind: kind,
    operands: @operands,
    results: @[],
    attrs: @attrs,
    regions: @regions,
  )
  result = newSeqOfCap[ShValueId](resultTypes.len)
  for ty in resultTypes:
    let v = freshValue(fn[], ty)
    op.results.add v
    result.add v.id
  if b.regionStack.len > 0:
    b.regionStack[^1].ops.add op
  else:
    fn[].ops.add op

proc emitOp(b: var ShBuilder; kind: ShOpKind;
    operands: openArray[ShValueId]; resultTypes: openArray[ShTensorType];
    attrs: openArray[ShAttrEntry];
    regions: openArray[ShRegion] = []): seq[ShValueId] =
  ## Append a tensor-result op to the current emission target.
  emitValueOp(b, kind, operands, tensorValueTypes(resultTypes), attrs, regions)

proc constant*(b: var ShBuilder; dtype: DType;
    shape: openArray[int]; data: openArray[byte]): ShValueId =
  ## Emit a `stablehlo.constant` of dense element type `dtype` and shape
  ## `shape`. `data` is the raw little-endian element bytes; its length
  ## must equal `numElements * dtype.byteSize`.
  let ty = initTensorType(dtype, shape)
  let expected = ty.numElements * dtype.byteSize
  if data.len != expected:
    raise newException(ShBuilderError,
      "constant data size mismatch: got " & $data.len &
      " bytes, expected " & $expected & " for " & $ty)
  let attrs = @[
    ShAttrEntry(name: "value", value: ShAttr(
      kind: akDenseElements,
      denseDtype: dtype,
      denseShape: @shape,
      denseBytes: @data,
    )),
  ]
  emitOp(b, okConstant, [], [ty], attrs)[0]

proc binaryElementwise(b: var ShBuilder; kind: ShOpKind;
    lhs, rhs: ShValueId; opName: string): ShValueId =
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt != rt:
    raise newException(ShBuilderError,
      opName & ": operand types differ — " & $lt & " vs " & $rt)
  emitOp(b, kind, [lhs, rhs], [lt], [])[0]

proc add*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.add`. Operand types must match exactly; v1 has no
  ## broadcasting at the IR level (broadcast ops live in Phase 4).
  binaryElementwise(b, okAdd, lhs, rhs, "add")

proc sub*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.subtract`.
  binaryElementwise(b, okSub, lhs, rhs, "sub")

proc mul*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.multiply`.
  binaryElementwise(b, okMul, lhs, rhs, "mul")

proc neg*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.negate`. Result type matches operand.
  let ty = getType(b, operand)
  emitOp(b, okNeg, [operand], [ty], [])[0]

# ---- Phase 4 binary elementwise ------------------------------------------

proc divide*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.divide`. Operand types must match exactly.
  binaryElementwise(b, okDiv, lhs, rhs, "div")

proc maximum*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.maximum`.
  binaryElementwise(b, okMax, lhs, rhs, "max")

proc minimum*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.minimum`.
  binaryElementwise(b, okMin, lhs, rhs, "min")

proc atan2*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.atan2`.
  binaryElementwise(b, okAtan2, lhs, rhs, "atan2")

proc power*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.power`.
  binaryElementwise(b, okPower, lhs, rhs, "power")

proc remainder*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.remainder`.
  binaryElementwise(b, okRemainder, lhs, rhs, "remainder")

proc andOp*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.and`.
  binaryElementwise(b, okAnd, lhs, rhs, "and")

proc orOp*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.or`.
  binaryElementwise(b, okOr, lhs, rhs, "or")

proc xorOp*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.xor`.
  binaryElementwise(b, okXor, lhs, rhs, "xor")

proc shiftLeft*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.shift_left`.
  binaryElementwise(b, okShiftLeft, lhs, rhs, "shift_left")

proc shiftRightArithmetic*(b: var ShBuilder;
    lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.shift_right_arithmetic`.
  binaryElementwise(b, okShiftRightArithmetic, lhs, rhs,
    "shift_right_arithmetic")

proc shiftRightLogical*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.shift_right_logical`.
  binaryElementwise(b, okShiftRightLogical, lhs, rhs,
    "shift_right_logical")

# ---- Phase 4 unary elementwise -------------------------------------------

proc unaryElementwise(b: var ShBuilder; kind: ShOpKind;
    operand: ShValueId): ShValueId =
  let ty = getType(b, operand)
  emitOp(b, kind, [operand], [ty], [])[0]

proc exponential*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.exponential` (the dialect spelling for `exp`).
  unaryElementwise(b, okExp, operand)

proc log*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.log`.
  unaryElementwise(b, okLog, operand)

proc sqrt*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.sqrt`.
  unaryElementwise(b, okSqrt, operand)

proc abs*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.abs`.
  unaryElementwise(b, okAbs, operand)

proc tanh*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.tanh`.
  unaryElementwise(b, okTanh, operand)

proc cbrt*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.cbrt`.
  unaryElementwise(b, okCbrt, operand)

proc ceil*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.ceil`.
  unaryElementwise(b, okCeil, operand)

proc exponentialMinusOne*(b: var ShBuilder;
    operand: ShValueId): ShValueId =
  ## Emit `stablehlo.exponential_minus_one` (the dialect spelling for
  ## `expm1`).
  unaryElementwise(b, okExponentialMinusOne, operand)

proc floor*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.floor`.
  unaryElementwise(b, okFloor, operand)

proc logPlusOne*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.log_plus_one` (the dialect spelling for `log1p`).
  unaryElementwise(b, okLogPlusOne, operand)

proc logistic*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.logistic`.
  unaryElementwise(b, okLogistic, operand)

proc tan*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.tan`.
  unaryElementwise(b, okTan, operand)

proc sign*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.sign`.
  unaryElementwise(b, okSign, operand)

proc roundNearestAfz*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.round_nearest_afz`.
  unaryElementwise(b, okRoundNearestAfz, operand)

proc roundNearestEven*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.round_nearest_even`.
  unaryElementwise(b, okRoundNearestEven, operand)

proc notOp*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.not`.
  unaryElementwise(b, okNot, operand)

proc countLeadingZeros*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.count_leading_zeros`.
  unaryElementwise(b, okCountLeadingZeros, operand)

proc popcnt*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.popcnt`.
  unaryElementwise(b, okPopcnt, operand)

proc optimizationBarrier*(b: var ShBuilder;
    operand: ShValueId): ShValueId =
  ## Emit single-result `stablehlo.optimization_barrier`.
  unaryElementwise(b, okOptimizationBarrier, operand)

proc convert*(b: var ShBuilder; operand: ShValueId; dtype: DType): ShValueId =
  ## Emit `stablehlo.convert`. Result shape matches the operand and the
  ## element type is changed to `dtype`.
  let inTy = getType(b, operand)
  let outTy = initTensorType(dtype, inTy.shape)
  emitOp(b, okConvert, [operand], [outTy], [])[0]

proc bitcastConvert*(b: var ShBuilder; operand: ShValueId; dtype: DType;
    outputShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.bitcast_convert`. The total input and output bit
  ## counts must match.
  let inTy = getType(b, operand)
  let outTy = initTensorType(dtype, outputShape)
  let inBits = inTy.numElements * inTy.dtype.bitWidth
  let outBits = outTy.numElements * outTy.dtype.bitWidth
  if inBits != outBits:
    raise newException(ShBuilderError,
      "bitcast_convert: input bit count " & $inBits &
        " differs from output bit count " & $outBits)
  emitOp(b, okBitcastConvert, [operand], [outTy], [])[0]

proc bitcastConvert*(b: var ShBuilder; operand: ShValueId;
    dtype: DType): ShValueId =
  ## Emit same-shape `stablehlo.bitcast_convert`.
  bitcastConvert(b, operand, dtype, getType(b, operand).shape)

proc isFinite*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.is_finite`. Operand must be floating point and the
  ## result is a same-shape boolean tensor.
  let inTy = getType(b, operand)
  if not inTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "is_finite: operand must be floating point, got " & $inTy)
  let outTy = initTensorType(dtBool, inTy.shape)
  emitOp(b, okIsFinite, [operand], [outTy], [])[0]

proc reducePrecision*(b: var ShBuilder; operand: ShValueId;
    exponentBits, mantissaBits: int): ShValueId =
  ## Emit `stablehlo.reduce_precision`. Operand and result types match.
  let inTy = getType(b, operand)
  if not inTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "reduce_precision: operand must be floating point, got " & $inTy)
  if exponentBits < 1:
    raise newException(ShBuilderError,
      "reduce_precision: exponentBits must be >= 1")
  if mantissaBits < 0:
    raise newException(ShBuilderError,
      "reduce_precision: mantissaBits must be >= 0")
  let attrs = @[
    ShAttrEntry(name: "exponent_bits",
      value: ShAttr(kind: akI64, i64: int64(exponentBits))),
    ShAttrEntry(name: "mantissa_bits",
      value: ShAttr(kind: akI64, i64: int64(mantissaBits))),
  ]
  emitOp(b, okReducePrecision, [operand], [inTy], attrs)[0]

proc batchNormInference*(b: var ShBuilder;
    operand, scale, offset, mean, variance: ShValueId;
    epsilon: float32; featureIndex: int): ShValueId =
  ## Emit `stablehlo.batch_norm_inference`.
  let operandTy = getType(b, operand)
  let scaleTy = getType(b, scale)
  let offsetTy = getType(b, offset)
  let meanTy = getType(b, mean)
  let varianceTy = getType(b, variance)
  if not operandTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "batch_norm_inference: operand must be floating point, got " &
        $operandTy)
  if featureIndex < 0 or featureIndex >= operandTy.shape.len:
    raise newException(ShBuilderError,
      "batch_norm_inference: featureIndex " & $featureIndex &
        " out of range for rank " & $operandTy.shape.len)
  let featureShape = @[operandTy.shape[featureIndex]]
  for input in [("scale", scaleTy), ("offset", offsetTy),
                ("mean", meanTy), ("variance", varianceTy)]:
    let inputName = input[0]
    let inputTy = input[1]
    if inputTy.dtype != operandTy.dtype:
      raise newException(ShBuilderError,
        "batch_norm_inference: " & inputName & " dtype " & $inputTy.dtype &
          " differs from operand dtype " & $operandTy.dtype)
    if inputTy.shape != featureShape:
      raise newException(ShBuilderError,
        "batch_norm_inference: " & inputName & " shape " & $inputTy.shape &
          " must be " & $featureShape)
  let attrs = @[
    ShAttrEntry(name: "epsilon",
      value: ShAttr(kind: akF64, f64: float64(epsilon))),
    ShAttrEntry(name: "feature_index",
      value: ShAttr(kind: akI64, i64: int64(featureIndex))),
  ]
  emitOp(b, okBatchNormInference,
    [operand, scale, offset, mean, variance], [operandTy], attrs)[0]

proc requireBatchNormFeatureInput(opName, inputName: string;
    operandTy, inputTy: ShTensorType; featureShape: openArray[int]) =
  if inputTy.dtype != operandTy.dtype:
    raise newException(ShBuilderError,
      opName & ": " & inputName & " dtype " & $inputTy.dtype &
        " differs from operand dtype " & $operandTy.dtype)
  if inputTy.shape != @featureShape:
    raise newException(ShBuilderError,
      opName & ": " & inputName & " shape " & $inputTy.shape &
        " must be " & $(@featureShape))

proc batchNormTraining*(b: var ShBuilder;
    operand, scale, offset: ShValueId;
    epsilon: float32; featureIndex: int): seq[ShValueId] =
  ## Emit `stablehlo.batch_norm_training`.
  let operandTy = getType(b, operand)
  let scaleTy = getType(b, scale)
  let offsetTy = getType(b, offset)
  if not operandTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "batch_norm_training: operand must be floating point, got " &
        $operandTy)
  if featureIndex < 0 or featureIndex >= operandTy.shape.len:
    raise newException(ShBuilderError,
      "batch_norm_training: featureIndex " & $featureIndex &
        " out of range for rank " & $operandTy.shape.len)
  let featureShape = @[operandTy.shape[featureIndex]]
  requireBatchNormFeatureInput("batch_norm_training", "scale",
    operandTy, scaleTy, featureShape)
  requireBatchNormFeatureInput("batch_norm_training", "offset",
    operandTy, offsetTy, featureShape)
  let featureTy = initTensorType(operandTy.dtype, featureShape)
  let attrs = @[
    ShAttrEntry(name: "epsilon",
      value: ShAttr(kind: akF64, f64: float64(epsilon))),
    ShAttrEntry(name: "feature_index",
      value: ShAttr(kind: akI64, i64: int64(featureIndex))),
  ]
  emitOp(b, okBatchNormTraining, [operand, scale, offset],
    [operandTy, featureTy, featureTy], attrs)

proc batchNormGrad*(b: var ShBuilder;
    operand, scale, mean, variance, gradOutput: ShValueId;
    epsilon: float32; featureIndex: int): seq[ShValueId] =
  ## Emit `stablehlo.batch_norm_grad`.
  let operandTy = getType(b, operand)
  let scaleTy = getType(b, scale)
  let meanTy = getType(b, mean)
  let varianceTy = getType(b, variance)
  let gradOutputTy = getType(b, gradOutput)
  if not operandTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "batch_norm_grad: operand must be floating point, got " & $operandTy)
  if gradOutputTy != operandTy:
    raise newException(ShBuilderError,
      "batch_norm_grad: gradOutput type " & $gradOutputTy &
        " must match operand type " & $operandTy)
  if featureIndex < 0 or featureIndex >= operandTy.shape.len:
    raise newException(ShBuilderError,
      "batch_norm_grad: featureIndex " & $featureIndex &
        " out of range for rank " & $operandTy.shape.len)
  let featureShape = @[operandTy.shape[featureIndex]]
  requireBatchNormFeatureInput("batch_norm_grad", "scale",
    operandTy, scaleTy, featureShape)
  requireBatchNormFeatureInput("batch_norm_grad", "mean",
    operandTy, meanTy, featureShape)
  requireBatchNormFeatureInput("batch_norm_grad", "variance",
    operandTy, varianceTy, featureShape)
  let featureTy = initTensorType(operandTy.dtype, featureShape)
  let attrs = @[
    ShAttrEntry(name: "epsilon",
      value: ShAttr(kind: akF64, f64: float64(epsilon))),
    ShAttrEntry(name: "feature_index",
      value: ShAttr(kind: akI64, i64: int64(featureIndex))),
  ]
  emitOp(b, okBatchNormGrad,
    [operand, scale, mean, variance, gradOutput],
    [operandTy, featureTy, featureTy], attrs)

proc dotOutputShape*(lhsShape, rhsShape: openArray[int]): seq[int] =
  ## Output shape for `stablehlo.dot` over rank-1/rank-2 operands.
  if lhsShape.len < 1 or lhsShape.len > 2:
    raise newException(ShBuilderError,
      "dot: lhs must be rank 1 or 2, got rank " & $lhsShape.len)
  if rhsShape.len < 1 or rhsShape.len > 2:
    raise newException(ShBuilderError,
      "dot: rhs must be rank 1 or 2, got rank " & $rhsShape.len)
  if lhsShape[^1] != rhsShape[0]:
    raise newException(ShBuilderError,
      "dot: contracting dim mismatch (" & $lhsShape[^1] &
        " vs " & $rhsShape[0] & ")")
  if lhsShape.len == 2:
    result.add lhsShape[0]
  if rhsShape.len == 2:
    result.add rhsShape[1]

proc dot*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.dot` for rank-1/rank-2 dot products.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt.dtype != rt.dtype:
    raise newException(ShBuilderError,
      "dot: dtype mismatch — " & $lt & " vs " & $rt)
  let outTy = initTensorType(lt.dtype, dotOutputShape(lt.shape, rt.shape))
  emitOp(b, okDot, [lhs, rhs], [outTy], [])[0]

proc cholesky*(b: var ShBuilder; operand: ShValueId;
    lower = true): ShValueId =
  ## Emit `stablehlo.cholesky` for a batch of square matrices.
  let ty = getType(b, operand)
  if not ty.dtype.isFloat:
    raise newException(ShBuilderError,
      "cholesky: operand must be floating point, got " & $ty)
  if ty.shape.len < 2:
    raise newException(ShBuilderError,
      "cholesky: operand rank must be at least 2, got " & $ty.shape.len)
  if ty.shape[^1] != ty.shape[^2]:
    raise newException(ShBuilderError,
      "cholesky: innermost matrix dimensions must be square, got " &
        $ty.shape)
  let attrs = @[
    ShAttrEntry(name: "lower",
      value: ShAttr(kind: akBool, b: lower)),
  ]
  emitOp(b, okCholesky, [operand], [ty], attrs)[0]

proc getDimensionSize*(b: var ShBuilder; operand: ShValueId;
    dimension: int): ShValueId =
  ## Emit `stablehlo.get_dimension_size`, returning a scalar `int32`.
  let ty = getType(b, operand)
  if dimension < 0 or dimension >= ty.shape.len:
    raise newException(ShBuilderError,
      "get_dimension_size: dimension " & $dimension &
        " out of range for rank " & $ty.shape.len)
  let attrs = @[
    ShAttrEntry(name: "dimension",
      value: ShAttr(kind: akI64, i64: int64(dimension))),
  ]
  emitOp(b, okGetDimensionSize, [operand],
    [initTensorType(dtInt32, [])], attrs)[0]

proc padOutputShape*(operandShape, edgePaddingLow, edgePaddingHigh,
    interiorPadding: openArray[int]): seq[int] =
  ## Computes the StableHLO `pad` result shape.
  if edgePaddingLow.len != operandShape.len or
      edgePaddingHigh.len != operandShape.len or
      interiorPadding.len != operandShape.len:
    raise newException(ShBuilderError,
      "pad: padding arrays must match operand rank " & $operandShape.len)
  result = newSeq[int](operandShape.len)
  for i, dim in operandShape:
    if interiorPadding[i] < 0:
      raise newException(ShBuilderError,
        "pad: interior padding at dim " & $i & " must be non-negative")
    result[i] = dim + edgePaddingLow[i] + edgePaddingHigh[i] +
      max(dim - 1, 0) * interiorPadding[i]
    if result[i] < 0:
      raise newException(ShBuilderError,
        "pad: result dimension " & $i & " is negative")

proc pad*(b: var ShBuilder; operand, paddingValue: ShValueId;
    edgePaddingLow, edgePaddingHigh,
    interiorPadding: openArray[int]): ShValueId =
  ## Emit `stablehlo.pad`.
  let operandTy = getType(b, operand)
  let paddingTy = getType(b, paddingValue)
  if paddingTy.dtype != operandTy.dtype or paddingTy.shape.len != 0:
    raise newException(ShBuilderError,
      "pad: paddingValue must be a scalar with operand dtype " &
        $operandTy.dtype & ", got " & $paddingTy)
  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)
  let outShape = padOutputShape(operandTy.shape, edgePaddingLow,
    edgePaddingHigh, interiorPadding)
  let attrs = @[
    ShAttrEntry(name: "edge_padding_low",
      value: ShAttr(kind: akI64Array, i64s: toI64(edgePaddingLow))),
    ShAttrEntry(name: "edge_padding_high",
      value: ShAttr(kind: akI64Array, i64s: toI64(edgePaddingHigh))),
    ShAttrEntry(name: "interior_padding",
      value: ShAttr(kind: akI64Array, i64s: toI64(interiorPadding))),
  ]
  emitOp(b, okPad, [operand, paddingValue],
    [initTensorType(operandTy.dtype, outShape)], attrs)[0]

proc broadcast*(b: var ShBuilder; operand: ShValueId;
    broadcastSizes: openArray[int]): ShValueId =
  ## Emit legacy `stablehlo.broadcast`.
  let operandTy = getType(b, operand)
  for i, d in broadcastSizes:
    if d < 0:
      raise newException(ShBuilderError,
        "broadcast: broadcast size #" & $i & " must be non-negative")
  var outShape = @broadcastSizes
  outShape.add operandTy.shape
  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)
  let attrs = @[
    ShAttrEntry(name: "broadcast_sizes",
      value: ShAttr(kind: akI64Array, i64s: toI64(broadcastSizes))),
  ]
  emitOp(b, okBroadcast, [operand],
    [initTensorType(operandTy.dtype, outShape)], attrs)[0]

proc dynamicSlice*(b: var ShBuilder; operand: ShValueId;
    startIndices: openArray[ShValueId];
    sliceSizes: openArray[int]): ShValueId =
  ## Emit `stablehlo.dynamic_slice`.
  let operandTy = getType(b, operand)
  if startIndices.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_slice: start index count " & $startIndices.len &
        " must match operand rank " & $operandTy.shape.len)
  if sliceSizes.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_slice: slice_sizes length " & $sliceSizes.len &
        " must match operand rank " & $operandTy.shape.len)
  for i, id in startIndices:
    let ty = getType(b, id)
    if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
        ty.shape.len != 0:
      raise newException(ShBuilderError,
        "dynamic_slice: start index #" & $i &
          " must be an integer scalar, got " & $ty)
  for i, size in sliceSizes:
    if size < 0 or size > operandTy.shape[i]:
      raise newException(ShBuilderError,
        "dynamic_slice: slice size " & $size &
          " out of range for dim " & $i & " of shape " & $operandTy.shape)
  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)
  var operands = @[operand]
  operands.add startIndices
  let attrs = @[
    ShAttrEntry(name: "slice_sizes",
      value: ShAttr(kind: akI64Array, i64s: toI64(sliceSizes))),
  ]
  emitOp(b, okDynamicSlice, operands,
    [initTensorType(operandTy.dtype, sliceSizes)], attrs)[0]

proc dynamicUpdateSlice*(b: var ShBuilder; operand, update: ShValueId;
    startIndices: openArray[ShValueId]): ShValueId =
  ## Emit `stablehlo.dynamic_update_slice`.
  let operandTy = getType(b, operand)
  let updateTy = getType(b, update)
  if updateTy.dtype != operandTy.dtype:
    raise newException(ShBuilderError,
      "dynamic_update_slice: update dtype " & $updateTy.dtype &
        " differs from operand dtype " & $operandTy.dtype)
  if updateTy.shape.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_update_slice: update rank must match operand rank")
  if startIndices.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_update_slice: start index count " & $startIndices.len &
        " must match operand rank " & $operandTy.shape.len)
  for i in 0 ..< operandTy.shape.len:
    if updateTy.shape[i] > operandTy.shape[i]:
      raise newException(ShBuilderError,
        "dynamic_update_slice: update dim " & $i &
          " is larger than operand dim")
  for i, id in startIndices:
    let ty = getType(b, id)
    if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
        ty.shape.len != 0:
      raise newException(ShBuilderError,
        "dynamic_update_slice: start index #" & $i &
          " must be an integer scalar, got " & $ty)
  var operands = @[operand, update]
  operands.add startIndices
  emitOp(b, okDynamicUpdateSlice, operands, [operandTy], [])[0]

proc iota*(b: var ShBuilder; dtype: DType; shape: openArray[int];
    dimension: int): ShValueId =
  ## Emit `stablehlo.iota`.
  if dtype == dtBool:
    raise newException(ShBuilderError,
      "iota: bool element type is not supported")
  if dimension < 0 or dimension >= shape.len:
    raise newException(ShBuilderError,
      "iota: dimension " & $dimension &
        " out of range for rank " & $shape.len)
  for i, d in shape:
    if d < 0:
      raise newException(ShBuilderError,
        "iota: shape dimension #" & $i & " must be non-negative")
  let attrs = @[
    ShAttrEntry(name: "iota_dimension",
      value: ShAttr(kind: akI64, i64: int64(dimension))),
  ]
  emitOp(b, okIota, [], [initTensorType(dtype, shape)], attrs)[0]

proc replicaId*(b: var ShBuilder): ShValueId =
  ## Emit `stablehlo.replica_id`.
  emitOp(b, okReplicaId, [], [initTensorType(dtUint32, [])], [])[0]

proc partitionId*(b: var ShBuilder): ShValueId =
  ## Emit `stablehlo.partition_id`.
  emitOp(b, okPartitionId, [], [initTensorType(dtUint32, [])], [])[0]

proc createToken*(b: var ShBuilder): ShValueId =
  ## Emit `stablehlo.create_token`.
  emitValueOp(b, okCreateToken, [], [initTokenType()], [])[0]

proc afterAll*(b: var ShBuilder;
    tokens: openArray[ShValueId]): ShValueId =
  ## Emit `stablehlo.after_all`. All operands must be token values.
  for i, id in tokens:
    let ty = getValueType(b, id)
    if not ty.isToken:
      raise newException(ShBuilderError,
        "after_all: operand #" & $i & " must be token, got " & $ty)
  emitValueOp(b, okAfterAll, tokens, [initTokenType()], [])[0]

proc tupleOp*(b: var ShBuilder;
    values: openArray[ShValueId]): ShValueId =
  ## Emit `stablehlo.tuple` for tensor, token, tuple and resource values.
  var elemTypes = newSeq[ShValueType](values.len)
  for i, id in values:
    elemTypes[i] = getValueType(b, id)
  emitValueOp(b, okTuple, values, [initTupleType(elemTypes)], [])[0]

proc getTupleElement*(b: var ShBuilder; operand: ShValueId;
    index: int): ShValueId =
  ## Emit `stablehlo.get_tuple_element`.
  let tupleTy = getValueType(b, operand)
  if not tupleTy.isTuple:
    raise newException(ShBuilderError,
      "get_tuple_element: operand must be tuple, got " & $tupleTy)
  if index < 0 or index >= tupleTy.elements.len:
    raise newException(ShBuilderError,
      "get_tuple_element: index " & $index &
        " out of range for tuple length " & $tupleTy.elements.len)
  let attrs = @[
    ShAttrEntry(name: "index",
      value: ShAttr(kind: akI64, i64: int64(index))),
  ]
  emitValueOp(b, okGetTupleElement, [operand],
    [tupleTy.elements[index]], attrs)[0]

proc complexOp*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Emit `stablehlo.complex`.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt != rt:
    raise newException(ShBuilderError,
      "complex: operand types differ — " & $lt & " vs " & $rt)
  if lt.dtype notin {dtFloat32, dtFloat64}:
    raise newException(ShBuilderError,
      "complex: operands must be float32 or float64 tensors, got " & $lt)
  let outTy = initTensorType(lt.dtype.complexDType, lt.shape)
  emitOp(b, okComplex, [lhs, rhs], [outTy], [])[0]

proc real*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.real`.
  let ty = getType(b, operand)
  if not (ty.dtype.isFloat or ty.dtype.isComplex):
    raise newException(ShBuilderError,
      "real: operand must be floating point or complex, got " & $ty)
  let outTy = initTensorType(ty.dtype.complexPartDType, ty.shape)
  emitOp(b, okReal, [operand], [outTy], [])[0]

proc imag*(b: var ShBuilder; operand: ShValueId): ShValueId =
  ## Emit `stablehlo.imag`.
  let ty = getType(b, operand)
  if not (ty.dtype.isFloat or ty.dtype.isComplex):
    raise newException(ShBuilderError,
      "imag: operand must be floating point or complex, got " & $ty)
  let outTy = initTensorType(ty.dtype.complexPartDType, ty.shape)
  emitOp(b, okImag, [operand], [outTy], [])[0]

proc dynamicBroadcastInDim*(b: var ShBuilder; operand, outputDimensions:
    ShValueId; resultShape, broadcastDimensions: openArray[int];
    knownExpandingDimensions: openArray[int] = [];
    knownNonexpandingDimensions: openArray[int] = []): ShValueId =
  ## Emit `stablehlo.dynamic_broadcast_in_dim`. `resultShape` is the
  ## statically-known result type while `outputDimensions` carries the
  ## runtime dimension sizes.
  let operandTy = getType(b, operand)
  let dimsTy = getType(b, outputDimensions)
  if not (dimsTy.dtype.isSignedInt or dimsTy.dtype.isUnsignedInt) or
      dimsTy.shape != @[resultShape.len]:
    raise newException(ShBuilderError,
      "dynamic_broadcast_in_dim: outputDimensions must be an integer " &
        "vector of length " & $resultShape.len & ", got " & $dimsTy)
  requireNonNegativeShape("dynamic_broadcast_in_dim", resultShape)
  if broadcastDimensions.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_broadcast_in_dim: broadcastDimensions length " &
        $broadcastDimensions.len & " does not match operand rank " &
        $operandTy.shape.len)
  var seenOut = newSeq[bool](resultShape.len)
  for i, d in broadcastDimensions:
    if d < 0 or d >= resultShape.len:
      raise newException(ShBuilderError,
        "dynamic_broadcast_in_dim: broadcast dim " & $d &
          " out of range for result rank " & $resultShape.len)
    if seenOut[d]:
      raise newException(ShBuilderError,
        "dynamic_broadcast_in_dim: result dim " & $d & " mapped twice")
    seenOut[d] = true
    let inDim = operandTy.shape[i]
    let outDim = resultShape[d]
    if inDim != 1 and inDim != outDim:
      raise newException(ShBuilderError,
        "dynamic_broadcast_in_dim: operand dim " & $i & " (size " &
          $inDim & ") cannot broadcast to result dim " & $d &
          " (size " & $outDim & ")")
  requireUniqueSortedDims("dynamic_broadcast_in_dim",
    "known_expanding_dimensions", knownExpandingDimensions,
    operandTy.shape.len)
  requireUniqueSortedDims("dynamic_broadcast_in_dim",
    "known_nonexpanding_dimensions", knownNonexpandingDimensions,
    operandTy.shape.len)
  var knownSeen = newSeq[bool](operandTy.shape.len)
  for d in knownExpandingDimensions:
    knownSeen[d] = true
  for d in knownNonexpandingDimensions:
    if knownSeen[d]:
      raise newException(ShBuilderError,
        "dynamic_broadcast_in_dim: known dimension " & $d &
          " cannot be both expanding and nonexpanding")
  var attrs = @[
    ShAttrEntry(name: "broadcast_dimensions",
      value: ShAttr(kind: akI64Array, i64s: toI64(broadcastDimensions))),
  ]
  if knownExpandingDimensions.len > 0:
    attrs.add ShAttrEntry(name: "known_expanding_dimensions",
      value: ShAttr(kind: akI64Array,
        i64s: toI64(knownExpandingDimensions)))
  if knownNonexpandingDimensions.len > 0:
    attrs.add ShAttrEntry(name: "known_nonexpanding_dimensions",
      value: ShAttr(kind: akI64Array,
        i64s: toI64(knownNonexpandingDimensions)))
  emitOp(b, okDynamicBroadcastInDim, [operand, outputDimensions],
    [initTensorType(operandTy.dtype, resultShape)], attrs)[0]

proc fftOutputType*(operandTy: ShTensorType; fftType: FftType;
    fftLength: openArray[int]): ShTensorType =
  ## Infer the StableHLO `fft` result type from the operand type,
  ## algorithm, and static FFT lengths.
  if fftLength.len < 1 or fftLength.len > 3:
    raise newException(ShBuilderError,
      "fft: fftLength must have length 1..3, got " & $fftLength.len)
  if fftLength.len > operandTy.shape.len:
    raise newException(ShBuilderError,
      "fft: fftLength rank " & $fftLength.len &
        " exceeds operand rank " & $operandTy.shape.len)
  for i, d in fftLength:
    if d < 0:
      raise newException(ShBuilderError,
        "fft: fftLength dimension #" & $i & " must be non-negative")
  let suffixStart = operandTy.shape.len - fftLength.len
  case fftType
  of ftFft, ftIfft:
    if not operandTy.dtype.isComplex:
      raise newException(ShBuilderError,
        "fft: FFT/IFFT operand must be complex, got " & $operandTy)
    for i, d in fftLength:
      if operandTy.shape[suffixStart + i] != d:
        raise newException(ShBuilderError,
          "fft: fftLength does not match operand suffix shape")
    result = operandTy
  of ftRfft:
    if not operandTy.dtype.isFloat:
      raise newException(ShBuilderError,
        "fft: RFFT operand must be floating point, got " & $operandTy)
    for i, d in fftLength:
      if operandTy.shape[suffixStart + i] != d:
        raise newException(ShBuilderError,
          "fft: fftLength does not match real operand suffix shape")
    var outShape = operandTy.shape
    outShape[^1] =
      if operandTy.shape[^1] == 0: 0
      else: operandTy.shape[^1] div 2 + 1
    result = initTensorType(operandTy.dtype.complexDType, outShape)
  of ftIrfft:
    if not operandTy.dtype.isComplex:
      raise newException(ShBuilderError,
        "fft: IRFFT operand must be complex, got " & $operandTy)
    var outShape = operandTy.shape
    for i, d in fftLength:
      outShape[suffixStart + i] = d
    let expectedLast =
      if outShape[^1] == 0: 0
      else: outShape[^1] div 2 + 1
    if operandTy.shape[^1] != expectedLast:
      raise newException(ShBuilderError,
        "fft: IRFFT operand last dim " & $operandTy.shape[^1] &
          " must equal result last dim / 2 + 1 (" & $expectedLast & ")")
    result = initTensorType(operandTy.dtype.complexPartDType, outShape)

proc fft*(b: var ShBuilder; operand: ShValueId; fftType: FftType;
    fftLength: openArray[int]): ShValueId =
  ## Emit `stablehlo.fft`.
  let operandTy = getType(b, operand)
  let outTy = fftOutputType(operandTy, fftType, fftLength)
  let attrs = @[
    ShAttrEntry(name: "fft_type",
      value: ShAttr(kind: akString, str: fftType.stablehloName)),
    ShAttrEntry(name: "fft_length",
      value: ShAttr(kind: akI64Array, i64s: toI64(fftLength))),
  ]
  emitOp(b, okFft, [operand], [outTy], attrs)[0]

proc triangularSolveOutputShape*(aTy, bTy: ShTensorType;
    leftSide: bool): seq[int] =
  ## Validate `triangular_solve` input shapes and return the output
  ## shape, which matches `b`.
  if aTy.dtype != bTy.dtype:
    raise newException(ShBuilderError,
      "triangular_solve: dtype mismatch — " & $aTy & " vs " & $bTy)
  if not (aTy.dtype.isFloat or aTy.dtype.isComplex):
    raise newException(ShBuilderError,
      "triangular_solve: operands must be floating point or complex, got " &
        $aTy.dtype)
  if aTy.shape.len < 2 or bTy.shape.len < 2:
    raise newException(ShBuilderError,
      "triangular_solve: operands must have rank >= 2")
  if aTy.shape.len != bTy.shape.len:
    raise newException(ShBuilderError,
      "triangular_solve: operand ranks must match")
  if aTy.shape[^1] != aTy.shape[^2]:
    raise newException(ShBuilderError,
      "triangular_solve: coefficient matrix must be square")
  for i in 0 ..< max(aTy.shape.len - 2, 0):
    if aTy.shape[i] != bTy.shape[i]:
      raise newException(ShBuilderError,
        "triangular_solve: batch dim " & $i & " mismatch")
  let matrixDim = aTy.shape[^1]
  let bAxis = if leftSide: bTy.shape.len - 2 else: bTy.shape.len - 1
  if bTy.shape[bAxis] != matrixDim:
    raise newException(ShBuilderError,
      "triangular_solve: b matrix dimension " & $bTy.shape[bAxis] &
        " must match a matrix size " & $matrixDim)
  result = bTy.shape

proc triangularSolve*(b: var ShBuilder; a, rhs: ShValueId;
    leftSide = true; lower = true; unitDiagonal = false;
    transposeA = tkNoTranspose): ShValueId =
  ## Emit `stablehlo.triangular_solve`.
  let aTy = getType(b, a)
  let rhsTy = getType(b, rhs)
  let outShape = triangularSolveOutputShape(aTy, rhsTy, leftSide)
  let attrs = @[
    ShAttrEntry(name: "left_side",
      value: ShAttr(kind: akBool, b: leftSide)),
    ShAttrEntry(name: "lower",
      value: ShAttr(kind: akBool, b: lower)),
    ShAttrEntry(name: "unit_diagonal",
      value: ShAttr(kind: akBool, b: unitDiagonal)),
    ShAttrEntry(name: "transpose_a",
      value: ShAttr(kind: akString, str: transposeA.stablehloName)),
  ]
  emitOp(b, okTriangularSolve, [a, rhs],
    [initTensorType(rhsTy.dtype, outShape)], attrs)[0]

proc einsum*(b: var ShBuilder; lhs, rhs: ShValueId; config: string;
    resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.einsum`. The result shape is explicit because the
  ## StableHLO op carries only the TF-style config string.
  if config.len == 0:
    raise newException(ShBuilderError, "einsum: config must not be empty")
  let lhsTy = getType(b, lhs)
  let rhsTy = getType(b, rhs)
  if lhsTy.dtype != rhsTy.dtype:
    raise newException(ShBuilderError,
      "einsum: dtype mismatch — " & $lhsTy & " vs " & $rhsTy)
  requireNonNegativeShape("einsum", resultShape)
  let attrs = @[
    ShAttrEntry(name: "einsum_config",
      value: ShAttr(kind: akString, str: config)),
  ]
  emitOp(b, okEinsum, [lhs, rhs],
    [initTensorType(lhsTy.dtype, resultShape)], attrs)[0]

proc unaryEinsum*(b: var ShBuilder; operand: ShValueId; config: string;
    resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.unary_einsum`.
  if config.len == 0:
    raise newException(ShBuilderError,
      "unary_einsum: config must not be empty")
  let operandTy = getType(b, operand)
  requireNonNegativeShape("unary_einsum", resultShape)
  let attrs = @[
    ShAttrEntry(name: "einsum_config",
      value: ShAttr(kind: akString, str: config)),
  ]
  emitOp(b, okUnaryEinsum, [operand],
    [initTensorType(operandTy.dtype, resultShape)], attrs)[0]

proc torchIndexSelectOutputShape*(operandShape, indexShape: openArray[int];
    dim, batchDims: int): seq[int] =
  ## Infer StableHLO `torch_index_select` output shape.
  if dim < 0 or dim >= operandShape.len:
    raise newException(ShBuilderError,
      "torch_index_select: dim " & $dim &
        " out of range for operand rank " & $operandShape.len)
  if batchDims < 0 or batchDims > dim or batchDims > indexShape.len:
    raise newException(ShBuilderError,
      "torch_index_select: invalid batchDims " & $batchDims)
  if batchDims > operandShape.len:
    raise newException(ShBuilderError,
      "torch_index_select: batchDims exceeds operand rank")
  for i in 0 ..< batchDims:
    if operandShape[i] != indexShape[i]:
      raise newException(ShBuilderError,
        "torch_index_select: batch dim " & $i & " mismatch")
  result = @[]
  for i in 0 ..< dim:
    result.add operandShape[i]
  for i in batchDims ..< indexShape.len:
    result.add indexShape[i]
  for i in dim + 1 ..< operandShape.len:
    result.add operandShape[i]

proc torchIndexSelect*(b: var ShBuilder; operand, index: ShValueId;
    dim, batchDims: int): ShValueId =
  ## Emit `stablehlo.torch_index_select`.
  let operandTy = getType(b, operand)
  let indexTy = getType(b, index)
  if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
    raise newException(ShBuilderError,
      "torch_index_select: index must be an integer tensor, got " & $indexTy)
  let outShape = torchIndexSelectOutputShape(operandTy.shape, indexTy.shape,
    dim, batchDims)
  let attrs = @[
    ShAttrEntry(name: "dim",
      value: ShAttr(kind: akI64, i64: int64(dim))),
    ShAttrEntry(name: "batch_dims",
      value: ShAttr(kind: akI64, i64: int64(batchDims))),
  ]
  emitOp(b, okTorchIndexSelect, [operand, index],
    [initTensorType(operandTy.dtype, outShape)], attrs)[0]

proc appendChannelAttr(attrs: var seq[ShAttrEntry]; ch: ChannelHandle) =
  if ch.hasChannelHandle:
    attrs.add rawAttr("channel_handle", channelHandleMlir(ch))

proc scalarElementType(ty: ShTensorType): ShValueType =
  initValueType(initTensorType(ty.dtype, []))

proc valueTypesOfIds(b: ShBuilder; ids: openArray[ShValueId]):
    seq[ShValueType] =
  result = newSeq[ShValueType](ids.len)
  for i, id in ids:
    result[i] = getValueType(b, id)

proc resultValueTypes(b: ShBuilder; ids: openArray[ShValueId]):
    seq[ShValueType] =
  valueTypesOfIds(b, ids)

proc checkResultTypes(b: ShBuilder; opName: string;
    ids: openArray[ShValueId]; expected: openArray[ShValueType]) =
  if ids.len != expected.len:
    raise newException(ShBuilderError,
      opName & ": region returned " & $ids.len & " value(s), expected " &
        $expected.len)
  for i, id in ids:
    let ty = getValueType(b, id)
    if ty != expected[i]:
      raise newException(ShBuilderError,
        opName & ": region result #" & $i & " type " & $ty &
          " does not match expected " & $expected[i])

proc rng*(b: var ShBuilder; a, bound, shape: ShValueId;
    distribution: RngDistribution; resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.rng`. `a` and `bound` are scalar bounds; `shape`
  ## is the runtime dimension tensor matching `resultShape.len`.
  let aTy = getType(b, a)
  let bTy = getType(b, bound)
  let shapeTy = getType(b, shape)
  if aTy != bTy:
    raise newException(ShBuilderError,
      "rng: bound operand types differ — " & $aTy & " vs " & $bTy)
  if aTy.shape.len != 0:
    raise newException(ShBuilderError,
      "rng: bound operands must be scalar tensors, got " & $aTy)
  if not (aTy.dtype == dtBool or aTy.dtype.isSignedInt or
      aTy.dtype.isUnsignedInt or aTy.dtype.isFloat):
    raise newException(ShBuilderError,
      "rng: unsupported bound dtype " & $aTy.dtype)
  if distribution == rdNormal and not aTy.dtype.isFloat:
    raise newException(ShBuilderError,
      "rng: NORMAL distribution requires floating-point bounds")
  if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
      shapeTy.shape != @[resultShape.len]:
    raise newException(ShBuilderError,
      "rng: shape must be an integer vector of length " & $resultShape.len)
  requireNonNegativeShape("rng", resultShape)
  let attrs = @[
    rawAttr("rng_distribution",
      "#stablehlo<rng_distribution " & distribution.stablehloName & ">"),
  ]
  emitOp(b, okRng, [a, bound, shape],
    [initTensorType(aTy.dtype, resultShape)], attrs)[0]

proc rngBitGenerator*(b: var ShBuilder; initialState: ShValueId;
    algorithm: RngAlgorithm; outputType: ShTensorType): seq[ShValueId] =
  ## Emit `stablehlo.rng_bit_generator`, returning
  ## `[output_state, output]`.
  let stateTy = getType(b, initialState)
  if not (stateTy.dtype.isSignedInt or stateTy.dtype.isUnsignedInt or
      stateTy.dtype.isFloat):
    raise newException(ShBuilderError,
      "rng_bit_generator: state must be int or float tensor, got " & $stateTy)
  if not (outputType.dtype.isSignedInt or outputType.dtype.isUnsignedInt or
      outputType.dtype.isFloat):
    raise newException(ShBuilderError,
      "rng_bit_generator: output must be int or float tensor, got " &
        $outputType)
  requireNonNegativeShape("rng_bit_generator", outputType.shape)
  let attrs = @[
    rawAttr("rng_algorithm",
      "#stablehlo<rng_algorithm " & algorithm.stablehloName & ">"),
  ]
  emitOp(b, okRngBitGenerator, [initialState], [stateTy, outputType], attrs)

proc gather*(b: var ShBuilder; operand, startIndices: ShValueId;
    dims: GatherDimensionNumbers; sliceSizes, resultShape: openArray[int];
    indicesAreSorted = false): ShValueId =
  ## Emit `stablehlo.gather`. Result shape is explicit; dimension-number
  ## consistency is checked by the verifier and upstream StableHLO.
  let operandTy = getType(b, operand)
  let indexTy = getType(b, startIndices)
  if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
    raise newException(ShBuilderError,
      "gather: startIndices must be an integer tensor, got " & $indexTy)
  if sliceSizes.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "gather: sliceSizes length must match operand rank")
  for i, s in sliceSizes:
    if s < 0 or s > operandTy.shape[i]:
      raise newException(ShBuilderError,
        "gather: slice size " & $s & " out of range for operand dim " & $i)
  requireNonNegativeShape("gather", resultShape)
  let attrs = @[
    rawAttr("dimension_numbers", gatherDimsMlir(dims)),
    i64ArrayAttr("slice_sizes", sliceSizes),
    boolAttr("indices_are_sorted", indicesAreSorted),
  ]
  emitOp(b, okGather, [operand, startIndices],
    [initTensorType(operandTy.dtype, resultShape)], attrs)[0]

proc dynamicGather*(b: var ShBuilder;
    operand, startIndices, sliceSizes: ShValueId;
    dims: GatherDimensionNumbers; resultShape: openArray[int];
    indicesAreSorted = false): ShValueId =
  ## Emit `stablehlo.dynamic_gather`, with `sliceSizes` supplied as a
  ## runtime integer vector.
  let operandTy = getType(b, operand)
  let indexTy = getType(b, startIndices)
  let sizesTy = getType(b, sliceSizes)
  if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
    raise newException(ShBuilderError,
      "dynamic_gather: startIndices must be integer tensor")
  if not (sizesTy.dtype.isSignedInt or sizesTy.dtype.isUnsignedInt) or
      sizesTy.shape != @[operandTy.shape.len]:
    raise newException(ShBuilderError,
      "dynamic_gather: sliceSizes must be an integer vector matching " &
        "operand rank")
  requireNonNegativeShape("dynamic_gather", resultShape)
  let attrs = @[
    rawAttr("dimension_numbers", gatherDimsMlir(dims)),
    boolAttr("indices_are_sorted", indicesAreSorted),
  ]
  emitOp(b, okDynamicGather, [operand, startIndices, sliceSizes],
    [initTensorType(operandTy.dtype, resultShape)], attrs)[0]

proc sort*(b: var ShBuilder; inputs: openArray[ShValueId];
    dimension = -1; isStable = false;
    comparator: ShArgRegionBuilder): seq[ShValueId] =
  ## Emit `stablehlo.sort`. The comparator receives two scalar values per
  ## input and must return a scalar bool.
  if inputs.len == 0:
    raise newException(ShBuilderError, "sort: inputs must not be empty")
  var inTypes = newSeq[ShTensorType](inputs.len)
  for i, id in inputs:
    inTypes[i] = getType(b, id)
    if inTypes[i].shape != inTypes[0].shape:
      raise newException(ShBuilderError,
        "sort: all inputs must have the same shape")
  let rank = inTypes[0].shape.len
  if dimension < -1 or dimension >= rank:
    raise newException(ShBuilderError,
      "sort: dimension out of range for rank " & $rank)
  var argTypes: seq[ShValueType] = @[]
  for ty in inTypes:
    argTypes.add scalarElementType(ty)
    argTypes.add scalarElementType(ty)
  let args = beginValueRegion(b, argTypes)
  let cmp = comparator(b, args)
  let boolScalar = initValueType(initTensorType(dtBool, []))
  checkResultTypes(b, "sort", cmp, [boolScalar])
  stablehloReturn(b, cmp)
  let region = endRegion(b)
  let attrs = @[
    i64Attr("dimension", dimension),
    boolAttr("is_stable", isStable),
  ]
  emitOp(b, okSort, inputs, inTypes, attrs, [region])

proc mapOp*(b: var ShBuilder; inputs: openArray[ShValueId];
    outputDtypes: openArray[DType]; dimensions: openArray[int];
    computation: ShArgRegionBuilder): seq[ShValueId] =
  ## Emit `stablehlo.map`. All inputs must have the same shape; result
  ## shapes mirror the inputs and result dtypes come from `outputDtypes`.
  if inputs.len == 0:
    raise newException(ShBuilderError, "map: inputs must not be empty")
  if outputDtypes.len == 0:
    raise newException(ShBuilderError, "map: outputs must not be empty")
  var inTypes = newSeq[ShTensorType](inputs.len)
  for i, id in inputs:
    inTypes[i] = getType(b, id)
    if inTypes[i].shape != inTypes[0].shape:
      raise newException(ShBuilderError,
        "map: all inputs must have the same shape")
  requireUniqueSortedDims("map", "dimensions", dimensions,
    inTypes[0].shape.len, sorted = true)
  var argTypes: seq[ShValueType] = @[]
  for ty in inTypes:
    argTypes.add scalarElementType(ty)
  let args = beginValueRegion(b, argTypes)
  let bodyResults = computation(b, args)
  var outTypes = newSeq[ShTensorType](outputDtypes.len)
  var expected = newSeq[ShValueType](outputDtypes.len)
  for i, dtype in outputDtypes:
    outTypes[i] = initTensorType(dtype, inTypes[0].shape)
    expected[i] = initValueType(initTensorType(dtype, []))
  checkResultTypes(b, "map", bodyResults, expected)
  stablehloReturn(b, bodyResults)
  let region = endRegion(b)
  emitOp(b, okMap, inputs, outTypes,
    [i64ArrayAttr("dimensions", dimensions)], [region])

proc caseOp*(b: var ShBuilder; index: ShValueId;
    branches: openArray[ShRegionBuilder]): seq[ShValueId] =
  ## Emit `stablehlo.case`. Branch result types are inferred from the
  ## first branch and checked against every other branch.
  if branches.len == 0:
    raise newException(ShBuilderError,
      "case: at least one branch is required")
  let indexTy = getType(b, index)
  if indexTy.dtype != dtInt32:
    raise newException(ShBuilderError,
      "case: index must be an int32 tensor, got " & $indexTy)
  var regions: seq[ShRegion] = @[]
  var outTypes: seq[ShValueType] = @[]
  for i, branch in branches:
    discard beginValueRegion(b, [])
    let ids = branch(b)
    if ids.len == 0:
      raise newException(ShBuilderError,
        "case: branch #" & $i & " returned no values")
    if i == 0:
      outTypes = resultValueTypes(b, ids)
    else:
      checkResultTypes(b, "case branch #" & $i, ids, outTypes)
    stablehloReturn(b, ids)
    regions.add endRegion(b)
  emitValueOp(b, okCase, [index], outTypes, [], regions)

proc scatter*(b: var ShBuilder; inputs: openArray[ShValueId];
    scatterIndices: ShValueId; updates: openArray[ShValueId];
    dims: ScatterDimensionNumbers; updateComputation: ShArgRegionBuilder;
    indicesAreSorted = false; uniqueIndices = false): seq[ShValueId] =
  ## Emit `stablehlo.scatter`. The update computation receives two
  ## scalar values per input/update pair and returns one scalar per input.
  if inputs.len == 0 or inputs.len != updates.len:
    raise newException(ShBuilderError,
      "scatter: inputs and updates must have the same non-zero length")
  let idxTy = getType(b, scatterIndices)
  if not (idxTy.dtype.isSignedInt or idxTy.dtype.isUnsignedInt):
    raise newException(ShBuilderError,
      "scatter: scatterIndices must be an integer tensor")
  var inputTypes = newSeq[ShTensorType](inputs.len)
  var argTypes: seq[ShValueType] = @[]
  var expected: seq[ShValueType] = @[]
  for i, id in inputs:
    inputTypes[i] = getType(b, id)
    let updateTy = getType(b, updates[i])
    if updateTy.dtype != inputTypes[i].dtype:
      raise newException(ShBuilderError,
        "scatter: update #" & $i & " dtype differs from input dtype")
    argTypes.add scalarElementType(inputTypes[i])
    argTypes.add scalarElementType(inputTypes[i])
    expected.add scalarElementType(inputTypes[i])
  let args = beginValueRegion(b, argTypes)
  let bodyResults = updateComputation(b, args)
  checkResultTypes(b, "scatter", bodyResults, expected)
  stablehloReturn(b, bodyResults)
  let region = endRegion(b)
  var operands = @inputs
  operands.add scatterIndices
  operands.add updates
  let attrs = @[
    rawAttr("scatter_dimension_numbers", scatterDimsMlir(dims)),
    boolAttr("indices_are_sorted", indicesAreSorted),
    boolAttr("unique_indices", uniqueIndices),
  ]
  emitOp(b, okScatter, operands, inputTypes, attrs, [region])

proc selectAndScatter*(b: var ShBuilder; operand, source, initValue: ShValueId;
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    selectComputation, scatterComputation: ShArgRegionBuilder): ShValueId =
  ## Emit `stablehlo.select_and_scatter`.
  let operandTy = getType(b, operand)
  let sourceTy = getType(b, source)
  let initTy = getType(b, initValue)
  if initTy.dtype != operandTy.dtype or initTy.shape.len != 0:
    raise newException(ShBuilderError,
      "select_and_scatter: initValue must be a scalar with operand dtype")
  if sourceTy.dtype != operandTy.dtype:
    raise newException(ShBuilderError,
      "select_and_scatter: source dtype must match operand")
  let rank = operandTy.shape.len
  if windowDimensions.len != rank or windowStrides.len != rank or
      padding.len != rank:
    raise newException(ShBuilderError,
      "select_and_scatter: window attributes must match operand rank")
  var padFlat = newSeq[int64](padding.len * 2)
  for i in 0 ..< padding.len:
    padFlat[i * 2] = int64(padding[i][0])
    padFlat[i * 2 + 1] = int64(padding[i][1])
  let elemTy = scalarElementType(operandTy)
  let selectArgs = beginValueRegion(b, [elemTy, elemTy])
  let pred = selectComputation(b, selectArgs)
  checkResultTypes(b, "select_and_scatter select", pred,
    [initValueType(initTensorType(dtBool, []))])
  stablehloReturn(b, pred)
  let selectRegion = endRegion(b)
  let scatterArgs = beginValueRegion(b, [elemTy, elemTy])
  let scatter = scatterComputation(b, scatterArgs)
  checkResultTypes(b, "select_and_scatter scatter", scatter, [elemTy])
  stablehloReturn(b, scatter)
  let scatterRegion = endRegion(b)
  let attrs = @[
    i64ArrayAttr("window_dimensions", windowDimensions),
    i64ArrayAttr("window_strides", windowStrides),
    ShAttrEntry(name: "padding",
      value: ShAttr(kind: akI64Matrix, matRows: padding.len, matCols: 2,
        matVals: padFlat)),
  ]
  emitOp(b, okSelectAndScatter, [operand, source, initValue],
    [operandTy], attrs, [selectRegion, scatterRegion])[0]

proc setDimensionSize*(b: var ShBuilder; operand, size: ShValueId;
    dimension: int): ShValueId =
  ## Emit `stablehlo.set_dimension_size`.
  let operandTy = getType(b, operand)
  let sizeTy = getType(b, size)
  if dimension < 0 or dimension >= operandTy.shape.len:
    raise newException(ShBuilderError,
      "set_dimension_size: dimension " & $dimension &
        " out of range for rank " & $operandTy.shape.len)
  if not (sizeTy.dtype.isSignedInt or sizeTy.dtype.isUnsignedInt) or
      sizeTy.shape.len != 0:
    raise newException(ShBuilderError,
      "set_dimension_size: size must be an integer scalar, got " & $sizeTy)
  let attrs = @[
    ShAttrEntry(name: "dimension",
      value: ShAttr(kind: akI64, i64: int64(dimension))),
  ]
  emitOp(b, okSetDimensionSize, [operand, size], [operandTy], attrs)[0]

proc dynamicReshape*(b: var ShBuilder; operand, outputShape: ShValueId;
    resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.dynamic_reshape`.
  let operandTy = getType(b, operand)
  let shapeTy = getType(b, outputShape)
  if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
      shapeTy.shape.len != 1:
    raise newException(ShBuilderError,
      "dynamic_reshape: output_shape must be a rank-1 integer tensor, got " &
        $shapeTy)
  if shapeTy.shape[0] != resultShape.len:
    raise newException(ShBuilderError,
      "dynamic_reshape: output_shape length " & $shapeTy.shape[0] &
        " must match result rank " & $resultShape.len)
  var inElements = operandTy.numElements
  var outElements = 1
  for d in resultShape:
    if d < 0:
      raise newException(ShBuilderError,
        "dynamic_reshape: result shape contains negative dim " & $d)
    outElements *= d
  if inElements != outElements:
    raise newException(ShBuilderError,
      "dynamic_reshape: element count mismatch")
  emitOp(b, okDynamicReshape, [operand, outputShape],
    [initTensorType(operandTy.dtype, resultShape)], [])[0]

proc dynamicPad*(b: var ShBuilder; operand, paddingValue, edgePaddingLow,
    edgePaddingHigh, interiorPadding: ShValueId;
    resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.dynamic_pad`.
  let operandTy = getType(b, operand)
  let paddingTy = getType(b, paddingValue)
  if paddingTy.dtype != operandTy.dtype or paddingTy.shape.len != 0:
    raise newException(ShBuilderError,
      "dynamic_pad: paddingValue must be scalar with operand dtype")
  for (inputName, id) in [("edge_padding_low", edgePaddingLow),
                         ("edge_padding_high", edgePaddingHigh),
                         ("interior_padding", interiorPadding)]:
    let ty = getType(b, id)
    if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
        ty.shape != @[operandTy.shape.len]:
      raise newException(ShBuilderError,
        "dynamic_pad: " & inputName & " must be an integer vector of length " &
          $operandTy.shape.len & ", got " & $ty)
  if resultShape.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "dynamic_pad: result rank must match operand rank")
  for i, d in resultShape:
    if d < 0:
      raise newException(ShBuilderError,
        "dynamic_pad: result dim #" & $i & " must be non-negative")
  emitOp(b, okDynamicPad,
    [operand, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding],
    [initTensorType(operandTy.dtype, resultShape)], [])[0]

proc dynamicIota*(b: var ShBuilder; dtype: DType; outputShape: ShValueId;
    resultShape: openArray[int]; dimension: int): ShValueId =
  ## Emit `stablehlo.dynamic_iota`.
  let shapeTy = getType(b, outputShape)
  if dtype == dtBool:
    raise newException(ShBuilderError,
      "dynamic_iota: bool element type is not supported")
  if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
      shapeTy.shape != @[resultShape.len]:
    raise newException(ShBuilderError,
      "dynamic_iota: output_shape must be an integer vector of length " &
        $resultShape.len)
  if dimension < 0 or dimension >= resultShape.len:
    raise newException(ShBuilderError,
      "dynamic_iota: dimension " & $dimension &
        " out of range for rank " & $resultShape.len)
  let attrs = @[
    ShAttrEntry(name: "iota_dimension",
      value: ShAttr(kind: akI64, i64: int64(dimension))),
  ]
  emitOp(b, okDynamicIota, [outputShape],
    [initTensorType(dtype, resultShape)], attrs)[0]

proc realDynamicSlice*(b: var ShBuilder; operand, startIndices,
    limitIndices, strides: ShValueId; resultShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.real_dynamic_slice`.
  let operandTy = getType(b, operand)
  for (inputName, id) in [("start_indices", startIndices),
                         ("limit_indices", limitIndices),
                         ("strides", strides)]:
    let ty = getType(b, id)
    if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
        ty.shape != @[operandTy.shape.len]:
      raise newException(ShBuilderError,
        "real_dynamic_slice: " & inputName &
          " must be an integer vector of length " & $operandTy.shape.len)
  if resultShape.len != operandTy.shape.len:
    raise newException(ShBuilderError,
      "real_dynamic_slice: result rank must match operand rank")
  emitOp(b, okRealDynamicSlice,
    [operand, startIndices, limitIndices, strides],
    [initTensorType(operandTy.dtype, resultShape)], [])[0]

# ---- Phase 4 shape ops ---------------------------------------------------

proc reshape*(b: var ShBuilder; operand: ShValueId;
    newShape: openArray[int]): ShValueId =
  ## Emit `stablehlo.reshape`. The result type carries the new shape;
  ## element count must match the operand.
  let inTy = getType(b, operand)
  let outTy = initTensorType(inTy.dtype, newShape)
  if outTy.numElements != inTy.numElements:
    raise newException(ShBuilderError,
      "reshape: element count mismatch — " & $inTy & " has " &
        $inTy.numElements & " elements, " & $outTy & " has " &
        $outTy.numElements)
  emitOp(b, okReshape, [operand], [outTy], [])[0]

proc transpose*(b: var ShBuilder; operand: ShValueId;
    permutation: openArray[int]): ShValueId =
  ## Emit `stablehlo.transpose`. `permutation` is a length-rank list of
  ## distinct axis indices.
  let inTy = getType(b, operand)
  if permutation.len != inTy.shape.len:
    raise newException(ShBuilderError,
      "transpose: permutation length " & $permutation.len &
        " does not match operand rank " & $inTy.shape.len)
  var seen = newSeq[bool](inTy.shape.len)
  var outShape = newSeq[int](inTy.shape.len)
  for i, p in permutation:
    if p < 0 or p >= inTy.shape.len:
      raise newException(ShBuilderError,
        "transpose: permutation index " & $p & " out of range for rank " &
          $inTy.shape.len)
    if seen[p]:
      raise newException(ShBuilderError,
        "transpose: permutation index " & $p & " repeated")
    seen[p] = true
    outShape[i] = inTy.shape[p]
  let outTy = initTensorType(inTy.dtype, outShape)
  var perm = newSeq[int64](permutation.len)
  for i, p in permutation: perm[i] = int64(p)
  let attrs = @[
    ShAttrEntry(name: "permutation",
      value: ShAttr(kind: akI64Array, i64s: perm)),
  ]
  emitOp(b, okTranspose, [operand], [outTy], attrs)[0]

proc reverse*(b: var ShBuilder; operand: ShValueId;
    dimensions: openArray[int]): ShValueId =
  ## Emit `stablehlo.reverse`. `dimensions` lists distinct axes to
  ## reverse; result type matches the operand.
  let inTy = getType(b, operand)
  if dimensions.len == 0:
    raise newException(ShBuilderError,
      "reverse: dimensions must not be empty")
  var seen = newSeq[bool](inTy.shape.len)
  var dims = newSeq[int64](dimensions.len)
  for i, d in dimensions:
    if d < 0 or d >= inTy.shape.len:
      raise newException(ShBuilderError,
        "reverse: dimension " & $d & " out of range for rank " &
          $inTy.shape.len)
    if seen[d]:
      raise newException(ShBuilderError,
        "reverse: dimension " & $d & " repeated")
    seen[d] = true
    dims[i] = int64(d)
  let attrs = @[
    ShAttrEntry(name: "dimensions",
      value: ShAttr(kind: akI64Array, i64s: dims)),
  ]
  emitOp(b, okReverse, [operand], [inTy], attrs)[0]

proc returnOp*(b: var ShBuilder; values: openArray[ShValueId]) =
  ## Emit `func.return`. There must be exactly one return per function
  ## and it must be the last op. The verifier checks the type match
  ## against `outputTypes`.
  discard emitOp(b, okReturn, values, [], [])

# ---- Phase 5c linear-algebra ops -----------------------------------------

proc dotGeneral*(b: var ShBuilder; lhs, rhs: ShValueId;
    lhsBatching, rhsBatching, lhsContracting, rhsContracting: openArray[int]):
    ShValueId =
  ## Emit `stablehlo.dot_general`. Computes a generalised matrix product
  ## with explicit batching and contracting dimension lists. Result
  ## shape is `lhsBatch ++ lhsRemaining ++ rhsRemaining`.
  ##
  ## Constraints (validated here, mirrored by the verifier):
  ## - dtype of lhs and rhs match.
  ## - the batching lists have equal length and matching dim sizes.
  ## - the contracting lists have equal length and matching dim sizes.
  ## - no index appears twice within a list, and batching/contracting
  ##   lists are disjoint per side.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt.dtype != rt.dtype:
    raise newException(ShBuilderError,
      "dot_general: dtype mismatch \u2014 " & $lt & " vs " & $rt)
  if lhsBatching.len != rhsBatching.len:
    raise newException(ShBuilderError,
      "dot_general: batching dim count mismatch (" &
        $lhsBatching.len & " vs " & $rhsBatching.len & ")")
  if lhsContracting.len != rhsContracting.len:
    raise newException(ShBuilderError,
      "dot_general: contracting dim count mismatch (" &
        $lhsContracting.len & " vs " & $rhsContracting.len & ")")

  proc validateAxisLists(side: string; rank: int;
      batching, contracting: openArray[int]) =
    var seen = newSeq[bool](rank)
    for d in batching:
      if d < 0 or d >= rank:
        raise newException(ShBuilderError,
          "dot_general: " & side & " batching dim " & $d &
            " out of range for rank " & $rank)
      if seen[d]:
        raise newException(ShBuilderError,
          "dot_general: " & side & " dim " & $d & " repeated")
      seen[d] = true
    for d in contracting:
      if d < 0 or d >= rank:
        raise newException(ShBuilderError,
          "dot_general: " & side & " contracting dim " & $d &
            " out of range for rank " & $rank)
      if seen[d]:
        raise newException(ShBuilderError,
          "dot_general: " & side & " dim " & $d & " repeated")
      seen[d] = true

  validateAxisLists("lhs", lt.shape.len, lhsBatching, lhsContracting)
  validateAxisLists("rhs", rt.shape.len, rhsBatching, rhsContracting)
  for i in 0 ..< lhsBatching.len:
    if lt.shape[lhsBatching[i]] != rt.shape[rhsBatching[i]]:
      raise newException(ShBuilderError,
        "dot_general: batching dim size mismatch at index " & $i &
          " \u2014 lhs[" & $lhsBatching[i] & "]=" &
          $lt.shape[lhsBatching[i]] & ", rhs[" & $rhsBatching[i] & "]=" &
          $rt.shape[rhsBatching[i]])
  for i in 0 ..< lhsContracting.len:
    if lt.shape[lhsContracting[i]] != rt.shape[rhsContracting[i]]:
      raise newException(ShBuilderError,
        "dot_general: contracting dim size mismatch at index " & $i &
          " \u2014 lhs[" & $lhsContracting[i] & "]=" &
          $lt.shape[lhsContracting[i]] & ", rhs[" & $rhsContracting[i] & "]=" &
          $rt.shape[rhsContracting[i]])

  proc remaining(rank: int; batching, contracting: openArray[int]): seq[int] =
    var used = newSeq[bool](rank)
    for d in batching: used[d] = true
    for d in contracting: used[d] = true
    result = @[]
    for i in 0 ..< rank:
      if not used[i]: result.add i

  let lhsRem = remaining(lt.shape.len, lhsBatching, lhsContracting)
  let rhsRem = remaining(rt.shape.len, rhsBatching, rhsContracting)
  var outShape = newSeqOfCap[int](
    lhsBatching.len + lhsRem.len + rhsRem.len)
  for d in lhsBatching: outShape.add lt.shape[d]
  for d in lhsRem: outShape.add lt.shape[d]
  for d in rhsRem: outShape.add rt.shape[d]
  let outTy = initTensorType(lt.dtype, outShape)

  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)

  let attrs = @[
    ShAttrEntry(name: "dot_dimension_numbers", value: ShAttr(
      kind: akDotDims,
      lhsBatchingDims: toI64(lhsBatching),
      rhsBatchingDims: toI64(rhsBatching),
      lhsContractingDims: toI64(lhsContracting),
      rhsContractingDims: toI64(rhsContracting),
    )),
  ]
  emitOp(b, okDotGeneral, [lhs, rhs], [outTy], attrs)[0]

proc matmul*(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
  ## Convenience wrapper over `dotGeneral` for the rank-2 \u00d7 rank-2
  ## case: contracting dim 1 of `lhs` against dim 0 of `rhs`. Useful for
  ## `Linear` layers where weights are stored `(in, out)`.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt.shape.len != 2 or rt.shape.len != 2:
    raise newException(ShBuilderError,
      "matmul: expected rank-2 operands, got " & $lt & " and " & $rt)
  dotGeneral(b, lhs, rhs, [], [], [1], [0])

proc broadcastInDim*(b: var ShBuilder; operand: ShValueId;
    outputShape: openArray[int];
    broadcastDimensions: openArray[int]): ShValueId =
  ## Emit `stablehlo.broadcast_in_dim`. Each operand axis `i` maps to
  ## output axis `broadcastDimensions[i]`; remaining output axes are
  ## broadcast (filled). Operand axes must be size 1 or match the
  ## corresponding output axis.
  let inTy = getType(b, operand)
  if broadcastDimensions.len != inTy.shape.len:
    raise newException(ShBuilderError,
      "broadcast_in_dim: broadcastDimensions length " &
        $broadcastDimensions.len & " does not match operand rank " &
        $inTy.shape.len)
  var seen = newSeq[bool](outputShape.len)
  for i, mapped in broadcastDimensions:
    if mapped < 0 or mapped >= outputShape.len:
      raise newException(ShBuilderError,
        "broadcast_in_dim: mapping " & $mapped &
          " out of range for output rank " & $outputShape.len)
    if seen[mapped]:
      raise newException(ShBuilderError,
        "broadcast_in_dim: output dim " & $mapped & " mapped twice")
    seen[mapped] = true
    let ind = inTy.shape[i]
    let outd = outputShape[mapped]
    if ind != 1 and ind != outd:
      raise newException(ShBuilderError,
        "broadcast_in_dim: operand dim " & $i & " (size " & $ind &
          ") cannot broadcast to output dim " & $mapped & " (size " &
          $outd & ")")
  let outTy = initTensorType(inTy.dtype, outputShape)
  var dims = newSeq[int64](broadcastDimensions.len)
  for i, d in broadcastDimensions: dims[i] = int64(d)
  let attrs = @[
    ShAttrEntry(name: "broadcast_dimensions",
      value: ShAttr(kind: akI64Array, i64s: dims)),
  ]
  emitOp(b, okBroadcastInDim, [operand], [outTy], attrs)[0]

# ---- Phase 5b reductions + region support --------------------------------

proc beginValueRegion*(b: var ShBuilder;
    argTypes: openArray[ShValueType]): seq[ShValueId] =
  ## Open a new nested region under the currently open function.
  ## Subsequent op emissions are appended into this region until
  ## `endRegion` is called. Returns the SSA ids of the region's
  ## entry-block parameters (allocated in the enclosing function's
  ## value table).
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "beginRegion called with no open function")
  var region = ShRegion(args: @[], ops: @[])
  result = newSeqOfCap[ShValueId](argTypes.len)
  var fn = addr b.module.funcs[b.curFn]
  for ty in argTypes:
    let v = freshValue(fn[], ty)
    region.args.add v
    result.add v.id
  b.regionStack.add region

proc beginRegion*(b: var ShBuilder;
    argTypes: openArray[ShTensorType]): seq[ShValueId] =
  ## Tensor-only compatibility wrapper around `beginValueRegion`.
  beginValueRegion(b, tensorValueTypes(argTypes))

proc endRegion*(b: var ShBuilder): ShRegion =
  ## Close the innermost open region and return its built-up body. The
  ## caller (typically a region-bearing op like `reduce`) attaches the
  ## returned region to its op via the appropriate field.
  if b.regionStack.len == 0:
    raise newException(ShBuilderError,
      "endRegion called with no open region")
  result = b.regionStack[^1]
  b.regionStack.setLen(b.regionStack.len - 1)

proc stablehloReturn*(b: var ShBuilder; values: openArray[ShValueId]) =
  ## Emit `stablehlo.return` for a region body. Distinct from
  ## `returnOp` (which is `func.return` for function bodies).
  if b.regionStack.len == 0:
    raise newException(ShBuilderError,
      "stablehloReturn must be called inside an open region")
  discard emitOp(b, okStablehloReturn, values, [], [])

proc reduce*(b: var ShBuilder; input, initValue: ShValueId;
    dimensions: openArray[int];
    body: proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId
      {.closure.}): ShValueId =
  ## Emit `stablehlo.reduce` with a user-provided reducer body. The
  ## body receives two scalar SSA ids of the input's element type and
  ## must return a single scalar SSA id of the same type. The returned
  ## value becomes the region's `stablehlo.return` operand.
  let inTy = getType(b, input)
  let initTy = getType(b, initValue)
  if initTy.shape.len != 0:
    raise newException(ShBuilderError,
      "reduce: init value must be a 0-rank scalar, got " & $initTy)
  if initTy.dtype != inTy.dtype:
    raise newException(ShBuilderError,
      "reduce: init dtype " & $initTy.dtype &
        " differs from input dtype " & $inTy.dtype)
  if dimensions.len == 0:
    raise newException(ShBuilderError,
      "reduce: must reduce over at least one dimension")
  var seenDim = newSeq[bool](inTy.shape.len)
  for d in dimensions:
    if d < 0 or d >= inTy.shape.len:
      raise newException(ShBuilderError,
        "reduce: dimension " & $d & " out of range for rank " &
          $inTy.shape.len)
    if seenDim[d]:
      raise newException(ShBuilderError,
        "reduce: dimension " & $d & " repeated")
    seenDim[d] = true
  var outShape: seq[int] = @[]
  for i, dim in inTy.shape:
    if not seenDim[i]: outShape.add dim
  let outTy = initTensorType(inTy.dtype, outShape)
  let elemTy = initTensorType(inTy.dtype, [])
  # Build the reducer region.
  let regionArgs = beginRegion(b, [elemTy, elemTy])
  let bodyResult = body(b, regionArgs[0], regionArgs[1])
  let bodyResTy = getType(b, bodyResult)
  if bodyResTy != elemTy:
    raise newException(ShBuilderError,
      "reduce: reducer body returned " & $bodyResTy &
        ", expected element type " & $elemTy)
  stablehloReturn(b, [bodyResult])
  let region = endRegion(b)
  var dims = newSeq[int64](dimensions.len)
  for i, d in dimensions: dims[i] = int64(d)
  let attrs = @[
    ShAttrEntry(name: "dimensions",
      value: ShAttr(kind: akI64Array, i64s: dims)),
  ]
  emitOp(b, okReduce, [input, initValue], [outTy], attrs, [region])[0]

# ---- Phase 7b control flow ----------------------------------------------

proc ifOp*(b: var ShBuilder; predicate: ShValueId;
    thenBuilder: proc(b: var ShBuilder): seq[ShValueId] {.closure.};
    elseBuilder: proc(b: var ShBuilder): seq[ShValueId] {.closure.}):
    seq[ShValueId] =
  ## Emit `stablehlo.if` with two single-block regions.
  ##
  ## The predicate must be a scalar `i1` (i.e. a 0-rank `bool` tensor).
  ## Each branch builder runs against the *same* `ShBuilder` with the
  ## branch's region open; the values it returns become the
  ## `stablehlo.return` operands. Both branches must yield the same
  ## number and types of values; those types become the op's results.
  let predTy = getType(b, predicate)
  if predTy.dtype != dtBool or predTy.shape.len != 0:
    raise newException(ShBuilderError,
      "ifOp: predicate must be a 0-rank bool tensor, got " & $predTy)

  discard beginRegion(b, [])
  let thenResults = thenBuilder(b)
  if thenResults.len == 0:
    raise newException(ShBuilderError,
      "ifOp: then-branch must yield at least one value")
  var thenTypes = newSeq[ShTensorType](thenResults.len)
  for i, id in thenResults: thenTypes[i] = getType(b, id)
  stablehloReturn(b, thenResults)
  let thenRegion = endRegion(b)

  discard beginRegion(b, [])
  let elseResults = elseBuilder(b)
  if elseResults.len != thenResults.len:
    raise newException(ShBuilderError,
      "ifOp: else-branch yielded " & $elseResults.len &
        " value(s), then-branch yielded " & $thenResults.len)
  for i, id in elseResults:
    let ty = getType(b, id)
    if ty != thenTypes[i]:
      raise newException(ShBuilderError,
        "ifOp: result #" & $i & " type mismatch \u2014 then=" &
          $thenTypes[i] & " else=" & $ty)
  stablehloReturn(b, elseResults)
  let elseRegion = endRegion(b)

  emitOp(b, okIf, [predicate], thenTypes, [], [thenRegion, elseRegion])

# ---- Phase 7b.2 control flow: compare + while ----------------------------

const CompareDirections* = ["LT", "LE", "GT", "GE", "EQ", "NE"]
  ## Allowed `direction` values for `compare`. Mirrors the StableHLO
  ## `comparison_direction` enum.

proc compare*(b: var ShBuilder; lhs, rhs: ShValueId;
    direction: string): ShValueId =
  ## Emit `stablehlo.compare`. Operands must have matching dtype and
  ## shape; the result is a same-shape `bool` tensor. `direction` must be
  ## one of `CompareDirections`.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt != rt:
    raise newException(ShBuilderError,
      "compare: operand types differ \u2014 " & $lt & " vs " & $rt)
  if direction notin CompareDirections:
    raise newException(ShBuilderError,
      "compare: invalid direction '" & direction & "'")
  let outTy = initTensorType(dtBool, lt.shape)
  let attrs = @[
    ShAttrEntry(name: "comparison_direction",
      value: ShAttr(kind: akString, str: direction)),
  ]
  emitOp(b, okCompare, [lhs, rhs], [outTy], attrs)[0]

proc select*(b: var ShBuilder; pred, onTrue, onFalse: ShValueId): ShValueId =
  ## Emit `stablehlo.select`. `pred` must be `i1` with the same shape as
  ## the value operands; `onTrue`/`onFalse` must agree on dtype + shape.
  ## The result mirrors the value-operand type.
  let pt = getType(b, pred)
  let at = getType(b, onTrue)
  let bt = getType(b, onFalse)
  if pt.dtype != dtBool:
    raise newException(ShBuilderError,
      "select: predicate must be i1, got " & $pt)
  if pt.shape != at.shape:
    raise newException(ShBuilderError,
      "select: predicate shape " & $pt.shape &
        " does not match value shape " & $at.shape)
  if at != bt:
    raise newException(ShBuilderError,
      "select: value operands differ \u2014 " & $at & " vs " & $bt)
  emitOp(b, okSelect, [pred, onTrue, onFalse], [at], [])[0]

proc whileOp*(b: var ShBuilder; initOperands: openArray[ShValueId];
    condBuilder: proc(b: var ShBuilder;
      args: openArray[ShValueId]): ShValueId {.closure.};
    bodyBuilder: proc(b: var ShBuilder;
      args: openArray[ShValueId]): seq[ShValueId] {.closure.}):
    seq[ShValueId] =
  ## Emit `stablehlo.while`. The op's operands and results share the
  ## same flat list of carried types: one entry per loop-carried value.
  ##
  ## - `condBuilder` runs against an open region whose entry args
  ##   mirror the carried types and must yield a single 0-rank `bool`
  ##   SSA id.
  ## - `bodyBuilder` runs against an open region whose entry args
  ##   mirror the carried types and must yield exactly N values
  ##   matching those types in order.
  if initOperands.len == 0:
    raise newException(ShBuilderError,
      "whileOp: must carry at least one value through the loop")
  var carriedTypes = newSeq[ShTensorType](initOperands.len)
  for i, id in initOperands:
    carriedTypes[i] = getType(b, id)

  # Cond region: takes the carried args, yields a 0-rank bool.
  let condArgs = beginRegion(b, carriedTypes)
  let predId = condBuilder(b, condArgs)
  let predTy = getType(b, predId)
  if predTy.dtype != dtBool or predTy.shape.len != 0:
    raise newException(ShBuilderError,
      "whileOp: condBuilder must yield a 0-rank bool, got " & $predTy)
  stablehloReturn(b, [predId])
  let condRegion = endRegion(b)

  # Body region: takes the carried args, yields N matching values.
  let bodyArgs = beginRegion(b, carriedTypes)
  let bodyResults = bodyBuilder(b, bodyArgs)
  if bodyResults.len != carriedTypes.len:
    raise newException(ShBuilderError,
      "whileOp: bodyBuilder yielded " & $bodyResults.len &
        " value(s), expected " & $carriedTypes.len)
  for i, id in bodyResults:
    let ty = getType(b, id)
    if ty != carriedTypes[i]:
      raise newException(ShBuilderError,
        "whileOp: body result #" & $i & " type " & $ty &
          " does not match carried type " & $carriedTypes[i])
  stablehloReturn(b, bodyResults)
  let bodyRegion = endRegion(b)

  emitOp(b, okWhile, initOperands, carriedTypes, [],
    [condRegion, bodyRegion])

proc concatenate*(b: var ShBuilder; operands: openArray[ShValueId];
    dimension: int): ShValueId =
  ## Concatenates tensors along `dimension`. All operands must share dtype
  ## and have identical shape except along the concatenation axis.
  if operands.len == 0:
    raise newException(ShBuilderError, "concatenate: no operands")
  let firstTy = b.getType(operands[0])
  var totalDim = 0
  for i, id in operands:
    let ty = b.getType(id)
    if ty.dtype != firstTy.dtype:
      raise newException(ShBuilderError,
        "concatenate: operand #" & $i & " dtype mismatch")
    if ty.shape.len != firstTy.shape.len:
      raise newException(ShBuilderError,
        "concatenate: operand #" & $i & " rank mismatch")
    for j in 0 ..< ty.shape.len:
      if j != dimension and ty.shape[j] != firstTy.shape[j]:
        raise newException(ShBuilderError,
          "concatenate: operand #" & $i & " shape mismatch at dim " & $j)
    totalDim += ty.shape[dimension]
  var outShape = firstTy.shape
  outShape[dimension] = totalDim
  let outTy = initTensorType(firstTy.dtype, outShape)
  let res = freshId(b.module.funcs[b.curFn], outTy)
  let op = ShOp(
    kind: okConcatenate,
    operands: @operands,
    results: @[res],
    attrs: @[ShAttrEntry(name: "dimension",
      value: ShAttr(kind: akI64, i64: int64(dimension)))],
  )
  if b.regionStack.len > 0:
    b.regionStack[^1].ops.add op
  else:
    b.module.funcs[b.curFn].ops.add op
  res.id

proc slice*(b: var ShBuilder; operand: ShValueId;
    startIndices, limitIndices, strides: openArray[int]): ShValueId =
  ## Static slice with start/limit/strides per dimension.
  let inTy = b.getType(operand)
  if startIndices.len != inTy.shape.len or
     limitIndices.len != inTy.shape.len or
     strides.len != inTy.shape.len:
    raise newException(ShBuilderError,
      "slice: index arrays must match operand rank " & $inTy.shape.len)
  var outShape = newSeq[int](inTy.shape.len)
  for i in 0 ..< inTy.shape.len:
    outShape[i] = (limitIndices[i] - startIndices[i] + strides[i] - 1) div strides[i]
  let outTy = initTensorType(inTy.dtype, outShape)
  let res = freshId(b.module.funcs[b.curFn], outTy)
  let op = ShOp(
    kind: okSlice,
    operands: @[operand],
    results: @[res],
    attrs: @[
      ShAttrEntry(name: "start_indices",
        value: ShAttr(kind: akI64Array, i64s: startIndices.mapIt(int64(it)))),
      ShAttrEntry(name: "limit_indices",
        value: ShAttr(kind: akI64Array, i64s: limitIndices.mapIt(int64(it)))),
      ShAttrEntry(name: "strides",
        value: ShAttr(kind: akI64Array, i64s: strides.mapIt(int64(it)))),
    ],
  )
  if b.regionStack.len > 0:
    b.regionStack[^1].ops.add op
  else:
    b.module.funcs[b.curFn].ops.add op
  res.id

proc sine*(b: var ShBuilder; operand: ShValueId): ShValueId =
  unaryElementwise(b, okSine, operand)

proc cosine*(b: var ShBuilder; operand: ShValueId): ShValueId =
  unaryElementwise(b, okCosine, operand)

proc rsqrt*(b: var ShBuilder; operand: ShValueId): ShValueId =
  unaryElementwise(b, okRsqrt, operand)

proc clamp*(b: var ShBuilder; minVal, operand, maxVal: ShValueId): ShValueId =
  ## Clamps operand element-wise between minVal and maxVal.
  let inTy = b.getType(operand)
  let outTy = inTy
  let res = freshId(b.module.funcs[b.curFn], outTy)
  let op = ShOp(
    kind: okClamp,
    operands: @[minVal, operand, maxVal],
    results: @[res],
    attrs: @[],
  )
  if b.regionStack.len > 0:
    b.regionStack[^1].ops.add op
  else:
    b.module.funcs[b.curFn].ops.add op
  res.id

# ---- Phase 11 (CNN ops): convolution + reduce_window ---------------------

type
  ConvDimensionNumbers* = object
    ## Friendly wrapper for the dimension-numbers attribute of
    ## `stablehlo.convolution`. All fields use operand-axis indices.
    inputBatch*: int
    inputFeature*: int
    inputSpatial*: seq[int]
    kernelInputFeature*: int
    kernelOutputFeature*: int
    kernelSpatial*: seq[int]
    outputBatch*: int
    outputFeature*: int
    outputSpatial*: seq[int]

func nhwcOIHWConvDims*(spatialRank: int = 2): ConvDimensionNumbers =
  ## NHWC layout with OIHW kernel: input `[N, H, W, C]`, kernel
  ## `[O, I, H, W]`, output `[N, H, W, C]`. The default for `Conv2d`.
  ## With `spatialRank = 2`, input/output spatial dims are `[1, 2]` and
  ## kernel spatial dims are `[2, 3]`.
  result.inputBatch = 0
  result.inputFeature = 1 + spatialRank
  result.inputSpatial = newSeq[int](spatialRank)
  for i in 0 ..< spatialRank: result.inputSpatial[i] = 1 + i
  result.kernelOutputFeature = 0
  result.kernelInputFeature = 1
  result.kernelSpatial = newSeq[int](spatialRank)
  for i in 0 ..< spatialRank: result.kernelSpatial[i] = 2 + i
  result.outputBatch = 0
  result.outputFeature = 1 + spatialRank
  result.outputSpatial = newSeq[int](spatialRank)
  for i in 0 ..< spatialRank: result.outputSpatial[i] = 1 + i

proc convolutionOutputShape*(lhsShape, rhsShape: openArray[int];
    windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    lhsDilation, rhsDilation: openArray[int];
    dims: ConvDimensionNumbers;
    featureGroupCount, batchGroupCount: int): seq[int] =
  ## Pure shape inference helper for `stablehlo.convolution`. Validates
  ## ranks and returns the output shape in result-axis order.
  let spatialRank = dims.inputSpatial.len
  if dims.outputSpatial.len != spatialRank or
      dims.kernelSpatial.len != spatialRank:
    raise newException(ShBuilderError,
      "convolution: spatial dim list length mismatch")
  if windowStrides.len != spatialRank or padding.len != spatialRank or
      lhsDilation.len != spatialRank or rhsDilation.len != spatialRank:
    raise newException(ShBuilderError,
      "convolution: window/dilation/padding length must equal spatial rank " &
        $spatialRank)
  let outRank = 2 + spatialRank
  if lhsShape.len != outRank:
    raise newException(ShBuilderError,
      "convolution: lhs rank " & $lhsShape.len &
        " does not match expected " & $outRank)
  if rhsShape.len != outRank:
    raise newException(ShBuilderError,
      "convolution: rhs rank " & $rhsShape.len &
        " does not match expected " & $outRank)
  let batch = lhsShape[dims.inputBatch]
  if batchGroupCount <= 0 or batch mod batchGroupCount != 0:
    raise newException(ShBuilderError,
      "convolution: batch (" & $batch & ") must be divisible by " &
        "batch_group_count " & $batchGroupCount)
  let outFeature = rhsShape[dims.kernelOutputFeature]
  result = newSeq[int](outRank)
  result[dims.outputBatch] = batch div batchGroupCount
  result[dims.outputFeature] = outFeature
  for i in 0 ..< spatialRank:
    let ld = lhsShape[dims.inputSpatial[i]]
    let kd = rhsShape[dims.kernelSpatial[i]]
    let dilatedInput =
      if ld == 0: 0
      else: (ld - 1) * lhsDilation[i] + 1
    let dilatedKernel =
      if kd == 0: 0
      else: (kd - 1) * rhsDilation[i] + 1
    let padded = padding[i][0] + dilatedInput + padding[i][1]
    let n =
      if padded == 0 or dilatedKernel > padded: 0
      else: (padded - dilatedKernel) div windowStrides[i] + 1
    result[dims.outputSpatial[i]] = n

proc convolution*(b: var ShBuilder; lhs, rhs: ShValueId;
    windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    lhsDilation, rhsDilation: openArray[int];
    dims: ConvDimensionNumbers;
    featureGroupCount: int = 1; batchGroupCount: int = 1;
    windowReversal: openArray[bool] = []): ShValueId =
  ## Emit `stablehlo.convolution`. `lhs` is the input, `rhs` is the
  ## kernel. Layout, strides, dilation, and padding are explicit; the
  ## output shape is inferred from those plus the operand shapes.
  ##
  ## `windowReversal` defaults to all-false (omitted from the IR). When
  ## any entry is true the kernel is convolved with its spatial axes
  ## reversed; this is the StableHLO primitive used to express the
  ## input gradient of a forward convolution.
  let lt = getType(b, lhs)
  let rt = getType(b, rhs)
  if lt.dtype != rt.dtype:
    raise newException(ShBuilderError,
      "convolution: dtype mismatch \u2014 " & $lt & " vs " & $rt)
  let outShape = convolutionOutputShape(lt.shape, rt.shape,
    windowStrides, padding, lhsDilation, rhsDilation, dims,
    featureGroupCount, batchGroupCount)
  let outTy = initTensorType(lt.dtype, outShape)
  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)
  var padFlat = newSeq[int64](padding.len * 2)
  for i in 0 ..< padding.len:
    padFlat[i * 2] = int64(padding[i][0])
    padFlat[i * 2 + 1] = int64(padding[i][1])
  var attrs = @[
    ShAttrEntry(name: "window_strides",
      value: ShAttr(kind: akI64Array, i64s: toI64(windowStrides))),
    ShAttrEntry(name: "padding",
      value: ShAttr(kind: akI64Matrix,
        matRows: padding.len, matCols: 2, matVals: padFlat)),
    ShAttrEntry(name: "lhs_dilation",
      value: ShAttr(kind: akI64Array, i64s: toI64(lhsDilation))),
    ShAttrEntry(name: "rhs_dilation",
      value: ShAttr(kind: akI64Array, i64s: toI64(rhsDilation))),
    ShAttrEntry(name: "dimension_numbers",
      value: ShAttr(kind: akConvDims,
        inputBatchDim: int64(dims.inputBatch),
        inputFeatureDim: int64(dims.inputFeature),
        inputSpatialDims: toI64(dims.inputSpatial),
        kernelInputFeatureDim: int64(dims.kernelInputFeature),
        kernelOutputFeatureDim: int64(dims.kernelOutputFeature),
        kernelSpatialDims: toI64(dims.kernelSpatial),
        outputBatchDim: int64(dims.outputBatch),
        outputFeatureDim: int64(dims.outputFeature),
        outputSpatialDims: toI64(dims.outputSpatial))),
    ShAttrEntry(name: "feature_group_count",
      value: ShAttr(kind: akI64, i64: int64(featureGroupCount))),
    ShAttrEntry(name: "batch_group_count",
      value: ShAttr(kind: akI64, i64: int64(batchGroupCount))),
  ]
  if windowReversal.len > 0:
    var anyReversed = false
    for r in windowReversal:
      if r: anyReversed = true; break
    if anyReversed:
      var revFlags = newSeq[int64](windowReversal.len)
      for i, r in windowReversal:
        revFlags[i] = (if r: 1'i64 else: 0'i64)
      attrs.add ShAttrEntry(name: "window_reversal",
        value: ShAttr(kind: akI64Array, i64s: revFlags))
  emitOp(b, okConvolution, [lhs, rhs], [outTy], attrs)[0]

proc reduceWindowOutputShape*(inShape: openArray[int];
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    baseDilations, windowDilations: openArray[int]): seq[int] =
  ## Pure shape helper for `stablehlo.reduce_window`. Mirrors the
  ## verifier's expected output shape.
  let r = inShape.len
  if windowDimensions.len != r or windowStrides.len != r or
      padding.len != r or baseDilations.len != r or
      windowDilations.len != r:
    raise newException(ShBuilderError,
      "reduce_window: window/stride/padding/dilation arrays must have rank " &
        $r)
  result = newSeq[int](r)
  for i in 0 ..< r:
    let dilatedInput =
      if inShape[i] == 0: 0
      else: (inShape[i] - 1) * baseDilations[i] + 1
    let padded = padding[i][0] + dilatedInput + padding[i][1]
    let dilatedWindow =
      if windowDimensions[i] == 0: 0
      else: (windowDimensions[i] - 1) * windowDilations[i] + 1
    result[i] =
      if padded == 0 or dilatedWindow > padded: 0
      else: (padded - dilatedWindow) div windowStrides[i] + 1

proc reduceWindow*(b: var ShBuilder; input, initValue: ShValueId;
    windowDimensions, windowStrides: openArray[int];
    padding: openArray[array[2, int]];
    baseDilations, windowDilations: openArray[int];
    body: proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId
      {.closure.}): ShValueId =
  ## Emit `stablehlo.reduce_window` with a user-provided reducer body.
  ## The body receives two scalar SSA ids of the input element type and
  ## must return one scalar result.
  let inTy = getType(b, input)
  let initTy = getType(b, initValue)
  if initTy.shape.len != 0:
    raise newException(ShBuilderError,
      "reduce_window: init value must be 0-rank, got " & $initTy)
  if initTy.dtype != inTy.dtype:
    raise newException(ShBuilderError,
      "reduce_window: init dtype " & $initTy.dtype &
        " differs from input dtype " & $inTy.dtype)
  let outShape = reduceWindowOutputShape(inTy.shape, windowDimensions,
    windowStrides, padding, baseDilations, windowDilations)
  let outTy = initTensorType(inTy.dtype, outShape)
  let elemTy = initTensorType(inTy.dtype, [])
  let regionArgs = beginRegion(b, [elemTy, elemTy])
  let bodyResult = body(b, regionArgs[0], regionArgs[1])
  let bodyResTy = getType(b, bodyResult)
  if bodyResTy != elemTy:
    raise newException(ShBuilderError,
      "reduce_window: reducer body returned " & $bodyResTy &
        ", expected element type " & $elemTy)
  stablehloReturn(b, [bodyResult])
  let region = endRegion(b)
  proc toI64(xs: openArray[int]): seq[int64] =
    result = newSeq[int64](xs.len)
    for i, x in xs: result[i] = int64(x)
  var padFlat = newSeq[int64](padding.len * 2)
  for i in 0 ..< padding.len:
    padFlat[i * 2] = int64(padding[i][0])
    padFlat[i * 2 + 1] = int64(padding[i][1])
  let attrs = @[
    ShAttrEntry(name: "window_dimensions",
      value: ShAttr(kind: akI64Array, i64s: toI64(windowDimensions))),
    ShAttrEntry(name: "window_strides",
      value: ShAttr(kind: akI64Array, i64s: toI64(windowStrides))),
    ShAttrEntry(name: "base_dilations",
      value: ShAttr(kind: akI64Array, i64s: toI64(baseDilations))),
    ShAttrEntry(name: "window_dilations",
      value: ShAttr(kind: akI64Array, i64s: toI64(windowDilations))),
    ShAttrEntry(name: "padding",
      value: ShAttr(kind: akI64Matrix,
        matRows: padding.len, matCols: 2, matVals: padFlat)),
  ]
  emitOp(b, okReduceWindow, [input, initValue], [outTy], attrs, [region])[0]

proc allGather*(b: var ShBuilder; operands: openArray[ShValueId];
    allGatherDim: int; resultShapes: openArray[seq[int]];
    replicaGroups: openArray[seq[int]];
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): seq[ShValueId] =
  ## Emit `stablehlo.all_gather` for one or more operands.
  if operands.len == 0 or operands.len != resultShapes.len:
    raise newException(ShBuilderError,
      "all_gather: operands and resultShapes must have same non-zero length")
  var outTypes = newSeq[ShTensorType](operands.len)
  for i, id in operands:
    let ty = getType(b, id)
    if allGatherDim < 0 or allGatherDim >= ty.shape.len:
      raise newException(ShBuilderError,
        "all_gather: allGatherDim out of range")
    requireNonNegativeShape("all_gather", resultShapes[i])
    if resultShapes[i].len != ty.shape.len:
      raise newException(ShBuilderError,
        "all_gather: result rank must match operand rank")
    outTypes[i] = initTensorType(ty.dtype, resultShapes[i])
  var attrs = @[
    i64Attr("all_gather_dim", allGatherDim),
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  appendChannelAttr(attrs, channelHandle)
  if useGlobalDeviceIds:
    attrs.add boolAttr("use_global_device_ids", true)
  emitOp(b, okAllGather, operands, outTypes, attrs)

proc allReduce*(b: var ShBuilder; operands: openArray[ShValueId];
    replicaGroups: openArray[seq[int]]; computation: ShArgRegionBuilder;
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): seq[ShValueId] =
  ## Emit `stablehlo.all_reduce`. The computation receives two scalar
  ## values per operand and returns one scalar per operand.
  if operands.len == 0:
    raise newException(ShBuilderError,
      "all_reduce: operands must not be empty")
  var outTypes = newSeq[ShTensorType](operands.len)
  var argTypes: seq[ShValueType] = @[]
  var expected: seq[ShValueType] = @[]
  for i, id in operands:
    outTypes[i] = getType(b, id)
    argTypes.add scalarElementType(outTypes[i])
    argTypes.add scalarElementType(outTypes[i])
    expected.add scalarElementType(outTypes[i])
  let args = beginValueRegion(b, argTypes)
  let ids = computation(b, args)
  checkResultTypes(b, "all_reduce", ids, expected)
  stablehloReturn(b, ids)
  let region = endRegion(b)
  var attrs = @[
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  appendChannelAttr(attrs, channelHandle)
  if useGlobalDeviceIds:
    attrs.add boolAttr("use_global_device_ids", true)
  emitOp(b, okAllReduce, operands, outTypes, attrs, [region])

proc reduceScatter*(b: var ShBuilder; operand: ShValueId;
    scatterDimension: int; resultShape: openArray[int];
    replicaGroups: openArray[seq[int]]; computation: ShArgRegionBuilder;
    channelHandle: ChannelHandle = NoChannelHandle;
    useGlobalDeviceIds = false): ShValueId =
  ## Emit `stablehlo.reduce_scatter`.
  let ty = getType(b, operand)
  if scatterDimension < 0 or scatterDimension >= ty.shape.len:
    raise newException(ShBuilderError,
      "reduce_scatter: scatterDimension out of range")
  requireNonNegativeShape("reduce_scatter", resultShape)
  let elemTy = scalarElementType(ty)
  let args = beginValueRegion(b, [elemTy, elemTy])
  let ids = computation(b, args)
  checkResultTypes(b, "reduce_scatter", ids, [elemTy])
  stablehloReturn(b, ids)
  let region = endRegion(b)
  var attrs = @[
    i64Attr("scatter_dimension", scatterDimension),
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  appendChannelAttr(attrs, channelHandle)
  if useGlobalDeviceIds:
    attrs.add boolAttr("use_global_device_ids", true)
  emitOp(b, okReduceScatter, [operand],
    [initTensorType(ty.dtype, resultShape)], attrs, [region])[0]

proc allToAll*(b: var ShBuilder; operands: openArray[ShValueId];
    splitDimension, concatDimension, splitCount: int;
    resultShapes: openArray[seq[int]];
    replicaGroups: openArray[seq[int]];
    channelHandle: ChannelHandle = NoChannelHandle): seq[ShValueId] =
  ## Emit `stablehlo.all_to_all`.
  if operands.len == 0 or operands.len != resultShapes.len:
    raise newException(ShBuilderError,
      "all_to_all: operands and resultShapes must have same non-zero length")
  if splitCount <= 0:
    raise newException(ShBuilderError,
      "all_to_all: splitCount must be positive")
  var outTypes = newSeq[ShTensorType](operands.len)
  for i, id in operands:
    let ty = getType(b, id)
    if splitDimension < 0 or splitDimension >= ty.shape.len or
        concatDimension < 0 or concatDimension >= ty.shape.len:
      raise newException(ShBuilderError,
        "all_to_all: split/concat dimensions out of range")
    requireNonNegativeShape("all_to_all", resultShapes[i])
    outTypes[i] = initTensorType(ty.dtype, resultShapes[i])
  var attrs = @[
    i64Attr("split_dimension", splitDimension),
    i64Attr("concat_dimension", concatDimension),
    i64Attr("split_count", splitCount),
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  appendChannelAttr(attrs, channelHandle)
  emitOp(b, okAllToAll, operands, outTypes, attrs)

proc collectiveBroadcast*(b: var ShBuilder; operand: ShValueId;
    replicaGroups: openArray[seq[int]];
    channelHandle: ChannelHandle = NoChannelHandle): ShValueId =
  ## Emit `stablehlo.collective_broadcast`.
  let ty = getType(b, operand)
  var attrs = @[
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  appendChannelAttr(attrs, channelHandle)
  emitOp(b, okCollectiveBroadcast, [operand], [ty], attrs)[0]

proc collectivePermute*(b: var ShBuilder; operand: ShValueId;
    sourceTargetPairs: openArray[array[2, int]];
    channelHandle: ChannelHandle = NoChannelHandle): ShValueId =
  ## Emit `stablehlo.collective_permute`.
  let ty = getType(b, operand)
  var attrs = @[matrixAttr("source_target_pairs", sourceTargetPairs)]
  appendChannelAttr(attrs, channelHandle)
  emitOp(b, okCollectivePermute, [operand], [ty], attrs)[0]

proc crossReplicaSum*(b: var ShBuilder; operand: ShValueId;
    replicaGroups: openArray[seq[int]]): ShValueId =
  ## Emit legacy `stablehlo.cross-replica-sum`.
  let ty = getType(b, operand)
  let attrs = @[
    rawAttr("replica_groups", replicaGroupsMlir(replicaGroups)),
  ]
  emitOp(b, okCrossReplicaSum, [operand], [ty], attrs)[0]

proc asyncStart*(b: var ShBuilder; operands: openArray[ShValueId];
    body: ShArgRegionBuilder): ShValueId =
  ## Emit `stablehlo.async_start`. The body receives region args
  ## matching `operands` and its returned values form the future type.
  var argTypes = valueTypesOfIds(b, operands)
  let args = beginValueRegion(b, argTypes)
  let ids = body(b, args)
  if ids.len == 0:
    raise newException(ShBuilderError,
      "async_start: body must return at least one value")
  let outTypes = resultValueTypes(b, ids)
  stablehloReturn(b, ids)
  let region = endRegion(b)
  emitValueOp(b, okAsyncStart, operands, [initFutureType(outTypes)],
    [], [region])[0]

proc asyncDone*(b: var ShBuilder; future: ShValueId): seq[ShValueId] =
  ## Emit `stablehlo.async_done`.
  let ty = getValueType(b, future)
  if not ty.isFuture:
    raise newException(ShBuilderError,
      "async_done: operand must be future, got " & $ty)
  emitValueOp(b, okAsyncDone, [future], ty.futureResults, [])

proc infeed*(b: var ShBuilder; token: ShValueId;
    resultTypes: openArray[ShValueType]; infeedConfig = ""): seq[ShValueId] =
  ## Emit `stablehlo.infeed`. `resultTypes` should usually end with a
  ## token result.
  let tokenTy = getValueType(b, token)
  if not tokenTy.isToken:
    raise newException(ShBuilderError,
      "infeed: operand must be token, got " & $tokenTy)
  var attrs: seq[ShAttrEntry] = @[]
  if infeedConfig.len > 0:
    attrs.add stringAttr("infeed_config", infeedConfig)
  emitValueOp(b, okInfeed, [token], resultTypes, attrs)

proc outfeed*(b: var ShBuilder; inputs: openArray[ShValueId];
    token: ShValueId; outfeedConfig = ""): ShValueId =
  ## Emit `stablehlo.outfeed`.
  let tokenTy = getValueType(b, token)
  if not tokenTy.isToken:
    raise newException(ShBuilderError,
      "outfeed: token operand expected, got " & $tokenTy)
  var operands = @inputs
  operands.add token
  var attrs: seq[ShAttrEntry] = @[]
  if outfeedConfig.len > 0:
    attrs.add stringAttr("outfeed_config", outfeedConfig)
  emitValueOp(b, okOutfeed, operands, [initTokenType()], attrs)[0]

proc send*(b: var ShBuilder; inputs: openArray[ShValueId];
    token: ShValueId; channelHandle: ChannelHandle;
    isHostTransfer = false;
    sourceTargetPairs: openArray[array[2, int]] = []): ShValueId =
  ## Emit `stablehlo.send`.
  let tokenTy = getValueType(b, token)
  if not tokenTy.isToken:
    raise newException(ShBuilderError,
      "send: token operand expected, got " & $tokenTy)
  var operands = @inputs
  operands.add token
  var attrs = @[
    rawAttr("channel_handle", channelHandleMlir(channelHandle)),
    boolAttr("is_host_transfer", isHostTransfer),
  ]
  if sourceTargetPairs.len > 0:
    attrs.add matrixAttr("source_target_pairs", sourceTargetPairs)
  emitValueOp(b, okSend, operands, [initTokenType()], attrs)[0]

proc recv*(b: var ShBuilder; token: ShValueId;
    resultTypes: openArray[ShValueType]; channelHandle: ChannelHandle;
    isHostTransfer = false;
    sourceTargetPairs: openArray[array[2, int]] = []): seq[ShValueId] =
  ## Emit `stablehlo.recv`. `resultTypes` should usually end with a
  ## token result.
  let tokenTy = getValueType(b, token)
  if not tokenTy.isToken:
    raise newException(ShBuilderError,
      "recv: token operand expected, got " & $tokenTy)
  var attrs = @[
    rawAttr("channel_handle", channelHandleMlir(channelHandle)),
    boolAttr("is_host_transfer", isHostTransfer),
  ]
  if sourceTargetPairs.len > 0:
    attrs.add matrixAttr("source_target_pairs", sourceTargetPairs)
  emitValueOp(b, okRecv, [token], resultTypes, attrs)

proc customCall*(b: var ShBuilder; inputs: openArray[ShValueId];
    resultTypes: openArray[ShValueType]; callTargetName: string;
    hasSideEffect = false; backendConfig = ""; apiVersion = 0;
    calledComputations: openArray[string] = []): seq[ShValueId] =
  ## Emit `stablehlo.custom_call` with explicit result value types.
  if callTargetName.len == 0:
    raise newException(ShBuilderError,
      "custom_call: callTargetName must not be empty")
  var attrs = @[
    stringAttr("call_target_name", callTargetName),
    boolAttr("has_side_effect", hasSideEffect),
  ]
  if backendConfig.len > 0:
    attrs.add stringAttr("backend_config", backendConfig)
  if apiVersion != 0:
    attrs.add rawAttr("api_version", $apiVersion & " : i32")
  if calledComputations.len > 0:
    var raw = "["
    for i, name in calledComputations:
      if i > 0: raw.add ", "
      raw.add '@'
      raw.add name
    raw.add ']'
    attrs.add rawAttr("called_computations", raw)
  emitValueOp(b, okCustomCall, inputs, resultTypes, attrs)

proc composite*(b: var ShBuilder; inputs: openArray[ShValueId];
    resultTypes: openArray[ShValueType]; name, decomposition: string;
    compositeAttributes = "{}"; version = 0): seq[ShValueId] =
  ## Emit `stablehlo.composite` with an explicit decomposition symbol.
  if name.len == 0 or decomposition.len == 0:
    raise newException(ShBuilderError,
      "composite: name and decomposition must not be empty")
  let symbol =
    if decomposition[0] == '@': decomposition
    else: "@" & decomposition
  let attrs = @[
    stringAttr("name", name),
    rawAttr("composite_attributes", compositeAttributes),
    rawAttr("decomposition", symbol),
    rawAttr("version", $version & " : i32"),
  ]
  emitValueOp(b, okComposite, inputs, resultTypes, attrs)

proc convDimsAttrMlir(dims: ConvDimensionNumbers; inputRank, kernelRank,
    outputRank: int): string =
  proc layoutFor(rank: int; batchOrIn, featureOrOut: int;
      spatial: openArray[int]; isKernel: bool): string =
    var labels = newSeq[string](rank)
    for i in 0 ..< rank: labels[i] = ""
    if isKernel:
      labels[batchOrIn] = "i"
      labels[featureOrOut] = "o"
    else:
      labels[batchOrIn] = "b"
      labels[featureOrOut] = "f"
    for sIdx, axis in spatial:
      labels[axis] = $sIdx
    result = "["
    for i in 0 ..< labels.len:
      if i > 0: result.add ", "
      result.add labels[i]
    result.add ']'
  "#stablehlo.conv<" &
    layoutFor(inputRank, dims.inputBatch, dims.inputFeature,
      dims.inputSpatial, false) & "x" &
    layoutFor(kernelRank, dims.kernelInputFeature, dims.kernelOutputFeature,
      dims.kernelSpatial, true) & "->" &
    layoutFor(outputRank, dims.outputBatch, dims.outputFeature,
      dims.outputSpatial, false) & ">"

proc dynamicConv*(b: var ShBuilder; lhs, rhs, padding: ShValueId;
    windowStrides, lhsDilation, rhsDilation: openArray[int];
    dims: ConvDimensionNumbers; resultShape: openArray[int];
    featureGroupCount: int = 1; batchGroupCount: int = 1;
    windowReversal: openArray[bool] = []): ShValueId =
  ## Emit `stablehlo.dynamic_conv`. Padding is a runtime rank-2 integer
  ## tensor, so the result shape is explicit.
  let lhsTy = getType(b, lhs)
  let rhsTy = getType(b, rhs)
  let paddingTy = getType(b, padding)
  if lhsTy.dtype != rhsTy.dtype:
    raise newException(ShBuilderError,
      "dynamic_conv: dtype mismatch — " & $lhsTy & " vs " & $rhsTy)
  if not (paddingTy.dtype.isSignedInt or paddingTy.dtype.isUnsignedInt) or
      paddingTy.shape.len != 2 or paddingTy.shape[1] != 2:
    raise newException(ShBuilderError,
      "dynamic_conv: padding must be a rank-2 integer tensor with width 2")
  let spatialRank = dims.inputSpatial.len
  if windowStrides.len != spatialRank or lhsDilation.len != spatialRank or
      rhsDilation.len != spatialRank:
    raise newException(ShBuilderError,
      "dynamic_conv: window/dilation attributes must match spatial rank")
  requireNonNegativeShape("dynamic_conv", resultShape)
  var attrs = @[
    i64ArrayAttr("window_strides", windowStrides),
    i64ArrayAttr("lhs_dilation", lhsDilation),
    i64ArrayAttr("rhs_dilation", rhsDilation),
    rawAttr("dimension_numbers",
      convDimsAttrMlir(dims, lhsTy.shape.len, rhsTy.shape.len,
        resultShape.len)),
    i64Attr("feature_group_count", featureGroupCount),
    i64Attr("batch_group_count", batchGroupCount),
  ]
  if windowReversal.len > 0:
    if windowReversal.len != spatialRank:
      raise newException(ShBuilderError,
        "dynamic_conv: windowReversal must match spatial rank")
    var raw = "array<i1: "
    for i, r in windowReversal:
      if i > 0: raw.add ", "
      raw.add (if r: "true" else: "false")
    raw.add '>'
    attrs.add rawAttr("window_reversal", raw)
  emitOp(b, okDynamicConv, [lhs, rhs, padding],
    [initTensorType(lhsTy.dtype, resultShape)], attrs)[0]

proc uniformQuantize*(b: var ShBuilder; operand: ShValueId;
    resultType: ShTensorType): ShValueId =
  ## Emit `stablehlo.uniform_quantize`. The current executable subset
  ## models the result with rew's concrete `DType`; full quantized element
  ## descriptors are carried by the public `Value` model.
  let operandTy = getType(b, operand)
  if resultType.shape != operandTy.shape:
    raise newException(ShBuilderError,
      "uniform_quantize: result shape must match operand shape")
  emitOp(b, okUniformQuantize, [operand], [resultType], [])[0]

proc uniformDequantize*(b: var ShBuilder; operand: ShValueId;
    resultDType: DType): ShValueId =
  ## Emit `stablehlo.uniform_dequantize` into a floating-point result.
  if not resultDType.isFloat:
    raise newException(ShBuilderError,
      "uniform_dequantize: result dtype must be floating point")
  let operandTy = getType(b, operand)
  emitOp(b, okUniformDequantize, [operand],
    [initTensorType(resultDType, operandTy.shape)], [])[0]

func build*(b: ShBuilder): ShModule =
  ## Snapshot the in-progress module. Caller is expected to have closed
  ## any open function with `endFunc` first.
  if b.curFn >= 0:
    raise newException(ShBuilderError,
      "build called while a function is still open")
  b.module

proc setModuleExecutionCounts*(b: var ShBuilder;
    numReplicas, numPartitions: int) =
  ## Records distributed execution counts on the module for textual MLIR
  ## metadata. Compile options remain the source of truth for PJRT.
  if numReplicas < 0 or numPartitions < 0:
    raise newException(ShBuilderError,
      "setModuleExecutionCounts: counts must be non-negative")
  b.module.numReplicas = numReplicas
  b.module.numPartitions = numPartitions

proc addShardyMeshOp*(b: var ShBuilder; meshOp: string) =
  ## Adds a module-level Shardy mesh definition if it is not already present.
  if meshOp.len == 0:
    return
  for existing in b.module.shardyMeshOps:
    if existing == meshOp:
      return
  b.module.shardyMeshOps.add meshOp

proc setCurrentInputShardings*(b: var ShBuilder;
    shardings: openArray[string]) =
  ## Replaces function-argument Shardy attributes on the function currently
  ## being built. Empty entries mean "no explicit sharding".
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "setCurrentInputShardings called with no open function")
  if shardings.len != b.module.funcs[b.curFn].inputTypes.len:
    raise newException(ShBuilderError,
      "setCurrentInputShardings: sharding count does not match inputs")
  b.module.funcs[b.curFn].inputShardings = @shardings

proc setCurrentOutputShardings*(b: var ShBuilder;
    shardings: openArray[string]) =
  ## Replaces function-result Shardy attributes on the function currently
  ## being built. Empty entries mean "no explicit sharding".
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "setCurrentOutputShardings called with no open function")
  if shardings.len != b.module.funcs[b.curFn].outputTypes.len:
    raise newException(ShBuilderError,
      "setCurrentOutputShardings: sharding count does not match outputs")
  b.module.funcs[b.curFn].outputShardings = @shardings

proc setCurrentOutputTypes*(b: var ShBuilder;
    outputs: openArray[ShTensorType]) =
  ## Replace the `outputTypes` of the function currently being built.
  ## Used by the dispatcher (`src/rew/dispatch.nim`) when a trace is
  ## opened with no known result types and the body discovers them
  ## dynamically. Raises `ShBuilderError` when no function is open.
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "setCurrentOutputTypes called with no open function")
  b.module.funcs[b.curFn].outputTypes = @outputs
  b.module.funcs[b.curFn].outputValueTypes = tensorValueTypes(outputs)
  b.module.funcs[b.curFn].outputShardings.setLen(outputs.len)

proc setCurrentOutputValueTypes*(b: var ShBuilder;
    outputs: openArray[ShValueType]) =
  ## Replace the general `outputValueTypes` of the function currently
  ## being built. Non-tensor outputs keep the legacy tensor table aligned
  ## with sentinel entries.
  if b.curFn < 0:
    raise newException(ShBuilderError,
      "setCurrentOutputValueTypes called with no open function")
  var tensorOutputs = newSeq[ShTensorType](outputs.len)
  for i, ty in outputs:
    tensorOutputs[i] = ty.tensorTypeOrDefault
  b.module.funcs[b.curFn].outputTypes = tensorOutputs
  b.module.funcs[b.curFn].outputValueTypes = @outputs
