## StableHLO module verifier.
##
## Runs **before** bytecode emission and produces Nim exceptions whose
## message references the user-facing op name and argument shapes —
## never the internal IR node id. Per the layer instructions, this
## module is pure data inspection: no PJRT, no bytecode.

import ../dtype
import ./ir

type
  StableHloError* = object of CatchableError
    ## Raised by `verify`. The dispatcher translates this to a clean
    ## per-op error at the user call site.

template fail(msg: string) = raise newException(StableHloError, msg)

func opName(kind: ShOpKind): string =
  case kind
  of okConstant: "stablehlo.constant"
  of okAdd: "stablehlo.add"
  of okSub: "stablehlo.subtract"
  of okMul: "stablehlo.multiply"
  of okNeg: "stablehlo.negate"
  of okReturn: "func.return"
  of okDiv: "stablehlo.divide"
  of okMax: "stablehlo.maximum"
  of okMin: "stablehlo.minimum"
  of okExp: "stablehlo.exponential"
  of okLog: "stablehlo.log"
  of okSqrt: "stablehlo.sqrt"
  of okAbs: "stablehlo.abs"
  of okTanh: "stablehlo.tanh"
  of okReshape: "stablehlo.reshape"
  of okTranspose: "stablehlo.transpose"
  of okReduce: "stablehlo.reduce"
  of okStablehloReturn: "stablehlo.return"
  of okDotGeneral: "stablehlo.dot_general"
  of okBroadcastInDim: "stablehlo.broadcast_in_dim"
  of okIf: "stablehlo.if"
  of okCompare: "stablehlo.compare"
  of okWhile: "stablehlo.while"
  of okSelect: "stablehlo.select"
  of okConcatenate: "stablehlo.concatenate"
  of okSlice: "stablehlo.slice"
  of okSine: "stablehlo.sine"
  of okCosine: "stablehlo.cosine"
  of okRsqrt: "stablehlo.rsqrt"
  of okClamp: "stablehlo.clamp"
  of okConvolution: "stablehlo.convolution"
  of okReduceWindow: "stablehlo.reduce_window"
  of okCbrt: "stablehlo.cbrt"
  of okCeil: "stablehlo.ceil"
  of okExponentialMinusOne: "stablehlo.exponential_minus_one"
  of okFloor: "stablehlo.floor"
  of okLogPlusOne: "stablehlo.log_plus_one"
  of okLogistic: "stablehlo.logistic"
  of okTan: "stablehlo.tan"
  of okAtan2: "stablehlo.atan2"
  of okPower: "stablehlo.power"
  of okRemainder: "stablehlo.remainder"
  of okSign: "stablehlo.sign"
  of okRoundNearestAfz: "stablehlo.round_nearest_afz"
  of okRoundNearestEven: "stablehlo.round_nearest_even"
  of okAnd: "stablehlo.and"
  of okOr: "stablehlo.or"
  of okXor: "stablehlo.xor"
  of okNot: "stablehlo.not"
  of okShiftLeft: "stablehlo.shift_left"
  of okShiftRightArithmetic: "stablehlo.shift_right_arithmetic"
  of okShiftRightLogical: "stablehlo.shift_right_logical"
  of okCountLeadingZeros: "stablehlo.count_leading_zeros"
  of okPopcnt: "stablehlo.popcnt"
  of okReverse: "stablehlo.reverse"
  of okOptimizationBarrier: "stablehlo.optimization_barrier"
  of okConvert: "stablehlo.convert"
  of okBitcastConvert: "stablehlo.bitcast_convert"
  of okIsFinite: "stablehlo.is_finite"
  of okReducePrecision: "stablehlo.reduce_precision"
  of okBatchNormInference: "stablehlo.batch_norm_inference"
  of okDot: "stablehlo.dot"
  of okBatchNormTraining: "stablehlo.batch_norm_training"
  of okBatchNormGrad: "stablehlo.batch_norm_grad"
  of okCholesky: "stablehlo.cholesky"
  of okGetDimensionSize: "stablehlo.get_dimension_size"
  of okPad: "stablehlo.pad"
  of okBroadcast: "stablehlo.broadcast"
  of okDynamicSlice: "stablehlo.dynamic_slice"
  of okDynamicUpdateSlice: "stablehlo.dynamic_update_slice"
  of okIota: "stablehlo.iota"
  of okReplicaId: "stablehlo.replica_id"
  of okPartitionId: "stablehlo.partition_id"
  of okSetDimensionSize: "stablehlo.set_dimension_size"
  of okDynamicReshape: "stablehlo.dynamic_reshape"
  of okDynamicPad: "stablehlo.dynamic_pad"
  of okDynamicIota: "stablehlo.dynamic_iota"
  of okRealDynamicSlice: "stablehlo.real_dynamic_slice"
  of okCreateToken: "stablehlo.create_token"
  of okAfterAll: "stablehlo.after_all"
  of okTuple: "stablehlo.tuple"
  of okGetTupleElement: "stablehlo.get_tuple_element"
  of okComplex: "stablehlo.complex"
  of okReal: "stablehlo.real"
  of okImag: "stablehlo.imag"
  of okDynamicBroadcastInDim: "stablehlo.dynamic_broadcast_in_dim"
  of okFft: "stablehlo.fft"
  of okTriangularSolve: "stablehlo.triangular_solve"
  of okEinsum: "stablehlo.einsum"
  of okUnaryEinsum: "stablehlo.unary_einsum"
  of okTorchIndexSelect: "stablehlo.torch_index_select"
  of okRng: "stablehlo.rng"
  of okRngBitGenerator: "stablehlo.rng_bit_generator"
  of okGather: "stablehlo.gather"
  of okDynamicGather: "stablehlo.dynamic_gather"
  of okSort: "stablehlo.sort"
  of okMap: "stablehlo.map"
  of okCase: "stablehlo.case"
  of okScatter: "stablehlo.scatter"
  of okSelectAndScatter: "stablehlo.select_and_scatter"
  of okAllGather: "stablehlo.all_gather"
  of okAllReduce: "stablehlo.all_reduce"
  of okAllToAll: "stablehlo.all_to_all"
  of okReduceScatter: "stablehlo.reduce_scatter"
  of okCollectiveBroadcast: "stablehlo.collective_broadcast"
  of okCollectivePermute: "stablehlo.collective_permute"
  of okCrossReplicaSum: "stablehlo.cross-replica-sum"
  of okAsyncStart: "stablehlo.async_start"
  of okAsyncDone: "stablehlo.async_done"
  of okInfeed: "stablehlo.infeed"
  of okOutfeed: "stablehlo.outfeed"
  of okSend: "stablehlo.send"
  of okRecv: "stablehlo.recv"
  of okCustomCall: "stablehlo.custom_call"
  of okComposite: "stablehlo.composite"
  of okDynamicConv: "stablehlo.dynamic_conv"
  of okUniformQuantize: "stablehlo.uniform_quantize"
  of okUniformDequantize: "stablehlo.uniform_dequantize"

func valueTypeOfId(fn: ShFunction; id: ShValueId): ShValueType =
  if not id.isValid:
    fail("invalid SSA value id " & $id & " in function '" & fn.name & "'")
  if fn.valueTypes.len > 0:
    if id.int >= fn.valueTypes.len:
      fail("invalid SSA value id " & $id & " in function '" & fn.name & "'")
    fn.valueTypes[id.int]
  else:
    if id.int >= fn.types.len:
      fail("invalid SSA value id " & $id & " in function '" & fn.name & "'")
    initValueType(fn.types[id.int])

func typeOfId(fn: ShFunction; id: ShValueId): ShTensorType =
  let ty = valueTypeOfId(fn, id)
  if not ty.isTensor:
    fail("value " & $id & " in function '" & fn.name &
      "' is not a tensor: " & $ty)
  ty.tensor

func arity(kind: ShOpKind): tuple[operands, results: int] =
  case kind
  of okIota, okReplicaId, okPartitionId, okCreateToken: (0, 1)
  of okConstant: (0, 1)
  of okAdd, okSub, okMul, okDiv, okMax, okMin,
     okAtan2, okPower, okRemainder,
     okAnd, okOr, okXor,
     okShiftLeft, okShiftRightArithmetic, okShiftRightLogical: (2, 1)
  of okComplex: (2, 1)
  of okNeg, okExp, okLog, okSqrt, okAbs, okTanh,
     okSine, okCosine, okRsqrt,
     okCbrt, okCeil, okExponentialMinusOne, okFloor,
     okLogPlusOne, okLogistic, okTan,
     okSign, okRoundNearestAfz, okRoundNearestEven, okNot,
     okCountLeadingZeros, okPopcnt, okOptimizationBarrier,
     okConvert, okBitcastConvert, okIsFinite,
     okReducePrecision,
     okReshape, okTranspose, okReverse,
     okCholesky, okGetDimensionSize,
     okBroadcast: (1, 1)
  of okReal, okImag: (1, 1)
  of okDynamicBroadcastInDim: (2, 1)
  of okFft: (1, 1)
  of okTriangularSolve: (2, 1)
  of okEinsum: (2, 1)
  of okUnaryEinsum: (1, 1)
  of okTorchIndexSelect: (2, 1)
  of okRng: (3, 1)
  of okRngBitGenerator: (1, 2)
  of okGather: (2, 1)
  of okDynamicGather: (3, 1)
  of okSelectAndScatter: (3, 1)
  of okCollectiveBroadcast, okCollectivePermute, okCrossReplicaSum: (1, 1)
  of okAsyncDone: (1, -1)
  of okDynamicConv: (3, 1)
  of okUniformQuantize, okUniformDequantize: (1, 1)
  of okSort, okMap, okCase, okScatter,
     okAllGather, okAllReduce, okAllToAll, okReduceScatter,
     okAsyncStart, okInfeed, okOutfeed, okSend, okRecv,
     okCustomCall, okComposite: (-1, -1)
  of okConcatenate: (-1, 1)  # variadic operands
  of okSlice: (1, 1)
  of okClamp: (3, 1)
  of okConvolution: (2, 1)
  of okSetDimensionSize, okDynamicReshape: (2, 1)
  of okReduceWindow: (2, 1)
  of okPad: (2, 1)
  of okDynamicPad: (5, 1)
  of okDynamicIota: (1, 1)
  of okRealDynamicSlice: (4, 1)
  of okAfterAll: (-1, 1)
  of okTuple: (-1, 1)
  of okGetTupleElement: (1, 1)
  of okBatchNormInference: (5, 1)
  of okBatchNormTraining: (3, 3)
  of okBatchNormGrad: (5, 3)
  of okReduce: (2, 1)
  of okDotGeneral: (2, 1)
  of okDot: (2, 1)
  of okBroadcastInDim: (1, 1)
  of okDynamicSlice, okDynamicUpdateSlice: (-1, 1)
  of okIf: (1, -1)  # 1 predicate operand, variadic results
  of okCompare: (2, 1)
  of okWhile: (-1, -1)  # variadic carried operands and results
  of okSelect: (3, 1)
  of okReturn, okStablehloReturn: (-1, 0)  # variadic operands, no results

proc verifyOp(fn: ShFunction; op: ShOp; isLast: bool;
    expectedReturns: openArray[ShValueType];
    inRegion: bool) {.raises: [StableHloError].} =
  let name = opName(op.kind)
  let (nOps, nRes) = arity(op.kind)

  if nOps >= 0 and op.operands.len != nOps:
    fail(name & ": expected " & $nOps & " operand(s), got " & $op.operands.len)
  if nRes >= 0 and op.results.len != nRes:
    fail(name & ": expected " & $nRes & " result(s), got " & $op.results.len)
  for i, res in op.results:
    let tableTy = valueTypeOfId(fn, res.id)
    let resultTy = res.valueTypeOf
    if resultTy != tableTy:
      fail(name & ": result #" & $i & " has type " & $resultTy &
        ", SSA table records " & $tableTy)

  proc verifyRegion(region: ShRegion; expected: openArray[ShValueType];
      label: string) =
    if region.ops.len == 0:
      fail(name & ": " & label & " region body is empty")
    for j, rop in region.ops:
      verifyOp(fn, rop, j == region.ops.high, expected, inRegion = true)
    let last = region.ops[^1]
    if last.kind != okStablehloReturn:
      fail(name & ": " & label & " region must end with stablehlo.return")
    if last.operands.len != expected.len:
      fail(name & ": " & label & " region returns " &
        $last.operands.len & " value(s), expected " & $expected.len)
    for i, id in last.operands:
      let actual = valueTypeOfId(fn, id)
      if actual != expected[i]:
        fail(name & ": " & label & " return #" & $i & " has type " &
          $actual & ", expected " & $expected[i])

  case op.kind
  of okCreateToken:
    let res = op.results[0].valueTypeOf
    if not res.isToken:
      fail(name & ": result type must be token, got " & $res)

  of okAfterAll:
    for i, id in op.operands:
      let ty = valueTypeOfId(fn, id)
      if not ty.isToken:
        fail(name & ": operand #" & $i & " must be token, got " & $ty)
    let res = op.results[0].valueTypeOf
    if not res.isToken:
      fail(name & ": result type must be token, got " & $res)

  of okTuple:
    let res = op.results[0].valueTypeOf
    if not res.isTuple:
      fail(name & ": result type must be tuple, got " & $res)
    if res.elements.len != op.operands.len:
      fail(name & ": result tuple element count " & $res.elements.len &
        " does not match operand count " & $op.operands.len)
    for i, id in op.operands:
      let elemTy = valueTypeOfId(fn, id)
      if res.elements[i] != elemTy:
        fail(name & ": tuple element #" & $i & " type " &
          $res.elements[i] & " does not match operand type " & $elemTy)

  of okGetTupleElement:
    let operandTy = valueTypeOfId(fn, op.operands[0])
    if not operandTy.isTuple:
      fail(name & ": operand must be tuple, got " & $operandTy)
    var
      hasIndex = false
      index: int64
    for a in op.attrs:
      if a.name == "index" and a.value.kind == akI64:
        hasIndex = true
        index = a.value.i64
        break
    if not hasIndex:
      fail(name & ": missing `index` attribute")
    if index < 0 or index.int >= operandTy.elements.len:
      fail(name & ": index " & $index &
        " out of range for tuple length " & $operandTy.elements.len)
    let res = op.results[0].valueTypeOf
    let expected = operandTy.elements[index.int]
    if res != expected:
      fail(name & ": result type " & $res &
        " does not match tuple element type " & $expected)

  of okComplex:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt != rt:
      fail(name & ": operand type mismatch — " & $lt & " vs " & $rt)
    if lt.dtype notin {dtFloat32, dtFloat64}:
      fail(name & ": operands must be float32 or float64 tensors, got " & $lt)
    let expected = initTensorType(lt.dtype.complexDType, lt.shape)
    let res = op.results[0].ty
    if res != expected:
      fail(name & ": result type " & $res &
        " does not match expected " & $expected)

  of okReal, okImag:
    let ty = typeOfId(fn, op.operands[0])
    if not (ty.dtype.isFloat or ty.dtype.isComplex):
      fail(name & ": operand must be floating point or complex, got " & $ty)
    let expected = initTensorType(ty.dtype.complexPartDType, ty.shape)
    let res = op.results[0].ty
    if res != expected:
      fail(name & ": result type " & $res &
        " does not match expected " & $expected)

  of okDynamicBroadcastInDim:
    let operandTy = typeOfId(fn, op.operands[0])
    let dimsTy = typeOfId(fn, op.operands[1])
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype:
      fail(name & ": result dtype differs from operand dtype")
    if not (dimsTy.dtype.isSignedInt or dimsTy.dtype.isUnsignedInt) or
        dimsTy.shape != @[res.shape.len]:
      fail(name & ": output_dimensions must be an integer vector " &
        "matching result rank")
    var
      dims: seq[int64]
      expanding: seq[int64]
      nonexpanding: seq[int64]
    for a in op.attrs:
      case a.name
      of "broadcast_dimensions":
        if a.value.kind == akI64Array: dims = a.value.i64s
      of "known_expanding_dimensions":
        if a.value.kind == akI64Array: expanding = a.value.i64s
      of "known_nonexpanding_dimensions":
        if a.value.kind == akI64Array: nonexpanding = a.value.i64s
      else: discard
    if dims.len != operandTy.shape.len:
      fail(name & ": broadcast_dimensions length " & $dims.len &
        " does not match operand rank " & $operandTy.shape.len)
    var seenResult = newSeq[bool](res.shape.len)
    for i, d in dims:
      if d < 0 or d.int >= res.shape.len:
        fail(name & ": broadcast dimension " & $d &
          " out of range for result rank " & $res.shape.len)
      if seenResult[d]:
        fail(name & ": result dim " & $d & " mapped twice")
      seenResult[d] = true
      let inDim = operandTy.shape[i]
      let outDim = res.shape[d]
      if inDim != 1 and inDim != outDim:
        fail(name & ": operand dim " & $i & " (size " & $inDim &
          ") cannot broadcast to result dim " & $d & " (size " &
          $outDim & ")")
    var seenKnown = newSeq[bool](operandTy.shape.len)
    for d in expanding:
      if d < 0 or d.int >= operandTy.shape.len:
        fail(name & ": known_expanding dimension out of range")
      if seenKnown[d]:
        fail(name & ": known dimension " & $d & " repeated")
      seenKnown[d] = true
    for d in nonexpanding:
      if d < 0 or d.int >= operandTy.shape.len:
        fail(name & ": known_nonexpanding dimension out of range")
      if seenKnown[d]:
        fail(name & ": known dimension " & $d &
          " appears in both expanding and nonexpanding sets")
      seenKnown[d] = true

  of okFft:
    let operandTy = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    var
      fftType = ""
      fftLength: seq[int64]
    for a in op.attrs:
      if a.name == "fft_type" and a.value.kind == akString:
        fftType = a.value.str
      elif a.name == "fft_length" and a.value.kind == akI64Array:
        fftLength = a.value.i64s
    if fftType notin ["FFT", "IFFT", "RFFT", "IRFFT"]:
      fail(name & ": invalid fft_type '" & fftType & "'")
    if fftLength.len < 1 or fftLength.len > 3:
      fail(name & ": fft_length must have length 1..3")
    if fftLength.len > operandTy.shape.len:
      fail(name & ": fft_length rank exceeds operand rank")
    let suffixStart = operandTy.shape.len - fftLength.len
    for i, d in fftLength:
      if d < 0:
        fail(name & ": fft_length entries must be non-negative")
      if fftType in ["FFT", "IFFT", "RFFT"] and
          operandTy.shape[suffixStart + i] != d.int:
        fail(name & ": fft_length does not match operand suffix shape")
    case fftType
    of "FFT", "IFFT":
      if not operandTy.dtype.isComplex:
        fail(name & ": FFT/IFFT operand must be complex")
      if res != operandTy:
        fail(name & ": result type must match complex operand")
    of "RFFT":
      if not operandTy.dtype.isFloat:
        fail(name & ": RFFT operand must be floating point")
      let expectedDtype = operandTy.dtype.complexDType
      var expectedShape = operandTy.shape
      expectedShape[^1] =
        if operandTy.shape[^1] == 0: 0
        else: operandTy.shape[^1] div 2 + 1
      if res.dtype != expectedDtype or res.shape != expectedShape:
        fail(name & ": RFFT result type mismatch")
    of "IRFFT":
      if not operandTy.dtype.isComplex:
        fail(name & ": IRFFT operand must be complex")
      var expectedShape = operandTy.shape
      for i, d in fftLength:
        expectedShape[suffixStart + i] = d.int
      let expectedLast =
        if expectedShape[^1] == 0: 0
        else: expectedShape[^1] div 2 + 1
      if operandTy.shape[^1] != expectedLast:
        fail(name & ": IRFFT operand last dimension mismatch")
      let expectedDtype = operandTy.dtype.complexPartDType
      if res.dtype != expectedDtype or res.shape != expectedShape:
        fail(name & ": IRFFT result type mismatch")
    else:
      discard

  of okTriangularSolve:
    let aTy = typeOfId(fn, op.operands[0])
    let bTy = typeOfId(fn, op.operands[1])
    let res = op.results[0].ty
    var
      hasLeft = false
      hasLower = false
      hasUnit = false
      leftSide = true
      transposeA = ""
    for a in op.attrs:
      case a.name
      of "left_side":
        if a.value.kind == akBool:
          hasLeft = true
          leftSide = a.value.b
      of "lower":
        if a.value.kind == akBool: hasLower = true
      of "unit_diagonal":
        if a.value.kind == akBool: hasUnit = true
      of "transpose_a":
        if a.value.kind == akString: transposeA = a.value.str
      else: discard
    if not hasLeft or not hasLower or not hasUnit:
      fail(name & ": missing boolean attributes")
    if transposeA notin ["NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"]:
      fail(name & ": invalid transpose_a '" & transposeA & "'")
    if aTy.dtype != bTy.dtype:
      fail(name & ": dtype mismatch")
    if not (aTy.dtype.isFloat or aTy.dtype.isComplex):
      fail(name & ": operands must be floating point or complex")
    if aTy.shape.len < 2 or bTy.shape.len < 2:
      fail(name & ": operands must have rank >= 2")
    if aTy.shape.len != bTy.shape.len:
      fail(name & ": operand ranks must match")
    if aTy.shape[^1] != aTy.shape[^2]:
      fail(name & ": coefficient matrix must be square")
    for i in 0 ..< max(aTy.shape.len - 2, 0):
      if aTy.shape[i] != bTy.shape[i]:
        fail(name & ": batch dim mismatch")
    let rhsAxis = if leftSide: bTy.shape.len - 2 else: bTy.shape.len - 1
    if bTy.shape[rhsAxis] != aTy.shape[^1]:
      fail(name & ": rhs matrix dimension must match coefficient matrix")
    if res != bTy:
      fail(name & ": result type must match rhs")

  of okEinsum, okUnaryEinsum:
    var config = ""
    for a in op.attrs:
      if a.name == "einsum_config" and a.value.kind == akString:
        config = a.value.str
        break
    if config.len == 0:
      fail(name & ": missing `einsum_config` attribute")
    let res = op.results[0].ty
    if op.kind == okEinsum:
      let lhsTy = typeOfId(fn, op.operands[0])
      let rhsTy = typeOfId(fn, op.operands[1])
      if lhsTy.dtype != rhsTy.dtype:
        fail(name & ": operand dtype mismatch")
      if res.dtype != lhsTy.dtype:
        fail(name & ": result dtype must match operands")
    else:
      let operandTy = typeOfId(fn, op.operands[0])
      if res.dtype != operandTy.dtype:
        fail(name & ": result dtype must match operand")
    for d in res.shape:
      if d < 0:
        fail(name & ": result shape must be non-negative")

  of okTorchIndexSelect:
    let operandTy = typeOfId(fn, op.operands[0])
    let indexTy = typeOfId(fn, op.operands[1])
    let res = op.results[0].ty
    if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
      fail(name & ": index must be an integer tensor")
    var
      hasDim = false
      hasBatchDims = false
      dim: int64
      batchDims: int64
    for a in op.attrs:
      if a.name == "dim" and a.value.kind == akI64:
        hasDim = true
        dim = a.value.i64
      elif a.name == "batch_dims" and a.value.kind == akI64:
        hasBatchDims = true
        batchDims = a.value.i64
    if not hasDim or not hasBatchDims:
      fail(name & ": missing `dim` or `batch_dims` attribute")
    if dim < 0 or dim.int >= operandTy.shape.len:
      fail(name & ": dim out of range")
    if batchDims < 0 or batchDims > dim or
        batchDims.int > indexTy.shape.len:
      fail(name & ": invalid batch_dims")
    for i in 0 ..< batchDims.int:
      if operandTy.shape[i] != indexTy.shape[i]:
        fail(name & ": batch dim mismatch")
    var expectedShape: seq[int] = @[]
    for i in 0 ..< dim.int:
      expectedShape.add operandTy.shape[i]
    for i in batchDims.int ..< indexTy.shape.len:
      expectedShape.add indexTy.shape[i]
    for i in dim.int + 1 ..< operandTy.shape.len:
      expectedShape.add operandTy.shape[i]
    if res.dtype != operandTy.dtype or res.shape != expectedShape:
      fail(name & ": result type does not match expected shape")

  of okRng:
    let aTy = typeOfId(fn, op.operands[0])
    let bTy = typeOfId(fn, op.operands[1])
    let shapeTy = typeOfId(fn, op.operands[2])
    let res = op.results[0].ty
    if aTy != bTy or aTy.shape.len != 0:
      fail(name & ": bounds must be same-typed scalar tensors")
    if not (aTy.dtype == dtBool or aTy.dtype.isSignedInt or
        aTy.dtype.isUnsignedInt or aTy.dtype.isFloat):
      fail(name & ": unsupported bound dtype")
    if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
        shapeTy.shape != @[res.shape.len]:
      fail(name & ": shape operand must be an integer vector matching result rank")
    if res.dtype != aTy.dtype:
      fail(name & ": result dtype must match bounds")

  of okRngBitGenerator:
    let stateTy = typeOfId(fn, op.operands[0])
    if op.results[0].ty != stateTy:
      fail(name & ": output_state type must match initial_state")
    let outTy = op.results[1].ty
    if not (outTy.dtype.isSignedInt or outTy.dtype.isUnsignedInt or
        outTy.dtype.isFloat):
      fail(name & ": output must have int or float element type")

  of okGather, okDynamicGather:
    let operandTy = typeOfId(fn, op.operands[0])
    let indexTy = typeOfId(fn, op.operands[1])
    if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
      fail(name & ": indices must be an integer tensor")
    if op.kind == okDynamicGather:
      let sizesTy = typeOfId(fn, op.operands[2])
      if not (sizesTy.dtype.isSignedInt or sizesTy.dtype.isUnsignedInt) or
          sizesTy.shape != @[operandTy.shape.len]:
        fail(name & ": dynamic slice_sizes must match operand rank")
    else:
      var sizes: seq[int64]
      for a in op.attrs:
        if a.name == "slice_sizes" and a.value.kind == akI64Array:
          sizes = a.value.i64s
          break
      if sizes.len != operandTy.shape.len:
        fail(name & ": slice_sizes length must match operand rank")
      for i, size in sizes:
        if size < 0 or size.int > operandTy.shape[i]:
          fail(name & ": slice_sizes entry out of range")
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype:
      fail(name & ": result dtype must match operand")

  of okSort:
    if op.operands.len == 0 or op.results.len != op.operands.len:
      fail(name & ": result count must match non-empty input count")
    let first = typeOfId(fn, op.operands[0])
    for i, id in op.operands:
      let ty = typeOfId(fn, id)
      if ty.shape != first.shape:
        fail(name & ": operand #" & $i & " shape mismatch")
      if op.results[i].ty != ty:
        fail(name & ": result #" & $i & " must match operand type")
    if op.regions.len != 1:
      fail(name & ": expected comparator region")
    verifyRegion(op.regions[0], [initValueType(initTensorType(dtBool, []))],
      "comparator")

  of okMap:
    if op.operands.len == 0 or op.results.len == 0:
      fail(name & ": expected at least one operand and one result")
    let first = typeOfId(fn, op.operands[0])
    for id in op.operands:
      if typeOfId(fn, id).shape != first.shape:
        fail(name & ": operand shapes must match")
    var expected = newSeq[ShValueType](op.results.len)
    for i, res in op.results:
      if res.ty.shape != first.shape:
        fail(name & ": result shape must match operand shape")
      expected[i] = initValueType(initTensorType(res.ty.dtype, []))
    if op.regions.len != 1:
      fail(name & ": expected computation region")
    verifyRegion(op.regions[0], expected, "computation")

  of okCase:
    let idxTy = typeOfId(fn, op.operands[0])
    if idxTy.dtype != dtInt32:
      fail(name & ": index must be int32")
    if op.regions.len == 0:
      fail(name & ": expected at least one branch")
    var expected = newSeq[ShValueType](op.results.len)
    for i, res in op.results:
      expected[i] = res.valueTypeOf
    for i, region in op.regions:
      verifyRegion(region, expected, "branch #" & $i)

  of okScatter:
    if op.results.len == 0:
      fail(name & ": expected at least one result")
    let n = op.results.len
    if op.operands.len != n * 2 + 1:
      fail(name & ": operand count must be inputs + indices + updates")
    let indexTy = typeOfId(fn, op.operands[n])
    if not (indexTy.dtype.isSignedInt or indexTy.dtype.isUnsignedInt):
      fail(name & ": scatter_indices must be integer tensor")
    var expected = newSeq[ShValueType](n)
    for i in 0 ..< n:
      let inTy = typeOfId(fn, op.operands[i])
      let updateTy = typeOfId(fn, op.operands[n + 1 + i])
      if updateTy.dtype != inTy.dtype:
        fail(name & ": update dtype mismatch")
      if op.results[i].ty != inTy:
        fail(name & ": result type must match input type")
      expected[i] = initValueType(initTensorType(inTy.dtype, []))
    if op.regions.len != 1:
      fail(name & ": expected update region")
    verifyRegion(op.regions[0], expected, "update")

  of okSelectAndScatter:
    let operandTy = typeOfId(fn, op.operands[0])
    let sourceTy = typeOfId(fn, op.operands[1])
    let initTy = typeOfId(fn, op.operands[2])
    if sourceTy.dtype != operandTy.dtype or initTy.dtype != operandTy.dtype or
        initTy.shape.len != 0:
      fail(name & ": source/init dtypes must match operand; init must be scalar")
    if op.results[0].ty != operandTy:
      fail(name & ": result type must match operand")
    if op.regions.len != 2:
      fail(name & ": expected select and scatter regions")
    verifyRegion(op.regions[0], [initValueType(initTensorType(dtBool, []))],
      "select")
    verifyRegion(op.regions[1],
      [initValueType(initTensorType(operandTy.dtype, []))], "scatter")

  of okAllGather, okAllToAll:
    if op.operands.len == 0 or op.results.len != op.operands.len:
      fail(name & ": result count must match non-empty operand count")
    for i, id in op.operands:
      let ty = typeOfId(fn, id)
      if op.results[i].ty.dtype != ty.dtype:
        fail(name & ": result dtype mismatch")

  of okAllReduce:
    if op.operands.len == 0 or op.results.len != op.operands.len:
      fail(name & ": result count must match non-empty operand count")
    var expected = newSeq[ShValueType](op.results.len)
    for i, id in op.operands:
      let ty = typeOfId(fn, id)
      if op.results[i].ty != ty:
        fail(name & ": result type must match operand")
      expected[i] = initValueType(initTensorType(ty.dtype, []))
    if op.regions.len != 1:
      fail(name & ": expected computation region")
    verifyRegion(op.regions[0], expected, "computation")

  of okReduceScatter:
    let ty = typeOfId(fn, op.operands[0])
    if op.results[0].ty.dtype != ty.dtype:
      fail(name & ": result dtype must match operand")
    if op.regions.len != 1:
      fail(name & ": expected computation region")
    verifyRegion(op.regions[0],
      [initValueType(initTensorType(ty.dtype, []))], "computation")

  of okCollectiveBroadcast, okCollectivePermute, okCrossReplicaSum:
    let ty = typeOfId(fn, op.operands[0])
    if op.results[0].ty != ty:
      fail(name & ": result type must match operand")

  of okAsyncStart:
    if op.results[0].valueTypeOf.kind != stkFuture:
      fail(name & ": result must be future")
    if op.regions.len != 1:
      fail(name & ": expected body region")
    verifyRegion(op.regions[0], op.results[0].valueTypeOf.futureResults,
      "body")

  of okAsyncDone:
    let futureTy = valueTypeOfId(fn, op.operands[0])
    if not futureTy.isFuture:
      fail(name & ": operand must be future")
    if op.results.len != futureTy.futureResults.len:
      fail(name & ": result count must match future result count")
    for i, res in op.results:
      if res.valueTypeOf != futureTy.futureResults[i]:
        fail(name & ": result type must match future payload")

  of okInfeed, okRecv:
    if not valueTypeOfId(fn, op.operands[0]).isToken:
      fail(name & ": operand must be token")
    if op.results.len == 0:
      fail(name & ": expected at least one result")

  of okOutfeed, okSend:
    if op.operands.len == 0 or
        not valueTypeOfId(fn, op.operands[^1]).isToken:
      fail(name & ": final operand must be token")
    if not op.results[0].valueTypeOf.isToken:
      fail(name & ": result must be token")

  of okCustomCall, okComposite:
    if op.results.len == 0:
      fail(name & ": expected at least one result")

  of okDynamicConv:
    let lhsTy = typeOfId(fn, op.operands[0])
    let rhsTy = typeOfId(fn, op.operands[1])
    let padTy = typeOfId(fn, op.operands[2])
    if lhsTy.dtype != rhsTy.dtype:
      fail(name & ": lhs/rhs dtype mismatch")
    if not (padTy.dtype.isSignedInt or padTy.dtype.isUnsignedInt) or
        padTy.shape.len != 2 or padTy.shape[1] != 2:
      fail(name & ": padding must be a rank-2 integer tensor with width 2")
    if op.results[0].ty.dtype != lhsTy.dtype:
      fail(name & ": result dtype mismatch")

  of okUniformQuantize:
    let operandTy = typeOfId(fn, op.operands[0])
    if op.results[0].ty.shape != operandTy.shape:
      fail(name & ": result shape must match operand")

  of okUniformDequantize:
    let operandTy = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if not res.dtype.isFloat:
      fail(name & ": result dtype must be floating point")
    if res.shape != operandTy.shape:
      fail(name & ": result shape must match operand")

  of okConstant:
    var hasValue = false
    for a in op.attrs:
      if a.name == "value" and a.value.kind == akDenseElements:
        hasValue = true
        let resTy = op.results[0].ty
        if resTy.dtype != a.value.denseDtype or
            resTy.shape != a.value.denseShape:
          fail(name & ": dense value " &
            initTensorType(a.value.denseDtype, a.value.denseShape).`$` &
            " does not match result type " & $resTy)
        let expected = resTy.numElements * resTy.dtype.byteSize
        if a.value.denseBytes.len != expected:
          fail(name & ": dense value byte length " &
            $a.value.denseBytes.len & " does not match expected " & $expected)
    if not hasValue:
      fail(name & ": missing required `value` attribute")

  of okAdd, okSub, okMul, okDiv, okMax, okMin,
     okAtan2, okPower, okRemainder,
     okAnd, okOr, okXor,
     okShiftLeft, okShiftRightArithmetic, okShiftRightLogical:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt != rt:
      fail(name & ": operand type mismatch — " & $lt & " vs " & $rt)
    let res = op.results[0].ty
    if res != lt:
      fail(name & ": result type " & $res & " does not match operand " & $lt)

  of okNeg, okExp, okLog, okSqrt, okAbs, okTanh,
     okSine, okCosine, okRsqrt,
     okCbrt, okCeil, okExponentialMinusOne, okFloor,
     okLogPlusOne, okLogistic, okTan,
     okSign, okRoundNearestAfz, okRoundNearestEven, okNot,
     okOptimizationBarrier:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res != ot:
      fail(name & ": result type " & $res & " does not match operand " & $ot)

  of okCountLeadingZeros, okPopcnt:
    let ot = typeOfId(fn, op.operands[0])
    if not (ot.dtype.isSignedInt or ot.dtype.isUnsignedInt):
      fail(name & ": operand must be an integer tensor, got " & $ot)
    let res = op.results[0].ty
    if res != ot:
      fail(name & ": result type " & $res & " does not match operand " & $ot)

  of okConvert:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res.shape != ot.shape:
      fail(name & ": result shape " & $res.shape &
        " must match operand shape " & $ot.shape)

  of okBitcastConvert:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    let inBits = ot.numElements * ot.dtype.bitWidth
    let outBits = res.numElements * res.dtype.bitWidth
    if inBits != outBits:
      fail(name & ": input bit count " & $inBits &
        " differs from result bit count " & $outBits)

  of okIsFinite:
    let ot = typeOfId(fn, op.operands[0])
    if not ot.dtype.isFloat:
      fail(name & ": operand must be a floating-point tensor, got " & $ot)
    let res = op.results[0].ty
    if res.dtype != dtBool or res.shape != ot.shape:
      fail(name & ": result type " & $res &
        " must be bool with operand shape " & $ot.shape)

  of okReducePrecision:
    let ot = typeOfId(fn, op.operands[0])
    if not ot.dtype.isFloat:
      fail(name & ": operand must be a floating-point tensor, got " & $ot)
    let res = op.results[0].ty
    if res != ot:
      fail(name & ": result type " & $res & " does not match operand " & $ot)
    var
      hasExponent = false
      hasMantissa = false
      exponentBits: int64
      mantissaBits: int64
    for a in op.attrs:
      if a.name == "exponent_bits" and a.value.kind == akI64:
        hasExponent = true
        exponentBits = a.value.i64
      elif a.name == "mantissa_bits" and a.value.kind == akI64:
        hasMantissa = true
        mantissaBits = a.value.i64
    if not hasExponent:
      fail(name & ": missing `exponent_bits` attribute")
    if not hasMantissa:
      fail(name & ": missing `mantissa_bits` attribute")
    if exponentBits < 1:
      fail(name & ": exponent_bits must be >= 1")
    if mantissaBits < 0:
      fail(name & ": mantissa_bits must be >= 0")

  of okBatchNormInference:
    let operandTy = typeOfId(fn, op.operands[0])
    let scaleTy = typeOfId(fn, op.operands[1])
    let offsetTy = typeOfId(fn, op.operands[2])
    let meanTy = typeOfId(fn, op.operands[3])
    let varianceTy = typeOfId(fn, op.operands[4])
    if not operandTy.dtype.isFloat:
      fail(name & ": operand must be floating point, got " & $operandTy)
    var
      hasEpsilon = false
      hasFeatureIndex = false
      featureIndex: int64
    for a in op.attrs:
      if a.name == "epsilon" and a.value.kind == akF64:
        hasEpsilon = true
      elif a.name == "feature_index" and a.value.kind == akI64:
        hasFeatureIndex = true
        featureIndex = a.value.i64
    if not hasEpsilon:
      fail(name & ": missing `epsilon` attribute")
    if not hasFeatureIndex:
      fail(name & ": missing `feature_index` attribute")
    if featureIndex < 0 or featureIndex.int >= operandTy.shape.len:
      fail(name & ": feature_index " & $featureIndex &
        " out of range for rank " & $operandTy.shape.len)
    let featureShape = @[operandTy.shape[featureIndex.int]]
    for input in [("scale", scaleTy), ("offset", offsetTy),
                  ("mean", meanTy), ("variance", varianceTy)]:
      let inputName = input[0]
      let inputTy = input[1]
      if inputTy.dtype != operandTy.dtype:
        fail(name & ": " & inputName & " dtype " & $inputTy.dtype &
          " differs from operand dtype " & $operandTy.dtype)
      if inputTy.shape != featureShape:
        fail(name & ": " & inputName & " shape " & $inputTy.shape &
          " must be " & $featureShape)
    let res = op.results[0].ty
    if res != operandTy:
      fail(name & ": result type " & $res &
        " does not match operand " & $operandTy)

  of okBatchNormTraining, okBatchNormGrad:
    let operandTy = typeOfId(fn, op.operands[0])
    if not operandTy.dtype.isFloat:
      fail(name & ": operand must be floating point, got " & $operandTy)
    var
      hasEpsilon = false
      hasFeatureIndex = false
      featureIndex: int64
    for a in op.attrs:
      if a.name == "epsilon" and a.value.kind == akF64:
        hasEpsilon = true
      elif a.name == "feature_index" and a.value.kind == akI64:
        hasFeatureIndex = true
        featureIndex = a.value.i64
    if not hasEpsilon:
      fail(name & ": missing `epsilon` attribute")
    if not hasFeatureIndex:
      fail(name & ": missing `feature_index` attribute")
    if featureIndex < 0 or featureIndex.int >= operandTy.shape.len:
      fail(name & ": feature_index " & $featureIndex &
        " out of range for rank " & $operandTy.shape.len)
    let featureShape = @[operandTy.shape[featureIndex.int]]
    let featureTy = initTensorType(operandTy.dtype, featureShape)
    if op.kind == okBatchNormTraining:
      let scaleTy = typeOfId(fn, op.operands[1])
      let offsetTy = typeOfId(fn, op.operands[2])
      for input in [("scale", scaleTy), ("offset", offsetTy)]:
        let inputName = input[0]
        let inputTy = input[1]
        if inputTy.dtype != operandTy.dtype:
          fail(name & ": " & inputName & " dtype " & $inputTy.dtype &
            " differs from operand dtype " & $operandTy.dtype)
        if inputTy.shape != featureShape:
          fail(name & ": " & inputName & " shape " & $inputTy.shape &
            " must be " & $featureShape)
    else:
      let scaleTy = typeOfId(fn, op.operands[1])
      let meanTy = typeOfId(fn, op.operands[2])
      let varianceTy = typeOfId(fn, op.operands[3])
      let gradOutputTy = typeOfId(fn, op.operands[4])
      if gradOutputTy != operandTy:
        fail(name & ": grad_output type " & $gradOutputTy &
          " must match operand type " & $operandTy)
      for input in [("scale", scaleTy), ("mean", meanTy),
                    ("variance", varianceTy)]:
        let inputName = input[0]
        let inputTy = input[1]
        if inputTy.dtype != operandTy.dtype:
          fail(name & ": " & inputName & " dtype " & $inputTy.dtype &
            " differs from operand dtype " & $operandTy.dtype)
        if inputTy.shape != featureShape:
          fail(name & ": " & inputName & " shape " & $inputTy.shape &
            " must be " & $featureShape)
    if op.results[0].ty != operandTy:
      fail(name & ": output type " & $op.results[0].ty &
        " does not match operand " & $operandTy)
    if op.results[1].ty != featureTy:
      fail(name & ": second result type " & $op.results[1].ty &
        " must be " & $featureTy)
    if op.results[2].ty != featureTy:
      fail(name & ": third result type " & $op.results[2].ty &
        " must be " & $featureTy)

  of okDot:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt.dtype != rt.dtype:
      fail(name & ": dtype mismatch — " & $lt & " vs " & $rt)
    if lt.shape.len < 1 or lt.shape.len > 2:
      fail(name & ": lhs must be rank 1 or 2, got rank " & $lt.shape.len)
    if rt.shape.len < 1 or rt.shape.len > 2:
      fail(name & ": rhs must be rank 1 or 2, got rank " & $rt.shape.len)
    if lt.shape[^1] != rt.shape[0]:
      fail(name & ": contracting dim mismatch")
    var expectedShape: seq[int] = @[]
    if lt.shape.len == 2:
      expectedShape.add lt.shape[0]
    if rt.shape.len == 2:
      expectedShape.add rt.shape[1]
    let res = op.results[0].ty
    if res.dtype != lt.dtype or res.shape != expectedShape:
      fail(name & ": result type " & $res &
        " does not match expected " & $initTensorType(lt.dtype, expectedShape))

  of okCholesky:
    let ty = typeOfId(fn, op.operands[0])
    if not ty.dtype.isFloat:
      fail(name & ": operand must be floating point, got " & $ty)
    if ty.shape.len < 2:
      fail(name & ": operand rank must be at least 2, got " & $ty.shape.len)
    if ty.shape[^1] != ty.shape[^2]:
      fail(name & ": innermost matrix dimensions must be square")
    var hasLower = false
    for a in op.attrs:
      if a.name == "lower" and a.value.kind == akBool:
        hasLower = true
        break
    if not hasLower:
      fail(name & ": missing `lower` attribute")
    let res = op.results[0].ty
    if res != ty:
      fail(name & ": result type " & $res & " does not match operand " & $ty)

  of okGetDimensionSize:
    let ty = typeOfId(fn, op.operands[0])
    var
      hasDim = false
      dimension: int64
    for a in op.attrs:
      if a.name == "dimension" and a.value.kind == akI64:
        hasDim = true
        dimension = a.value.i64
        break
    if not hasDim:
      fail(name & ": missing `dimension` attribute")
    if dimension < 0 or dimension.int >= ty.shape.len:
      fail(name & ": dimension " & $dimension &
        " out of range for rank " & $ty.shape.len)
    let res = op.results[0].ty
    let expected = initTensorType(dtInt32, @[])
    if res != expected:
      fail(name & ": result type " & $res & " must be " & $expected)

  of okPad:
    let operandTy = typeOfId(fn, op.operands[0])
    let paddingTy = typeOfId(fn, op.operands[1])
    if paddingTy.dtype != operandTy.dtype or paddingTy.shape.len != 0:
      fail(name & ": padding_value must be scalar with operand dtype")
    var lows, highs, interiors: seq[int64]
    for a in op.attrs:
      if a.name == "edge_padding_low" and a.value.kind == akI64Array:
        lows = a.value.i64s
      elif a.name == "edge_padding_high" and a.value.kind == akI64Array:
        highs = a.value.i64s
      elif a.name == "interior_padding" and a.value.kind == akI64Array:
        interiors = a.value.i64s
    if lows.len != operandTy.shape.len or highs.len != operandTy.shape.len or
        interiors.len != operandTy.shape.len:
      fail(name & ": padding arrays must match operand rank " &
        $operandTy.shape.len)
    var expectedShape = newSeq[int](operandTy.shape.len)
    for i, dim in operandTy.shape:
      if interiors[i] < 0:
        fail(name & ": interior padding at dim " & $i &
          " must be non-negative")
      let outDim = int64(dim) + lows[i] + highs[i] +
        int64(max(dim - 1, 0)) * interiors[i]
      if outDim < 0:
        fail(name & ": result dimension " & $i & " is negative")
      expectedShape[i] = int(outDim)
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype or res.shape != expectedShape:
      fail(name & ": result type " & $res &
        " does not match expected " &
        $initTensorType(operandTy.dtype, expectedShape))

  of okBroadcast:
    let operandTy = typeOfId(fn, op.operands[0])
    var sizes: seq[int64]
    for a in op.attrs:
      if a.name == "broadcast_sizes" and a.value.kind == akI64Array:
        sizes = a.value.i64s
        break
    var expectedShape: seq[int] = @[]
    for i, d in sizes:
      if d < 0:
        fail(name & ": broadcast size #" & $i & " must be non-negative")
      expectedShape.add int(d)
    expectedShape.add operandTy.shape
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype or res.shape != expectedShape:
      fail(name & ": result type " & $res &
        " does not match expected " &
        $initTensorType(operandTy.dtype, expectedShape))

  of okDynamicSlice:
    if op.operands.len < 2:
      fail(name & ": expected operand plus start indices")
    let operandTy = typeOfId(fn, op.operands[0])
    if op.operands.len != operandTy.shape.len + 1:
      fail(name & ": start index count " & $(op.operands.len - 1) &
        " must match operand rank " & $operandTy.shape.len)
    var sizes: seq[int64]
    for a in op.attrs:
      if a.name == "slice_sizes" and a.value.kind == akI64Array:
        sizes = a.value.i64s
        break
    if sizes.len != operandTy.shape.len:
      fail(name & ": slice_sizes length " & $sizes.len &
        " must match operand rank " & $operandTy.shape.len)
    var expectedShape = newSeq[int](sizes.len)
    for i, size in sizes:
      if size < 0 or size > operandTy.shape[i]:
        fail(name & ": slice size " & $size & " out of range at dim " & $i)
      expectedShape[i] = int(size)
    for i in 1 ..< op.operands.len:
      let idxTy = typeOfId(fn, op.operands[i])
      if not (idxTy.dtype.isSignedInt or idxTy.dtype.isUnsignedInt) or
          idxTy.shape.len != 0:
        fail(name & ": start index #" & $(i - 1) &
          " must be an integer scalar, got " & $idxTy)
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype or res.shape != expectedShape:
      fail(name & ": result type " & $res &
        " does not match expected " &
        $initTensorType(operandTy.dtype, expectedShape))

  of okDynamicUpdateSlice:
    if op.operands.len < 3:
      fail(name & ": expected operand, update, and start indices")
    let operandTy = typeOfId(fn, op.operands[0])
    let updateTy = typeOfId(fn, op.operands[1])
    if updateTy.dtype != operandTy.dtype:
      fail(name & ": update dtype " & $updateTy.dtype &
        " differs from operand dtype " & $operandTy.dtype)
    if updateTy.shape.len != operandTy.shape.len:
      fail(name & ": update rank must match operand rank")
    if op.operands.len != operandTy.shape.len + 2:
      fail(name & ": start index count " & $(op.operands.len - 2) &
        " must match operand rank " & $operandTy.shape.len)
    for i in 0 ..< operandTy.shape.len:
      if updateTy.shape[i] > operandTy.shape[i]:
        fail(name & ": update dim " & $i & " is larger than operand dim")
    for i in 2 ..< op.operands.len:
      let idxTy = typeOfId(fn, op.operands[i])
      if not (idxTy.dtype.isSignedInt or idxTy.dtype.isUnsignedInt) or
          idxTy.shape.len != 0:
        fail(name & ": start index #" & $(i - 2) &
          " must be an integer scalar, got " & $idxTy)
    let res = op.results[0].ty
    if res != operandTy:
      fail(name & ": result type " & $res &
        " does not match operand " & $operandTy)

  of okIota:
    let res = op.results[0].ty
    if res.dtype == dtBool:
      fail(name & ": bool element type is not supported")
    var
      hasDimension = false
      dimension: int64
    for a in op.attrs:
      if a.name == "iota_dimension" and a.value.kind == akI64:
        hasDimension = true
        dimension = a.value.i64
        break
    if not hasDimension:
      fail(name & ": missing `iota_dimension` attribute")
    if dimension < 0 or dimension.int >= res.shape.len:
      fail(name & ": iota_dimension " & $dimension &
        " out of range for rank " & $res.shape.len)

  of okReplicaId, okPartitionId:
    let res = op.results[0].ty
    let expected = initTensorType(dtUint32, @[])
    if res != expected:
      fail(name & ": result type " & $res & " must be " & $expected)

  of okSetDimensionSize:
    let operandTy = typeOfId(fn, op.operands[0])
    let sizeTy = typeOfId(fn, op.operands[1])
    if not (sizeTy.dtype.isSignedInt or sizeTy.dtype.isUnsignedInt) or
        sizeTy.shape.len != 0:
      fail(name & ": size must be an integer scalar, got " & $sizeTy)
    var
      hasDimension = false
      dimension: int64
    for a in op.attrs:
      if a.name == "dimension" and a.value.kind == akI64:
        hasDimension = true
        dimension = a.value.i64
        break
    if not hasDimension:
      fail(name & ": missing `dimension` attribute")
    if dimension < 0 or dimension.int >= operandTy.shape.len:
      fail(name & ": dimension " & $dimension &
        " out of range for rank " & $operandTy.shape.len)
    let res = op.results[0].ty
    if res != operandTy:
      fail(name & ": result type " & $res &
        " does not match operand " & $operandTy)

  of okDynamicReshape:
    let operandTy = typeOfId(fn, op.operands[0])
    let shapeTy = typeOfId(fn, op.operands[1])
    let res = op.results[0].ty
    if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
        shapeTy.shape.len != 1:
      fail(name & ": output_shape must be a rank-1 integer tensor")
    if shapeTy.shape[0] != res.shape.len:
      fail(name & ": output_shape length must match result rank")
    if res.dtype != operandTy.dtype:
      fail(name & ": result dtype differs from operand dtype")
    if res.numElements != operandTy.numElements:
      fail(name & ": element count mismatch")

  of okDynamicPad:
    let operandTy = typeOfId(fn, op.operands[0])
    let paddingTy = typeOfId(fn, op.operands[1])
    if paddingTy.dtype != operandTy.dtype or paddingTy.shape.len != 0:
      fail(name & ": padding_value must be scalar with operand dtype")
    for i in 2 .. 4:
      let ty = typeOfId(fn, op.operands[i])
      if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
          ty.shape != @[operandTy.shape.len]:
        fail(name & ": padding operand #" & $(i - 2) &
          " must be an integer vector of length " & $operandTy.shape.len)
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype:
      fail(name & ": result dtype differs from operand dtype")
    if res.shape.len != operandTy.shape.len:
      fail(name & ": result rank must match operand rank")

  of okDynamicIota:
    let shapeTy = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res.dtype == dtBool:
      fail(name & ": bool element type is not supported")
    if not (shapeTy.dtype.isSignedInt or shapeTy.dtype.isUnsignedInt) or
        shapeTy.shape != @[res.shape.len]:
      fail(name & ": output_shape must be an integer vector matching result rank")
    var
      hasDimension = false
      dimension: int64
    for a in op.attrs:
      if a.name == "iota_dimension" and a.value.kind == akI64:
        hasDimension = true
        dimension = a.value.i64
        break
    if not hasDimension:
      fail(name & ": missing `iota_dimension` attribute")
    if dimension < 0 or dimension.int >= res.shape.len:
      fail(name & ": iota_dimension out of range")

  of okRealDynamicSlice:
    let operandTy = typeOfId(fn, op.operands[0])
    for i in 1 .. 3:
      let ty = typeOfId(fn, op.operands[i])
      if not (ty.dtype.isSignedInt or ty.dtype.isUnsignedInt) or
          ty.shape != @[operandTy.shape.len]:
        fail(name & ": slice operand #" & $(i - 1) &
          " must be an integer vector of length " & $operandTy.shape.len)
    let res = op.results[0].ty
    if res.dtype != operandTy.dtype:
      fail(name & ": result dtype differs from operand dtype")
    if res.shape.len != operandTy.shape.len:
      fail(name & ": result rank must match operand rank")

  of okReshape:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res.dtype != ot.dtype:
      fail(name & ": dtype mismatch — " & $ot & " vs " & $res)
    if res.numElements != ot.numElements:
      fail(name & ": element count mismatch — " & $ot & " (" &
        $ot.numElements & ") vs " & $res & " (" & $res.numElements & ")")

  of okTranspose:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res.dtype != ot.dtype:
      fail(name & ": dtype mismatch — " & $ot & " vs " & $res)
    var perm: seq[int64] = @[]
    for a in op.attrs:
      if a.name == "permutation" and a.value.kind == akI64Array:
        perm = a.value.i64s
        break
    if perm.len != ot.shape.len:
      fail(name & ": permutation length " & $perm.len &
        " does not match operand rank " & $ot.shape.len)
    var seenAxes = newSeq[bool](ot.shape.len)
    var expectedShape = newSeq[int](ot.shape.len)
    for i, p in perm:
      if p < 0 or p.int >= ot.shape.len:
        fail(name & ": permutation index " & $p & " out of range")
      if seenAxes[p]:
        fail(name & ": permutation index " & $p & " repeated")
      seenAxes[p] = true
      expectedShape[i] = ot.shape[p]
    if res.shape != expectedShape:
      fail(name & ": result shape " & $res.shape &
        " does not match permuted shape " & $expectedShape)

  of okReverse:
    let ot = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res != ot:
      fail(name & ": result type " & $res & " does not match operand " & $ot)
    var dims: seq[int64] = @[]
    for a in op.attrs:
      if a.name == "dimensions" and a.value.kind == akI64Array:
        dims = a.value.i64s
        break
    if dims.len == 0:
      fail(name & ": missing or empty `dimensions` attribute")
    var seenAxes = newSeq[bool](ot.shape.len)
    for d in dims:
      if d < 0 or d.int >= ot.shape.len:
        fail(name & ": dimension " & $d &
          " out of range for rank " & $ot.shape.len)
      if seenAxes[d]:
        fail(name & ": dimension " & $d & " repeated")
      seenAxes[d] = true

  of okReturn:
    if inRegion:
      fail(name & " must appear in a function body, not a region")
    if not isLast:
      fail(name & " must be the last op in the function")
    if op.operands.len != expectedReturns.len:
      fail(name & ": expected " & $expectedReturns.len &
        " return value(s), got " & $op.operands.len)
    for i, id in op.operands:
      let actual = valueTypeOfId(fn, id)
      if actual != expectedReturns[i]:
        fail(name & ": return value #" & $i & " has type " & $actual &
          ", function declares " & $expectedReturns[i])

  of okStablehloReturn:
    if not inRegion:
      fail(name & " may only appear inside a region body")
    if not isLast:
      fail(name & " must be the last op in its region")
    # Operand types are checked by the enclosing op (e.g. reduce).

  of okReduce:
    let inTy = typeOfId(fn, op.operands[0])
    let initTy = typeOfId(fn, op.operands[1])
    if initTy.shape.len != 0:
      fail(name & ": init value must be 0-rank, got " & $initTy)
    if initTy.dtype != inTy.dtype:
      fail(name & ": init dtype " & $initTy.dtype &
        " differs from input dtype " & $inTy.dtype)
    var dims: seq[int64] = @[]
    for a in op.attrs:
      if a.name == "dimensions" and a.value.kind == akI64Array:
        dims = a.value.i64s
        break
    if dims.len == 0:
      fail(name & ": missing or empty `dimensions` attribute")
    var seenDim = newSeq[bool](inTy.shape.len)
    var expectedShape: seq[int] = @[]
    for d in dims:
      if d < 0 or d.int >= inTy.shape.len:
        fail(name & ": dimension " & $d & " out of range for rank " &
          $inTy.shape.len)
      if seenDim[d]:
        fail(name & ": dimension " & $d & " repeated")
      seenDim[d] = true
    for i, dim in inTy.shape:
      if not seenDim[i]: expectedShape.add dim
    let res = op.results[0].ty
    if res.dtype != inTy.dtype:
      fail(name & ": result dtype " & $res.dtype &
        " differs from input dtype " & $inTy.dtype)
    if res.shape != expectedShape:
      fail(name & ": result shape " & $res.shape &
        " does not match reduced shape " & $expectedShape)
    if op.regions.len != 1:
      fail(name & ": expected exactly 1 region, got " & $op.regions.len)
    let reducer = op.regions[0]
    let elemTy = initTensorType(inTy.dtype, @[])
    if reducer.args.len != 2:
      fail(name & ": reducer region must have 2 args, got " &
        $reducer.args.len)
    for i, a in reducer.args:
      if a.ty != elemTy:
        fail(name & ": reducer arg #" & $i & " has type " & $a.ty &
          ", expected " & $elemTy)
    if reducer.ops.len == 0:
      fail(name & ": reducer region body is empty")
    for j, rop in reducer.ops:
      let isLastInRegion = j == reducer.ops.high
      verifyOp(fn, rop, isLastInRegion, expectedReturns, inRegion = true)
    let last = reducer.ops[^1]
    if last.kind != okStablehloReturn:
      fail(name & ": reducer region must end with stablehlo.return")
    if last.operands.len != 1:
      fail(name & ": reducer return must yield exactly 1 value")
    let retTy = typeOfId(fn, last.operands[0])
    if retTy != elemTy:
      fail(name & ": reducer return type " & $retTy &
        " differs from element type " & $elemTy)

  of okDotGeneral:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt.dtype != rt.dtype:
      fail(name & ": dtype mismatch \u2014 " & $lt & " vs " & $rt)
    var dims: ShAttr
    var hasDims = false
    for a in op.attrs:
      if a.name == "dot_dimension_numbers" and a.value.kind == akDotDims:
        dims = a.value
        hasDims = true
        break
    if not hasDims:
      fail(name & ": missing `dot_dimension_numbers` attribute")
    if dims.lhsBatchingDims.len != dims.rhsBatchingDims.len:
      fail(name & ": batching dim count mismatch")
    if dims.lhsContractingDims.len != dims.rhsContractingDims.len:
      fail(name & ": contracting dim count mismatch")
    var lhsSeen = newSeq[bool](lt.shape.len)
    var rhsSeen = newSeq[bool](rt.shape.len)
    for d in dims.lhsBatchingDims:
      if d < 0 or d.int >= lt.shape.len:
        fail(name & ": lhs batching dim " & $d & " out of range")
      if lhsSeen[d]: fail(name & ": lhs dim " & $d & " repeated")
      lhsSeen[d] = true
    for d in dims.lhsContractingDims:
      if d < 0 or d.int >= lt.shape.len:
        fail(name & ": lhs contracting dim " & $d & " out of range")
      if lhsSeen[d]: fail(name & ": lhs dim " & $d & " repeated")
      lhsSeen[d] = true
    for d in dims.rhsBatchingDims:
      if d < 0 or d.int >= rt.shape.len:
        fail(name & ": rhs batching dim " & $d & " out of range")
      if rhsSeen[d]: fail(name & ": rhs dim " & $d & " repeated")
      rhsSeen[d] = true
    for d in dims.rhsContractingDims:
      if d < 0 or d.int >= rt.shape.len:
        fail(name & ": rhs contracting dim " & $d & " out of range")
      if rhsSeen[d]: fail(name & ": rhs dim " & $d & " repeated")
      rhsSeen[d] = true
    for i in 0 ..< dims.lhsBatchingDims.len:
      if lt.shape[dims.lhsBatchingDims[i]] !=
          rt.shape[dims.rhsBatchingDims[i]]:
        fail(name & ": batching dim size mismatch at index " & $i)
    for i in 0 ..< dims.lhsContractingDims.len:
      if lt.shape[dims.lhsContractingDims[i]] !=
          rt.shape[dims.rhsContractingDims[i]]:
        fail(name & ": contracting dim size mismatch at index " & $i)
    var expectedShape: seq[int] = @[]
    for d in dims.lhsBatchingDims:
      expectedShape.add lt.shape[d]
    for i in 0 ..< lt.shape.len:
      if not lhsSeen[i]: expectedShape.add lt.shape[i]
    for i in 0 ..< rt.shape.len:
      if not rhsSeen[i]: expectedShape.add rt.shape[i]
    let res = op.results[0].ty
    if res.dtype != lt.dtype:
      fail(name & ": result dtype " & $res.dtype &
        " differs from operand dtype " & $lt.dtype)
    if res.shape != expectedShape:
      fail(name & ": result shape " & $res.shape &
        " does not match expected " & $expectedShape)

  of okBroadcastInDim:
    let inTy = typeOfId(fn, op.operands[0])
    let res = op.results[0].ty
    if res.dtype != inTy.dtype:
      fail(name & ": dtype mismatch \u2014 " & $inTy & " vs " & $res)
    var dims: seq[int64] = @[]
    for a in op.attrs:
      if a.name == "broadcast_dimensions" and a.value.kind == akI64Array:
        dims = a.value.i64s
        break
    if dims.len != inTy.shape.len:
      fail(name & ": broadcast_dimensions length " & $dims.len &
        " does not match operand rank " & $inTy.shape.len)
    var seenAxes = newSeq[bool](res.shape.len)
    for i, d in dims:
      if d < 0 or d.int >= res.shape.len:
        fail(name & ": dimension " & $d &
          " out of range for output rank " & $res.shape.len)
      if seenAxes[d]:
        fail(name & ": output dim " & $d & " mapped twice")
      seenAxes[d] = true
      let ind = inTy.shape[i]
      let outd = res.shape[d]
      if ind != 1 and ind != outd:
        fail(name & ": operand dim " & $i & " (size " & $ind &
          ") cannot broadcast to output dim " & $d & " (size " &
          $outd & ")")

  of okIf:
    let predTy = typeOfId(fn, op.operands[0])
    if predTy.dtype != dtBool or predTy.shape.len != 0:
      fail(name & ": predicate must be 0-rank bool, got " & $predTy)
    if op.regions.len != 2:
      fail(name & ": expected exactly 2 regions, got " & $op.regions.len)
    if op.results.len == 0:
      fail(name & ": must yield at least one result")
    for branchIdx, region in op.regions:
      let branchName = if branchIdx == 0: "then" else: "else"
      if region.args.len != 0:
        fail(name & ": " & branchName & " region takes no args, got " &
          $region.args.len)
      if region.ops.len == 0:
        fail(name & ": " & branchName & " region body is empty")
      for j, rop in region.ops:
        let isLastInRegion = j == region.ops.high
        verifyOp(fn, rop, isLastInRegion, expectedReturns, inRegion = true)
      let last = region.ops[^1]
      if last.kind != okStablehloReturn:
        fail(name & ": " & branchName &
          " region must end with stablehlo.return")
      if last.operands.len != op.results.len:
        fail(name & ": " & branchName & " region returns " &
          $last.operands.len & " value(s), op declares " & $op.results.len)
      for i, id in last.operands:
        let actual = typeOfId(fn, id)
        let expected = op.results[i].ty
        if actual != expected:
          fail(name & ": " & branchName & " return #" & $i & " has type " &
            $actual & ", expected " & $expected)

  of okCompare:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt != rt:
      fail(name & ": operand type mismatch \u2014 " & $lt & " vs " & $rt)
    let res = op.results[0].ty
    if res.dtype != dtBool or res.shape != lt.shape:
      fail(name & ": result type " & $res &
        " must be tensor<" & $lt.shape & "xi1>")
    var hasDir = false
    for a in op.attrs:
      if a.name == "comparison_direction" and a.value.kind == akString:
        hasDir = true
        case a.value.str
        of "LT", "LE", "GT", "GE", "EQ", "NE": discard
        else:
          fail(name & ": invalid comparison_direction '" & a.value.str & "'")
        break
    if not hasDir:
      fail(name & ": missing `comparison_direction` attribute")

  of okWhile:
    if op.operands.len == 0:
      fail(name & ": must carry at least one value")
    if op.results.len != op.operands.len:
      fail(name & ": result count " & $op.results.len &
        " does not match operand count " & $op.operands.len)
    var carriedTypes = newSeq[ShTensorType](op.operands.len)
    for i, id in op.operands:
      carriedTypes[i] = typeOfId(fn, id)
      if op.results[i].ty != carriedTypes[i]:
        fail(name & ": carried value #" & $i & " operand type " &
          $carriedTypes[i] & " differs from result type " &
          $op.results[i].ty)
    if op.regions.len != 2:
      fail(name & ": expected exactly 2 regions, got " & $op.regions.len)
    let boolScalar = initTensorType(dtBool, @[])
    for regionIdx, region in op.regions:
      let regionName = if regionIdx == 0: "cond" else: "body"
      if region.args.len != carriedTypes.len:
        fail(name & ": " & regionName & " region has " & $region.args.len &
          " args, expected " & $carriedTypes.len)
      for i, a in region.args:
        if a.ty != carriedTypes[i]:
          fail(name & ": " & regionName & " region arg #" & $i & " has type " &
            $a.ty & ", expected " & $carriedTypes[i])
      if region.ops.len == 0:
        fail(name & ": " & regionName & " region body is empty")
      for j, rop in region.ops:
        let isLastInRegion = j == region.ops.high
        verifyOp(fn, rop, isLastInRegion, expectedReturns, inRegion = true)
      let last = region.ops[^1]
      if last.kind != okStablehloReturn:
        fail(name & ": " & regionName &
          " region must end with stablehlo.return")
      if regionIdx == 0:
        if last.operands.len != 1:
          fail(name & ": cond region must yield exactly 1 value")
        let retTy = typeOfId(fn, last.operands[0])
        if retTy != boolScalar:
          fail(name & ": cond region return type " & $retTy &
            " must be " & $boolScalar)
      else:
        if last.operands.len != carriedTypes.len:
          fail(name & ": body region returns " & $last.operands.len &
            " value(s), op carries " & $carriedTypes.len)
        for i, id in last.operands:
          let actual = typeOfId(fn, id)
          if actual != carriedTypes[i]:
            fail(name & ": body return #" & $i & " has type " & $actual &
              ", expected " & $carriedTypes[i])

  of okSelect:
    let pt = typeOfId(fn, op.operands[0])
    let at = typeOfId(fn, op.operands[1])
    let bt = typeOfId(fn, op.operands[2])
    if pt.dtype != dtBool:
      fail(name & ": predicate must be i1, got " & $pt)
    if pt.shape != at.shape:
      fail(name & ": predicate shape " & $pt.shape &
        " does not match value shape " & $at.shape)
    if at != bt:
      fail(name & ": value-operand type mismatch \u2014 " & $at & " vs " & $bt)
    let res = op.results[0].ty
    if res != at:
      fail(name & ": result type " & $res & " must equal value type " & $at)

  of okConcatenate:
    if op.operands.len == 0:
      fail(name & ": must have at least one operand")
    var dim: int64 = 0
    for a in op.attrs:
      if a.name == "dimension" and a.value.kind == akI64:
        dim = a.value.i64
        break
    let firstTy = typeOfId(fn, op.operands[0])
    if dim < 0 or dim.int >= firstTy.shape.len:
      fail(name & ": dimension " & $dim & " out of range for rank " &
        $firstTy.shape.len)
    var totalDim = 0
    for i, id in op.operands:
      let ty = typeOfId(fn, id)
      if ty.dtype != firstTy.dtype:
        fail(name & ": operand #" & $i & " dtype mismatch")
      if ty.shape.len != firstTy.shape.len:
        fail(name & ": operand #" & $i & " rank mismatch")
      for j in 0 ..< ty.shape.len:
        if j != dim.int and ty.shape[j] != firstTy.shape[j]:
          fail(name & ": operand #" & $i & " shape mismatch at dim " & $j)
      totalDim += ty.shape[dim.int]
    let res = op.results[0].ty
    var expectedShape = firstTy.shape
    expectedShape[dim.int] = totalDim
    if res.dtype != firstTy.dtype:
      fail(name & ": result dtype mismatch")
    if res.shape != expectedShape:
      fail(name & ": result shape " & $res.shape &
        " does not match expected " & $expectedShape)

  of okSlice:
    let inTy = typeOfId(fn, op.operands[0])
    var starts, limits, strides: seq[int64]
    for a in op.attrs:
      if a.name == "start_indices" and a.value.kind == akI64Array:
        starts = a.value.i64s
      elif a.name == "limit_indices" and a.value.kind == akI64Array:
        limits = a.value.i64s
      elif a.name == "strides" and a.value.kind == akI64Array:
        strides = a.value.i64s
    if starts.len != inTy.shape.len or limits.len != inTy.shape.len or
       strides.len != inTy.shape.len:
      fail(name & ": index arrays must match operand rank " &
        $inTy.shape.len)
    var expectedShape = newSeq[int](inTy.shape.len)
    for i in 0 ..< inTy.shape.len:
      if starts[i] < 0 or limits[i] > inTy.shape[i].int64 or
         starts[i] >= limits[i] or strides[i] <= 0:
        fail(name & ": invalid slice bounds at dim " & $i)
      expectedShape[i] = int((limits[i] - starts[i] + strides[i] - 1) div strides[i])
    let res = op.results[0].ty
    if res.dtype != inTy.dtype:
      fail(name & ": dtype mismatch")
    if res.shape != expectedShape:
      fail(name & ": result shape " & $res.shape &
        " does not match expected " & $expectedShape)

  of okClamp:
    let minTy = typeOfId(fn, op.operands[0])
    let opTy = typeOfId(fn, op.operands[1])
    let maxTy = typeOfId(fn, op.operands[2])
    if minTy.dtype != opTy.dtype or maxTy.dtype != opTy.dtype:
      fail(name & ": dtype mismatch among operands")
    let res = op.results[0].ty
    if res != opTy:
      fail(name & ": result type " & $res &
        " does not match operand type " & $opTy)

  of okConvolution:
    let lt = typeOfId(fn, op.operands[0])
    let rt = typeOfId(fn, op.operands[1])
    if lt.dtype != rt.dtype:
      fail(name & ": dtype mismatch \u2014 " & $lt & " vs " & $rt)
    var
      cd: ShAttr
      hasCd = false
      windowStrides: seq[int64]
      lhsDilation: seq[int64]
      rhsDilation: seq[int64]
      windowReversal: seq[int64]
      padRows = 0
      padVals: seq[int64]
      featureGroupCount: int64 = 1
      batchGroupCount: int64 = 1
    for a in op.attrs:
      case a.name
      of "dimension_numbers":
        if a.value.kind == akConvDims:
          cd = a.value
          hasCd = true
      of "window_strides":
        if a.value.kind == akI64Array: windowStrides = a.value.i64s
      of "lhs_dilation":
        if a.value.kind == akI64Array: lhsDilation = a.value.i64s
      of "rhs_dilation":
        if a.value.kind == akI64Array: rhsDilation = a.value.i64s
      of "window_reversal":
        if a.value.kind == akI64Array: windowReversal = a.value.i64s
      of "padding":
        if a.value.kind == akI64Matrix:
          padRows = a.value.matRows
          padVals = a.value.matVals
      of "feature_group_count":
        if a.value.kind == akI64: featureGroupCount = a.value.i64
      of "batch_group_count":
        if a.value.kind == akI64: batchGroupCount = a.value.i64
      else: discard
    if not hasCd:
      fail(name & ": missing `dimension_numbers` attribute")
    let spatialRank = cd.inputSpatialDims.len
    if cd.outputSpatialDims.len != spatialRank or
        cd.kernelSpatialDims.len != spatialRank:
      fail(name & ": spatial dim list lengths must match")
    if windowStrides.len != spatialRank or lhsDilation.len != spatialRank or
        rhsDilation.len != spatialRank or padRows != spatialRank:
      fail(name & ": window/dilation/padding length must equal spatial rank " &
        $spatialRank)
    if windowReversal.len > 0 and windowReversal.len != spatialRank:
      fail(name & ": window_reversal length " & $windowReversal.len &
        " must equal spatial rank " & $spatialRank)
    let outRank = 2 + spatialRank
    if lt.shape.len != outRank or rt.shape.len != outRank:
      fail(name & ": operand rank does not match expected " & $outRank)
    let res = op.results[0].ty
    if res.dtype != lt.dtype:
      fail(name & ": result dtype " & $res.dtype &
        " differs from operand dtype " & $lt.dtype)
    if res.shape.len != outRank:
      fail(name & ": result rank does not match expected " & $outRank)
    if batchGroupCount <= 0 or featureGroupCount <= 0:
      fail(name & ": batch_group_count and feature_group_count must be positive")
    let batch = lt.shape[cd.inputBatchDim]
    if batch mod batchGroupCount.int != 0:
      fail(name & ": batch (" & $batch & ") not divisible by batch_group_count " &
        $batchGroupCount)
    if res.shape[cd.outputBatchDim] != batch div batchGroupCount.int:
      fail(name & ": result batch dim mismatch")
    if res.shape[cd.outputFeatureDim] != rt.shape[cd.kernelOutputFeatureDim]:
      fail(name & ": result feature dim does not match kernel output feature")
    for i in 0 ..< spatialRank:
      let ld = lt.shape[cd.inputSpatialDims[i]]
      let kd = rt.shape[cd.kernelSpatialDims[i]]
      let dilatedInput =
        if ld == 0: 0
        else: (ld - 1) * lhsDilation[i].int + 1
      let dilatedKernel =
        if kd == 0: 0
        else: (kd - 1) * rhsDilation[i].int + 1
      let padded = padVals[i * 2].int + dilatedInput + padVals[i * 2 + 1].int
      let expected =
        if padded == 0 or dilatedKernel > padded: 0
        else: (padded - dilatedKernel) div windowStrides[i].int + 1
      if res.shape[cd.outputSpatialDims[i]] != expected:
        fail(name & ": result spatial dim " & $i &
          " has size " & $res.shape[cd.outputSpatialDims[i]] &
          ", expected " & $expected)

  of okReduceWindow:
    let inTy = typeOfId(fn, op.operands[0])
    let initTy = typeOfId(fn, op.operands[1])
    if initTy.shape.len != 0:
      fail(name & ": init value must be 0-rank, got " & $initTy)
    if initTy.dtype != inTy.dtype:
      fail(name & ": init dtype " & $initTy.dtype &
        " differs from input dtype " & $inTy.dtype)
    var
      windowDims: seq[int64]
      windowStrides: seq[int64]
      baseDilations: seq[int64]
      windowDilations: seq[int64]
      padRows = 0
      padVals: seq[int64]
    for a in op.attrs:
      case a.name
      of "window_dimensions":
        if a.value.kind == akI64Array: windowDims = a.value.i64s
      of "window_strides":
        if a.value.kind == akI64Array: windowStrides = a.value.i64s
      of "base_dilations":
        if a.value.kind == akI64Array: baseDilations = a.value.i64s
      of "window_dilations":
        if a.value.kind == akI64Array: windowDilations = a.value.i64s
      of "padding":
        if a.value.kind == akI64Matrix:
          padRows = a.value.matRows
          padVals = a.value.matVals
      else: discard
    let r = inTy.shape.len
    if windowDims.len != r or windowStrides.len != r or
        baseDilations.len != r or windowDilations.len != r or padRows != r:
      fail(name & ": window/stride/padding/dilation arrays must have rank " & $r)
    let res = op.results[0].ty
    if res.dtype != inTy.dtype:
      fail(name & ": result dtype mismatch")
    if res.shape.len != r:
      fail(name & ": result rank mismatch")
    for i in 0 ..< r:
      let dilatedInput =
        if inTy.shape[i] == 0: 0
        else: (inTy.shape[i] - 1) * baseDilations[i].int + 1
      let padded = padVals[i * 2].int + dilatedInput + padVals[i * 2 + 1].int
      let dilatedWindow =
        if windowDims[i] == 0: 0
        else: (windowDims[i].int - 1) * windowDilations[i].int + 1
      let expected =
        if padded == 0 or dilatedWindow > padded: 0
        else: (padded - dilatedWindow) div windowStrides[i].int + 1
      if res.shape[i] != expected:
        fail(name & ": result dim " & $i & " has size " & $res.shape[i] &
          ", expected " & $expected)
    if op.regions.len != 1:
      fail(name & ": expected exactly 1 region, got " & $op.regions.len)
    let reducer = op.regions[0]
    let elemTy = initTensorType(inTy.dtype, @[])
    if reducer.args.len != 2:
      fail(name & ": reducer region must have 2 args, got " & $reducer.args.len)
    for i, a in reducer.args:
      if a.ty != elemTy:
        fail(name & ": reducer arg #" & $i & " has type " & $a.ty &
          ", expected " & $elemTy)
    if reducer.ops.len == 0:
      fail(name & ": reducer region body is empty")
    for j, rop in reducer.ops:
      let isLastInRegion = j == reducer.ops.high
      verifyOp(fn, rop, isLastInRegion, expectedReturns, inRegion = true)
    let last = reducer.ops[^1]
    if last.kind != okStablehloReturn:
      fail(name & ": reducer region must end with stablehlo.return")
    if last.operands.len != 1:
      fail(name & ": reducer return must yield exactly 1 value")
    let retTy = typeOfId(fn, last.operands[0])
    if retTy != elemTy:
      fail(name & ": reducer return type " & $retTy &
        " differs from element type " & $elemTy)

proc verifyFunction(fn: ShFunction) {.raises: [StableHloError].} =
  if fn.name.len == 0:
    fail("function has empty name")
  if fn.inputValueTypes.len > 0 and fn.inputValueTypes.len != fn.inputTypes.len:
    fail("function '" & fn.name &
      "': general input type count disagrees with tensor input count")
  if fn.outputValueTypes.len > 0 and
      fn.outputValueTypes.len != fn.outputTypes.len:
    fail("function '" & fn.name &
      "': general output type count disagrees with tensor output count")
  if fn.valueTypes.len > 0 and fn.valueTypes.len != fn.types.len:
    fail("function '" & fn.name &
      "': general SSA type table count disagrees with tensor type table")
  let inputVts = inputValueTypesOf(fn)
  let outputVts = outputValueTypesOf(fn)
  for i, ty in inputVts:
    let tensorTy = fn.inputTypes[i]
    if ty.isTensor:
      if ty.tensor != tensorTy:
        fail("function '" & fn.name & "' input #" & $i &
          " general type " & $ty & " disagrees with tensor type " &
          $tensorTy)
    elif tensorTy != ShTensorType():
      fail("function '" & fn.name & "' input #" & $i &
        " non-tensor type " & $ty &
        " must use a sentinel tensor table entry")
  for i, ty in outputVts:
    let tensorTy = fn.outputTypes[i]
    if ty.isTensor:
      if ty.tensor != tensorTy:
        fail("function '" & fn.name & "' output #" & $i &
          " general type " & $ty & " disagrees with tensor type " &
          $tensorTy)
    elif tensorTy != ShTensorType():
      fail("function '" & fn.name & "' output #" & $i &
        " non-tensor type " & $ty &
        " must use a sentinel tensor table entry")
  let valueVts =
    if fn.valueTypes.len > 0: fn.valueTypes
    else: tensorValueTypes(fn.types)
  for i, ty in valueVts:
    if i == 0:
      continue
    let tensorTy = fn.types[i]
    if ty.isTensor:
      if ty.tensor != tensorTy:
        fail("function '" & fn.name & "' SSA value #" & $i &
          " general type " & $ty & " disagrees with tensor type " &
          $tensorTy)
    elif tensorTy != ShTensorType():
      fail("function '" & fn.name & "' SSA value #" & $i &
        " non-tensor type " & $ty &
        " must use a sentinel tensor table entry")
  if fn.args.len != inputVts.len:
    fail("function '" & fn.name & "': arg count " & $fn.args.len &
      " disagrees with declared input count " & $inputVts.len)
  for i, a in fn.args:
    if a.valueTypeOf != inputVts[i]:
      fail("function '" & fn.name & "' arg #" & $i &
        " has type " & $a.valueTypeOf & ", declared " & $inputVts[i])
    let generalTy = valueTypeOfId(fn, a.id)
    if a.valueTypeOf != generalTy:
      fail("function '" & fn.name & "' arg #" & $i &
        " general type " & $a.valueTypeOf &
        " disagrees with SSA table type " & $generalTy)
  if fn.ops.len == 0:
    fail("function '" & fn.name & "' has empty body")
  for i, op in fn.ops:
    let isLast = i == fn.ops.high
    verifyOp(fn, op, isLast, outputVts, inRegion = false)
  if fn.ops[^1].kind != okReturn:
    fail("function '" & fn.name & "' must end with func.return")

proc verify*(m: ShModule) {.raises: [StableHloError].} =
  ## Validates the whole module. Raises `StableHloError` on the first
  ## problem found; later passes do not run.
  if m.funcs.len == 0:
    fail("module '" & m.name & "' has no functions")
  var seen: seq[string] = @[]
  for fn in m.funcs:
    if fn.name in seen:
      fail("module '" & m.name & "' has duplicate function '" & fn.name & "'")
    seen.add fn.name
    verifyFunction(fn)
