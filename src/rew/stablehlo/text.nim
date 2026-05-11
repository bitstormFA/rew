## StableHLO **textual MLIR** emitter.
##
## This is the v1 lowering surface: take a verified `ShModule` and produce
## the `module { func.func @main(...) { stablehlo.add ... } }` text that
## every PJRT plugin will accept via `PjrtClient::Compile`.
##
## Bytecode emission is deferred to Phase 9 and will reuse the same IR.
## The MLIR-bytecode primitives in `mlirbc.nim` are already validated so
## that work is a focused encoder, not a fresh container project.
##
## The emitter assumes the module has been validated by `verify.nim`. It
## still raises `StableHloEmitError` for shapes the textual format itself
## cannot represent (e.g. dtypes the StableHLO dialect spells differently
## from our enum).

import ../dtype
import ./ir

type
  StableHloEmitError* = object of CatchableError
    ## Raised when the IR is well-formed but the textual lowering cannot
    ## represent it. Distinct from `StableHloError` (verifier).

func mlirElementType(dt: DType): string =
  ## StableHLO uses the standard MLIR builtin element-type spellings.
  case dt
  of dtBool: "i1"
  of dtInt4: "i4"
  of dtInt8: "i8"
  of dtInt16: "i16"
  of dtInt32: "i32"
  of dtInt64: "i64"
  of dtUint4: "ui4"
  of dtUint8: "ui8"
  of dtUint16: "ui16"
  of dtUint32: "ui32"
  of dtUint64: "ui64"
  of dtFloat16: "f16"
  of dtBFloat16: "bf16"
  of dtFloat32: "f32"
  of dtFloat64: "f64"
  of dtComplex64: "complex<f32>"
  of dtComplex128: "complex<f64>"
  of dtNF4: "i4"
  of dtFloat8E4M3Fn: "f8E4M3FN"
  of dtFloat8E5M2: "f8E5M2"

func mlirTensorType(t: ShTensorType): string =
  ## Produces e.g. `tensor<2x3xf32>` or `tensor<f32>` (rank-0).
  if t.shape.len == 0:
    return "tensor<" & mlirElementType(t.dtype) & ">"
  result = "tensor<"
  for d in t.shape:
    result.add $d
    result.add 'x'
  result.add mlirElementType(t.dtype)
  result.add '>'

proc mlirValueType(t: ShValueType): string =
  ## Produces the MLIR spelling for any currently lowerable value type.
  case t.kind
  of stkTensor:
    result = mlirTensorType(t.tensor)
  of stkToken:
    result = "!stablehlo.token"
  of stkTuple:
    result = "tuple<"
    for i, elem in t.elements:
      if i > 0: result.add ", "
      result.add mlirValueType(elem)
    result.add '>'
  of stkFuture:
    result = "!stablehlo.future<"
    if t.futureResults.len == 1:
      result.add mlirValueType(t.futureResults[0])
    else:
      result.add '('
      for i, elem in t.futureResults:
        if i > 0: result.add ", "
        result.add mlirValueType(elem)
      result.add ')'
    result.add '>'
  of stkResource:
    raise newException(StableHloEmitError,
      "resource value type cannot be emitted as textual StableHLO: " & $t)

proc mlirShardedValueType(t: ShValueType; sharding: string): string =
  ## Renders a value type, optionally attaching an SDY sharding attribute.
  result = mlirValueType(t)
  if sharding.len > 0:
    result.add " {sdy.sharding = #sdy.sharding"
    result.add sharding
    result.add "}"

proc valueTypeOfId(fn: ShFunction; id: ShValueId): ShValueType =
  if fn.valueTypes.len > 0 and id.int > 0 and id.int < fn.valueTypes.len:
    fn.valueTypes[id.int]
  elif id.int > 0 and id.int < fn.types.len:
    initValueType(fn.types[id.int])
  else:
    raise newException(StableHloEmitError,
      "invalid SSA value id " & $id & " in function '" & fn.name & "'")

func valueRef(id: ShValueId): string =
  ## SSA value name. We use `%v<id>` so they line up with the dense ids
  ## the builder hands out, making IR↔text correspondence easy to read.
  "%v" & $id.int

func hexByte(b: byte): string =
  const HexChars = "0123456789ABCDEF"
  result = newString(2)
  result[0] = HexChars[int(b shr 4)]
  result[1] = HexChars[int(b and 0x0F)]

func denseHex(bytes: openArray[byte]): string =
  ## Render dense data in StableHLO's `dense<"0x...">` raw-bytes form.
  ## This avoids having to reinterpret integers/floats by dtype here;
  ## the parser on the consumer side does the bit reinterpretation.
  result = "dense<\"0x"
  for b in bytes: result.add hexByte(b)
  result.add "\">"

func renderI64Array(xs: openArray[int64]): string =
  result = "array<i64: "
  for i, x in xs:
    if i > 0: result.add ", "
    result.add $x
  result.add ">"

func mlirString(s: string): string =
  result = "\""
  for ch in s:
    case ch
    of '\\':
      result.add "\\\\"
    of '"':
      result.add "\\\""
    of '\n':
      result.add "\\0A"
    else:
      result.add ch
  result.add '"'

func opMnemonic(kind: ShOpKind): string =
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

proc emitOp(s: var string; op: ShOp; fn: ShFunction; indent: string)

proc emitStablehloReturn(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  ## `stablehlo.return %v : tensor<...>` (single-value form for reducer
  ## bodies; multi-value reducers add commas).
  s.add indent
  s.add "stablehlo.return"
  if op.operands.len > 0:
    s.add ' '
    for i, operand in op.operands:
      if i > 0: s.add ", "
      s.add valueRef(operand)
    s.add " : "
    for i, operand in op.operands:
      if i > 0: s.add ", "
      s.add mlirValueType(valueTypeOfId(fn, operand))
  s.add '\n'

proc emitReduce(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  ## Pretty form:
  ##   %r = stablehlo.reduce(%input init: %init) across dimensions = [..] :
  ##     (tensor<...>, tensor<f32>) -> tensor<...>
  ##     reducer(%a: tensor<f32>, %b: tensor<f32>) {
  ##       <body ops>
  ##     }
  if op.results.len != 1 or op.operands.len != 2 or op.regions.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.reduce arity mismatch")
  var dims: seq[int64] = @[]
  for a in op.attrs:
    if a.name == "dimensions" and a.value.kind == akI64Array:
      dims = a.value.i64s
      break
  if dims.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.reduce missing `dimensions` attribute")
  let inTy = fn.types[op.operands[0].int]
  let initTy = fn.types[op.operands[1].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.reduce("
  s.add valueRef(op.operands[0])
  s.add " init: "
  s.add valueRef(op.operands[1])
  s.add ") across dimensions = ["
  for i, d in dims:
    if i > 0: s.add ", "
    s.add $d
  s.add "] : ("
  s.add mlirTensorType(inTy)
  s.add ", "
  s.add mlirTensorType(initTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'
  let region = op.regions[0]
  s.add indent
  s.add "  reducer("
  for i, a in region.args:
    if i > 0: s.add ", "
    s.add valueRef(a.id)
    s.add ": "
    s.add mlirTensorType(a.ty)
  s.add ") {\n"
  let bodyIndent = indent & "    "
  for rop in region.ops:
    emitOp(s, rop, fn, bodyIndent)
  s.add indent
  s.add "  }\n"

proc emitElementwiseAt(s: var string; op: ShOp; indent: string) =
  ## Indent-aware variant of `emitElementwise` for use inside region
  ## bodies. Same textual form as the function-body case.
  if op.results.len != 1:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " expected 1 result, got " & $op.results.len)
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = "
  s.add opMnemonic(op.kind)
  s.add ' '
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add " : "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitConstantAt(s: var string; op: ShOp; indent: string) =
  if op.results.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.constant must have exactly one result")
  let res = op.results[0]
  var valueAttr = ""
  for a in op.attrs:
    if a.name == "value" and a.value.kind == akDenseElements:
      valueAttr = denseHex(a.value.denseBytes)
      break
  if valueAttr.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.constant missing dense `value` attribute")
  s.add indent
  s.add valueRef(res.id)
  s.add " = stablehlo.constant "
  s.add valueAttr
  s.add " : "
  s.add mlirTensorType(res.ty)
  s.add '\n'

proc emitReshapeAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError, "stablehlo.reshape arity mismatch")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.reshape "
  s.add valueRef(op.operands[0])
  s.add " : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitTypeChangingUnaryAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " arity mismatch")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  s.add valueRef(op.operands[0])
  s.add ") : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitReducePrecisionAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.reduce_precision arity mismatch")
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
  if not hasExponent or not hasMantissa:
    raise newException(StableHloEmitError,
      "stablehlo.reduce_precision missing bit-width attributes")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.reduce_precision\"("
  s.add valueRef(op.operands[0])
  s.add ") {exponent_bits = "
  s.add $exponentBits
  s.add " : i32, mantissa_bits = "
  s.add $mantissaBits
  s.add " : i32} : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitBatchNormInferenceAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 5:
    raise newException(StableHloEmitError,
      "stablehlo.batch_norm_inference arity mismatch")
  var
    hasEpsilon = false
    hasFeatureIndex = false
    epsilon: float64
    featureIndex: int64
  for a in op.attrs:
    if a.name == "epsilon" and a.value.kind == akF64:
      hasEpsilon = true
      epsilon = a.value.f64
    elif a.name == "feature_index" and a.value.kind == akI64:
      hasFeatureIndex = true
      featureIndex = a.value.i64
  if not hasEpsilon or not hasFeatureIndex:
    raise newException(StableHloEmitError,
      "stablehlo.batch_norm_inference missing attributes")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.batch_norm_inference\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") {epsilon = "
  s.add $epsilon
  s.add " : f32, feature_index = "
  s.add $featureIndex
  s.add " : i64} : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitBatchNormMultiResultAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 3:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " expected 3 results, got " & $op.results.len)
  if op.kind == okBatchNormTraining and op.operands.len != 3:
    raise newException(StableHloEmitError,
      "stablehlo.batch_norm_training arity mismatch")
  if op.kind == okBatchNormGrad and op.operands.len != 5:
    raise newException(StableHloEmitError,
      "stablehlo.batch_norm_grad arity mismatch")
  var
    hasEpsilon = false
    hasFeatureIndex = false
    epsilon: float64
    featureIndex: int64
  for a in op.attrs:
    if a.name == "epsilon" and a.value.kind == akF64:
      hasEpsilon = true
      epsilon = a.value.f64
    elif a.name == "feature_index" and a.value.kind == akI64:
      hasFeatureIndex = true
      featureIndex = a.value.i64
  if not hasEpsilon or not hasFeatureIndex:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " missing attributes")
  s.add indent
  for i, res in op.results:
    if i > 0: s.add ", "
    s.add valueRef(res.id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") {epsilon = "
  s.add $epsilon
  s.add " : f32, feature_index = "
  s.add $featureIndex
  s.add " : i64} : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> ("
  for i, res in op.results:
    if i > 0: s.add ", "
    s.add mlirTensorType(res.ty)
  s.add ")\n"

proc emitCholeskyAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.cholesky arity mismatch")
  var
    hasLower = false
    lower = true
  for a in op.attrs:
    if a.name == "lower" and a.value.kind == akBool:
      hasLower = true
      lower = a.value.b
      break
  if not hasLower:
    raise newException(StableHloEmitError,
      "stablehlo.cholesky missing `lower` attribute")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.cholesky\"("
  s.add valueRef(op.operands[0])
  s.add ") {lower = "
  s.add (if lower: "true" else: "false")
  s.add "} : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitGetDimensionSizeAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.get_dimension_size arity mismatch")
  var
    hasDimension = false
    dimension: int64
  for a in op.attrs:
    if a.name == "dimension" and a.value.kind == akI64:
      hasDimension = true
      dimension = a.value.i64
      break
  if not hasDimension:
    raise newException(StableHloEmitError,
      "stablehlo.get_dimension_size missing `dimension` attribute")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.get_dimension_size\"("
  s.add valueRef(op.operands[0])
  s.add ") {dimension = "
  s.add $dimension
  s.add " : i64} : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitPadAt(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError, "stablehlo.pad arity mismatch")
  var lows, highs, interiors: seq[int64]
  for a in op.attrs:
    if a.name == "edge_padding_low" and a.value.kind == akI64Array:
      lows = a.value.i64s
    elif a.name == "edge_padding_high" and a.value.kind == akI64Array:
      highs = a.value.i64s
    elif a.name == "interior_padding" and a.value.kind == akI64Array:
      interiors = a.value.i64s
  if lows.len == 0 or highs.len == 0 or interiors.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.pad missing padding attributes")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.pad\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") {edge_padding_low = "
  s.add renderI64Array(lows)
  s.add ", edge_padding_high = "
  s.add renderI64Array(highs)
  s.add ", interior_padding = "
  s.add renderI64Array(interiors)
  s.add "} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ", "
  s.add mlirTensorType(fn.types[op.operands[1].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitBroadcastAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.broadcast arity mismatch")
  var sizes: seq[int64]
  for a in op.attrs:
    if a.name == "broadcast_sizes" and a.value.kind == akI64Array:
      sizes = a.value.i64s
      break
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.broadcast\"("
  s.add valueRef(op.operands[0])
  s.add ") {broadcast_sizes = "
  s.add renderI64Array(sizes)
  s.add "} : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitDynamicSliceAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len < 2:
    raise newException(StableHloEmitError,
      "stablehlo.dynamic_slice arity mismatch")
  var sizes: seq[int64]
  for a in op.attrs:
    if a.name == "slice_sizes" and a.value.kind == akI64Array:
      sizes = a.value.i64s
      break
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.dynamic_slice\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") {slice_sizes = "
  s.add renderI64Array(sizes)
  s.add "} : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitDynamicUpdateSliceAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len < 3:
    raise newException(StableHloEmitError,
      "stablehlo.dynamic_update_slice arity mismatch")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.dynamic_update_slice\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitIotaAt(s: var string; op: ShOp; indent: string) =
  if op.results.len != 1 or op.operands.len != 0:
    raise newException(StableHloEmitError, "stablehlo.iota arity mismatch")
  var
    hasDimension = false
    dimension: int64
  for a in op.attrs:
    if a.name == "iota_dimension" and a.value.kind == akI64:
      hasDimension = true
      dimension = a.value.i64
      break
  if not hasDimension:
    raise newException(StableHloEmitError,
      "stablehlo.iota missing `iota_dimension` attribute")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.iota\"() {iota_dimension = "
  s.add $dimension
  s.add " : i64} : () -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitNullaryTensorAt(s: var string; op: ShOp; indent: string) =
  if op.results.len != 1 or op.operands.len != 0:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " arity mismatch")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"() : () -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitSetDimensionSizeAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.set_dimension_size arity mismatch")
  var
    hasDimension = false
    dimension: int64
  for a in op.attrs:
    if a.name == "dimension" and a.value.kind == akI64:
      hasDimension = true
      dimension = a.value.i64
      break
  if not hasDimension:
    raise newException(StableHloEmitError,
      "stablehlo.set_dimension_size missing `dimension` attribute")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.set_dimension_size\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") {dimension = "
  s.add $dimension
  s.add " : i64} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ", "
  s.add mlirTensorType(fn.types[op.operands[1].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitDynamicIotaAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.dynamic_iota arity mismatch")
  var
    hasDimension = false
    dimension: int64
  for a in op.attrs:
    if a.name == "iota_dimension" and a.value.kind == akI64:
      hasDimension = true
      dimension = a.value.i64
      break
  if not hasDimension:
    raise newException(StableHloEmitError,
      "stablehlo.dynamic_iota missing `iota_dimension` attribute")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.dynamic_iota\"("
  s.add valueRef(op.operands[0])
  s.add ") {iota_dimension = "
  s.add $dimension
  s.add " : i64} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitGenericVariadicAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitGenericBinaryAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " arity mismatch")
  let lhsTy = fn.types[op.operands[0].int]
  let rhsTy = fn.types[op.operands[1].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") : ("
  s.add mlirTensorType(lhsTy)
  s.add ", "
  s.add mlirTensorType(rhsTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitComplexAt(s: var string; op: ShOp; indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.complex arity mismatch")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.complex "
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add " : "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitTransposeAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.transpose arity mismatch")
  var perm: seq[int64] = @[]
  for a in op.attrs:
    if a.name == "permutation" and a.value.kind == akI64Array:
      perm = a.value.i64s
      break
  if perm.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.transpose missing `permutation` attribute")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.transpose "
  s.add valueRef(op.operands[0])
  s.add ", dims = ["
  for i, p in perm:
    if i > 0: s.add ", "
    s.add $p
  s.add "] : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitReverseAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError, "stablehlo.reverse arity mismatch")
  var dims: seq[int64] = @[]
  for a in op.attrs:
    if a.name == "dimensions" and a.value.kind == akI64Array:
      dims = a.value.i64s
      break
  if dims.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.reverse missing `dimensions` attribute")
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.reverse "
  s.add valueRef(op.operands[0])
  s.add ", dims = ["
  for i, d in dims:
    if i > 0: s.add ", "
    s.add $d
  s.add "] : "
  s.add mlirTensorType(inTy)
  s.add '\n'

proc emitReturnAt(s: var string; op: ShOp; outputs: openArray[ShValueType];
    indent: string) =
  s.add indent
  s.add "func.return"
  if op.operands.len > 0:
    s.add ' '
    for i, operand in op.operands:
      if i > 0: s.add ", "
      s.add valueRef(operand)
    s.add " : "
    for i, ty in outputs:
      if i > 0: s.add ", "
      s.add mlirValueType(ty)
  s.add '\n'

proc emitDotGeneral(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  ## `%r = stablehlo.dot_general %lhs, %rhs,
  ##     batching_dims = [..] x [..],
  ##     contracting_dims = [..] x [..]
  ##     : (tensor<...>, tensor<...>) -> tensor<...>`
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.dot_general arity mismatch")
  var dims: ShAttr
  var hasDims = false
  for a in op.attrs:
    if a.name == "dot_dimension_numbers" and a.value.kind == akDotDims:
      dims = a.value
      hasDims = true
      break
  if not hasDims:
    raise newException(StableHloEmitError,
      "stablehlo.dot_general missing `dot_dimension_numbers` attribute")
  let lhsTy = fn.types[op.operands[0].int]
  let rhsTy = fn.types[op.operands[1].int]
  proc renderList(xs: openArray[int64]): string =
    result = "["
    for i, x in xs:
      if i > 0: result.add ", "
      result.add $x
    result.add ']'
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.dot_general "
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ", batching_dims = "
  s.add renderList(dims.lhsBatchingDims)
  s.add " x "
  s.add renderList(dims.rhsBatchingDims)
  s.add ", contracting_dims = "
  s.add renderList(dims.lhsContractingDims)
  s.add " x "
  s.add renderList(dims.rhsContractingDims)
  s.add " : ("
  s.add mlirTensorType(lhsTy)
  s.add ", "
  s.add mlirTensorType(rhsTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitBroadcastInDim(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  ## `%r = stablehlo.broadcast_in_dim %in, dims = [..] :
  ##     (tensor<...>) -> tensor<...>`
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.broadcast_in_dim arity mismatch")
  var dims: seq[int64] = @[]
  for a in op.attrs:
    if a.name == "broadcast_dimensions" and a.value.kind == akI64Array:
      dims = a.value.i64s
      break
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.broadcast_in_dim "
  s.add valueRef(op.operands[0])
  s.add ", dims = ["
  for i, d in dims:
    if i > 0: s.add ", "
    s.add $d
  s.add "] : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitIf(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  ## ```
  ## %r:N = "stablehlo.if"(%pred) ({
  ##   <then ops>
  ## }, {
  ##   <else ops>
  ## }) : (tensor<i1>) -> (tensor<...>, ...)
  ## ```
  if op.operands.len != 1 or op.regions.len != 2 or op.results.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.if arity mismatch")
  let predTy = fn.types[op.operands[0].int]
  s.add indent
  if op.results.len == 1:
    s.add valueRef(op.results[0].id)
  else:
    s.add valueRef(op.results[0].id)
    s.add ":"
    s.add $op.results.len
  s.add " = \"stablehlo.if\"("
  s.add valueRef(op.operands[0])
  s.add ") ({\n"
  let bodyIndent = indent & "  "
  for rop in op.regions[0].ops:
    emitOp(s, rop, fn, bodyIndent)
  s.add indent
  s.add "}, {\n"
  for rop in op.regions[1].ops:
    emitOp(s, rop, fn, bodyIndent)
  s.add indent
  s.add "}) : ("
  s.add mlirTensorType(predTy)
  s.add ") -> ("
  for i, r in op.results:
    if i > 0: s.add ", "
    s.add mlirTensorType(r.ty)
  s.add ")\n"

proc emitCompare(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  ## `%r = stablehlo.compare LT, %l, %r : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xi1>`
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.compare arity mismatch")
  var direction = ""
  for a in op.attrs:
    if a.name == "comparison_direction" and a.value.kind == akString:
      direction = a.value.str
      break
  if direction.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.compare missing `comparison_direction` attribute")
  let lhsTy = fn.types[op.operands[0].int]
  let rhsTy = fn.types[op.operands[1].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.compare "
  s.add direction
  s.add ", "
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add " : ("
  s.add mlirTensorType(lhsTy)
  s.add ", "
  s.add mlirTensorType(rhsTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitSelect(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  ## `%r = stablehlo.select %pred, %a, %b : tensor<NxI1>, tensor<NxF32>`
  if op.results.len != 1 or op.operands.len != 3:
    raise newException(StableHloEmitError,
      "stablehlo.select arity mismatch")
  let predTy = fn.types[op.operands[0].int]
  let valTy = op.results[0].ty
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.select "
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ", "
  s.add valueRef(op.operands[2])
  s.add " : "
  s.add mlirTensorType(predTy)
  s.add ", "
  s.add mlirTensorType(valTy)
  s.add '\n'

proc emitWhileRegion(s: var string; region: ShRegion; fn: ShFunction;
    indent: string; label: string) =
  s.add indent
  s.add "^"
  s.add label
  s.add '('
  for i, a in region.args:
    if i > 0: s.add ", "
    s.add valueRef(a.id)
    s.add ": "
    s.add mlirTensorType(a.ty)
  s.add "):\n"
  let bodyIndent = indent & "  "
  for rop in region.ops:
    emitOp(s, rop, fn, bodyIndent)

proc emitWhile(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  ## ```
  ## %r:N = "stablehlo.while"(%init0, %init1) ({
  ##   ^cond(%c0: tensor<...>, %c1: tensor<...>):
  ##     <cond ops>
  ##     "stablehlo.return"(%pred) : (tensor<i1>) -> ()
  ## }, {
  ##   ^body(%b0: tensor<...>, %b1: tensor<...>):
  ##     <body ops>
  ##     "stablehlo.return"(%n0, %n1) : (tensor<...>, tensor<...>) -> ()
  ## }) : (tensor<...>, ...) -> (tensor<...>, ...)
  ## ```
  if op.regions.len != 2 or op.operands.len == 0 or
      op.results.len != op.operands.len:
    raise newException(StableHloEmitError,
      "stablehlo.while arity mismatch")
  s.add indent
  if op.results.len == 1:
    s.add valueRef(op.results[0].id)
  else:
    s.add valueRef(op.results[0].id)
    s.add ":"
    s.add $op.results.len
  s.add " = \"stablehlo.while\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") ({\n"
  let regionIndent = indent & "  "
  emitWhileRegion(s, op.regions[0], fn, regionIndent, "cond")
  s.add indent
  s.add "}, {\n"
  emitWhileRegion(s, op.regions[1], fn, regionIndent, "body")
  s.add indent
  s.add "}) : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> ("
  for i, r in op.results:
    if i > 0: s.add ", "
    s.add mlirTensorType(r.ty)
  s.add ")\n"

proc emitConcatenate(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.concatenate arity mismatch")
  var dim: int64 = 0
  for a in op.attrs:
    if a.name == "dimension" and a.value.kind == akI64:
      dim = a.value.i64
      break
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.concatenate "
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ", dim = "
  s.add $dim
  s.add " : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitSlice(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.slice arity mismatch")
  var starts, limits, strides: seq[int64]
  for a in op.attrs:
    if a.name == "start_indices" and a.value.kind == akI64Array:
      starts = a.value.i64s
    elif a.name == "limit_indices" and a.value.kind == akI64Array:
      limits = a.value.i64s
    elif a.name == "strides" and a.value.kind == akI64Array:
      strides = a.value.i64s
  let inTy = fn.types[op.operands[0].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.slice "
  s.add valueRef(op.operands[0])
  s.add " ["
  for i in 0 ..< starts.len:
    if i > 0: s.add ", "
    s.add $starts[i]
    s.add ":"
    s.add $limits[i]
    if strides[i] != 1:
      s.add ":"
      s.add $strides[i]
  s.add "] : ("
  s.add mlirTensorType(inTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitClamp(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  if op.results.len != 1 or op.operands.len != 3:
    raise newException(StableHloEmitError,
      "stablehlo.clamp arity mismatch")
  let minTy = fn.types[op.operands[0].int]
  let opTy = fn.types[op.operands[1].int]
  let maxTy = fn.types[op.operands[2].int]
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.clamp "
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ", "
  s.add valueRef(op.operands[2])
  s.add " : ("
  s.add mlirTensorType(minTy)
  s.add ", "
  s.add mlirTensorType(opTy)
  s.add ", "
  s.add mlirTensorType(maxTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitConvolution(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  ## ```
  ## %r = stablehlo.convolution(%lhs, %rhs)
  ##   dim_numbers = [b, 0, 1, f]x[o, i, 0, 1]->[b, 0, 1, f],
  ##   window = {stride = [1, 1], pad = [[0, 0], [0, 0]],
  ##             lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
  ##   {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  ##   : (tensor<...>, tensor<...>) -> tensor<...>
  ## ```
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.convolution arity mismatch")
  var
    convDims: ShAttr
    hasConvDims = false
    windowStrides: seq[int64]
    lhsDilation: seq[int64]
    rhsDilation: seq[int64]
    windowReversal: seq[int64]
    hasReversal = false
    padRows = 0
    padVals: seq[int64]
    featureGroupCount: int64 = 1
    batchGroupCount: int64 = 1
  for a in op.attrs:
    case a.name
    of "dimension_numbers":
      if a.value.kind == akConvDims:
        convDims = a.value
        hasConvDims = true
    of "window_strides":
      if a.value.kind == akI64Array: windowStrides = a.value.i64s
    of "lhs_dilation":
      if a.value.kind == akI64Array: lhsDilation = a.value.i64s
    of "rhs_dilation":
      if a.value.kind == akI64Array: rhsDilation = a.value.i64s
    of "window_reversal":
      if a.value.kind == akI64Array:
        windowReversal = a.value.i64s
        hasReversal = true
    of "padding":
      if a.value.kind == akI64Matrix:
        padRows = a.value.matRows
        padVals = a.value.matVals
    of "feature_group_count":
      if a.value.kind == akI64: featureGroupCount = a.value.i64
    of "batch_group_count":
      if a.value.kind == akI64: batchGroupCount = a.value.i64
    else: discard
  if not hasConvDims:
    raise newException(StableHloEmitError,
      "stablehlo.convolution missing `dimension_numbers` attribute")
  let lhsTy = fn.types[op.operands[0].int]
  let rhsTy = fn.types[op.operands[1].int]
  let outTy = op.results[0].ty
  let spatialRank = convDims.inputSpatialDims.len
  let inRank = lhsTy.shape.len
  let kernelRank = rhsTy.shape.len
  let outRank = outTy.shape.len

  proc layoutFor(rank: int; batchOrIn, featureOrOut: int64;
      spatial: openArray[int64]; isKernel: bool): string =
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
    for i in 0 ..< rank:
      if i > 0: result.add ", "
      result.add labels[i]
    result.add ']'

  let inputLayout = layoutFor(inRank,
    convDims.inputBatchDim, convDims.inputFeatureDim,
    convDims.inputSpatialDims, isKernel = false)
  let kernelLayout = layoutFor(kernelRank,
    convDims.kernelInputFeatureDim, convDims.kernelOutputFeatureDim,
    convDims.kernelSpatialDims, isKernel = true)
  let outputLayout = layoutFor(outRank,
    convDims.outputBatchDim, convDims.outputFeatureDim,
    convDims.outputSpatialDims, isKernel = false)

  proc renderI64Array(xs: openArray[int64]): string =
    result = "["
    for i, x in xs:
      if i > 0: result.add ", "
      result.add $x
    result.add ']'

  proc renderPadding(rows: int; vals: openArray[int64]): string =
    result = "["
    for r in 0 ..< rows:
      if r > 0: result.add ", "
      result.add '['
      result.add $vals[r * 2]
      result.add ", "
      result.add $vals[r * 2 + 1]
      result.add ']'
    result.add ']'

  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.convolution("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ")\n"
  s.add indent
  s.add "    dim_numbers = "
  s.add inputLayout
  s.add 'x'
  s.add kernelLayout
  s.add "->"
  s.add outputLayout
  s.add ",\n"
  s.add indent
  s.add "    window = {stride = "
  s.add renderI64Array(windowStrides)
  s.add ", pad = "
  s.add renderPadding(padRows, padVals)
  if lhsDilation.len > 0:
    var nonOne = false
    for d in lhsDilation:
      if d != 1: nonOne = true; break
    if nonOne:
      s.add ", lhs_dilate = "
      s.add renderI64Array(lhsDilation)
  if rhsDilation.len > 0:
    var nonOne = false
    for d in rhsDilation:
      if d != 1: nonOne = true; break
    if nonOne:
      s.add ", rhs_dilate = "
      s.add renderI64Array(rhsDilation)
  if hasReversal:
    s.add ", reverse = ["
    for i, r in windowReversal:
      if i > 0: s.add ", "
      s.add (if r != 0: "true" else: "false")
    s.add ']'
  s.add "}\n"
  s.add indent
  s.add "    {batch_group_count = "
  s.add $batchGroupCount
  s.add " : i64, feature_group_count = "
  s.add $featureGroupCount
  s.add " : i64}\n"
  s.add indent
  s.add "    : ("
  s.add mlirTensorType(lhsTy)
  s.add ", "
  s.add mlirTensorType(rhsTy)
  s.add ") -> "
  s.add mlirTensorType(outTy)
  s.add '\n'
  discard spatialRank

proc emitReduceWindow(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  ## ```
  ## %r = "stablehlo.reduce_window"(%input, %init) ({
  ##   ^bb0(%a: tensor<f32>, %b: tensor<f32>):
  ##     %0 = stablehlo.maximum %a, %b : tensor<f32>
  ##     "stablehlo.return"(%0) : (tensor<f32>) -> ()
  ## }) {
  ##   window_dimensions = array<i64: 1, 2, 2, 1>,
  ##   window_strides = array<i64: 1, 2, 2, 1>,
  ##   base_dilations = array<i64: 1, 1, 1, 1>,
  ##   window_dilations = array<i64: 1, 1, 1, 1>,
  ##   padding = dense<[[0, 0], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
  ## } : (tensor<...>, tensor<f32>) -> tensor<...>
  ## ```
  if op.results.len != 1 or op.operands.len != 2 or op.regions.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.reduce_window arity mismatch")
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
  let inTy = fn.types[op.operands[0].int]
  let initTy = fn.types[op.operands[1].int]
  let r = inTy.shape.len

  proc denseI64Array(xs: openArray[int64]): string =
    result = "array<i64: "
    for i, x in xs:
      if i > 0: result.add ", "
      result.add $x
    result.add '>'

  proc densePadding(rows: int; vals: openArray[int64]): string =
    result = "dense<["
    for ri in 0 ..< rows:
      if ri > 0: result.add ", "
      result.add '['
      result.add $vals[ri * 2]
      result.add ", "
      result.add $vals[ri * 2 + 1]
      result.add ']'
    result.add "]> : tensor<"
    result.add $rows
    result.add "x2xi64>"

  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.reduce_window\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") ({\n"
  let region = op.regions[0]
  s.add indent
  s.add "  ^bb0("
  for i, a in region.args:
    if i > 0: s.add ", "
    s.add valueRef(a.id)
    s.add ": "
    s.add mlirTensorType(a.ty)
  s.add "):\n"
  let bodyIndent = indent & "    "
  for rop in region.ops:
    emitOp(s, rop, fn, bodyIndent)
  s.add indent
  s.add "}) {\n"
  s.add indent
  s.add "  window_dimensions = "
  s.add denseI64Array(windowDims)
  s.add ",\n"
  s.add indent
  s.add "  window_strides = "
  s.add denseI64Array(windowStrides)
  s.add ",\n"
  s.add indent
  s.add "  base_dilations = "
  s.add denseI64Array(baseDilations)
  s.add ",\n"
  s.add indent
  s.add "  window_dilations = "
  s.add denseI64Array(windowDilations)
  s.add ",\n"
  s.add indent
  s.add "  padding = "
  s.add densePadding(padRows, padVals)
  s.add "\n"
  s.add indent
  s.add "} : ("
  s.add mlirTensorType(inTy)
  s.add ", "
  s.add mlirTensorType(initTy)
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'
  discard r

proc emitCreateTokenAt(s: var string; op: ShOp; indent: string) =
  if op.results.len != 1 or op.operands.len != 0:
    raise newException(StableHloEmitError,
      "stablehlo.create_token arity mismatch")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = stablehlo.create_token : "
  s.add mlirValueType(op.results[0].valueTypeOf)
  s.add '\n'

proc emitAfterAllAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.after_all must have exactly one result")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.after_all\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirValueType(valueTypeOfId(fn, operand))
  s.add ") -> "
  s.add mlirValueType(op.results[0].valueTypeOf)
  s.add '\n'

proc emitTupleAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.tuple must have exactly one result")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.tuple\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirValueType(valueTypeOfId(fn, operand))
  s.add ") -> "
  s.add mlirValueType(op.results[0].valueTypeOf)
  s.add '\n'

proc emitGetTupleElementAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError,
      "stablehlo.get_tuple_element arity mismatch")
  var
    hasIndex = false
    index: int64
  for a in op.attrs:
    if a.name == "index" and a.value.kind == akI64:
      hasIndex = true
      index = a.value.i64
      break
  if not hasIndex:
    raise newException(StableHloEmitError,
      "stablehlo.get_tuple_element missing `index` attribute")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.get_tuple_element\"("
  s.add valueRef(op.operands[0])
  s.add ") <{index = "
  s.add $index
  s.add " : i32}> : ("
  s.add mlirValueType(valueTypeOfId(fn, op.operands[0]))
  s.add ") -> "
  s.add mlirValueType(op.results[0].valueTypeOf)
  s.add '\n'

proc emitDynamicBroadcastInDimAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.dynamic_broadcast_in_dim arity mismatch")
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
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.dynamic_broadcast_in_dim\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") {broadcast_dimensions = "
  s.add renderI64Array(dims)
  if expanding.len > 0:
    s.add ", known_expanding_dimensions = "
    s.add renderI64Array(expanding)
  if nonexpanding.len > 0:
    s.add ", known_nonexpanding_dimensions = "
    s.add renderI64Array(nonexpanding)
  s.add "} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ", "
  s.add mlirTensorType(fn.types[op.operands[1].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitFftAt(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  if op.results.len != 1 or op.operands.len != 1:
    raise newException(StableHloEmitError, "stablehlo.fft arity mismatch")
  var
    fftType = ""
    fftLength: seq[int64]
  for a in op.attrs:
    if a.name == "fft_type" and a.value.kind == akString:
      fftType = a.value.str
    elif a.name == "fft_length" and a.value.kind == akI64Array:
      fftLength = a.value.i64s
  if fftType.len == 0 or fftLength.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.fft missing required attributes")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.fft\"("
  s.add valueRef(op.operands[0])
  s.add ") {fft_type = #stablehlo<fft_type "
  s.add fftType
  s.add ">, fft_length = "
  s.add renderI64Array(fftLength)
  s.add "} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitTriangularSolveAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.triangular_solve arity mismatch")
  var
    hasLeft = false
    hasLower = false
    hasUnit = false
    leftSide = true
    lower = true
    unitDiagonal = false
    transposeA = ""
  for a in op.attrs:
    case a.name
    of "left_side":
      if a.value.kind == akBool:
        hasLeft = true
        leftSide = a.value.b
    of "lower":
      if a.value.kind == akBool:
        hasLower = true
        lower = a.value.b
    of "unit_diagonal":
      if a.value.kind == akBool:
        hasUnit = true
        unitDiagonal = a.value.b
    of "transpose_a":
      if a.value.kind == akString:
        transposeA = a.value.str
    else: discard
  if not hasLeft or not hasLower or not hasUnit or transposeA.len == 0:
    raise newException(StableHloEmitError,
      "stablehlo.triangular_solve missing required attributes")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.triangular_solve\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") {left_side = "
  s.add (if leftSide: "true" else: "false")
  s.add ", lower = "
  s.add (if lower: "true" else: "false")
  s.add ", unit_diagonal = "
  s.add (if unitDiagonal: "true" else: "false")
  s.add ", transpose_a = #stablehlo<transpose "
  s.add transposeA
  s.add ">} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ", "
  s.add mlirTensorType(fn.types[op.operands[1].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitEinsumAt(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  if op.results.len != 1:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " arity mismatch")
  if (op.kind == okEinsum and op.operands.len != 2) or
      (op.kind == okUnaryEinsum and op.operands.len != 1):
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " operand count mismatch")
  var config = ""
  for a in op.attrs:
    if a.name == "einsum_config" and a.value.kind == akString:
      config = a.value.str
      break
  if config.len == 0:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " missing `einsum_config` attribute")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ") {einsum_config = "
  s.add mlirString(config)
  s.add "} : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirTensorType(fn.types[operand.int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

proc emitTorchIndexSelectAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len != 1 or op.operands.len != 2:
    raise newException(StableHloEmitError,
      "stablehlo.torch_index_select arity mismatch")
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
    raise newException(StableHloEmitError,
      "stablehlo.torch_index_select missing required attributes")
  s.add indent
  s.add valueRef(op.results[0].id)
  s.add " = \"stablehlo.torch_index_select\"("
  s.add valueRef(op.operands[0])
  s.add ", "
  s.add valueRef(op.operands[1])
  s.add ") {dim = "
  s.add $dim
  s.add " : i64, batch_dims = "
  s.add $batchDims
  s.add " : i64} : ("
  s.add mlirTensorType(fn.types[op.operands[0].int])
  s.add ", "
  s.add mlirTensorType(fn.types[op.operands[1].int])
  s.add ") -> "
  s.add mlirTensorType(op.results[0].ty)
  s.add '\n'

func renderI64Matrix(rows, cols: int; vals: openArray[int64]): string =
  result = "dense<["
  for r in 0 ..< rows:
    if r > 0: result.add ", "
    result.add '['
    for c in 0 ..< cols:
      if c > 0: result.add ", "
      result.add $vals[r * cols + c]
    result.add ']'
  result.add "]> : tensor<"
  result.add $rows
  result.add 'x'
  result.add $cols
  result.add "xi64>"

proc renderAttrValue(attr: ShAttr): string =
  case attr.kind
  of akI64:
    $attr.i64 & " : i64"
  of akI64Array:
    renderI64Array(attr.i64s)
  of akF64:
    $attr.f64 & " : f64"
  of akBool:
    if attr.b: "true" else: "false"
  of akString:
    mlirString(attr.str)
  of akRawMlir:
    attr.mlir
  of akI64Matrix:
    renderI64Matrix(attr.matRows, attr.matCols, attr.matVals)
  of akDenseElements:
    denseHex(attr.denseBytes)
  of akDotDims, akConvDims:
    raise newException(StableHloEmitError,
      "generic emitter cannot render structured StableHLO attribute")

proc emitRegionList(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  for i, region in op.regions:
    if i > 0:
      s.add ", "
    s.add "{\n"
    s.add indent
    s.add "  ^bb0"
    if region.args.len > 0:
      s.add '('
      for j, arg in region.args:
        if j > 0: s.add ", "
        s.add valueRef(arg.id)
        s.add ": "
        s.add mlirValueType(arg.valueTypeOf)
      s.add ')'
    s.add ":\n"
    let bodyIndent = indent & "    "
    for rop in region.ops:
      emitOp(s, rop, fn, bodyIndent)
    s.add indent
    s.add "}"

proc emitGenericQuotedAt(s: var string; op: ShOp; fn: ShFunction;
    indent: string) =
  if op.results.len == 0:
    raise newException(StableHloEmitError,
      opMnemonic(op.kind) & " expected at least one result")
  s.add indent
  for i, res in op.results:
    if i > 0: s.add ", "
    s.add valueRef(res.id)
  s.add " = \""
  s.add opMnemonic(op.kind)
  s.add "\"("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add valueRef(operand)
  s.add ')'
  if op.regions.len > 0:
    s.add " ("
    emitRegionList(s, op, fn, indent)
    s.add ')'
  if op.attrs.len > 0:
    s.add " {"
    for i, attr in op.attrs:
      if i > 0: s.add ", "
      s.add attr.name
      s.add " = "
      s.add renderAttrValue(attr.value)
    s.add '}'
  s.add " : ("
  for i, operand in op.operands:
    if i > 0: s.add ", "
    s.add mlirValueType(valueTypeOfId(fn, operand))
  s.add ") -> "
  if op.results.len > 1:
    s.add '('
  for i, res in op.results:
    if i > 0: s.add ", "
    s.add mlirValueType(res.valueTypeOf)
  if op.results.len > 1:
    s.add ')'
  s.add '\n'

proc emitOp(s: var string; op: ShOp; fn: ShFunction; indent: string) =
  case op.kind
  of okConstant: emitConstantAt(s, op, indent)
  of okAdd, okSub, okMul, okNeg,
     okDiv, okMax, okMin,
     okAtan2, okPower, okRemainder,
     okAnd, okOr, okXor,
     okShiftLeft, okShiftRightArithmetic, okShiftRightLogical,
     okExp, okLog, okSqrt, okAbs, okTanh,
     okSine, okCosine, okRsqrt,
     okCbrt, okCeil, okExponentialMinusOne, okFloor,
     okLogPlusOne, okLogistic, okTan,
     okSign, okRoundNearestAfz, okRoundNearestEven, okNot,
     okCountLeadingZeros, okPopcnt, okOptimizationBarrier:
    emitElementwiseAt(s, op, indent)
  of okConvert, okBitcastConvert, okIsFinite:
    emitTypeChangingUnaryAt(s, op, fn, indent)
  of okReducePrecision:
    emitReducePrecisionAt(s, op, fn, indent)
  of okBatchNormInference:
    emitBatchNormInferenceAt(s, op, fn, indent)
  of okBatchNormTraining, okBatchNormGrad:
    emitBatchNormMultiResultAt(s, op, fn, indent)
  of okDot:
    emitGenericBinaryAt(s, op, fn, indent)
  of okCholesky:
    emitCholeskyAt(s, op, fn, indent)
  of okGetDimensionSize:
    emitGetDimensionSizeAt(s, op, fn, indent)
  of okPad:
    emitPadAt(s, op, fn, indent)
  of okBroadcast:
    emitBroadcastAt(s, op, fn, indent)
  of okDynamicSlice:
    emitDynamicSliceAt(s, op, fn, indent)
  of okDynamicUpdateSlice:
    emitDynamicUpdateSliceAt(s, op, fn, indent)
  of okIota:
    emitIotaAt(s, op, indent)
  of okReplicaId, okPartitionId:
    emitNullaryTensorAt(s, op, indent)
  of okSetDimensionSize:
    emitSetDimensionSizeAt(s, op, fn, indent)
  of okDynamicReshape, okDynamicPad:
    emitGenericVariadicAt(s, op, fn, indent)
  of okDynamicIota:
    emitDynamicIotaAt(s, op, fn, indent)
  of okRealDynamicSlice:
    emitGenericVariadicAt(s, op, fn, indent)
  of okCreateToken:
    emitCreateTokenAt(s, op, indent)
  of okAfterAll:
    emitAfterAllAt(s, op, fn, indent)
  of okTuple:
    emitTupleAt(s, op, fn, indent)
  of okGetTupleElement:
    emitGetTupleElementAt(s, op, fn, indent)
  of okComplex:
    emitComplexAt(s, op, indent)
  of okReal, okImag:
    emitTypeChangingUnaryAt(s, op, fn, indent)
  of okDynamicBroadcastInDim:
    emitDynamicBroadcastInDimAt(s, op, fn, indent)
  of okFft:
    emitFftAt(s, op, fn, indent)
  of okTriangularSolve:
    emitTriangularSolveAt(s, op, fn, indent)
  of okEinsum, okUnaryEinsum:
    emitEinsumAt(s, op, fn, indent)
  of okTorchIndexSelect:
    emitTorchIndexSelectAt(s, op, fn, indent)
  of okRng, okRngBitGenerator,
     okGather, okDynamicGather,
     okSort, okMap, okCase, okScatter, okSelectAndScatter,
     okAllGather, okAllReduce, okAllToAll, okReduceScatter,
     okCollectiveBroadcast, okCollectivePermute, okCrossReplicaSum,
     okAsyncStart, okAsyncDone,
     okInfeed, okOutfeed, okSend, okRecv,
     okCustomCall, okComposite,
     okDynamicConv, okUniformQuantize, okUniformDequantize:
    emitGenericQuotedAt(s, op, fn, indent)
  of okReshape: emitReshapeAt(s, op, fn, indent)
  of okTranspose: emitTransposeAt(s, op, fn, indent)
  of okReverse: emitReverseAt(s, op, fn, indent)
  of okReduce: emitReduce(s, op, fn, indent)
  of okStablehloReturn: emitStablehloReturn(s, op, fn, indent)
  of okDotGeneral: emitDotGeneral(s, op, fn, indent)
  of okBroadcastInDim: emitBroadcastInDim(s, op, fn, indent)
  of okIf: emitIf(s, op, fn, indent)
  of okCompare: emitCompare(s, op, fn, indent)
  of okWhile: emitWhile(s, op, fn, indent)
  of okSelect: emitSelect(s, op, fn, indent)
  of okConcatenate: emitConcatenate(s, op, fn, indent)
  of okSlice: emitSlice(s, op, fn, indent)
  of okClamp: emitClamp(s, op, fn, indent)
  of okConvolution: emitConvolution(s, op, fn, indent)
  of okReduceWindow: emitReduceWindow(s, op, fn, indent)
  of okReturn: emitReturnAt(s, op, outputValueTypesOf(fn), indent)

proc emitFunction(s: var string; fn: ShFunction) =
  s.add "  func.func "
  if fn.visibility == svPrivate:
    s.add "private "
  s.add '@'
  s.add fn.name
  s.add '('
  for i, arg in fn.args:
    if i > 0: s.add ", "
    s.add valueRef(arg.id)
    s.add ": "
    let sharding =
      if i < fn.inputShardings.len: fn.inputShardings[i] else: ""
    s.add mlirShardedValueType(arg.valueTypeOf, sharding)
  s.add ')'
  let outputs = outputValueTypesOf(fn)
  if outputs.len > 0:
    s.add " -> "
    if outputs.len > 1: s.add '('
    for i, ty in outputs:
      if i > 0: s.add ", "
      let sharding =
        if i < fn.outputShardings.len: fn.outputShardings[i] else: ""
      s.add mlirShardedValueType(ty, sharding)
    if outputs.len > 1: s.add ')'
  s.add " {\n"
  for op in fn.ops:
    emitOp(s, op, fn, "    ")
  s.add "  }\n"

proc emitText*(m: ShModule): string =
  ## Render `m` as textual MLIR. Caller is expected to have run
  ## `verify(m)` first; this proc does no IR validation of its own.
  result = "module @"
  if m.name.len == 0:
    result.add "module"
  else:
    result.add m.name
  var attrs: seq[string] = @[]
  if m.numReplicas > 0:
    attrs.add "mhlo.num_replicas = " & $m.numReplicas & " : i32"
  if m.numPartitions > 0:
    attrs.add "mhlo.num_partitions = " & $m.numPartitions & " : i32"
  if attrs.len > 0:
    result.add " attributes {"
    for i, attr in attrs:
      if i > 0: result.add ", "
      result.add attr
    result.add "}"
  result.add " {\n"
  for meshOp in m.shardyMeshOps:
    result.add "  "
    result.add meshOp
    result.add "\n"
  for fn in m.funcs:
    emitFunction(result, fn)
  result.add "}\n"
