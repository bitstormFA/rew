## StableHLO IR — plain value `object`s.
##
## Phase 2a scope: data structures only. The graph is built by appending
## nodes into `seq`s, never by linking refs. SSA values are addressed by
## opaque `ShValueId`, which is dense per-`ShFunction`.
##
## This module is pure Nim and does not import from `src/rew/pjrt/`. It
## also does not import the bytecode emitter — the dependency goes the
## other way.

import ../dtype

type
  ShTensorType* = object
    ## Ranked tensor type. v1 carries only static shapes (no dynamic
    ## dims). The dtype/shape pair is what verification compares.
    dtype*: DType
    shape*: seq[int]

  ShTypeError* = object of CatchableError
    ## Raised when a caller asks for a concrete StableHLO type kind that a
    ## general value descriptor does not carry.

  ShTypeKind* = enum
    ## StableHLO value categories. The executable path still consumes
    ## `ShTensorType`; this descriptor lets the IR/modeling layer represent
    ## token, tuple, future and resource values while op lowering migrates.
    stkTensor
    stkToken
    stkTuple
    stkFuture
    stkResource

  ShValueType* = object
    ## General StableHLO value type descriptor.
    case kind*: ShTypeKind
    of stkTensor:
      tensor*: ShTensorType
    of stkToken:
      discard
    of stkTuple:
      elements*: seq[ShValueType]
    of stkFuture:
      futureResults*: seq[ShValueType]
    of stkResource:
      resourceName*: string

  ShValueId* = distinct int
    ## SSA value identifier, dense per `ShFunction`. The 0 id is reserved
    ## as "invalid" so default-initialized fields are visibly wrong.

  ShValue* = object
    ## A typed SSA value. Lives inside a `ShFunction` block. The `id`
    ## indexes into the function's value table for type lookup.
    id*: ShValueId
    ty*: ShTensorType
      ## Tensor-only view retained for the current emitter/verifier path.
    valueType*: ShValueType
      ## General value descriptor used while full StableHLO coverage
      ## migrates beyond tensor-only ops.

  ShOpKind* = enum
    ## Closed enum of ops that the builder can emit. Order is stable so
    ## bytecode constants in `mlirbc.nim` can index by ord. New op kinds
    ## are appended at the end.
    okConstant
    okAdd
    okSub
    okMul
    okNeg
    okReturn
    # --- Phase 4 -----------------------------------------------------------
    okDiv
    okMax
    okMin
    okExp
    okLog
    okSqrt
    okAbs
    okTanh
    okReshape
    okTranspose
    # --- Phase 5b ----------------------------------------------------------
    okReduce
    okStablehloReturn
    # --- Phase 5c ----------------------------------------------------------
    okDotGeneral
    okBroadcastInDim
    # --- Phase 7b ----------------------------------------------------------
    okIf
    # --- Phase 7b.2 --------------------------------------------------------
    okCompare
    okWhile
    # --- Phase 8 -----------------------------------------------------------
    okSelect
    # --- Phase 10 (nn parity) ----------------------------------------------
    okConcatenate
    okSlice
    okSine
    okCosine
    okRsqrt
    okClamp
    # --- Phase 11 (CNN ops) ------------------------------------------------
    okConvolution
    okReduceWindow
    # --- OpenXLA coverage --------------------------------------------------
    okCbrt
    okCeil
    okExponentialMinusOne
    okFloor
    okLogPlusOne
    okLogistic
    okTan
    okAtan2
    okPower
    okRemainder
    okSign
    okRoundNearestAfz
    okRoundNearestEven
    okAnd
    okOr
    okXor
    okNot
    okShiftLeft
    okShiftRightArithmetic
    okShiftRightLogical
    okCountLeadingZeros
    okPopcnt
    okReverse
    okOptimizationBarrier
    okConvert
    okBitcastConvert
    okIsFinite
    okReducePrecision
    okBatchNormInference
    okDot
    okBatchNormTraining
    okBatchNormGrad
    okCholesky
    okGetDimensionSize
    okPad
    okBroadcast
    okDynamicSlice
    okDynamicUpdateSlice
    okIota
    okReplicaId
    okPartitionId
    okSetDimensionSize
    okDynamicReshape
    okDynamicPad
    okDynamicIota
    okRealDynamicSlice
    okCreateToken
    okAfterAll
    okTuple
    okGetTupleElement
    okComplex
    okReal
    okImag
    okDynamicBroadcastInDim
    okFft
    okTriangularSolve
    okEinsum
    okUnaryEinsum
    okTorchIndexSelect
    okRng
    okRngBitGenerator
    okGather
    okDynamicGather
    okSort
    okMap
    okCase
    okScatter
    okSelectAndScatter
    okAllGather
    okAllReduce
    okAllToAll
    okReduceScatter
    okCollectiveBroadcast
    okCollectivePermute
    okCrossReplicaSum
    okAsyncStart
    okAsyncDone
    okInfeed
    okOutfeed
    okSend
    okRecv
    okCustomCall
    okComposite
    okDynamicConv
    okUniformQuantize
    okUniformDequantize

  ShAttrKind* = enum
    ## Attribute payload variant. Kept narrow on purpose; new kinds are
    ## added when an op needs them.
    akI64
    akI64Array
    akF64
    akBool
    akString
    akRawMlir
    akDenseElements
    akDotDims
    akI64Matrix
    akConvDims

  ShAttr* = object
    ## A small attribute payload attached to an op (e.g. constant data,
    ## dimension lists). Plain value object so attrs `seq` is cheap.
    case kind*: ShAttrKind
    of akI64:
      i64*: int64
    of akI64Array:
      i64s*: seq[int64]
    of akF64:
      f64*: float64
    of akBool:
      b*: bool
    of akString:
      str*: string
    of akRawMlir:
      mlir*: string
    of akDenseElements:
      denseDtype*: DType
      denseShape*: seq[int]
      denseBytes*: seq[byte]
    of akDotDims:
      ## `dot_general`'s dimension-numbers attribute, split into the four
      ## index lists the StableHLO dialect declares. Empty `seq`s are
      ## valid (e.g. matmul has no batching dims).
      lhsBatchingDims*: seq[int64]
      rhsBatchingDims*: seq[int64]
      lhsContractingDims*: seq[int64]
      rhsContractingDims*: seq[int64]
    of akI64Matrix:
      ## A 2-D `si64` constant attribute, used by `convolution` and
      ## `reduce_window` for `padding = dense<[[lo, hi], ...]>`. Stored
      ## as a flat row-major buffer with `matRows * matCols == matVals.len`.
      matRows*: int
      matCols*: int
      matVals*: seq[int64]
    of akConvDims:
      ## `stablehlo.convolution`'s dimension-numbers attribute. Encodes
      ## the spatial layout of `lhs` (input), `rhs` (kernel) and the
      ## result. Spatial dim indices are stored in operand order
      ## (e.g. `[1, 2]` for NHWC with dims H, W in positions 1 and 2).
      inputBatchDim*: int64
      inputFeatureDim*: int64
      inputSpatialDims*: seq[int64]
      kernelInputFeatureDim*: int64
      kernelOutputFeatureDim*: int64
      kernelSpatialDims*: seq[int64]
      outputBatchDim*: int64
      outputFeatureDim*: int64
      outputSpatialDims*: seq[int64]

  ShAttrEntry* = object
    ## Named attribute on an op. Kept as a flat `seq[ShAttrEntry]` rather
    ## than a `Table` so the IR stays a value type and order is stable.
    name*: string
    value*: ShAttr

  ShOp* = object
    ## A single SSA op inside a function body.
    kind*: ShOpKind
    operands*: seq[ShValueId]
    results*: seq[ShValue]
    attrs*: seq[ShAttrEntry]
    regions*: seq[ShRegion]
      ## Nested regions (e.g. the reducer body of `stablehlo.reduce`).
      ## Empty for region-less ops.

  ShRegion* = object
    ## A single-block region. v1 carries exactly one entry block per
    ## region — sufficient for `reduce`, `scan`, `while`, `if/else`.
    ## Multi-block regions arrive only if/when control flow needs them.
    args*: seq[ShValue]
      ## Entry-block parameters. Their `ShValueId`s live in the enclosing
      ## function's flat `types` table.
    ops*: seq[ShOp]
      ## Body ops; the last must be `okStablehloReturn`.

  ShVisibility* = enum
    svPublic
    svPrivate

  ShFunction* = object
    ## A StableHLO `func.func`. The body is a single block whose entry
    ## arguments mirror `inputTypes`; nested regions arrive in a later
    ## phase together with `cond`/`scan`/`fori`.
    name*: string
    visibility*: ShVisibility
    inputTypes*: seq[ShTensorType]
      ## Tensor-only input types used by the current textual emitter.
    inputValueTypes*: seq[ShValueType]
      ## General input value descriptors mirroring `inputTypes`.
    inputShardings*: seq[string]
      ## Optional textual Shardy sharding attributes for each input.
      ## Entries are the payload inside `#sdy.sharding<...>`.
    outputTypes*: seq[ShTensorType]
      ## Tensor-only result types used by the current textual emitter.
    outputValueTypes*: seq[ShValueType]
      ## General result value descriptors mirroring `outputTypes`.
    outputShardings*: seq[string]
      ## Optional textual Shardy sharding attributes for each result.
      ## Entries are the payload inside `#sdy.sharding<...>`.
    args*: seq[ShValue]
    ops*: seq[ShOp]
    types*: seq[ShTensorType]
      ## Indexed by `ShValueId.int`. `types[0]` is a sentinel.
    valueTypes*: seq[ShValueType]
      ## General SSA type table indexed by `ShValueId.int`. `valueTypes[0]`
      ## mirrors the sentinel in `types[0]`.

  ShModule* = object
    ## The top-level `builtin.module`.
    name*: string
    numReplicas*: int
      ## Optional XLA execution metadata emitted as module attributes.
      ## Zero means "do not emit"; compile options remain authoritative.
    numPartitions*: int
      ## Optional XLA execution metadata emitted as module attributes.
      ## Zero means "do not emit"; compile options remain authoritative.
    shardyMeshOps*: seq[string]
      ## Module-level `sdy.mesh` definitions rendered before functions.
    funcs*: seq[ShFunction]

const
  InvalidShValueId* = ShValueId(0)
    ## Sentinel returned by error paths and used for default fields.

func `==`*(a, b: ShValueId): bool {.borrow.}
func `$`*(id: ShValueId): string {.borrow.}
func isValid*(id: ShValueId): bool =
  ## True iff `id` refers to a real SSA value (not the sentinel).
  id.int != 0

func `==`*(a, b: ShTensorType): bool =
  ## Structural equality on dtype + shape. Used by the verifier.
  a.dtype == b.dtype and a.shape == b.shape

func `==`*(a, b: ShValueType): bool =
  ## Structural equality on general StableHLO value type descriptors.
  if a.kind != b.kind:
    return false
  case a.kind
  of stkTensor:
    result = a.tensor == b.tensor
  of stkToken:
    result = true
  of stkTuple:
    if a.elements.len != b.elements.len:
      return false
    for i in 0 ..< a.elements.len:
      if a.elements[i] != b.elements[i]:
        return false
    result = true
  of stkFuture:
    if a.futureResults.len != b.futureResults.len:
      return false
    for i in 0 ..< a.futureResults.len:
      if a.futureResults[i] != b.futureResults[i]:
        return false
    result = true
  of stkResource:
    result = a.resourceName == b.resourceName

func initTensorType*(dtype: DType, shape: openArray[int]): ShTensorType =
  ## Convenience constructor that copies `shape`.
  ShTensorType(dtype: dtype, shape: @shape)

func initValueType*(tensor: ShTensorType): ShValueType =
  ## Wraps a ranked tensor type as a general StableHLO value type.
  ShValueType(kind: stkTensor, tensor: tensor)

func tensorValueTypes*(types: openArray[ShTensorType]): seq[ShValueType] =
  ## Converts tensor-only descriptors into general StableHLO value types.
  result = newSeq[ShValueType](types.len)
  for i, ty in types:
    result[i] = initValueType(ty)

func initShValue*(id: ShValueId; ty: ShTensorType): ShValue =
  ## Creates an SSA value with both tensor-only and general descriptors.
  ShValue(id: id, ty: ty, valueType: initValueType(ty))

func tensorTypeOrDefault*(t: ShValueType): ShTensorType =
  ## Returns the tensor payload, or the default sentinel for non-tensor
  ## values. This keeps the legacy tensor table aligned with the general
  ## value table while token/tuple/resource lowering migrates in.
  case t.kind
  of stkTensor:
    t.tensor
  of stkToken, stkTuple, stkFuture, stkResource:
    ShTensorType()

func initShValue*(id: ShValueId; ty: ShValueType): ShValue =
  ## Creates an SSA value from a general StableHLO value descriptor.
  ShValue(id: id, ty: ty.tensorTypeOrDefault, valueType: ty)

func initTokenType*(): ShValueType =
  ## Creates a StableHLO token type descriptor.
  ShValueType(kind: stkToken)

func initTupleType*(elements: openArray[ShValueType]): ShValueType =
  ## Creates a StableHLO tuple type descriptor.
  ShValueType(kind: stkTuple, elements: @elements)

func initFutureType*(results: openArray[ShValueType]): ShValueType =
  ## Creates a StableHLO future type descriptor.
  ShValueType(kind: stkFuture, futureResults: @results)

func initResourceType*(name: string): ShValueType =
  ## Creates a named resource type descriptor.
  ShValueType(kind: stkResource, resourceName: name)

func isTensor*(t: ShValueType): bool =
  ## True when `t` is a ranked tensor descriptor.
  t.kind == stkTensor

func isToken*(t: ShValueType): bool =
  ## True when `t` is a token descriptor.
  t.kind == stkToken

func isTuple*(t: ShValueType): bool =
  ## True when `t` is a tuple descriptor.
  t.kind == stkTuple

func isFuture*(t: ShValueType): bool =
  ## True when `t` is an asynchronous future descriptor.
  t.kind == stkFuture

func isResource*(t: ShValueType): bool =
  ## True when `t` is a resource descriptor.
  t.kind == stkResource

func valueTypeOf*(v: ShValue): ShValueType =
  ## Returns the general descriptor for an SSA value.
  ##
  ## Hand-built legacy tests may fill only `ty`; in that case synthesize the
  ## descriptor from the tensor view.
  let defaultValueType = initValueType(ShTensorType())
  if v.valueType == defaultValueType and v.ty != ShTensorType():
    initValueType(v.ty)
  else:
    v.valueType

func inputValueTypesOf*(fn: ShFunction): seq[ShValueType] =
  ## Returns general input descriptors, falling back to legacy tensor fields.
  if fn.inputValueTypes.len == fn.inputTypes.len:
    fn.inputValueTypes
  else:
    tensorValueTypes(fn.inputTypes)

func outputValueTypesOf*(fn: ShFunction): seq[ShValueType] =
  ## Returns general output descriptors, falling back to legacy tensor fields.
  if fn.outputValueTypes.len == fn.outputTypes.len:
    fn.outputValueTypes
  else:
    tensorValueTypes(fn.outputTypes)

func numElements*(t: ShTensorType): int =
  ## Product of `shape`. Returns 1 for scalar (rank-0) tensors.
  result = 1
  for d in t.shape: result *= d

func `$`*(t: ShTensorType): string =
  ## Pretty-prints as `f32[2,3]` for diagnostics.
  result = t.dtype.name & "["
  for i, d in t.shape:
    if i > 0: result.add ','
    result.add $d
  result.add ']'

func `$`*(t: ShValueType): string =
  ## Pretty-prints a general StableHLO value type for diagnostics.
  case t.kind
  of stkTensor:
    result = $t.tensor
  of stkToken:
    result = "token"
  of stkTuple:
    result = "tuple<"
    for i, elem in t.elements:
      if i > 0: result.add ','
      result.add $elem
    result.add '>'
  of stkFuture:
    result = "future<"
    for i, elem in t.futureResults:
      if i > 0: result.add ','
      result.add $elem
    result.add '>'
  of stkResource:
    result = "resource<" & t.resourceName & ">"

proc requireTensorType*(t: ShValueType; context: string): ShTensorType =
  ## Returns the tensor payload or raises a typed error with `context`.
  if t.kind != stkTensor:
    raise newException(ShTypeError,
      context & ": expected tensor type, got " & $t)
  t.tensor
