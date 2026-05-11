## General OpenXLA value model.
##
## `Tensor` remains the ergonomic high-level array type. `Value` is the
## lower-level representation needed for full StableHLO coverage: tokens,
## tuples, resources, dynamic dimensions, complex/quantized/index/FP8 tensor
## element descriptions, and sharding annotations.

import ./dtype
import ./device
import ./sharding
import ./stablehlo/ir

type
  ValueTypeError* = object of CatchableError
    ## Raised when a general `ValueType` cannot be lowered to the current
    ## StableHLO IR type surface.

  DimKind* = enum
    dkStatic
    dkDynamic

  Dim* = object
    ## Static or dynamic tensor dimension.
    case kind*: DimKind
    of dkStatic:
      size*: int
    of dkDynamic:
      name*: string

  ElementTypeKind* = enum
    etkDType
    etkComplex
    etkQuantized
    etkIndex
    etkFloat8

  Float8Format* = enum
    f8E4M3Fn
    f8E5M2

  ElementType* = object
    ## Tensor element type descriptor beyond the current `DType` enum.
    case kind*: ElementTypeKind
    of etkDType:
      dtype*: DType
    of etkComplex:
      complexPart*: DType
    of etkQuantized:
      storageType*: DType
      expressedType*: DType
      scale*: float64
      zeroPoint*: int64
    of etkIndex:
      discard
    of etkFloat8:
      float8Format*: Float8Format

  ValueKind* = enum
    vkTensor
    vkToken
    vkTuple
    vkFuture
    vkResource

  ValueType* = object
    ## Type-level shape of an OpenXLA value.
    case kind*: ValueKind
    of vkTensor:
      element*: ElementType
      dims*: seq[Dim]
    of vkToken:
      discard
    of vkTuple:
      elements*: seq[ValueType]
    of vkFuture:
      futureResults*: seq[ValueType]
    of vkResource:
      resourceName*: string

  Value* = object
    ## Runtime or trace value. `traceId` is valid for SSA-backed values.
    valueType*: ValueType
    device*: Device
    sharding*: Sharding
    traceId*: ShValueId

func initStaticDim*(size: int): Dim =
  ## Creates a static tensor dimension.
  Dim(kind: dkStatic, size: size)

func initDynamicDim*(name: string): Dim =
  ## Creates a named dynamic tensor dimension.
  Dim(kind: dkDynamic, name: name)

func initElementType*(dtype: DType): ElementType =
  ## Creates a normal dtype-backed element type.
  ElementType(kind: etkDType, dtype: dtype)

func initComplexElementType*(part: DType): ElementType =
  ## Creates a complex element type with float32 or float64 parts.
  ElementType(kind: etkComplex, complexPart: part)

func initQuantizedElementType*(storageType, expressedType: DType;
    scale: float64; zeroPoint: int64 = 0): ElementType =
  ## Creates a uniform quantized element type descriptor.
  ElementType(
    kind: etkQuantized,
    storageType: storageType,
    expressedType: expressedType,
    scale: scale,
    zeroPoint: zeroPoint)

func initIndexElementType*(): ElementType =
  ## Creates an MLIR index element type descriptor.
  ElementType(kind: etkIndex)

func initFloat8ElementType*(format: Float8Format): ElementType =
  ## Creates a float8 element type descriptor.
  ElementType(kind: etkFloat8, float8Format: format)

func staticDims*(shape: openArray[int]): seq[Dim] =
  ## Converts a static integer shape into general dimensions.
  result = newSeq[Dim](shape.len)
  for i, d in shape:
    result[i] = initStaticDim(d)

proc staticShape*(dims: openArray[Dim];
    context: string = "staticShape"): seq[int] =
  ## Converts general dimensions to a static shape.
  ##
  ## The current executable StableHLO path stores only static integer
  ## shapes. Dynamic dimensions remain representable in `ValueType`, but
  ## callers must keep them at the modeling/tooling layer until lowering
  ## gains dynamic shape support.
  for dim in dims:
    case dim.kind
    of dkStatic:
      result.add dim.size
    of dkDynamic:
      raise newException(ValueTypeError,
        context & ": dynamic dimension '" & dim.name &
        "' cannot be lowered to ShTensorType")

func initTensorValueType*(element: ElementType;
    dims: openArray[Dim]): ValueType =
  ## Creates a tensor value type.
  ValueType(kind: vkTensor, element: element, dims: @dims)

func initTensorValueType*(dtype: DType; shape: openArray[int]): ValueType =
  ## Creates a static tensor value type from today's `DType` shape.
  initTensorValueType(initElementType(dtype), staticDims(shape))

func initTokenValueType*(): ValueType =
  ## Creates a StableHLO token value type.
  ValueType(kind: vkToken)

func initTupleValueType*(elements: openArray[ValueType]): ValueType =
  ## Creates a tuple value type.
  ValueType(kind: vkTuple, elements: @elements)

func initFutureValueType*(results: openArray[ValueType]): ValueType =
  ## Creates an asynchronous future value type.
  ValueType(kind: vkFuture, futureResults: @results)

func initResourceValueType*(name: string): ValueType =
  ## Creates a resource value type.
  ValueType(kind: vkResource, resourceName: name)

proc toShValueType*(ty: ValueType): ShValueType =
  ## Converts a public OpenXLA value type into the StableHLO IR descriptor.
  ##
  ## Tensor lowering accepts today's `DType` + static-shape subset. Tokens,
  ## tuples and resources lower directly because they do not depend on the
  ## tensor emitter yet.
  case ty.kind
  of vkTensor:
    if ty.element.kind != etkDType:
      raise newException(ValueTypeError,
        "toShValueType: extended tensor element type cannot be lowered " &
        "to ShTensorType")
    initValueType(initTensorType(ty.element.dtype,
      staticShape(ty.dims, "toShValueType")))
  of vkToken:
    initTokenType()
  of vkTuple:
    var converted = newSeq[ShValueType](ty.elements.len)
    for i, elem in ty.elements:
      converted[i] = toShValueType(elem)
    initTupleType(converted)
  of vkFuture:
    var converted = newSeq[ShValueType](ty.futureResults.len)
    for i, elem in ty.futureResults:
      converted[i] = toShValueType(elem)
    initFutureType(converted)
  of vkResource:
    initResourceType(ty.resourceName)

proc toValueType*(ty: ShValueType): ValueType =
  ## Converts a StableHLO IR descriptor back into the public value model.
  case ty.kind
  of stkTensor:
    initTensorValueType(ty.tensor.dtype, ty.tensor.shape)
  of stkToken:
    initTokenValueType()
  of stkTuple:
    var converted = newSeq[ValueType](ty.elements.len)
    for i, elem in ty.elements:
      converted[i] = toValueType(elem)
    initTupleValueType(converted)
  of stkFuture:
    var converted = newSeq[ValueType](ty.futureResults.len)
    for i, elem in ty.futureResults:
      converted[i] = toValueType(elem)
    initFutureValueType(converted)
  of stkResource:
    initResourceValueType(ty.resourceName)

func initValue*(valueType: ValueType; device: Device;
    sharding: Sharding = initReplicated();
    traceId: ShValueId = InvalidShValueId): Value =
  ## Creates a general OpenXLA value.
  Value(
    valueType: valueType,
    device: device,
    sharding: sharding,
    traceId: traceId)

func isTensor*(v: Value): bool =
  ## True when `v` is a tensor value.
  v.valueType.kind == vkTensor

func isToken*(v: Value): bool =
  ## True when `v` is a token value.
  v.valueType.kind == vkToken

func isFuture*(v: Value): bool =
  ## True when `v` is an asynchronous future value.
  v.valueType.kind == vkFuture
