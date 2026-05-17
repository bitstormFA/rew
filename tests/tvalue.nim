## General OpenXLA value model.

import rew
import rew/xla

block static_tensor_value_type:
  let ty = initTensorValueType(dtFloat32, [2, 3])
  doAssert ty.kind == vkTensor
  doAssert ty.element.kind == etkDType
  doAssert ty.element.dtype == dtFloat32
  doAssert ty.dims.len == 2
  doAssert ty.dims[0].kind == dkStatic
  doAssert ty.dims[0].size == 2

block tuple_token_resource_types:
  let token = initTokenValueType()
  let resource = initResourceValueType("infeed")
  let tup = initTupleValueType([token, resource])
  doAssert tup.kind == vkTuple
  doAssert tup.elements.len == 2
  doAssert tup.elements[0].kind == vkToken
  doAssert tup.elements[1].resourceName == "infeed"

block value_with_sharding:
  let mesh = initMesh("data", ["x"], [2])
  let spec = initPartitionSpec(["x"])
  let sharding = initPartitioned(mesh, spec)
  let v = initValue(initTensorValueType(dtFloat32, [8]), cpu(), sharding)
  doAssert v.isTensor
  doAssert v.sharding.kind == skPartitioned
  doAssert not v.sharding.isReplicated

block extended_element_types:
  doAssert initComplexElementType(dtFloat32).kind == etkComplex
  doAssert initIndexElementType().kind == etkIndex
  doAssert initFloat8ElementType(f8E4M3Fn).float8Format == f8E4M3Fn
  let q = initQuantizedElementType(dtInt8, dtFloat32, 0.25)
  doAssert q.kind == etkQuantized
  doAssert q.scale == 0.25

block stablehlo_tensor_type_bridge:
  let ty = initTensorValueType(dtFloat32, [2, 3])
  let sh = ty.toShValueType()
  doAssert sh.kind == stkTensor
  doAssert sh.isTensor
  doAssert sh.tensor == initTensorType(dtFloat32, [2, 3])
  doAssert sh.requireTensorType("bridge") == initTensorType(dtFloat32, [2, 3])
  doAssert $sh == "float32[2,3]"

  let roundTrip = sh.toValueType()
  doAssert roundTrip.kind == vkTensor
  doAssert roundTrip.element.kind == etkDType
  doAssert roundTrip.element.dtype == dtFloat32
  doAssert roundTrip.dims.len == 2
  doAssert roundTrip.dims[0].size == 2
  doAssert roundTrip.dims[1].size == 3

block stablehlo_non_tensor_type_bridge:
  let token = initTokenValueType().toShValueType()
  doAssert token.kind == stkToken
  doAssert token.isToken
  doAssert $token == "token"
  doAssert token.toValueType().kind == vkToken

  let resource = initResourceValueType("infeed").toShValueType()
  doAssert resource.kind == stkResource
  doAssert resource.isResource
  doAssert resource.resourceName == "infeed"
  doAssert $resource == "resource<infeed>"
  doAssert resource.toValueType().resourceName == "infeed"

  let tupleTy = initTupleValueType([
    initTokenValueType(),
    initResourceValueType("rng"),
  ])
  let shTuple = tupleTy.toShValueType()
  doAssert shTuple.kind == stkTuple
  doAssert shTuple.isTuple
  doAssert shTuple.elements.len == 2
  doAssert $shTuple == "tuple<token,resource<rng>>"

  let roundTrip = shTuple.toValueType()
  doAssert roundTrip.kind == vkTuple
  doAssert roundTrip.elements[0].kind == vkToken
  doAssert roundTrip.elements[1].resourceName == "rng"

  let futureTy = initFutureValueType([
    initTensorValueType(dtFloat32, [2]),
    initTokenValueType(),
  ])
  let shFuture = futureTy.toShValueType()
  doAssert shFuture.kind == stkFuture
  doAssert shFuture.isFuture
  doAssert shFuture.futureResults.len == 2
  doAssert $shFuture == "future<float32[2],token>"
  let futureRoundTrip = shFuture.toValueType()
  doAssert futureRoundTrip.kind == vkFuture
  doAssert futureRoundTrip.futureResults[0].kind == vkTensor
  doAssert futureRoundTrip.futureResults[1].kind == vkToken

block stablehlo_bridge_rejects_unlowered_tensor_forms:
  let dynamicTy = initTensorValueType(initElementType(dtFloat32), [
    initStaticDim(2),
    initDynamicDim("batch"),
  ])
  doAssertRaises(ValueTypeError):
    discard dynamicTy.toShValueType()

  let complexTy = initTensorValueType(initComplexElementType(dtFloat32), [
    initStaticDim(2),
  ])
  doAssertRaises(ValueTypeError):
    discard complexTy.toShValueType()

  doAssertRaises(ShTypeError):
    discard initTokenType().requireTensorType("bridge")

echo "tvalue: OK"
