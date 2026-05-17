## StableHLO general value coverage: tokens and tuples.

import rew
import rew/xla
import std/strutils

block create_token_after_all_tuple_and_get_tuple_element:
  var b = initBuilder("values")
  let tokenTy = initTokenType()
  let tupleTy = initTupleType([tokenTy, tokenTy])
  discard b.beginValueFunc("main", [], [tupleTy, tokenTy])
  let first = b.createToken()
  let second = b.afterAll([first])
  let pair = b.tupleOp([first, second])
  let extracted = b.getTupleElement(pair, 1)
  let joined = b.afterAll([extracted])
  b.returnOp([pair, joined])
  b.endFunc()

  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "stablehlo.create_token" in text, text
  doAssert "\"stablehlo.after_all\"" in text, text
  doAssert "\"stablehlo.tuple\"" in text, text
  doAssert "\"stablehlo.get_tuple_element\"" in text, text
  doAssert "!stablehlo.token" in text, text
  doAssert "tuple<!stablehlo.token, !stablehlo.token>" in text, text
  doAssert "index = 1 : i32" in text, text

block tuple_can_mix_tensor_and_token:
  var b = initBuilder("mixed_values")
  let f32 = initTensorType(dtFloat32, [])
  let tokenTy = initTokenType()
  let mixedTy = initTupleType([initValueType(f32), tokenTy])
  let args = b.beginValueFunc("main", [initValueType(f32)],
    [mixedTy, initValueType(f32), tokenTy])
  let token = b.createToken()
  let mixed = b.tupleOp([args[0], token])
  let tensor = b.getTupleElement(mixed, 0)
  let tokenOut = b.getTupleElement(mixed, 1)
  b.returnOp([mixed, tensor, tokenOut])
  b.endFunc()

  let mod1 = b.build()
  verify(mod1)
  let text = emitText(mod1)
  doAssert "tuple<tensor<f32>, !stablehlo.token>" in text, text
  doAssert "-> (tuple<tensor<f32>, !stablehlo.token>, tensor<f32>, !stablehlo.token)" in text, text

block non_tensor_builder_errors:
  var b = initBuilder("errors")
  let f32 = initTensorType(dtFloat32, [])
  let args = b.beginFunc("main", [f32], [])
  let token = b.createToken()
  doAssertRaises(ShBuilderError):
    discard b.afterAll([args[0]])
  doAssertRaises(ShBuilderError):
    discard b.getTupleElement(token, 0)
  let pair = b.tupleOp([token])
  doAssertRaises(ShBuilderError):
    discard b.getTupleElement(pair, 1)
  b.returnOp([])
  b.endFunc()
  verify(b.build())

echo "tstablehlo_values: OK"
