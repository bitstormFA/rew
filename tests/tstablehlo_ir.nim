## Phase 2a — StableHLO IR construction via the builder API.

import rew

block build_simple_add:
  var b = initBuilder("test_add")
  let f32x3 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32x3, f32x3], [f32x3])
  doAssert args.len == 2
  let r = b.add(args[0], args[1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  doAssert m.name == "test_add"
  doAssert m.funcs.len == 1
  doAssert m.funcs[0].name == "main"
  doAssert m.funcs[0].ops.len == 2
  doAssert m.funcs[0].ops[0].kind == okAdd
  doAssert m.funcs[0].ops[1].kind == okReturn
  verify(m)

block constant_then_neg:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  discard b.beginFunc("k", [], [f32])
  let bytes: array[4, byte] = [0x00'u8, 0x00, 0x80, 0x3f]  # 1.0f32 LE
  let c = b.constant(dtFloat32, [], bytes)
  let n = b.neg(c)
  b.returnOp([n])
  b.endFunc()
  let m = b.build()
  verify(m)
  doAssert m.funcs[0].ops[0].kind == okConstant
  doAssert m.funcs[0].ops[0].attrs[0].name == "value"

block constant_size_mismatch_raises:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [2])
  discard b.beginFunc("k", [], [f32])
  var raised = false
  try:
    discard b.constant(dtFloat32, [2], [0x00'u8, 0x00, 0x00])  # 3 bytes, need 8
  except ShBuilderError:
    raised = true
  doAssert raised

block typeOf_lookup:
  var b = initBuilder()
  let f32x4 = initTensorType(dtFloat32, [4])
  let args = b.beginFunc("k", [f32x4], [f32x4])
  doAssert b.getType(args[0]) == f32x4
  doAssert b.getValueType(args[0]) == initValueType(f32x4)
  let neg = b.neg(args[0])
  doAssert b.getType(neg) == f32x4
  doAssert b.getValueType(neg) == initValueType(f32x4)
  b.returnOp([neg])
  b.endFunc()
  discard b.build()

block general_value_type_tables_mirror_tensor_tables:
  var b = initBuilder()
  let f32x2 = initTensorType(dtFloat32, [2])
  let f32s = initTensorType(dtFloat32, [])
  let args = b.beginFunc("k", [f32x2], [f32s])
  let zero = b.constant(dtFloat32, [], @[0'u8, 0, 0, 0])
  let r = b.reduce(args[0], zero, [0],
    proc(b: var ShBuilder; lhs, rhs: ShValueId): ShValueId =
      b.add(lhs, rhs))
  b.setCurrentOutputTypes([f32s])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  let fn = m.funcs[0]
  doAssert fn.inputValueTypesOf == @[initValueType(f32x2)]
  doAssert fn.outputValueTypesOf == @[initValueType(f32s)]
  doAssert fn.types.len == fn.valueTypes.len
  for i in 1 ..< fn.types.len:
    doAssert fn.valueTypes[i] == initValueType(fn.types[i])
  doAssert fn.args[0].valueTypeOf == initValueType(f32x2)
  doAssert fn.ops[0].results[0].valueTypeOf == initValueType(f32s)

block legacy_shvalue_value_type_falls_back_to_tensor_view:
  let f32 = initTensorType(dtFloat32, [3])
  let v = ShValue(id: ShValueId(7), ty: f32)
  doAssert v.valueTypeOf == initValueType(f32)

block builder_misuse_no_open_func:
  var b = initBuilder()
  var raised = false
  try:
    b.returnOp([])
  except ShBuilderError:
    raised = true
  doAssert raised

block builder_misuse_double_open:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  discard b.beginFunc("a", [], [f32])
  var raised = false
  try:
    discard b.beginFunc("b", [], [f32])
  except ShBuilderError:
    raised = true
  doAssert raised

block builder_misuse_build_open_func:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  discard b.beginFunc("a", [], [f32])
  var raised = false
  try:
    discard b.build()
  except ShBuilderError:
    raised = true
  doAssert raised

block tensortype_helpers:
  let t = initTensorType(dtFloat32, [2, 3])
  doAssert t.numElements == 6
  doAssert $t == "float32[2,3]"
  let scalar = initTensorType(dtInt32, [])
  doAssert scalar.numElements == 1
  doAssert $scalar == "int32[]"
