## Phase 2a — StableHLO verifier. Builds intentionally broken modules
## by hand (bypassing the builder's checks) and asserts that `verify`
## catches them with messages that mention user-facing op names.

import rew
import rew/xla
import std/strutils

template assertRaises(body: untyped; mustContain: string) =
  var raised = false
  var msg = ""
  try:
    body
  except StableHloError as e:
    raised = true
    msg = e.msg
  doAssert raised, "expected StableHloError, none raised"
  doAssert mustContain in msg,
    "expected error message to contain '" & mustContain & "', got: " & msg

block empty_module_fails:
  let m = ShModule(name: "empty", funcs: @[])
  assertRaises(verify(m), "no functions")

block empty_function_body_fails:
  var fn = ShFunction(
    name: "f",
    inputTypes: @[],
    outputTypes: @[],
    args: @[],
    ops: @[],
    types: @[ShTensorType()],
  )
  let m = ShModule(name: "m", funcs: @[fn])
  assertRaises(verify(m), "empty body")

block general_type_table_mismatch_fails:
  let f32 = initTensorType(dtFloat32, [])
  let i32 = initTensorType(dtInt32, [])
  var fn = ShFunction(
    name: "k",
    inputTypes: @[f32],
    inputValueTypes: @[initValueType(i32)],
    outputTypes: @[f32],
    outputValueTypes: @[initValueType(f32)],
    args: @[initShValue(ShValueId(1), f32)],
    ops: @[
      ShOp(kind: okReturn, operands: @[ShValueId(1)],
        results: @[], attrs: @[]),
    ],
    types: @[ShTensorType(), f32],
    valueTypes: @[initValueType(ShTensorType()), initValueType(f32)],
  )
  let m = ShModule(name: "m", funcs: @[fn])
  assertRaises(verify(m), "input #0 general type")

block missing_return_fails:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  let args = b.beginFunc("k", [f32], [f32])
  discard b.neg(args[0])
  b.endFunc()
  # Note: the builder's `endFunc` does not require a return, but verify does.
  let m = b.build()
  assertRaises(verify(m), "func.return")

block return_type_mismatch:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  let i32 = initTensorType(dtInt32, [])
  let args = b.beginFunc("k", [f32], [i32])
  b.returnOp([args[0]])
  b.endFunc()
  let m = b.build()
  assertRaises(verify(m), "func.return")

block duplicate_function_name:
  var b = initBuilder()
  let f32 = initTensorType(dtFloat32, [])
  discard b.beginFunc("dup", [f32], [f32])
  b.returnOp([ShValueId(1)])
  b.endFunc()
  discard b.beginFunc("dup", [f32], [f32])
  b.returnOp([ShValueId(1)])
  b.endFunc()
  let m = b.build()
  assertRaises(verify(m), "duplicate function 'dup'")

block constant_dense_byte_length_mismatch:
  ## Hand-built broken constant: result type says f32[2] (8 bytes) but
  ## attribute carries 4 bytes.
  var fn = ShFunction(
    name: "k",
    inputTypes: @[],
    outputTypes: @[initTensorType(dtFloat32, [2])],
    args: @[],
    ops: @[
      ShOp(
        kind: okConstant,
        operands: @[],
        results: @[ShValue(id: ShValueId(1),
          ty: initTensorType(dtFloat32, [2]))],
        attrs: @[ShAttrEntry(name: "value", value: ShAttr(
          kind: akDenseElements,
          denseDtype: dtFloat32,
          denseShape: @[2],
          denseBytes: @[0'u8, 0, 0, 0],  # only 4 of 8
        ))],
      ),
      ShOp(kind: okReturn, operands: @[ShValueId(1)],
        results: @[], attrs: @[]),
    ],
    types: @[ShTensorType(), initTensorType(dtFloat32, [2])],
  )
  let m = ShModule(name: "m", funcs: @[fn])
  assertRaises(verify(m), "stablehlo.constant")

block binary_op_type_mismatch_via_typeOf:
  ## Builder catches this at op-construction time, but the verifier
  ## must also catch it on hand-built IR.
  var fn = ShFunction(
    name: "k",
    inputTypes: @[initTensorType(dtFloat32, [3]),
                  initTensorType(dtFloat32, [4])],
    outputTypes: @[initTensorType(dtFloat32, [3])],
    args: @[
      ShValue(id: ShValueId(1), ty: initTensorType(dtFloat32, [3])),
      ShValue(id: ShValueId(2), ty: initTensorType(dtFloat32, [4])),
    ],
    ops: @[
      ShOp(kind: okAdd,
        operands: @[ShValueId(1), ShValueId(2)],
        results: @[ShValue(id: ShValueId(3),
          ty: initTensorType(dtFloat32, [3]))],
        attrs: @[]),
      ShOp(kind: okReturn, operands: @[ShValueId(3)],
        results: @[], attrs: @[]),
    ],
    types: @[
      ShTensorType(),
      initTensorType(dtFloat32, [3]),
      initTensorType(dtFloat32, [4]),
      initTensorType(dtFloat32, [3]),
    ],
  )
  let m = ShModule(name: "m", funcs: @[fn])
  assertRaises(verify(m), "stablehlo.add")

block bytecode_emit_stub_raises_typed_error:
  ## Bytecode emission is deferred to Phase 9. The textual emitter is the
  ## v1 surface; it has its own dedicated test file (`tstablehlo_text.nim`).
  discard
