## Phase 2c — StableHLO textual emitter.
##
## We compare against exact expected strings for tiny modules (so a
## regression in spacing/spelling is loud) and check for substring
## presence on more complex modules.

import rew
import std/strutils

template assertContains(haystack, needle: string) =
  doAssert needle in haystack,
    "expected to find " & needle.escape & " in:\n" & haystack

block emit_minimal_passthrough:
  var b = initBuilder("m")
  let f32 = initTensorType(dtFloat32, [])
  let args = b.beginFunc("main", [f32], [f32])
  b.returnOp([args[0]])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  let expected =
    "module @m {\n" &
    "  func.func @main(%v1: tensor<f32>) -> tensor<f32> {\n" &
    "    func.return %v1 : tensor<f32>\n" &
    "  }\n" &
    "}\n"
  doAssert text == expected, "got:\n" & text

block emit_add_two_vectors:
  var b = initBuilder("addmod")
  let f32x3 = initTensorType(dtFloat32, [3])
  let args = b.beginFunc("main", [f32x3, f32x3], [f32x3])
  let r = b.add(args[0], args[1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  let expected =
    "module @addmod {\n" &
    "  func.func @main(%v1: tensor<3xf32>, %v2: tensor<3xf32>) -> tensor<3xf32> {\n" &
    "    %v3 = stablehlo.add %v1, %v2 : tensor<3xf32>\n" &
    "    func.return %v3 : tensor<3xf32>\n" &
    "  }\n" &
    "}\n"
  doAssert text == expected, "got:\n" & text

block emit_constant_then_neg:
  var b = initBuilder("kmod")
  let f32 = initTensorType(dtFloat32, [])
  discard b.beginFunc("main", [], [f32])
  let bytes: array[4, byte] = [0x00'u8, 0x00, 0x80, 0x3f]  # 1.0f32 LE
  let c = b.constant(dtFloat32, [], bytes)
  let n = b.neg(c)
  b.returnOp([n])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "stablehlo.constant dense<\"0x0000803F\"> : tensor<f32>"
  assertContains text, "stablehlo.negate %v1 : tensor<f32>"
  assertContains text, "func.return %v2 : tensor<f32>"

block emit_mul_then_sub_higher_rank:
  var b = initBuilder("mathmod")
  let i32mat = initTensorType(dtInt32, [2, 3])
  let args = b.beginFunc("main", [i32mat, i32mat, i32mat], [i32mat])
  let m1 = b.mul(args[0], args[1])
  let r = b.sub(m1, args[2])
  b.returnOp([r])
  b.endFunc()
  let modu = b.build()
  verify(modu)
  let text = emitText(modu)
  assertContains text, "tensor<2x3xi32>"
  assertContains text, "stablehlo.multiply %v1, %v2 : tensor<2x3xi32>"
  assertContains text, "stablehlo.subtract %v4, %v3 : tensor<2x3xi32>"

block emit_dtype_spellings:
  ## Make sure we emit the canonical MLIR spellings for every dtype
  ## (`i1`, `bf16`, `ui16`, …). Build one trivial constant per dtype.
  let cases = @[
    (dtBool, "i1", 1),
    (dtInt8, "i8", 1),
    (dtUint8, "ui8", 1),
    (dtInt16, "i16", 2),
    (dtUint16, "ui16", 2),
    (dtFloat16, "f16", 2),
    (dtBFloat16, "bf16", 2),
    (dtInt32, "i32", 4),
    (dtUint32, "ui32", 4),
    (dtFloat32, "f32", 4),
    (dtInt64, "i64", 8),
    (dtUint64, "ui64", 8),
    (dtFloat64, "f64", 8),
    (dtComplex64, "complex<f32>", 8),
    (dtComplex128, "complex<f64>", 16),
  ]
  for (dt, mlirName, sz) in cases:
    var b = initBuilder()
    let ty = initTensorType(dt, [])
    discard b.beginFunc("main", [], [ty])
    let bytes = newSeq[byte](sz)
    let c = b.constant(dt, [], bytes)
    b.returnOp([c])
    b.endFunc()
    let m = b.build()
    verify(m)
    let text = emitText(m)
    assertContains text, "tensor<" & mlirName & ">"

block emit_multi_function_module:
  var b = initBuilder("multi")
  let f32 = initTensorType(dtFloat32, [])
  let argsA = b.beginFunc("a", [f32], [f32])
  let na = b.neg(argsA[0])
  b.returnOp([na])
  b.endFunc()
  let argsB = b.beginFunc("b", [f32, f32], [f32], visibility = svPrivate)
  let r = b.add(argsB[0], argsB[1])
  b.returnOp([r])
  b.endFunc()
  let m = b.build()
  verify(m)
  let text = emitText(m)
  assertContains text, "func.func @a("
  assertContains text, "func.func private @b("
  assertContains text, "stablehlo.negate"
  assertContains text, "stablehlo.add"

block emit_does_not_check_validity:
  ## `emitText` assumes `verify` already ran; supply hand-built bad IR
  ## and ensure we still produce *some* output without crashing. This is
  ## a guard against accidental re-verification creep in the emitter.
  let fn = ShFunction(
    name: "k",
    inputTypes: @[],
    outputTypes: @[],
    args: @[],
    ops: @[ShOp(kind: okReturn, operands: @[], results: @[], attrs: @[])],
    types: @[ShTensorType()],
  )
  let m = ShModule(name: "x", funcs: @[fn])
  let text = emitText(m)
  assertContains text, "func.func @k() {"
  assertContains text, "func.return"
