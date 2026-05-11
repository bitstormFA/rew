## Phase 7a \u2014 jit transform: trace + cache + lower.

import rew
import std/[os, strutils]

let TestDevice = cpu(0)

proc makeInput(dtype: DType; shape: openArray[int];
    dev = TestDevice; sharding: Sharding = initReplicated()): Tensor =
  ## Builds a placeholder input tensor outside any trace. We only ever
  ## hand these to `jit` to derive a signature; no buffer is materialised.
  initTraceTensor(ShValueId(1), dtype, @shape, dev, sharding)

block jit_lower_caches:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[add(args[0], args[1])]
  let j = jit(f, "addPair")
  let a = makeInput(dtFloat32, [2, 3])
  let b = makeInput(dtFloat32, [2, 3])
  doAssert j.cacheSize == 0
  let m1 = j.lower([a, b])
  doAssert j.cacheSize == 1
  let m2 = j.lower([a, b])
  doAssert j.cacheSize == 1, "second call must hit cache"
  doAssert m1.funcs[0].name == "addPair"
  doAssert m2.funcs[0].name == "addPair"
  let text = emitText(m1)
  doAssert "stablehlo.add" in text
  doAssert "@addPair" in text or "func.func" in text

block lazy_lower_caches:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    let y = add(args[0], args[1])
    @[mul(y, y)]
  let lz = lazy(f, "lazySquare")
  let a = makeInput(dtFloat32, [2, 3])
  let b = makeInput(dtFloat32, [2, 3])
  doAssert lz.cacheSize == 0
  let m1 = lz.lower([a, b])
  doAssert lz.cacheSize == 1
  let m2 = lz.lower([a, b])
  doAssert lz.cacheSize == 1
  doAssert emitText(m1) == emitText(m2)
  let text = lz.text([a, b])
  doAssert "stablehlo.add" in text
  doAssert "stablehlo.multiply" in text

block jit_text_and_dump_helpers:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[neg(args[0])]
  let j = jit(f, "dumpPair")
  let x = makeInput(dtFloat32, [2])
  doAssert "stablehlo.negate" in j.text([x])
  doAssert "@main" in j.executableText([x])
  let path = getTempDir() / "rew_jit_dump.mlir"
  discard j.dumpHlo([x], path)
  doAssert "stablehlo.negate" in readFile(path)
  removeFile(path)

block jit_signature_specialisation:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[mul(args[0], args[0])]
  let j = jit(f)
  let small = makeInput(dtFloat32, [2])
  let big = makeInput(dtFloat32, [4])
  discard j.lower([small])
  discard j.lower([big])
  doAssert j.cacheSize == 2, "different shapes must recompile"
  discard j.lower([small])
  doAssert j.cacheSize == 2, "same signature must reuse"

block jit_signature_string:
  let a = makeInput(dtFloat32, [2, 3])
  let b = makeInput(dtFloat32, [3])
  let s = signatureOf([a, b])
  doAssert "float32:[2,3]" in s
  doAssert "float32:[3]" in s
  doAssert "@cpu:0" in s
  doAssert "#replicated" in s

block jit_signature_includes_sharding:
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[add(args[0], args[0])]
  let j = jit(f, "shardedAdd")
  let mesh = initMesh("m", ["x"], [2])
  let sharded = initPartitioned(mesh, initPartitionSpec(["x"]))
  let xRep = makeInput(dtFloat32, [8])
  let xShard = makeInput(dtFloat32, [8], sharding = sharded)

  discard j.lower([xRep])
  discard j.lower([xShard])
  doAssert j.cacheSize == 2, "same shape with different sharding recompiles"
  let text = emitText(j.lower([xShard]))
  doAssert "stablehlo.add" in text

block jit_grad_compose:
  ## `jit(grad(loss))` — the canonical training step.
  let loss = proc(args: openArray[Tensor]): Tensor =
    reduceSum(mul(args[0], args[0]), [0])
  let trainStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
    grad(loss, args))
  let x = makeInput(dtFloat32, [4])
  let m = trainStep.lower([x])
  let text = emitText(m)
  # Forward x*x and the backward 2x both lower to multiplies.
  doAssert text.count("stablehlo.multiply") >= 2

block jit_call_requires_eager_inputs:
  ## Calling `jit` with placeholder (trace) tensors traces + caches the
  ## module, then fails when execution discovers the inputs aren't eager
  ## tensors. The cache is still primed.
  let f = proc(args: openArray[Tensor]): seq[Tensor] = @[args[0]]
  let j = jit(f)
  let x = makeInput(dtFloat32, [2])
  doAssertRaises(TensorModeError):
    discard j.call([x])
  doAssert j.cacheSize == 1

block jit_donate_validates:
  let f = proc(args: openArray[Tensor]): seq[Tensor] = @[args[0]]
  let j = jit(f, donateArgs = [3])
  let x = makeInput(dtFloat32, [2])
  doAssertRaises(JitError):
    discard j.lower([x])

block jit_zero_inputs_rejected:
  let f = proc(args: openArray[Tensor]): seq[Tensor] = @[]
  let j = jit(f)
  doAssertRaises(JitError):
    discard j.lower([])

block jit_clear_cache:
  let f = proc(args: openArray[Tensor]): seq[Tensor] = @[args[0]]
  let j = jit(f)
  let x = makeInput(dtFloat32, [2])
  discard j.lower([x])
  doAssert j.cacheSize == 1
  j.clearCache()
  doAssert j.cacheSize == 0
