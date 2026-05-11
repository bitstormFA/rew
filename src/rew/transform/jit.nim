## `jit` transform — pure runtime tracing with a per-signature compile cache.
##
## ## Invariant #3
## `jit` is **runtime tracing only** (no macros). The user supplies a Nim
## proc taking `openArray[Tensor]` and returning `seq[Tensor]`; on each
## call we open a fresh `withTrace`, build placeholder inputs of the
## right dtype/shape/device, run the proc, and close out a StableHLO
## function. The resulting `ShModule` is cached against the input
## signature (dtype + shape + device per arg). Subsequent calls with a
## matching signature reuse the cached module.
##
## ## Composition with `grad`
## `jit(proc(args: openArray[Tensor]): seq[Tensor] = grad(loss, args))`
## is the canonical training-step idiom: `grad` opens a tape inside the
## active `withTrace`, so forward and backward end up in the same
## StableHLO function.
##
## ## Execution
## `call(jit, args)` traces on cache miss, compiles the StableHLO through
## the PJRT eager layer, and executes it. `lower(jit, args)` and
## `text(jit, args)` remain available for inspection-only workflows.
##
## ## Donation
## `donateArgs` records the indices of inputs whose buffers may be
## donated to the executable on dispatch. The execute path marks donated
## input buffers after a successful PJRT call and later reuse raises
## `BufferDonatedError`.

import std/os
import std/tables
import ../tensor
import ../dtype
import ../device
import ../sharding
import ../dispatch
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../stablehlo/verify
import ../stablehlo/text
import ../eager

type
  JitFn* = proc(args: openArray[Tensor]): seq[Tensor] {.closure.}
    ## Signature of a function passed to `jit`. Use a one-element seq for
    ## single-output functions; the cache and lowering treat both the same.

  JitCacheEntry* = object
    ## Cached lowering for one input signature.
    module*: ShModule
    text*: string
      ## Pretty-printed StableHLO. Cached so repeated `text(jit, args)`
      ## calls don't re-walk the IR.
    outDtypes*: seq[DType]
    outShapes*: seq[seq[int]]
      ## Output dtypes and shapes captured at trace time so that the
      ## execute path can reconstruct eager `Tensor`s without a second
      ## walk through the IR.
    outShardings*: seq[Sharding]
      ## Output sharding annotations captured at trace time.

  JitFunction* = ref object
    ## A traceable function with a per-signature compile cache.
    ##
    ## `JitFunction` is a `ref` because it owns mutable state (the cache)
    ## that needs to be shared across all callers of the same handle.
    fn*: JitFn
    funcName*: string
    donateArgs*: seq[int]
    cache*: TableRef[string, JitCacheEntry]

  JitError* = object of CatchableError
    ## Raised by `jit` when an input is not eligible (wrong device,
    ## donation index out of bounds, etc.).

# ---- construction ---------------------------------------------------------

proc jit*(fn: JitFn; funcName = "jit_fn";
    donateArgs: openArray[int] = []): JitFunction =
  ## Wraps `fn` in a `JitFunction` with an empty cache.
  ##
  ## `funcName` is the name used for the emitted StableHLO `func.func`
  ## (helpful when reading the lowered text). `donateArgs` records the
  ## input positions whose buffers may be donated on dispatch; they are
  ## validated at lowering time and consumed by the execute path.
  result = JitFunction(
    fn: fn,
    funcName: funcName,
    donateArgs: @donateArgs,
    cache: newTable[string, JitCacheEntry](),
  )

# ---- signature / cache key ------------------------------------------------

func signatureOf*(args: openArray[Tensor]): string =
  ## Builds the cache key for one argument list. Format:
  ## `f32:[2,3]@cpu:0|f32:[3]@cpu:0`. Different shapes, dtypes, or
  ## devices map to distinct cache entries (shape specialization).
  result = ""
  for i, a in args:
    if i > 0: result.add '|'
    result.add a.dtype.name
    result.add ":["
    for j, d in a.shape:
      if j > 0: result.add ','
      result.add $d
    result.add "]@"
    result.add $a.device
    result.add "#"
    result.add shardingKey(a.sharding)

# ---- lowering -------------------------------------------------------------

proc validateDonation(jit: JitFunction; argCount: int) =
  for idx in jit.donateArgs:
    if idx < 0 or idx >= argCount:
      raise newException(JitError,
        "jit: donateArgs index " & $idx & " out of range for " &
          $argCount & " input(s)")

proc traceOnce(jit: JitFunction; args: openArray[Tensor]): JitCacheEntry =
  ## Runs the user fn once under a fresh `withTrace`, returning the
  ## verified module + cached text. Does not touch the cache.
  validateDonation(jit, args.len)
  if args.len == 0:
    raise newException(JitError, "jit: at least one input required")
  let dev = args[0].device
  for i in 1 ..< args.len:
    if args[i].device != dev:
      raise newException(JitError,
        "jit: all inputs must share a device; arg 0 = " &
          $dev & ", arg " & $i & " = " & $args[i].device)
  var dtypes = newSeq[DType](args.len)
  var shapes = newSeq[seq[int]](args.len)
  var shardings = newSeq[Sharding](args.len)
  for i, a in args:
    validateSharding(a.sharding, a.shape.len)
    dtypes[i] = a.dtype
    shapes[i] = a.shape
    shardings[i] = a.sharding
  withTrace ctx, jit.funcName, dev:
    let inputs = ctx.traceInputs(dtypes, shapes, shardings)
    let outputs = jit.fn(inputs)
    if outputs.len == 0:
      raise newException(JitError,
        "jit: function returned no outputs")
    result.outDtypes = newSeq[DType](outputs.len)
    result.outShapes = newSeq[seq[int]](outputs.len)
    result.outShardings = newSeq[Sharding](outputs.len)
    for i, t in outputs:
      validateSharding(t.sharding, t.shape.len)
      result.outDtypes[i] = t.dtype
      result.outShapes[i] = t.shape
      result.outShardings[i] = t.sharding
    ctx.traceReturn(outputs)
  let m = ctx.builder.build()
  verify(m)
  result.module = m
  result.text = emitText(m)

proc compileFor*(jit: JitFunction; args: openArray[Tensor]): JitCacheEntry =
  ## Returns the cache entry for this input signature, tracing on miss.
  let key = signatureOf(args)
  if key in jit.cache:
    return jit.cache[key]
  let entry = traceOnce(jit, args)
  jit.cache[key] = entry
  entry

proc lower*(jit: JitFunction; args: openArray[Tensor]): ShModule =
  ## Returns the lowered StableHLO module for this signature. Caches on
  ## first call; subsequent calls with a matching signature are O(1).
  compileFor(jit, args).module

proc text*(jit: JitFunction; args: openArray[Tensor]): string =
  ## Returns the cached StableHLO text for this input signature.
  compileFor(jit, args).text

proc cacheSize*(jit: JitFunction): int =
  ## Number of distinct input signatures currently cached.
  jit.cache.len

proc clearCache*(jit: JitFunction) =
  ## Drops every cached lowering. The next call re-traces from scratch.
  jit.cache.clear()

# ---- execution ------------------------------------------------------------

proc executableText(entry: JitCacheEntry): string =
  ## PJRT's StableHLO importer expects the executable entry function to be
  ## named `main`; keep the user-facing lowered module name intact.
  var m = entry.module
  if m.funcs.len > 0:
    m.funcs[0].name = "main"
  emitText(m)

proc executableText*(jit: JitFunction; args: openArray[Tensor]): string =
  ## Returns the StableHLO text rew sends to PJRT for this signature.
  executableText(compileFor(jit, args))

proc dumpHlo*(jit: JitFunction; args: openArray[Tensor]; path: string;
    executable = false): string =
  ## Writes StableHLO text for this signature to `path` and returns `path`.
  ##
  ## By default the dumped module preserves the user-facing function name.
  ## Set `executable = true` to dump the PJRT-ready variant whose entry
  ## function is renamed to `main`.
  let dir = parentDir(path)
  if dir.len > 0:
    createDir(dir)
  let contents =
    if executable: jit.executableText(args)
    else: jit.text(args)
  writeFile(path, contents)
  path

proc call*(jit: JitFunction; args: openArray[Tensor]): seq[Tensor] =
  ## Executes the cached program against `args` via the PJRT eager
  ## backend.
  ##
  ## Trace-on-miss + execute. `args` must be eager tensors on the same
  ## device; trace tensors raise `TensorModeError`. Indices in
  ## `donateArgs` are marked donated on a successful execute, so
  ## subsequent observation of those input buffers raises
  ## `BufferDonatedError`.
  let entry = compileFor(jit, args)
  let key = signatureOf(args)
  executeJit(jit.funcName, key, executableText(entry), args,
             entry.outDtypes, entry.outShapes, jit.donateArgs,
             entry.outShardings)
