## Dispatch — eager-vs-trace switching for op invocations.
##
## ## Invariant #2 and #3
## Eager dispatch is per-op compile-and-cache; `jit` is pure runtime tracing
## via dispatcher swap. This module is the swap point. Public ops in
## `src/rew/ops/*.nim` ask `currentMode()` and route through either the
## eager backend (PJRT execute, registered separately) or the trace backend
## (emit a StableHLO op into the active builder).
##
## ## Trace context
## A `TraceContext` owns one `ShBuilder` and the (still-open) `ShFunction`
## being built. Trace tensors carry SSA ids that index into that function.
## Trace contexts nest via the `withTrace` template; each nested call
## starts a fresh context and restores the previous one on exit.
##
## ## Eager backend
## The eager backend is installed explicitly by `eager.installEagerBackend`
## once the caller has selected a PJRT-capable device. Until that happens,
## eager ops raise `NoEagerBackendError`; trace mode remains available for
## lowering-only workflows.

import ./tensor
import ./dtype
import ./device
import ./sharding
import ./stablehlo/ir
import ./stablehlo/ops as shops

type
  DispatchMode* = enum
    ## Which dispatch path an op should take.
    dmEager   ## Per-op compile + execute on the device.
    dmTrace   ## Emit into the active StableHLO builder.

  NoEagerBackendError* = object of CatchableError
    ## Raised when an eager op runs without a registered execution
    ## backend.

  TraceContext* = object
    ## State carried for the duration of a `withTrace` block.
    builder*: ShBuilder
    funcName*: string
    device*: Device
      ## Device assigned to every trace tensor produced inside this
      ## context. All input tensors must agree.

  EagerBackend* = proc (op: string; operands: openArray[Tensor];
      attrs: openArray[(string, string)]): seq[Tensor] {.nimcall,
      raises: [CatchableError].}
    ## Function the dispatcher calls to execute an eager op. `nimcall`
    ## keeps it non-capturing.

var currentTrace {.threadvar.}: ref TraceContext
var eagerBackend {.threadvar.}: EagerBackend

proc currentMode*(): DispatchMode =
  ## Returns the dispatch mode for ops invoked right now on this thread.
  if currentTrace.isNil: dmEager else: dmTrace

proc currentTraceContext*(): ref TraceContext =
  ## Returns the active trace context. Only call after checking
  ## `currentMode() == dmTrace`. Returns `nil` when in eager mode.
  currentTrace

proc setEagerBackend*(backend: EagerBackend) =
  ## Registers the per-thread eager execution backend. Tests use this to
  ## inject a fake backend.
  eagerBackend = backend

proc clearEagerBackend*() =
  ## Removes any registered eager backend. Tests use this to assert the
  ## "no backend" error path.
  eagerBackend = nil

proc dispatchEager*(op: string; operands: openArray[Tensor];
    attrs: openArray[(string, string)] = []): seq[Tensor] =
  ## Routes an eager op through the registered backend, or raises
  ## `NoEagerBackendError` when no backend is wired. Used by every
  ## `{.rewOp.}` proc in eager mode.
  if eagerBackend.isNil:
    raise newException(NoEagerBackendError,
      "eager op '" & op & "': no PJRT execution backend registered. " &
        "call installEagerBackend() after selecting a PJRT device, or " &
        "use trace mode (`withTrace`) for lowering only.")
  result = eagerBackend(op, operands, attrs)

proc beginTrace*(funcName: string; device: Device): ref TraceContext =
  ## Constructs a fresh trace context. Public so that higher-level
  ## entry points (`jit`, `traceModule`) can manage their own scope; user
  ## code should prefer the `withTrace` template.
  result = (ref TraceContext)(
    builder: initBuilder("trace_module"),
    funcName: funcName,
    device: device,
  )

proc enterTrace*(ctx: ref TraceContext) =
  ## Installs `ctx` as the current trace context. Pairs with `exitTrace`.
  currentTrace = ctx

proc exitTrace*(prev: ref TraceContext) =
  ## Restores `prev` as the current trace context.
  currentTrace = prev

template withTrace*(ctxVar, funcName, device, body: untyped): untyped =
  ## Runs `body` with a fresh trace context bound to `ctxVar`.
  ##
  ## Inside the block, every op registered with `{.rewOp.}` builds a
  ## StableHLO op instead of executing on the device. After the block,
  ## the previous context (or `nil`) is restored. The caller closes the
  ## function on `ctxVar.builder` and calls `build()` to obtain the
  ## `ShModule`.
  let ctxVar {.inject.} = beginTrace(funcName, device)
  let prevTrace = currentTrace
  enterTrace(ctxVar)
  try:
    body
  finally:
    exitTrace(prevTrace)

proc traceInputs*(ctx: ref TraceContext; dtypes: openArray[DType];
    shapes: openArray[seq[int]];
    shardings: openArray[Sharding]): seq[Tensor] =
  ## Opens the function on `ctx.builder` with one argument per
  ## `(dtype, shape)` pair and returns the matching trace tensors. Output
  ## types are filled in lazily by `traceReturn` once the body has run.
  doAssert dtypes.len == shapes.len,
    "traceInputs: dtypes and shapes length mismatch"
  doAssert dtypes.len == shardings.len,
    "traceInputs: dtypes and shardings length mismatch"
  var inTypes = newSeq[ShTensorType](dtypes.len)
  var inputAttrs = newSeq[string](dtypes.len)
  for i in 0 ..< dtypes.len:
    validateSharding(shardings[i], shapes[i].len)
    inTypes[i] = initTensorType(dtypes[i], shapes[i])
    inputAttrs[i] = shardyTensorSharding(shardings[i], shapes[i].len)
    if inputAttrs[i].len > 0:
      ctx.builder.addShardyMeshOp(shardyMeshOp(shardings[i].activeMesh()))
  let argIds = ctx.builder.beginFunc(ctx.funcName, inTypes, [])
  ctx.builder.setCurrentInputShardings(inputAttrs)
  result = newSeq[Tensor](dtypes.len)
  for i, id in argIds:
    result[i] = initTraceTensor(id, dtypes[i], shapes[i], ctx.device,
      shardings[i])

proc traceInputs*(ctx: ref TraceContext; dtypes: openArray[DType];
    shapes: openArray[seq[int]]): seq[Tensor] =
  ## Opens replicated trace inputs. Kept for callers that do not yet carry
  ## explicit sharding metadata.
  var shardings = newSeq[Sharding](dtypes.len)
  for i in 0 ..< shardings.len:
    shardings[i] = initReplicated()
  traceInputs(ctx, dtypes, shapes, shardings)

proc traceReturn*(ctx: ref TraceContext; results: openArray[Tensor]) =
  ## Emits a `func.return` for the trace function and patches the
  ## function's output types from the result tensors. Must be called
  ## exactly once per traced function, after which `endFunc` is safe.
  var ids = newSeq[ShValueId](results.len)
  var outTypes = newSeq[ShTensorType](results.len)
  var outputAttrs = newSeq[string](results.len)
  for i, t in results:
    requireTrace(t, "traceReturn")
    ids[i] = t.traceId
    outTypes[i] = tensorTypeOf(t)
    outputAttrs[i] = shardyTensorSharding(t.sharding, t.shape.len)
    if outputAttrs[i].len > 0:
      ctx.builder.addShardyMeshOp(shardyMeshOp(t.sharding.activeMesh()))
  setCurrentOutputTypes(ctx.builder, outTypes)
  setCurrentOutputShardings(ctx.builder, outputAttrs)
  ctx.builder.returnOp(ids)
  ctx.builder.endFunc()
