## Functional `vjp` and `grad` transforms.
##
## Both run inside an active trace (the user opens a `withTrace` block;
## the transform opens a `gradMode` block on top of it). Forward and
## backward end up in the same StableHLO module, ready to be lowered to
## a single fused program by `jit` in Phase 7.
##
## ## API
## - `vjp(fn, primals)`: runs `fn(primals)`, returns `(output, vjpFn)`
##   where `vjpFn(cotangent)` builds the input cotangents and returns
##   them as a `seq[Tensor]`.
## - `grad(fn, primals)`: convenience for scalar-output `fn`. Returns
##   the gradient of `fn` w.r.t. each entry of `primals`. The implicit
##   seed cotangent is `1.0` of the output shape (a scalar).
## - `valueAndGrad(fn, primals)`: same scalar-output precondition as
##   `grad`, but also returns the forward value.
##
## Both transforms pause the tape while replaying vjp rules so the
## gradient subgraph does not feed back into the forward tape.

import std/tables
import ../tensor
import ../ops/literal
import ../ops/arith
import ./tape
import ./registry

type
  GradError* = object of CatchableError
    ## Raised when a transform precondition fails: scalar output
    ## expected, primal not on the tape, etc.

template gradMode*(body: untyped): untyped =
  ## Opens a fresh tape and runs `body` with recording active. Used by
  ## the transform layer; user code typically doesn't call this
  ## directly \u2014 prefer `vjp(fn, primals)` or `grad(fn, primals)`.
  let prevTape = currentTape()
  let tape {.inject.} = beginTape()
  try:
    body
  finally:
    endTape(prevTape)

# ---- cotangent accumulation ----------------------------------------------

proc accumulate(cots: CotangentMap; key: int; value: Tensor) =
  ## Adds `value` into the cotangent stored for SSA id `key`. If no
  ## cotangent has been seen yet, this becomes the first.
  if key in cots:
    cots[key] = add(cots[key], value)
  else:
    cots[key] = value

proc replay(tape: GradTape; outputs: openArray[Tensor];
    seeds: openArray[Tensor]): CotangentMap =
  ## Walks `tape.entries` in reverse, calling each op's vjp rule and
  ## accumulating cotangents into a map keyed by SSA id. `outputs` and
  ## `seeds` must be the same length.
  doAssert outputs.len == seeds.len,
    "vjp: outputs and seeds length mismatch"
  result = newCotangentMap()
  for i, outT in outputs:
    if not isTrace(outT):
      raise newException(GradError,
        "vjp: output #" & $i & " is not a trace tensor")
    accumulate(result, outT.traceId.int, seeds[i])
  withPausedTape:
    for i in countdown(tape.entries.high, 0):
      let entry = tape.entries[i]
      let outKey = entry.output.traceId.int
      if outKey notin result:
        # No cotangent flowed back to this op's output \u2014 nothing to do.
        continue
      let outCt = result[outKey]
      let rule = getVjpRule(entry.opName)
      let inCts = rule(entry.primals, entry.output, outCt, entry.intAttrs)
      doAssert inCts.len == entry.primals.len,
        "vjp rule for '" & entry.opName & "' returned " & $inCts.len &
          " cotangents, expected " & $entry.primals.len
      for j, ct in inCts:
        let primalKey = entry.primals[j].traceId.int
        if primalKey == 0: continue  # not a trace tensor; skip
        accumulate(result, primalKey, ct)

# ---- vjp ------------------------------------------------------------------

type
  VjpResult* = object
    ## Output of `vjp(fn, primals)`. `output` is the forward result;
    ## `pullback` runs the backward pass for a chosen cotangent.
    output*: Tensor
    pullback*: proc(cotangent: Tensor): seq[Tensor] {.closure.}

  ValueAndGradResult* = object
    ## Output of `valueAndGrad(fn, primals)`.
    ##
    ## `value` is the scalar forward result of `fn`; `grads` contains one
    ## gradient tensor per primal.
    value*: Tensor
    grads*: seq[Tensor]

proc vjp*(fn: proc(args: openArray[Tensor]): Tensor {.closure.};
    primals: openArray[Tensor]): VjpResult =
  ## Trace-mode reverse-mode transform for a single-output function.
  ## The returned `pullback` builds the gradient subgraph in the same
  ## trace and returns one cotangent per primal.
  let prevTape = currentTape()
  let tape = beginTape()
  var capturedPrimals = newSeq[Tensor](primals.len)
  for i, p in primals: capturedPrimals[i] = p
  let output = fn(primals)
  endTape(prevTape)
  let outputCaptured = output
  result.output = output
  result.pullback = proc(cotangent: Tensor): seq[Tensor] =
    if cotangent.shape != outputCaptured.shape:
      raise newException(GradError,
        "vjp.pullback: cotangent shape " & $cotangent.shape &
          " does not match output shape " & $outputCaptured.shape)
    let cots = replay(tape, [outputCaptured], [cotangent])
    var grads = newSeq[Tensor](capturedPrimals.len)
    for i, p in capturedPrimals:
      let key = p.traceId.int
      if key in cots:
        grads[i] = cots[key]
      else:
        # No path to this primal \u2014 zero gradient with primal shape.
        var zeroData = newSeq[float32](p.numElements)
        grads[i] = constantF32(p.shape, zeroData)
    grads

# ---- value and grad --------------------------------------------------------

proc valueAndGrad*(fn: proc(args: openArray[Tensor]): Tensor {.closure.};
    primals: openArray[Tensor]): ValueAndGradResult =
  ## Runs `fn` once and returns both its scalar output and gradients.
  ##
  ## This is the usual training-step convenience: it shares the same
  ## reverse-mode implementation as `vjp`/`grad` and returns one gradient
  ## tensor per primal.
  let v = vjp(fn, primals)
  if v.output.shape.len != 0:
    raise newException(GradError,
      "valueAndGrad: function output must be scalar (0-d), got shape " &
        $v.output.shape)
  ValueAndGradResult(
    value: v.output,
    grads: v.pullback(scalarF32(1'f32)),
  )

# ---- grad -----------------------------------------------------------------

proc grad*(fn: proc(args: openArray[Tensor]): Tensor {.closure.};
    primals: openArray[Tensor]): seq[Tensor] =
  ## Convenience for scalar-output `fn`: seeds the cotangent with `1.0`
  ## (matching the output shape) and returns one gradient tensor per
  ## primal.
  valueAndGrad(fn, primals).grads
