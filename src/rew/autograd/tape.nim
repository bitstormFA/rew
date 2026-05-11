## Autograd tape \u2014 records primitive op calls during a `gradMode`
## block so the `grad` / `vjp` transform can walk them in reverse.
##
## ## Why a tape (and not a graph)
## We already build a StableHLO graph through the dispatcher. The tape
## is a *parallel* recording that knows the op name, its primal inputs,
## its primal output, and any non-tensor int attrs (axes, shapes, dim
## lists). The forward graph stays in the StableHLO builder; the tape
## is what lets the transform layer look up vjp rules and accumulate
## cotangents back to the inputs.
##
## ## Pause flag
## Vjp rules emit StableHLO ops via the public op API \u2014 those calls
## must NOT recurse into the tape. `pauseTape` / `resumeTape` bracket
## rule application so the tape stays a strict primal log.

import std/tables
import ../tensor

type
  IntAttrs* = seq[(string, seq[int])]
    ## Non-tensor attributes the vjp rule may need (axes, shapes, dim
    ## lists). Strings are short keys; values are int sequences.

  TapeEntry* = object
    ## One recorded primitive op invocation.
    opName*: string
    primals*: seq[Tensor]
    output*: Tensor
    intAttrs*: IntAttrs

  GradTape* = ref object
    ## The thread-local tape. `entries` is append-only during the
    ## forward pass and walked in reverse by the transform.
    entries*: seq[TapeEntry]

var activeTape {.threadvar.}: GradTape
var paused {.threadvar.}: int  ## nesting count

proc currentTape*(): GradTape =
  ## Returns the active tape (or `nil` when no `gradMode` block is open
  ## or the tape is paused). Ops use this to decide whether to record.
  if paused > 0: nil else: activeTape

proc isRecording*(): bool =
  ## True iff a tape is active *and* not paused.
  not currentTape().isNil

proc beginTape*(): GradTape =
  ## Installs a fresh tape and returns it. Pairs with `endTape`. Public
  ## so the transform layer can manage the lifetime explicitly.
  result = GradTape(entries: @[])
  activeTape = result

proc endTape*(prev: GradTape) =
  ## Restores `prev` as the active tape (typically `nil`).
  activeTape = prev

proc pauseTape*() =
  ## Suppress recording while a vjp rule is running. Reentrant.
  inc paused

proc resumeTape*() =
  ## Counterpart to `pauseTape`. Reentrant.
  if paused > 0: dec paused

template withPausedTape*(body: untyped): untyped =
  ## Runs `body` with the tape paused. Vjp rules use this so the
  ## primitives they emit do not get recorded as new tape entries.
  pauseTape()
  try:
    body
  finally:
    resumeTape()

proc recordTraceOp*(name: string; primals: openArray[Tensor];
    output: Tensor; intAttrs: IntAttrs = @[]) =
  ## Append one tape entry. No-op when no tape is active or recording is
  ## paused. Called by every `{.rewOp.}` proc immediately after building
  ## its trace-mode result.
  let tape = currentTape()
  if tape.isNil: return
  var captured = newSeq[Tensor](primals.len)
  for i, p in primals: captured[i] = p
  tape.entries.add TapeEntry(
    opName: name,
    primals: captured,
    output: output,
    intAttrs: intAttrs,
  )

# ---- cotangent accumulator (used by transform.nim) ------------------------

type
  CotangentMap* = TableRef[int, Tensor]
    ## Maps trace SSA id (`Tensor.traceId.int`) to the accumulated
    ## cotangent for that intermediate. Keys with no entry have an
    ## implicit zero cotangent.

proc newCotangentMap*(): CotangentMap = newTable[int, Tensor]()
