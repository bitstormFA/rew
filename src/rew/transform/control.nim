## In-graph control flow: `cond`, `whileLoop`, `fori`.
##
## ## Invariant #3 follow-through
## Python-level `if`/`while` cannot drive in-graph branching because
## traced predicates are SSA values, not bools. These combinators let a
## traced function express conditional sub-graphs and bounded iteration
## directly in StableHLO via `stablehlo.if` and `stablehlo.while`.
##
## ## API
## - `cond(pred, thenFn, elseFn) -> Tensor`: single-output if/else.
## - `condN(pred, thenFn, elseFn) -> seq[Tensor]`: multi-output variant.
## - `whileLoop(init, cond, body) -> seq[Tensor]`: low-level loop with
##   user-supplied trace-tensor predicate.
## - `fori(low, high, init, body) -> seq[Tensor]`: integer-counter loop
##   over `[low, high)`. Counter is `int32`.
##
## ## Constraints
## - All combinators are valid in trace mode only.
## - `cond.pred` must be a 0-rank `bool` tensor.
## - Both `cond` branches and the `whileLoop` body must return matching
##   dtype/shape/device per output position.
## - Tape recording for ops emitted *inside* a branch or loop body is
##   currently paused (so reverse-mode `grad` through these combinators
##   is not yet supported \u2014 it raises a clean `CondError`).
##   Differentiable variants land alongside `scan`.

import ../tensor
import ../dispatch
import ../dtype
import ../stablehlo/ir
import ../stablehlo/ops as shops
import ../autograd/tape
import ../ops/literal
import ../ops/arith
import ../ops/compare
import ../ops/shape
import ../ops/linalg
import ../ops/unary

type
  CondError* = object of CatchableError
    ## Misuse of `cond`: wrong dtype/shape/device, mode, or mismatched
    ## branch outputs.

template requireTraceMode(opName: string) =
  if currentMode() != dmTrace:
    raise newException(CondError,
      opName & ": only valid in trace mode (open a `withTrace`/`jit` block)")

proc condN*(pred: Tensor;
    thenFn: proc(): seq[Tensor] {.closure.};
    elseFn: proc(): seq[Tensor] {.closure.}): seq[Tensor] =
  ## Multi-output `if/else`. Runs `thenFn` and `elseFn` once each to
  ## materialise both branches in the StableHLO `if` regions.
  requireTraceMode("cond")
  requireTrace(pred, "cond")
  if pred.dtype != dtBool or pred.shape.len != 0:
    raise newException(CondError,
      "cond: predicate must be a 0-rank bool tensor, got dtype=" &
        $pred.dtype & " shape=" & $pred.shape)
  let ctx = currentTraceContext()
  let dev = pred.device
  var thenOuts: seq[Tensor]
  var elseOuts: seq[Tensor]
  let resIds = ifOp(ctx.builder, pred.traceId,
    proc(b: var ShBuilder): seq[ShValueId] =
      withPausedTape:
        thenOuts = thenFn()
      if thenOuts.len == 0:
        raise newException(CondError, "cond: then-branch returned no values")
      result = newSeq[ShValueId](thenOuts.len)
      for i, t in thenOuts:
        if not isTrace(t):
          raise newException(CondError,
            "cond: then-branch result #" & $i & " is not a trace tensor")
        result[i] = t.traceId,
    proc(b: var ShBuilder): seq[ShValueId] =
      withPausedTape:
        elseOuts = elseFn()
      if elseOuts.len != thenOuts.len:
        raise newException(CondError,
          "cond: else-branch returned " & $elseOuts.len &
            " value(s), then-branch returned " & $thenOuts.len)
      result = newSeq[ShValueId](elseOuts.len)
      for i, t in elseOuts:
        if not isTrace(t):
          raise newException(CondError,
            "cond: else-branch result #" & $i & " is not a trace tensor")
        if t.dtype != thenOuts[i].dtype or t.shape != thenOuts[i].shape:
          raise newException(CondError,
            "cond: branch result #" & $i &
              " type mismatch \u2014 then=" & $thenOuts[i].dtype &
              $thenOuts[i].shape & " else=" & $t.dtype & $t.shape)
        if t.device != thenOuts[i].device:
          raise newException(CondError,
            "cond: branch result #" & $i & " device mismatch")
        result[i] = t.traceId)
  result = newSeq[Tensor](resIds.len)
  for i, id in resIds:
    result[i] = initTraceTensor(id, thenOuts[i].dtype, thenOuts[i].shape,
      dev, thenOuts[i].sharding)

proc cond*(pred: Tensor;
    thenFn: proc(): Tensor {.closure.};
    elseFn: proc(): Tensor {.closure.}): Tensor =
  ## Single-output `if/else`. The two branches must return tensors with
  ## matching dtype/shape/device.
  let outs = condN(pred,
    proc(): seq[Tensor] = @[thenFn()],
    proc(): seq[Tensor] = @[elseFn()])
  outs[0]

# ---- Phase 7b.2: whileLoop + fori ---------------------------------------

proc whileLoop*(init: openArray[Tensor];
    condFn: proc(carry: openArray[Tensor]): Tensor {.closure.};
    bodyFn: proc(carry: openArray[Tensor]): seq[Tensor] {.closure.}):
    seq[Tensor] =
  ## Lower a loop into `stablehlo.while`. `init` is the initial loop
  ## carry; `condFn` returns the continuation predicate (a 0-rank bool
  ## trace tensor); `bodyFn` returns the next carry. All carry tensors
  ## must have matching dtype/shape/device per position across `init`,
  ## the cond's input view, and the body's output. Both `condFn` and
  ## `bodyFn` are traced once each; tape recording is paused inside.
  requireTraceMode("whileLoop")
  if init.len == 0:
    raise newException(CondError,
      "whileLoop: must carry at least one value")
  for i, t in init:
    requireTrace(t, "whileLoop")
    if i > 0 and t.device != init[0].device:
      raise newException(CondError,
        "whileLoop: init #" & $i & " device mismatch")
  let ctx = currentTraceContext()
  let dev = init[0].device
  let initSeq = @init
  var carryDtypes = newSeq[DType](initSeq.len)
  var carryShapes = newSeq[seq[int]](initSeq.len)
  for i, t in initSeq:
    carryDtypes[i] = t.dtype
    carryShapes[i] = t.shape
  var initIds = newSeq[ShValueId](initSeq.len)
  for i, t in initSeq: initIds[i] = t.traceId

  let resIds = whileOp(ctx.builder, initIds,
    proc(b: var ShBuilder; args: openArray[ShValueId]): ShValueId =
      var carry = newSeq[Tensor](args.len)
      for i, id in args:
        carry[i] = initTraceTensor(id, carryDtypes[i], carryShapes[i],
          dev, initSeq[i].sharding)
      var pred: Tensor
      withPausedTape:
        pred = condFn(carry)
      if not isTrace(pred):
        raise newException(CondError,
          "whileLoop: condFn must return a trace tensor")
      if pred.dtype != dtBool or pred.shape.len != 0:
        raise newException(CondError,
          "whileLoop: condFn must return a 0-rank bool, got dtype=" &
            $pred.dtype & " shape=" & $pred.shape)
      pred.traceId,
    proc(b: var ShBuilder; args: openArray[ShValueId]): seq[ShValueId] =
      var carry = newSeq[Tensor](args.len)
      for i, id in args:
        carry[i] = initTraceTensor(id, carryDtypes[i], carryShapes[i],
          dev, initSeq[i].sharding)
      var nextCarry: seq[Tensor]
      withPausedTape:
        nextCarry = bodyFn(carry)
      if nextCarry.len != carry.len:
        raise newException(CondError,
          "whileLoop: bodyFn returned " & $nextCarry.len &
            " value(s), expected " & $carry.len)
      result = newSeq[ShValueId](nextCarry.len)
      for i, t in nextCarry:
        if not isTrace(t):
          raise newException(CondError,
            "whileLoop: body result #" & $i & " is not a trace tensor")
        if t.dtype != carryDtypes[i] or t.shape != carryShapes[i]:
          raise newException(CondError,
            "whileLoop: body result #" & $i &
              " type mismatch \u2014 carry=" & $carryDtypes[i] &
              $carryShapes[i] & " body=" & $t.dtype & $t.shape)
        if t.device != dev:
          raise newException(CondError,
            "whileLoop: body result #" & $i & " device mismatch")
        result[i] = t.traceId)

  result = newSeq[Tensor](resIds.len)
  for i, id in resIds:
    result[i] = initTraceTensor(id, carryDtypes[i], carryShapes[i],
      dev, initSeq[i].sharding)

proc fori*(low: int32; high: int32; init: openArray[Tensor];
    body: proc(i: Tensor; carry: openArray[Tensor]): seq[Tensor] {.closure.}):
    seq[Tensor] =
  ## Iterate `i` over `[low, high)` with an `int32` counter, threading
  ## `init` as the loop carry. The body sees the current counter as a
  ## 0-rank `int32` trace tensor and returns the updated carry. Lowers
  ## to `stablehlo.while` over a `(counter, carry...)` tuple; the
  ## counter is dropped from the result.
  requireTraceMode("fori")
  let counter0 = scalarI32(low)
  let upper = scalarI32(high)
  let one = scalarI32(1'i32)
  var fullInit = newSeqOfCap[Tensor](init.len + 1)
  fullInit.add counter0
  for t in init: fullInit.add t

  let outs = whileLoop(fullInit,
    proc(carry: openArray[Tensor]): Tensor =
      compare(carry[0], upper, "LT"),
    proc(carry: openArray[Tensor]): seq[Tensor] =
      let i = carry[0]
      var userCarry = newSeq[Tensor](carry.len - 1)
      for k in 1 ..< carry.len: userCarry[k - 1] = carry[k]
      let next = body(i, userCarry)
      if next.len != userCarry.len:
        raise newException(CondError,
          "fori: body returned " & $next.len &
            " value(s), expected " & $userCarry.len)
      result = newSeq[Tensor](carry.len)
      result[0] = add(i, one)
      for k, t in next: result[k + 1] = t)

  result = newSeq[Tensor](outs.len - 1)
  for k in 1 ..< outs.len: result[k - 1] = outs[k]

proc caseOp*(index: Tensor;
    branches: openArray[proc (): seq[Tensor] {.closure.}]): seq[Tensor] =
  ## Multi-way branch combinator: lowers to `stablehlo.case`. `index` must
  ## be a 0-rank `int32` trace tensor. Each branch returns a `seq[Tensor]`;
  ## all branches must yield the same number and types of results.
  ## Trace-mode only.
  requireTraceMode("caseOp")
  requireTrace(index, "caseOp")
  if index.dtype != dtInt32 or index.shape.len != 0:
    raise newException(CondError,
      "caseOp: index must be a 0-rank int32 tensor, got " & $index.dtype &
        " shape " & $index.shape)
  if branches.len == 0:
    raise newException(CondError, "caseOp: at least one branch is required")
  let ctx = currentTraceContext()
  let dev = index.device
  var branchBuilders: seq[ShRegionBuilder] = @[]
  var expectedTypes: seq[ShTensorType]
  for bi, branch in branches:
    branchBuilders.add proc(b: var ShBuilder): seq[ShValueId] {.closure.} =
      var outs: seq[Tensor]
      withPausedTape:
        outs = branch()
      if outs.len == 0:
        raise newException(CondError,
          "caseOp: branch #" & $bi & " returned no values")
      if bi == 0:
        expectedTypes = newSeq[ShTensorType](outs.len)
        for i, t in outs:
          if not isTrace(t):
            raise newException(CondError,
              "caseOp: branch #" & $bi & " result #" & $i &
                " is not a trace tensor")
          expectedTypes[i] = tensorTypeOf(t)
      else:
        if outs.len != expectedTypes.len:
          raise newException(CondError,
            "caseOp: branch #" & $bi & " returned " & $outs.len &
              " value(s), expected " & $expectedTypes.len)
        for i, t in outs:
          if not isTrace(t):
            raise newException(CondError,
              "caseOp: branch #" & $bi & " result #" & $i &
                " is not a trace tensor")
          let ty = tensorTypeOf(t)
          if ty != expectedTypes[i]:
            raise newException(CondError,
              "caseOp: branch #" & $bi & " result #" & $i &
                " type mismatch — expected " & $expectedTypes[i] &
                ", got " & $ty)
      result = newSeq[ShValueId](outs.len)
      for i, t in outs: result[i] = t.traceId
  let ids = shops.caseOp(ctx.builder, index.traceId, branchBuilders)
  result = newSeq[Tensor](ids.len)
  for i, id in ids:
    result[i] = initTraceTensor(id, expectedTypes[i].dtype,
      expectedTypes[i].shape, dev, index.sharding)

# ---- scan ---------------------------------------------------------------

type
  ScanBody* = proc(carry: openArray[Tensor]; x: openArray[Tensor]):
    tuple[carry: seq[Tensor], y: seq[Tensor]] {.closure.}
    ## Body passed to `scan`: receives current carry and the current
    ## slice of `xs` along axis 0, returns updated carry and an output
    ## slice.

proc scan*(init: openArray[Tensor]; xs: openArray[Tensor];
    body: ScanBody): tuple[carry: seq[Tensor], ys: seq[Tensor]] =
  ## Sequential scan over the leading axis of `xs`. At each step the
  ## body receives the current carry and one slice of `xs` and returns
  ## the updated carry and an output slice. The output slices are
  ## stacked along axis 0.
  ##
  ## Lowers to `stablehlo.while` with dynamic slice/update. Tape
  ## recording is paused inside the body (same limitation as
  ## `whileLoop` in v1).
  requireTraceMode("scan")
  if xs.len == 0:
    raise newException(CondError, "scan: at least one xs tensor required")
  let n = xs[0].shape[0]
  let dev = xs[0].device
  for x in xs:
    requireTrace(x, "scan")
    if x.shape[0] != n:
      raise newException(CondError,
        "scan: all xs must have the same leading dim size")
    if x.device != dev:
      raise newException(CondError, "scan: xs device mismatch")
  for t in init:
    requireTrace(t, "scan")
    if t.device != dev:
      raise newException(CondError, "scan: init device mismatch")

  # Determine output shapes by running body once on the 0-th slice.
  var startZeros: seq[Tensor] = @[]
  for _ in 0 ..< xs[0].shape.len:
    startZeros.add(scalarI32(0'i32))
  var sliceSizes: seq[int] = @[1]
  for i in 1 ..< xs[0].shape.len:
    sliceSizes.add(xs[0].shape[i])
  var x0Slices: seq[Tensor] = @[]
  for x in xs:
    x0Slices.add(squeeze(dynamicSlice(x, startZeros, sliceSizes), 0))
  var dummyCarry: seq[Tensor]
  var dummyY: seq[Tensor]
  withPausedTape:
    (dummyCarry, dummyY) = body(@init, x0Slices)
  if dummyCarry.len != init.len:
    raise newException(CondError,
      "scan: body returned " & $dummyCarry.len &
        " carry value(s), expected " & $init.len)

  # Build whileLoop: carry = (counter, carry..., ys_accum...).
  let counter0 = scalarI32(0'i32)
  let nVal = scalarI32(int32(n))
  let one = scalarI32(1'i32)
  let carryLen = init.len
  let ysLen = dummyY.len

  var fullInit: seq[Tensor] = @[counter0]
  for t in @init: fullInit.add t
  for t in dummyY:
    var ysShape = @[n]
    for d in t.shape: ysShape.add d
    if t.dtype == dtFloat32:
      fullInit.add(broadcastTo(scalarF32(0'f32), ysShape, @[]))
    elif t.dtype == dtInt32:
      fullInit.add(broadcastTo(scalarI32(0'i32), ysShape, @[]))
    else:
      fullInit.add(astype(
        broadcastTo(scalarF32(0'f32), ysShape, @[]), t.dtype))

  let outs = whileLoop(fullInit,
    proc(args: openArray[Tensor]): Tensor =
      compare(args[0], nVal, "LT"),
    proc(args: openArray[Tensor]): seq[Tensor] =
      let counter = args[0]
      # Extract carry.
      var carry = newSeq[Tensor](carryLen)
      for i in 0 ..< carryLen:
        carry[i] = args[1 + i]
      # Extract x slices at position counter.
      var xSlices: seq[Tensor] = @[]
      for xi, x in xs:
        var starts: seq[Tensor] = @[counter]
        for _ in 1 ..< x.shape.len:
          starts.add(scalarI32(0'i32))
        xSlices.add(squeeze(dynamicSlice(x, starts, sliceSizes), 0))
      # Call user body (tape is paused by whileLoop).
      let (newCarry, ysOut) = body(carry, xSlices)
      if newCarry.len != carryLen:
        raise newException(CondError,
          "scan: body returned " & $newCarry.len &
            " carry value(s), expected " & $carryLen)
      if ysOut.len != ysLen:
        raise newException(CondError,
          "scan: body returned " & $ysOut.len &
            " y value(s), expected " & $ysLen)
      # Build result: counter+1, newCarry..., updatedYs...
      result = newSeq[Tensor](args.len)
      result[0] = add(counter, one)
      for i in 0 ..< carryLen:
        result[1 + i] = newCarry[i]
      # Update ys accumulators with dynamic_update_slice.
      for yi, y in ysOut:
        let accumIdx = 1 + carryLen + yi
        var starts: seq[Tensor] = @[counter]
        for _ in 1 ..< y.shape.len:
          starts.add(scalarI32(0'i32))
        result[accumIdx] = dynamicUpdateSlice(
          args[accumIdx], unsqueeze(y, 0), starts))

  result.carry = newSeq[Tensor](carryLen)
  for i in 0 ..< carryLen:
    result.carry[i] = outs[1 + i]
  result.ys = newSeq[Tensor](ysLen)
  for i in 0 ..< ysLen:
    result.ys[i] = outs[1 + carryLen + i]

