## `vmap` — loop-based automatic batching transform.
##
## `vmap(fn)` returns a new function that expects inputs with an
## extra leading batch dimension.  Internally it slices along axis 0,
## calls `fn` for each slice inside a `fori`-lowered loop, and stacks
## the results.  Because the loop body is traced once, the whole
## computation lives in a single StableHLO module.
##
## ## Limitations
## - Only batch axis 0 is supported in v1.
## - The inner function must be trace-compatible (no eager-mode ops).
## - Differentiability through the batching loop follows the same
##   rules as `fori` / `whileLoop` (tape paused inside the body).
##

import ../tensor
import ../dtype
import ../dispatch
import ./control
import ../ops/literal
import ../ops/linalg
import ../ops/shape
import ../ops/unary

proc vmap*(fn: proc(args: openArray[Tensor]): seq[Tensor] {.closure.};
    inAxes: openArray[int] = @[]):
    proc(args: openArray[Tensor]): seq[Tensor] {.closure.} =
  ## Wrap `fn` so it operates over a leading batch dimension.
  ##
  ## The returned function expects each input to have an extra axis-0
  ## batch dimension.  When called in trace mode it lowers to a
  ## `fori` loop; outside trace mode it raises an error.
  ##
  ## `inAxes` (unused in v1) would specify a different batch axis per
  ## input; in v1 all inputs must use axis 0.

  result = proc(args: openArray[Tensor]): seq[Tensor] =
    if currentMode() != dmTrace:
      raise newException(TensorError,
        "vmap: only supported in trace/jit mode")
    let batchSize = args[0].shape[0]
    if batchSize == 0:
      return @[]

    # ---- extract 0-th slice to determine output shapes -----------------
    var firstSlice: seq[Tensor] = @[]
    for a in args:
      var starts: seq[Tensor] = @[]
      for _ in 0 ..< a.shape.len:
        starts.add(scalarI32(0'i32))
      var sizes: seq[int] = @[1]
      for i in 1 ..< a.shape.len:
        sizes.add(a.shape[i])
      firstSlice.add(squeeze(dynamicSlice(a, starts, sizes), 0))

    var outDtypes: seq[DType]
    var outShapes: seq[seq[int]]
    block:
      let firstOuts = fn(firstSlice)
      for o in firstOuts:
        outDtypes.add(o.dtype)
        var ysShape = @[batchSize]
        for d in o.shape: ysShape.add d
        outShapes.add(ysShape)

    # ---- build zero accumulators for outputs ---------------------------
    var accums: seq[Tensor] = @[]
    for i, yshape in outShapes:
      if outDtypes[i] == dtFloat32:
        accums.add(broadcastTo(scalarF32(0'f32), yshape, @[]))
      elif outDtypes[i] == dtInt32:
        accums.add(broadcastTo(scalarI32(0'i32), yshape, @[]))
      else:
        accums.add(astype(
          broadcastTo(scalarF32(0'f32), yshape, @[]), outDtypes[i]))

    # ---- loop over batch dimension ------------------------------------
    let outs = fori(0'i32, int32(batchSize), accums,
      proc(i: Tensor; carry: openArray[Tensor]): seq[Tensor] =
        # Extract slice at position i.
        var slices: seq[Tensor] = @[]
        for a in args:
          var starts: seq[Tensor] = @[i]
          for _ in 1 ..< a.shape.len:
            starts.add(scalarI32(0'i32))
          var sizes: seq[int] = @[1]
          for j in 1 ..< a.shape.len:
            sizes.add(a.shape[j])
          slices.add(squeeze(dynamicSlice(a, starts, sizes), 0))
        let outsI = fn(slices)
        # Write each output into its accumulator.
        result = newSeq[Tensor](carry.len)
        for oi, o in outsI:
          var starts: seq[Tensor] = @[i]
          for _ in 1 ..< o.shape.len:
            starts.add(scalarI32(0'i32))
          result[oi] = dynamicUpdateSlice(
            carry[oi], unsqueeze(o, 0), starts))
    outs
