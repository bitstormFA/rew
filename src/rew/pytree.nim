## Pytree — generic flatten/unflatten over tensor-bearing value types.
##
## ## Why
## A "pytree" is any value type whose fields are either `Tensor`s, other
## pytrees, or `seq`/`array`/tuples of pytrees. The framework uses a
## single pytree walker to drive optimizer state updates, `grad` over
## arbitrary parameter containers, serialization, and device transfer —
## without a `Module` base class and without registering types.
##
## ## API
## - `treeFlatten(t)` returns the flat `seq[Tensor]` of leaves in
##   field-declaration order.
## - `treeUnflatten(t, leaves)` returns a copy of `t` with its tensor
##   leaves replaced (in the same order) by the values from `leaves`. It
##   raises `PytreeError` if the count does not match.
## - `treeMap(t, fn)` returns a copy of `t` with `fn` applied to every
##   tensor leaf.
##
## Both procs recurse through nested `object`/`tuple` fields and through
## `seq`/`array` of pytrees. Non-tensor scalar fields (ints, floats,
## strings, enums, …) are preserved verbatim.

import ./tensor

type
  PytreeError* = object of CatchableError
    ## Raised by `treeUnflatten` when the leaf count does not match the
    ## tree's structure.

# ---- treeFlatten ----------------------------------------------------------

proc flattenInto(dst: var seq[Tensor]; v: Tensor) =
  dst.add(v)

proc flattenInto[T](dst: var seq[Tensor]; v: T)

proc flattenInto[T](dst: var seq[Tensor]; v: seq[T]) =
  for item in v: flattenInto(dst, item)

proc flattenInto[N, T](dst: var seq[Tensor]; v: array[N, T]) =
  for item in v: flattenInto(dst, item)

proc flattenInto[T](dst: var seq[Tensor]; v: T) =
  when T is Tensor:
    dst.add(v)
  elif T is (object | tuple):
    for _, field in fieldPairs(v):
      flattenInto(dst, field)
  else:
    discard

proc treeFlatten*[T](v: T): seq[Tensor] =
  ## Returns the flat `seq[Tensor]` of leaves in `v`, in field-declaration
  ## order. Non-tensor scalar fields are skipped.
  result = @[]
  flattenInto(result, v)

# ---- treeUnflatten --------------------------------------------------------

proc unflattenInto(dst: var Tensor; leaves: seq[Tensor]; idx: var int) =
  if idx >= leaves.len:
    raise newException(PytreeError,
      "treeUnflatten: too few leaves (need at least " & $(idx + 1) & ")")
  dst = leaves[idx]
  inc idx

proc unflattenInto[T](dst: var T; leaves: seq[Tensor]; idx: var int)

proc unflattenInto[T](dst: var seq[T]; leaves: seq[Tensor]; idx: var int) =
  for i in 0 ..< dst.len:
    unflattenInto(dst[i], leaves, idx)

proc unflattenInto[N, T](dst: var array[N, T]; leaves: seq[Tensor];
    idx: var int) =
  for i in 0 ..< dst.len:
    unflattenInto(dst[i], leaves, idx)

proc unflattenInto[T](dst: var T; leaves: seq[Tensor]; idx: var int) =
  when T is Tensor:
    if idx >= leaves.len:
      raise newException(PytreeError,
        "treeUnflatten: too few leaves (need at least " & $(idx + 1) & ")")
    dst = leaves[idx]
    inc idx
  elif T is (object | tuple):
    for _, field in fieldPairs(dst):
      unflattenInto(field, leaves, idx)
  else:
    discard

proc treeUnflatten*[T](structure: T; leaves: seq[Tensor]): T =
  ## Returns a copy of `structure` with its tensor leaves replaced by the
  ## entries of `leaves`, in the same order `treeFlatten` produced. Raises
  ## `PytreeError` if `leaves.len` does not match the leaf count of
  ## `structure`.
  result = structure
  var idx = 0
  unflattenInto(result, leaves, idx)
  if idx != leaves.len:
    raise newException(PytreeError,
      "treeUnflatten: too many leaves (got " & $leaves.len &
        ", consumed " & $idx & ")")

# ---- treeMap --------------------------------------------------------------

proc treeMap*[T](v: T; fn: proc(x: Tensor): Tensor): T =
  ## Returns a copy of `v` with `fn` applied to every tensor leaf, in
  ## field-declaration order.
  let leaves = treeFlatten(v)
  var mapped = newSeq[Tensor](leaves.len)
  for i, t in leaves: mapped[i] = fn(t)
  result = treeUnflatten(v, mapped)

proc treeLeafCount*[T](v: T): int =
  ## Number of `Tensor` leaves in `v`. Convenience for diagnostics.
  treeFlatten(v).len
