## Pytree - generic flatten/unflatten over tensor-bearing value types.
##
## ## Why
## A "pytree" is any value type whose fields are either `Tensor`s,
## `Param[Tensor]`, `Buffer[Tensor]`, other pytrees, or `seq`/`array`/tuples
## of pytrees. The framework uses a single pytree walker to drive optimizer
## state updates, `grad` over arbitrary parameter containers, serialization,
## and device transfer - without a `Module` base class and without registering
## types.
##
## ## API
## - `Param[T]` marks trainable state.
## - `Buffer[T]` marks non-trainable state that still participates in device
##   transfer and checkpointing.
## - Bare `Tensor` leaves are structural leaves. They are flattened for
##   transforms and serialization, but they are not trainable model leaves.
## - `treeFlatten(t)` returns the flat `seq[Tensor]` of all tensor-bearing
##   leaves in field-declaration order.
## - `treeLeaves(t)` returns path/kind metadata for every tensor-bearing leaf.
## - `treeUnflatten(t, leaves)` returns a copy of `t` with its tensor
##   leaves replaced (in the same order) by the values from `leaves`. It
##   raises `PytreeError` if the count does not match.
## - `treeMap(t, fn)` returns a copy of `t` with `fn` applied to every
##   tensor leaf.
##
## Both procs recurse through nested `object`/`tuple` fields and through
## `seq`/`array` of pytrees. Non-tensor scalar fields (ints, floats,
## strings, enums, ...) are preserved verbatim.

import std/strutils
import ./tensor
import ./dtype
import ./device

type
  PytreeError* = object of CatchableError
    ## Raised by `treeUnflatten` when the leaf count does not match the
    ## tree's structure.

  Param*[T] = object
    ## Trainable pytree leaf wrapper.
    value*: T

  Buffer*[T] = object
    ## Non-trainable pytree leaf wrapper.
    value*: T

  TreePath* = distinct string
    ## Stable field path for a tensor-bearing pytree leaf.

  PathSet* = object
    ## Small ordered set of pytree paths used by donation, freezing,
    ## partitioning, sharding, and checkpoint policies.
    paths*: seq[TreePath]

  TreeLeafKind* = enum
    tlTensor
    tlParam
    tlBuffer

  TreeLeaf* = object
    ## Named tensor-bearing leaf discovered during a pytree walk.
    path*: TreePath
    tensor*: Tensor
    kind*: TreeLeafKind

  TreePartition* = object
    ## Result of selecting leaves by path/kind.
    selected*: seq[TreeLeaf]
    rest*: seq[TreeLeaf]

func param*[T](value: T): Param[T] =
  ## Wraps `value` as trainable state.
  Param[T](value: value)

func buffer*[T](value: T): Buffer[T] =
  ## Wraps `value` as non-trainable state.
  Buffer[T](value: value)

func treePath*(value: string): TreePath =
  ## Converts a string into a typed pytree path.
  TreePath(value)

converter treePathToString*(path: TreePath): string =
  string(path)

func `$`*(path: TreePath): string =
  string(path)

func `==`*(left, right: TreePath): bool =
  string(left) == string(right)

func initPathSet*(paths: openArray[TreePath]): PathSet =
  ## Creates an ordered path set, keeping the first occurrence of each path.
  for path in paths:
    if path notin result.paths:
      result.paths.add path

func initPathSet*(paths: openArray[string]): PathSet =
  ## Creates an ordered path set from string paths.
  for path in paths:
    let typed = treePath(path)
    if typed notin result.paths:
      result.paths.add typed

func pathSet*(paths: openArray[TreePath]): PathSet =
  ## Convenience alias for `initPathSet`.
  initPathSet(paths)

func pathSet*(paths: openArray[string]): PathSet =
  ## Convenience alias for `initPathSet`.
  initPathSet(paths)

func len*(paths: PathSet): int =
  paths.paths.len

iterator items*(paths: PathSet): TreePath =
  for path in paths.paths:
    yield path

func contains*(paths: PathSet; path: TreePath): bool =
  for item in paths.paths:
    if item == path:
      return true

func contains*(paths: PathSet; path: string): bool =
  paths.contains(treePath(path))

proc incl*(paths: var PathSet; path: TreePath) =
  ## Adds `path` if it is not already present.
  if path notin paths.paths:
    paths.paths.add path

proc incl*(paths: var PathSet; path: string) =
  paths.incl(treePath(path))

func matchesPrefix*(prefix, path: TreePath): bool =
  ## True when `prefix` names `path` or an ancestor of `path`.
  let p = string(prefix)
  let value = string(path)
  value == p or value.startsWith(p & ".") or value.startsWith(p & "[")

converter paramToTensor*(p: Param[Tensor]): Tensor =
  p.value

converter bufferToTensor*(b: Buffer[Tensor]): Tensor =
  b.value

func shape*(p: Param[Tensor]): seq[int] =
  p.value.shape

func shape*(b: Buffer[Tensor]): seq[int] =
  b.value.shape

func dtype*(p: Param[Tensor]): DType =
  p.value.dtype

func dtype*(b: Buffer[Tensor]): DType =
  b.value.dtype

func device*(p: Param[Tensor]): Device =
  p.value.device

func device*(b: Buffer[Tensor]): Device =
  b.value.device

# ---- treeFlatten ----------------------------------------------------------

proc flattenInto(dst: var seq[Tensor]; v: Tensor) =
  dst.add(v)

proc flattenInto[T](dst: var seq[Tensor]; v: T)

proc flattenInto[T](dst: var seq[Tensor]; v: Param[T]) =
  flattenInto(dst, v.value)

proc flattenInto[T](dst: var seq[Tensor]; v: Buffer[T]) =
  flattenInto(dst, v.value)

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

# ---- treeLeaves / treePaths -----------------------------------------------

proc childPath(base: TreePath; name: string): TreePath =
  if string(base).len == 0:
    treePath(name)
  else:
    treePath(string(base) & "." & name)

proc indexPath(base: TreePath; i: int): TreePath =
  if string(base).len == 0:
    treePath("[" & $i & "]")
  else:
    treePath(string(base) & "[" & $i & "]")

proc collectLeaves(dst: var seq[TreeLeaf]; path: TreePath; v: Tensor;
    kind: TreeLeafKind = tlTensor) =
  dst.add TreeLeaf(path: path, tensor: v, kind: kind)

proc collectLeaves[T](dst: var seq[TreeLeaf]; path: TreePath; v: T;
    kind: TreeLeafKind = tlTensor)

proc collectLeaves[T](dst: var seq[TreeLeaf]; path: TreePath; v: Param[T];
    kind: TreeLeafKind = tlTensor) =
  collectLeaves(dst, path, v.value, tlParam)

proc collectLeaves[T](dst: var seq[TreeLeaf]; path: TreePath; v: Buffer[T];
    kind: TreeLeafKind = tlTensor) =
  collectLeaves(dst, path, v.value, tlBuffer)

proc collectLeaves[T](dst: var seq[TreeLeaf]; path: TreePath; v: seq[T];
    kind: TreeLeafKind = tlTensor) =
  for i, item in v:
    collectLeaves(dst, indexPath(path, i), item, kind)

proc collectLeaves[N, T](dst: var seq[TreeLeaf]; path: TreePath; v: array[N, T];
    kind: TreeLeafKind = tlTensor) =
  for i, item in v:
    collectLeaves(dst, indexPath(path, i), item, kind)

proc collectLeaves[T](dst: var seq[TreeLeaf]; path: TreePath; v: T;
    kind: TreeLeafKind = tlTensor) =
  when T is Tensor:
    dst.add TreeLeaf(path: path, tensor: v, kind: kind)
  elif T is (object | tuple):
    for name, field in fieldPairs(v):
      collectLeaves(dst, childPath(path, name), field, kind)
  else:
    discard

proc treeLeaves*[T](v: T): seq[TreeLeaf] =
  ## Returns tensor-bearing leaves with stable field paths.
  result = @[]
  collectLeaves(result, treePath(""), v)

proc treePaths*[T](v: T): PathSet =
  ## Returns paths for every tensor-bearing leaf.
  for leaf in treeLeaves(v):
    result.incl leaf.path

proc paramsOf*[T](v: T): PathSet =
  ## Returns paths for trainable leaves.
  for leaf in treeLeaves(v):
    if leaf.kind == tlParam:
      result.incl leaf.path

proc buffersOf*[T](v: T): PathSet =
  ## Returns paths for non-trainable buffer leaves.
  for leaf in treeLeaves(v):
    if leaf.kind == tlBuffer:
      result.incl leaf.path

proc treePartition*[T](v: T;
    pred: proc(path: TreePath; kind: TreeLeafKind): bool {.closure.}):
    TreePartition =
  ## Splits leaves into `selected` and `rest` using a path/kind predicate.
  for leaf in treeLeaves(v):
    if pred(leaf.path, leaf.kind):
      result.selected.add leaf
    else:
      result.rest.add leaf

# ---- treeUnflatten --------------------------------------------------------

proc unflattenInto(dst: var Tensor; leaves: seq[Tensor]; idx: var int) =
  if idx >= leaves.len:
    raise newException(PytreeError,
      "treeUnflatten: too few leaves (need at least " & $(idx + 1) & ")")
  dst = leaves[idx]
  inc idx

proc unflattenInto[T](dst: var T; leaves: seq[Tensor]; idx: var int)

proc unflattenInto[T](dst: var Param[T]; leaves: seq[Tensor]; idx: var int) =
  unflattenInto(dst.value, leaves, idx)

proc unflattenInto[T](dst: var Buffer[T]; leaves: seq[Tensor]; idx: var int) =
  unflattenInto(dst.value, leaves, idx)

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
      when compiles(unflattenInto(field, leaves, idx)):
        unflattenInto(field, leaves, idx)
      else:
        discard
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

proc treeMapWithPath*[T](v: T;
    fn: proc(path: TreePath; kind: TreeLeafKind; x: Tensor): Tensor): T =
  ## Returns a copy of `v` with `fn` applied to every tensor leaf, passing
  ## stable path and leaf-kind metadata.
  let leaves = treeLeaves(v)
  var mapped = newSeq[Tensor](leaves.len)
  for i, leaf in leaves:
    mapped[i] = fn(leaf.path, leaf.kind, leaf.tensor)
  result = treeUnflatten(v, mapped)

proc treeLeafCount*[T](v: T): int =
  ## Number of `Tensor` leaves in `v`. Convenience for diagnostics.
  treeFlatten(v).len
