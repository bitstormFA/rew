## Phase 5a — pytree flatten/unflatten/map over tensor-bearing types.

import rew
import rew/xla

let TestDevice = cpu(0)

proc fakeTensor(id: int; shape: openArray[int] = [2]): Tensor =
  ## Trace-mode tensor with a fabricated SSA id; sufficient for testing
  ## the structural walker since pytree never inspects buffers.
  initTraceTensor(ShValueId(id), dtFloat32, shape, TestDevice)

block flatten_object_tensors:
  type Pair = object
    a, b: Tensor
  let p = Pair(a: fakeTensor(1), b: fakeTensor(2))
  let leaves = treeFlatten(p)
  doAssert leaves.len == 2
  doAssert leaves[0].traceId.int == 1
  doAssert leaves[1].traceId.int == 2

block flatten_nested_object:
  type
    Inner = object
      w: Tensor
      bias: Tensor
    Outer = object
      lhs: Inner
      rhs: Inner
      tag: int
  let v = Outer(
    lhs: Inner(w: fakeTensor(10), bias: fakeTensor(11)),
    rhs: Inner(w: fakeTensor(20), bias: fakeTensor(21)),
    tag: 7,
  )
  let leaves = treeFlatten(v)
  doAssert leaves.len == 4
  doAssert leaves[0].traceId.int == 10
  doAssert leaves[3].traceId.int == 21
  doAssert treeLeafCount(v) == 4

block flatten_seq_and_array:
  type Layer = object
    weights: seq[Tensor]
    snapshot: array[2, Tensor]
  let l = Layer(
    weights: @[fakeTensor(1), fakeTensor(2), fakeTensor(3)],
    snapshot: [fakeTensor(4), fakeTensor(5)],
  )
  let leaves = treeFlatten(l)
  doAssert leaves.len == 5
  for i, t in leaves:
    doAssert t.traceId.int == i + 1

block unflatten_round_trip:
  type Pair = object
    a, b: Tensor
    name: string
  let p = Pair(a: fakeTensor(1), b: fakeTensor(2), name: "demo")
  let leaves = treeFlatten(p)
  doAssert leaves.len == 2
  let mapped = @[fakeTensor(99), fakeTensor(100)]
  let p2 = treeUnflatten(p, mapped)
  doAssert p2.a.traceId.int == 99
  doAssert p2.b.traceId.int == 100
  doAssert p2.name == "demo"

block unflatten_nested_seq:
  type Bag = object
    items: seq[Tensor]
  let b = Bag(items: @[fakeTensor(1), fakeTensor(2), fakeTensor(3)])
  let leaves = treeFlatten(b)
  doAssert leaves.len == 3
  let b2 = treeUnflatten(b, @[fakeTensor(7), fakeTensor(8), fakeTensor(9)])
  doAssert b2.items[0].traceId.int == 7
  doAssert b2.items[2].traceId.int == 9

block unflatten_count_mismatch_raises:
  type Pair = object
    a, b: Tensor
  let p = Pair(a: fakeTensor(1), b: fakeTensor(2))
  doAssertRaises(PytreeError):
    discard treeUnflatten(p, @[fakeTensor(99)])
  doAssertRaises(PytreeError):
    discard treeUnflatten(p, @[fakeTensor(99), fakeTensor(100), fakeTensor(101)])

block tree_map_replaces_leaves:
  type Pair = object
    a, b: Tensor
  let p = Pair(a: fakeTensor(1), b: fakeTensor(2))
  proc bumpId(t: Tensor): Tensor =
    initTraceTensor(ShValueId(t.traceId.int + 100), t.dtype, t.shape,
      t.device, t.sharding)
  let p2 = treeMap(p, bumpId)
  doAssert p2.a.traceId.int == 101
  doAssert p2.b.traceId.int == 102

block scalar_only_object_has_no_leaves:
  type Plain = object
    x, y: int
    name: string
  let v = Plain(x: 1, y: 2, name: "hi")
  doAssert treeFlatten(v).len == 0
  doAssert treeUnflatten(v, @[]) == v
