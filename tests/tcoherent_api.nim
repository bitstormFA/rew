## Coherent high-level API vocabulary tests.

import std/[os, strutils]
import rew

type
  TinyState = object
    weight: Param[Tensor]
    runningMean: Buffer[Tensor]
    bias: Tensor

block pytree_paths_and_partitions:
  let w = initTraceTensor(ShValueId(1), dtFloat32, @[2, 2], cpu(0))
  let mean = initTraceTensor(ShValueId(2), dtFloat32, @[2], cpu(0))
  let bias = initTraceTensor(ShValueId(3), dtFloat32, @[2], cpu(0))
  let state = TinyState(weight: param(w), runningMean: buffer(mean), bias: bias)

  doAssert treePaths(state) == @["weight", "runningMean", "bias"]
  doAssert paramsOf(state) == @["weight", "bias"]
  doAssert buffersOf(state) == @["runningMean"]

  let partitioned = treePartition(state,
    proc(path: string; kind: TreeLeafKind): bool =
      kind == tlParam or path == "bias")
  doAssert partitioned.selected.len == 2
  doAssert partitioned.rest.len == 1
  doAssert partitioned.rest[0].path == "runningMean"

block train_state_initializes_transform_state:
  let w = initTraceTensor(ShValueId(4), dtFloat32, @[2, 2], cpu(0))
  let lr = initTraceTensor(ShValueId(5), dtFloat32, @[], cpu(0))
  let state = initTrainState(TinyState(weight: param(w),
    runningMean: buffer(w), bias: w), chain(clipByGlobalNorm(1'f32), sgd(lr)))
  doAssert state.step == 0
  doAssert state.opt.kind == gtkChain
  doAssert state.optState.kind == gtkChain

block coherent_example_hides_raw_jitfn:
  let src = readFile(getCurrentDir() / "examples" / "mnist_coherent_api.nim")
  doAssert "JitFn" notin src
  doAssert "compileTrainStep" in src
  doAssert "initTrainState" in src
