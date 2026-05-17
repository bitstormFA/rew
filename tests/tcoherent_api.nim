## Coherent high-level API vocabulary tests.

import std/[os, strutils]
import rew

static:
  doAssert not compiles(block:
    var f: JitFunction
    discard f)
  doAssert not compiles(block:
    var f: JitFn
    discard f)
  doAssert not compiles(initBuilder("raw"))
  doAssert not compiles(registerVjp("raw.op"))
  doAssert not compiles(openXlaManifest())

type
  TinyState = object
    weight: Param[Tensor]
    runningMean: Buffer[Tensor]
    bias: Tensor

func fakeTensor(shape: openArray[int]): Tensor =
  Tensor(dtype: dtFloat32, shape: @shape, device: cpu(0))

block pytree_paths_and_partitions:
  let w = fakeTensor([2, 2])
  let mean = fakeTensor([2])
  let bias = fakeTensor([2])
  let state = TinyState(weight: param(w), runningMean: buffer(mean), bias: bias)

  doAssert treePaths(state) == @["weight", "runningMean", "bias"]
  doAssert paramsOf(state) == @["weight"]
  doAssert buffersOf(state) == @["runningMean"]

  let partitioned = treePartition(state,
    proc(path: string; kind: TreeLeafKind): bool =
      kind == tlParam)
  doAssert partitioned.selected.len == 1
  doAssert partitioned.rest.len == 2
  doAssert partitioned.rest[0].path == "runningMean"
  doAssert partitioned.rest[1].path == "bias"

block train_state_initializes_transform_state:
  let w = fakeTensor([2, 2])
  let lr = fakeTensor([])
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
