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

  doAssert treePaths(state).paths == @[
    treePath("weight"),
    treePath("runningMean"),
    treePath("bias"),
  ]
  doAssert paramsOf(state).paths == @[treePath("weight")]
  doAssert buffersOf(state).paths == @[treePath("runningMean")]

  let partitioned = treePartition(state,
    proc(path: TreePath; kind: TreeLeafKind): bool =
      discard path
      kind == tlParam)
  doAssert partitioned.selected.len == 1
  doAssert partitioned.rest.len == 2
  doAssert partitioned.rest[0].path == treePath("runningMean")
  doAssert partitioned.rest[1].path == treePath("bias")

block train_state_initializes_transform_state:
  let w = fakeTensor([2, 2])
  let lr = fakeTensor([])
  let state = initTrainState(TinyState(weight: param(w),
    runningMean: buffer(w), bias: w), chain(clipByGlobalNorm(1'f32), sgd(lr)))
  doAssert state.step == 0
  doAssert state.opt.transforms.len == 2
  doAssert state.opt.transforms[0].kind == gtkClipByGlobalNorm
  doAssert state.optState.states.len == 2

block coherent_example_hides_raw_jitfn:
  let src = readFile(getCurrentDir() / "examples" / "mnist_coherent_api.nim")
  doAssert "JitFn" notin src
  doAssert "compileTrainStep" in src
  doAssert "initTrainState" in src
