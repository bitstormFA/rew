{.experimental: "callOperator".}

## High-level typed training state and compiled steps.
##
## This module keeps the user-facing training language typed: users write a
## loss over a model and batch, while rew handles pytree flattening, `grad`,
## `jit`, optimizer state, and donation at the boundary.

import ../tensor
import ../pytree
import ../rng
import ../autograd/transform
import ../transform/jit
import ../optim/transform as optim_tx
import ./runtime
import ./context

type
  CallCtx* = object
    ## Execution context passed into typed losses and custom steps.
    ## Dynamic counters and PRNG state live on `TrainState`.
    runtime*: Runtime
    mode*: TrainMode

  TrainState*[M] = object
    ## Model, optimizer, optimizer state, PRNG key, and global step.
    model*: M
    opt*: GradientTransform
    optState*: OptimState
    step*: int
    key*: Key

  StepMetric* = object
    ## Tensor metric returned by a typed training step.
    name*: string
    value*: Tensor
    progBar*: bool

  StepResult*[S] = object
    ## Result of one high-level training step.
    state*: S
    loss*: Tensor
    metrics*: seq[StepMetric]

  LossFn*[M, B] = proc(model: M; batch: B; ctx: CallCtx): Tensor {.closure.}

  TrainStepFn*[M, B] = proc(state: TrainState[M]; batch: B;
    ctx: CallCtx): StepResult[TrainState[M]] {.closure.}

  CompiledTrainStep*[M, B] = object
    ## Cached typed training step.
    loss*: LossFn[M, B]
    runtime*: Runtime
    donate*: seq[string]
    compiled: JitFunction
    modelProto*: M
    batchProto*: B
    optStateProto*: OptimState
    opt*: GradientTransform
    prepared: bool

func initCallCtx*(runtime: Runtime; mode: TrainMode = tmFit): CallCtx =
  ## Creates an execution context for typed steps.
  CallCtx(runtime: runtime, mode: mode)

proc initTrainState*[M](model: M; opt: GradientTransform;
    key: Key = initKey(0)): TrainState[M] =
  ## Initializes optimizer state for `model`.
  TrainState[M](
    model: model,
    opt: opt,
    optState: initState(opt, model),
    step: 0,
    key: key,
  )

func containsPath(paths: openArray[string]; path: string): bool =
  for item in paths:
    if item == path:
      return true

proc donateIndices[M](model: M; donate: openArray[string]): seq[int] =
  let leaves = treeLeaves(model)
  for i, leaf in leaves:
    if containsPath(donate, leaf.path):
      result.add i

proc tensorSlice(args: openArray[Tensor]; start, count: int): seq[Tensor] =
  result = newSeq[Tensor](count)
  for i in 0 ..< count:
    result[i] = args[start + i]

proc concatTensors(parts: openArray[seq[Tensor]]): seq[Tensor] =
  for part in parts:
    for t in part:
      result.add t

proc buildJit[M, B](step: var CompiledTrainStep[M, B]) =
  let modelProto = step.modelProto
  let batchProto = step.batchProto
  let optStateProto = step.optStateProto
  let opt = step.opt
  let loss = step.loss
  let runtime = step.runtime
  let modelCount = treeLeafCount(modelProto)
  let batchCount = treeLeafCount(batchProto)
  let optStateCount = treeLeafCount(optStateProto)
  let optCount = treeLeafCount(opt)

  let fn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    var offset = 0
    let modelLeaves = tensorSlice(args, offset, modelCount)
    offset += modelCount
    let batchLeaves = tensorSlice(args, offset, batchCount)
    offset += batchCount
    let optStateLeaves = tensorSlice(args, offset, optStateCount)
    offset += optStateCount
    let optLeaves = tensorSlice(args, offset, optCount)

    let model = treeUnflatten(modelProto, modelLeaves)
    let batch = treeUnflatten(batchProto, batchLeaves)
    let optState = treeUnflatten(optStateProto, optStateLeaves)
    let tracedOpt = treeUnflatten(opt, optLeaves)
    let ctx = initCallCtx(runtime, tmFit)

    let lossForGrad = proc(flatParams: openArray[Tensor]): Tensor =
      let tracedModel = treeUnflatten(modelProto, @flatParams)
      loss(tracedModel, batch, ctx)

    let lossValue = loss(model, batch, ctx)
    let grads = treeUnflatten(model, grad(lossForGrad, modelLeaves))
    var effectiveOpt = tracedOpt
    let buffers = buffersOf(model)
    if buffers.len > 0:
      effectiveOpt = chain(freeze(buffers), opt)
    let (newModel, newOptState) =
      effectiveOpt.update(grads, optState, model)

    result = @[lossValue]
    result.add treeFlatten(newModel)
    result.add treeFlatten(newOptState)

  step.compiled = jit(fn, "rew_train_step",
    donateArgs = donateIndices(modelProto, step.donate))
  step.prepared = true

proc compileTrainStep*[M, B](loss: LossFn[M, B]; state: TrainState[M];
    runtime: Runtime = initRuntime(); donate: openArray[string] = []):
    CompiledTrainStep[M, B] =
  ## Creates a cached typed training step from a loss function.
  CompiledTrainStep[M, B](
    loss: loss,
    runtime: runtime,
    donate: @donate,
    modelProto: state.model,
    optStateProto: state.optState,
    opt: state.opt,
  )

proc call*[M, B](step: var CompiledTrainStep[M, B]; state: TrainState[M];
    batch: B): StepResult[TrainState[M]] =
  ## Runs one compiled typed training step.
  if not step.prepared:
    step.modelProto = state.model
    step.batchProto = batch
    step.optStateProto = state.optState
    step.opt = state.opt
    step.buildJit()

  let args = concatTensors([
    treeFlatten(state.model),
    treeFlatten(batch),
    treeFlatten(state.optState),
    treeFlatten(state.opt),
  ])
  let outs = step.compiled.call(args)
  let modelCount = treeLeafCount(state.model)
  let optStateCount = treeLeafCount(state.optState)
  let newModel = treeUnflatten(state.model, tensorSlice(outs, 1, modelCount))
  let newOptState = treeUnflatten(state.optState,
    tensorSlice(outs, 1 + modelCount, optStateCount))
  let newState = TrainState[M](
    model: newModel,
    opt: state.opt,
    optState: newOptState,
    step: state.step + 1,
    key: foldIn(state.key, uint64(state.step + 1)),
  )
  StepResult[TrainState[M]](
    state: newState,
    loss: outs[0],
    metrics: @[StepMetric(name: "loss", value: outs[0], progBar: true)],
  )

proc `()`*[M, B](step: var CompiledTrainStep[M, B]; state: TrainState[M];
    batch: B): StepResult[TrainState[M]] =
  ## Callable sugar for `step.call(state, batch)`.
  step.call(state, batch)
