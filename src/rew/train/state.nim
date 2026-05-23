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

  TrainState*[M, O = GradientTransform, S = OptimState] = object
    ## Model, optimizer, optimizer state, PRNG key, and global step.
    model*: M
    opt*: O
    optState*: S
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

  TrainStepFn*[M, B, O = GradientTransform, S = OptimState] = proc(
    state: TrainState[M, O, S]; batch: B;
    ctx: CallCtx): StepResult[TrainState[M, O, S]] {.closure.}

  CompiledTrainStep*[M, B, O = GradientTransform, S = OptimState] = object
    ## Cached typed training step.
    loss*: LossFn[M, B]
    runtime*: Runtime
    donate*: PathSet
    compiled: JitFunction
    modelProto*: M
    batchProto*: B
    optStateProto*: S
    opt*: O
    prepared: bool

func initCallCtx*(runtime: Runtime; mode: TrainMode = tmFit): CallCtx =
  ## Creates an execution context for typed steps.
  CallCtx(runtime: runtime, mode: mode)

proc initTrainState*[M, O](model: M; opt: O;
    key: Key = initKey(0)): auto =
  ## Initializes optimizer state for `model`.
  let optState = initState(opt, model)
  TrainState[M, O, typeof(optState)](
    model: model,
    opt: opt,
    optState: optState,
    step: 0,
    key: key,
  )

proc donateIndices[M](model: M; donate: PathSet): seq[int] =
  let leaves = treeLeaves(model)
  for i, leaf in leaves:
    if donate.contains(leaf.path):
      result.add i

proc tensorSlice(args: openArray[Tensor]; start, count: int): seq[Tensor] =
  result = newSeq[Tensor](count)
  for i in 0 ..< count:
    result[i] = args[start + i]

proc concatTensors(parts: openArray[seq[Tensor]]): seq[Tensor] =
  for part in parts:
    for t in part:
      result.add t

proc buildJit[M, B, O, S](step: var CompiledTrainStep[M, B, O, S]) =
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
    let (newModel, newOptState) =
      tracedOpt.update(grads, optState, model)

    result = @[lossValue]
    result.add treeFlatten(newModel)
    result.add treeFlatten(newOptState)

  step.compiled = jit(fn, "rew_train_step",
    donateArgs = donateIndices(modelProto, step.donate))
  step.prepared = true

proc compileTrainStep*[M, B, O, S](loss: LossFn[M, B];
    state: TrainState[M, O, S]; runtime: Runtime = initRuntime();
    donate: PathSet = PathSet()):
    CompiledTrainStep[M, B, O, S] =
  ## Creates a cached typed training step from a loss function.
  CompiledTrainStep[M, B, O, S](
    loss: loss,
    runtime: runtime,
    donate: donate,
    modelProto: state.model,
    optStateProto: state.optState,
    opt: state.opt,
  )

proc compileTrainStep*[M, B, O, S](loss: LossFn[M, B];
    state: TrainState[M, O, S]; runtime: Runtime;
    donate: openArray[string]): CompiledTrainStep[M, B, O, S] =
  ## Creates a cached typed training step with string path donation.
  compileTrainStep(loss, state, runtime, pathSet(donate))

proc call*[M, B, O, S](step: var CompiledTrainStep[M, B, O, S];
    state: TrainState[M, O, S]; batch: B):
    StepResult[TrainState[M, O, S]] =
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
  let newState = TrainState[M, O, S](
    model: newModel,
    opt: state.opt,
    optState: newOptState,
    step: state.step + 1,
    key: foldIn(state.key, uint64(state.step + 1)),
  )
  StepResult[TrainState[M, O, S]](
    state: newState,
    loss: outs[0],
    metrics: @[StepMetric(name: "loss", value: outs[0], progBar: true)],
  )

proc `()`*[M, B, O, S](step: var CompiledTrainStep[M, B, O, S];
    state: TrainState[M, O, S]; batch: B):
    StepResult[TrainState[M, O, S]] =
  ## Callable sugar for `step.call(state, batch)`.
  step.call(state, batch)
