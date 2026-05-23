## Composable optimizer protocol.
##
## The public optimizer style is open and Nim-native: concrete optimizer
## values implement `initState(opt, params)` and
## `update(opt, grads, state, params)`. Heterogeneous `chain` and
## `partition` use a small type-erased wrapper internally because Nim needs a
## single element type for runtime-length sequences.

import ../tensor
import ../pytree
import ../ops/literal
import ../ops/arith
import ../ops/factory
import ../ops/linalg
import ./sgd as sgd_impl
import ./adam
import ./adamw as adamw_impl
import ./clip

type
  EmptyOptState* = object
    ## Empty optimizer state for stateless transforms.

  ClipByGlobalNorm* = object
    ## Gradient transform that clips by global L2 norm.
    maxNorm*: float32

  Freeze* = object
    ## Gradient transform that zeros gradients whose paths match `paths`.
    paths*: PathSet

  ScheduleFn* = proc(step: int): Tensor {.closure.}
    ## Tensor-valued schedule used inside traced updates.

  ScaleBySchedule* = object
    ## Gradient transform that scales all gradient leaves by `schedule(step)`.
    schedule*: ScheduleFn

  ScaleByScheduleState* = object
    ## State for `ScaleBySchedule`.
    step*: int

  TransformKind* = enum
    ## Type-erased optimizer/gradient transform kind.
    gtkSgd
    gtkAdamW
    gtkClipByGlobalNorm
    gtkFreeze
    gtkScaleBySchedule
    gtkPartition
    gtkChain

  GradientTransform* = object
    ## Type-erased transform used only for heterogeneous chains/partitions.
    case kind*: TransformKind
    of gtkSgd:
      sgd*: Sgd
    of gtkAdamW:
      adamw*: AdamW
    of gtkClipByGlobalNorm:
      clip*: ClipByGlobalNorm
    of gtkFreeze:
      frozen*: Freeze
    of gtkScaleBySchedule:
      scaled*: ScaleBySchedule
    of gtkPartition:
      partitions*: seq[(TreePath, GradientTransform)]
    of gtkChain:
      transforms*: seq[GradientTransform]

  OptimState* = object
    ## Type-erased optimizer state matching `GradientTransform`.
    case kind*: TransformKind
    of gtkSgd, gtkClipByGlobalNorm, gtkFreeze:
      empty*: EmptyOptState
    of gtkAdamW:
      adamState*: AdamState
    of gtkScaleBySchedule:
      scheduleState*: ScaleByScheduleState
    of gtkPartition:
      partitionStates*: seq[(TreePath, OptimState)]
    of gtkChain:
      states*: seq[OptimState]

  Chain* = object
    ## Heterogeneous sequence of gradient transforms and optimizers.
    transforms*: seq[GradientTransform]

  ChainState* = object
    ## State for `Chain`.
    states*: seq[OptimState]

  Partition* = object
    ## Path-prefix partitioned optimizer policies.
    partitions*: seq[(TreePath, GradientTransform)]

  PartitionState* = object
    ## State for `Partition`.
    states*: seq[(TreePath, OptimState)]

proc sgd*(lr: Tensor): Sgd =
  ## Creates a vanilla SGD optimizer.
  initSgd(lr)

proc sgd*(lr: float32): Sgd =
  ## Creates a vanilla SGD optimizer from a host scalar learning rate.
  initSgd(scalarF32(lr))

proc adamw*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32;
    weightDecay: float32 = 0.01'f32): AdamW =
  ## Creates an AdamW optimizer.
  initAdamW(lr, beta1, beta2, eps, weightDecay)

proc adamw*(lr: float32; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32;
    weightDecay: float32 = 0.01'f32): AdamW =
  ## Creates an AdamW optimizer from a host scalar learning rate.
  initAdamW(scalarF32(lr), beta1, beta2, eps, weightDecay)

proc clipByGlobalNorm*(maxNorm: float32): ClipByGlobalNorm =
  ## Clips gradients by global norm before a later optimizer transform.
  ClipByGlobalNorm(maxNorm: maxNorm)

proc freeze*(paths: PathSet): Freeze =
  ## Zeros gradients for matching tree paths.
  Freeze(paths: paths)

proc freeze*(paths: openArray[TreePath]): Freeze =
  ## Zeros gradients for matching tree paths.
  Freeze(paths: pathSet(paths))

proc freeze*(paths: openArray[string]): Freeze =
  ## Zeros gradients for matching tree paths.
  Freeze(paths: pathSet(paths))

proc scaleBySchedule*(schedule: ScheduleFn): ScaleBySchedule =
  ## Scales gradients by the tensor returned from `schedule(step)`.
  ScaleBySchedule(schedule: schedule)

proc toGradientTransform*(tx: GradientTransform): GradientTransform =
  tx

proc toGradientTransform*(tx: Sgd): GradientTransform =
  GradientTransform(kind: gtkSgd, sgd: tx)

proc toGradientTransform*(tx: AdamW): GradientTransform =
  GradientTransform(kind: gtkAdamW, adamw: tx)

proc toGradientTransform*(tx: ClipByGlobalNorm): GradientTransform =
  GradientTransform(kind: gtkClipByGlobalNorm, clip: tx)

proc toGradientTransform*(tx: Freeze): GradientTransform =
  GradientTransform(kind: gtkFreeze, frozen: tx)

proc toGradientTransform*(tx: ScaleBySchedule): GradientTransform =
  GradientTransform(kind: gtkScaleBySchedule, scaled: tx)

proc toGradientTransform*(tx: Chain): GradientTransform =
  GradientTransform(kind: gtkChain, transforms: tx.transforms)

proc toGradientTransform*(tx: Partition): GradientTransform =
  GradientTransform(kind: gtkPartition, partitions: tx.partitions)

proc chain*(transforms: varargs[GradientTransform, toGradientTransform]):
    Chain =
  ## Applies transformations in order.
  Chain(transforms: @transforms)

proc scaleBySchedule*(base: GradientTransform; schedule: ScheduleFn): Chain =
  ## Applies `schedule` before `base`.
  chain(scaleBySchedule(schedule), base)

proc scaleBySchedule*(base: Chain; schedule: ScheduleFn): Chain =
  ## Applies `schedule` before `base`.
  chain(scaleBySchedule(schedule), base)

proc partition*(parts: varargs[(TreePath, GradientTransform)]): Partition =
  ## Records path-prefix partitions for fine-grained updates.
  Partition(partitions: @parts)

proc partition*(parts: varargs[(string, GradientTransform)]): Partition =
  ## Records path-prefix partitions for fine-grained updates.
  for (prefix, tx) in parts:
    result.partitions.add (treePath(prefix), tx)

proc partitionPart*(prefix: string; tx: GradientTransform):
    (TreePath, GradientTransform) =
  ## Builds one partition entry.
  (treePath(prefix), tx)

proc partitionPart*(prefix: string; tx: Sgd): (TreePath, GradientTransform) =
  partitionPart(prefix, toGradientTransform(tx))

proc partitionPart*(prefix: string; tx: AdamW): (TreePath, GradientTransform) =
  partitionPart(prefix, toGradientTransform(tx))

proc partitionPart*(prefix: string; tx: Chain): (TreePath, GradientTransform) =
  partitionPart(prefix, toGradientTransform(tx))

proc initState*[P](tx: Sgd; params: P): EmptyOptState =
  discard tx
  discard params
  EmptyOptState()

proc initState*[P](tx: AdamW; params: P): AdamState =
  discard tx
  initAdamState(params)

proc initState*[P](tx: ClipByGlobalNorm; params: P): EmptyOptState =
  discard tx
  discard params
  EmptyOptState()

proc initState*[P](tx: Freeze; params: P): EmptyOptState =
  discard tx
  discard params
  EmptyOptState()

proc initState*[P](tx: ScaleBySchedule; params: P): ScaleByScheduleState =
  discard tx
  discard params
  ScaleByScheduleState(step: 0)

proc initState*[P](tx: GradientTransform; params: P): OptimState =
  ## Initializes type-erased optimizer state for `params`.
  case tx.kind
  of gtkSgd:
    OptimState(kind: gtkSgd, empty: initState(tx.sgd, params))
  of gtkAdamW:
    OptimState(kind: gtkAdamW, adamState: initState(tx.adamw, params))
  of gtkClipByGlobalNorm:
    OptimState(kind: gtkClipByGlobalNorm, empty: initState(tx.clip, params))
  of gtkFreeze:
    OptimState(kind: gtkFreeze, empty: initState(tx.frozen, params))
  of gtkScaleBySchedule:
    OptimState(kind: gtkScaleBySchedule,
      scheduleState: initState(tx.scaled, params))
  of gtkPartition:
    var states: seq[(TreePath, OptimState)]
    for (prefix, childTx) in tx.partitions:
      states.add (prefix, initState(childTx, params))
    OptimState(kind: gtkPartition, partitionStates: states)
  of gtkChain:
    var states: seq[OptimState]
    for child in tx.transforms:
      states.add initState(child, params)
    OptimState(kind: gtkChain, states: states)

proc initState*[P](tx: Chain; params: P): ChainState =
  for child in tx.transforms:
    result.states.add initState(child, params)

proc initState*[P](tx: Partition; params: P): PartitionState =
  for (prefix, childTx) in tx.partitions:
    result.states.add (prefix, initState(childTx, params))

proc zeroLike(t: Tensor): Tensor =
  zerosLike(t)

proc maskNonParamGrads[P](grads: P; params: P): P =
  let paramLeaves = treeLeaves(params)
  let gradLeaves = treeLeaves(grads)
  if paramLeaves.len != gradLeaves.len:
    raise newException(PytreeError,
      "optimizer update: leaf-count mismatch (" &
        $paramLeaves.len & " vs " & $gradLeaves.len & ")")
  var mapped = newSeq[Tensor](gradLeaves.len)
  for i, leaf in gradLeaves:
    mapped[i] =
      if paramLeaves[i].kind == tlParam: leaf.tensor
      else: zeroLike(leaf.tensor)
  treeUnflatten(grads, mapped)

proc restoreNonParamLeaves[P](params: P; updated: P): P =
  let paramLeaves = treeLeaves(params)
  let updatedLeaves = treeLeaves(updated)
  if paramLeaves.len != updatedLeaves.len:
    raise newException(PytreeError,
      "optimizer update: updated leaf-count mismatch (" &
        $paramLeaves.len & " vs " & $updatedLeaves.len & ")")
  var mapped = newSeq[Tensor](updatedLeaves.len)
  for i, leaf in updatedLeaves:
    mapped[i] =
      if paramLeaves[i].kind == tlParam: leaf.tensor
      else: paramLeaves[i].tensor
  treeUnflatten(updated, mapped)

proc scaleTree[P](tree: P; scale: Tensor): P =
  let leaves = treeFlatten(tree)
  var mapped = newSeq[Tensor](leaves.len)
  for i, leaf in leaves:
    var bdims: seq[int] = @[]
    let scaleB = broadcastTo(scale, leaf.shape, bdims)
    mapped[i] = mul(scaleB, leaf)
  treeUnflatten(tree, mapped)

proc update*[P](tx: Sgd; grads: P; state: EmptyOptState;
    params: P): (P, EmptyOptState) =
  ## Applies SGD and returns `(newParams, nextState)`.
  discard state
  let maskedGrads = maskNonParamGrads(grads, params)
  let updated = tx.step(params, maskedGrads)
  (restoreNonParamLeaves(params, updated), EmptyOptState())

proc update*[P](tx: AdamW; grads: P; state: AdamState;
    params: P): (P, AdamState) =
  ## Applies AdamW and returns `(newParams, nextState)`.
  let maskedGrads = maskNonParamGrads(grads, params)
  let (updated, nextState) = tx.step(params, maskedGrads, state)
  (restoreNonParamLeaves(params, updated), nextState)

proc update*[P](tx: ClipByGlobalNorm; grads: P; state: EmptyOptState;
    params: P): (P, EmptyOptState) =
  ## Clips gradients and returns `(newGrads, nextState)`.
  discard state
  discard params
  (clipGradNorm(grads, tx.maxNorm), EmptyOptState())

proc update*[P](tx: Freeze; grads: P; state: EmptyOptState;
    params: P): (P, EmptyOptState) =
  ## Zeros gradients whose paths match `tx.paths`.
  discard state
  discard params
  (treeMapWithPath(grads,
    proc(path: TreePath; kind: TreeLeafKind; x: Tensor): Tensor =
      discard kind
      for prefix in tx.paths:
        if matchesPrefix(prefix, path):
          return zeroLike(x)
      x),
    EmptyOptState()
  )

proc update*[P](tx: ScaleBySchedule; grads: P; state: ScaleByScheduleState;
    params: P): (P, ScaleByScheduleState) =
  ## Scales gradients by `schedule(step)`.
  discard params
  let lr = tx.schedule(state.step)
  (scaleTree(grads, lr), ScaleByScheduleState(step: state.step + 1))

proc update*[P](tx: GradientTransform; grads: P; state: OptimState;
    params: P): (P, OptimState) =
  ## Applies a type-erased transform.
  case tx.kind
  of gtkSgd:
    let childState =
      if state.kind == gtkSgd: state.empty else: initState(tx.sgd, params)
    let (newParams, next) = tx.sgd.update(grads, childState, params)
    (newParams, OptimState(kind: gtkSgd, empty: next))
  of gtkAdamW:
    let childState =
      if state.kind == gtkAdamW: state.adamState
      else: initState(tx.adamw, params)
    let (newParams, next) = tx.adamw.update(grads, childState, params)
    (newParams, OptimState(kind: gtkAdamW, adamState: next))
  of gtkClipByGlobalNorm:
    let childState =
      if state.kind == gtkClipByGlobalNorm: state.empty
      else: initState(tx.clip, params)
    let (newGrads, next) = tx.clip.update(grads, childState, params)
    (newGrads, OptimState(kind: gtkClipByGlobalNorm, empty: next))
  of gtkFreeze:
    let childState =
      if state.kind == gtkFreeze: state.empty else: initState(tx.frozen, params)
    let (newGrads, next) = tx.frozen.update(grads, childState, params)
    (newGrads, OptimState(kind: gtkFreeze, empty: next))
  of gtkScaleBySchedule:
    let childState =
      if state.kind == gtkScaleBySchedule: state.scheduleState
      else: initState(tx.scaled, params)
    let (newGrads, next) = tx.scaled.update(grads, childState, params)
    (newGrads, OptimState(kind: gtkScaleBySchedule, scheduleState: next))
  of gtkPartition:
    var partitionState =
      if state.kind == gtkPartition: PartitionState(states: state.partitionStates)
      else: initState(Partition(partitions: tx.partitions), params)
    let (newParams, next) =
      Partition(partitions: tx.partitions).update(grads, partitionState, params)
    (newParams, OptimState(kind: gtkPartition, partitionStates: next.states))
  of gtkChain:
    var chainState =
      if state.kind == gtkChain: ChainState(states: state.states)
      else: initState(Chain(transforms: tx.transforms), params)
    let (newParams, next) =
      Chain(transforms: tx.transforms).update(grads, chainState, params)
    (newParams, OptimState(kind: gtkChain, states: next.states))

proc update*[P](tx: Chain; grads: P; state: ChainState;
    params: P): (P, ChainState) =
  ## Applies child transforms in order.
  var currentParams = params
  var currentGrads = grads
  for i, child in tx.transforms:
    let childState =
      if i < state.states.len: state.states[i]
      else: initState(child, currentParams)
    case child.kind
    of gtkClipByGlobalNorm, gtkFreeze, gtkScaleBySchedule:
      let (newGrads, nextState) =
        child.update(currentGrads, childState, currentParams)
      currentGrads = newGrads
      result.states.add nextState
    else:
      let (newParams, nextState) =
        child.update(currentGrads, childState, currentParams)
      currentParams = newParams
      result.states.add nextState
  (currentParams, result)

proc update*[P](tx: Partition; grads: P; state: PartitionState;
    params: P): (P, PartitionState) =
  ## Applies the first matching partition. v1 preserves existing behavior.
  if tx.partitions.len == 0:
    return (params, state)
  let (prefix, childTx) = tx.partitions[0]
  let childState =
    if state.states.len == 0: initState(childTx, params)
    else: state.states[0][1]
  let (newParams, newChildState) =
    childTx.update(grads, childState, params)
  (newParams, PartitionState(states: @[(prefix, newChildState)]))
