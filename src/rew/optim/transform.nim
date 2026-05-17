## Composable gradient transformations.
##
## This is the high-level optimizer language used by `TrainState` and the
## typed training-step API. Optimizers are plain values; all mutable optimizer
## statistics live in an explicit `OptimState` pytree-like value.

import std/strutils
import ../tensor
import ../pytree
import ../ops/linalg
import ../ops/arith
import ../ops/factory
import ./sgd as sgd_impl
import ./adam
import ./adamw as adamw_impl
import ./clip

type
  TransformKind* = enum
    gtkSgd
    gtkAdamW
    gtkClipByGlobalNorm
    gtkFreeze
    gtkChain
    gtkScaleBySchedule
    gtkPartition

  GradientTransform* = object
    ## A composable optimizer/update transformation.
    case kind*: TransformKind
    of gtkSgd:
      sgd*: Sgd
    of gtkAdamW:
      adamw*: AdamW
    of gtkClipByGlobalNorm:
      maxNorm*: float32
    of gtkFreeze:
      freezePaths*: seq[string]
    of gtkScaleBySchedule:
      schedule*: proc(step: int): Tensor {.closure.}
    of gtkPartition:
      partitions*: seq[(string, GradientTransform)]
    of gtkChain:
      transforms*: seq[GradientTransform]

  OptimState* = object
    ## Explicit optimizer state matching a `GradientTransform`.
    case kind*: TransformKind
    of gtkSgd, gtkClipByGlobalNorm, gtkFreeze:
      discard
    of gtkAdamW:
      adamState*: AdamState
    of gtkScaleBySchedule:
      step*: int
    of gtkPartition:
      partitionStates*: seq[(string, OptimState)]
    of gtkChain:
      states*: seq[OptimState]

proc sgd*(lr: Tensor): GradientTransform =
  ## Creates a vanilla SGD transform.
  GradientTransform(kind: gtkSgd, sgd: initSgd(lr))

proc adamw*(lr: Tensor; beta1: float32 = 0.9'f32;
    beta2: float32 = 0.999'f32; eps: float32 = 1e-8'f32;
    weightDecay: float32 = 0.01'f32): GradientTransform =
  ## Creates an AdamW transform.
  GradientTransform(kind: gtkAdamW,
    adamw: initAdamW(lr, beta1, beta2, eps, weightDecay))

proc clipByGlobalNorm*(maxNorm: float32): GradientTransform =
  ## Clips gradients by global norm before a later optimizer transform.
  GradientTransform(kind: gtkClipByGlobalNorm, maxNorm: maxNorm)

proc freeze*(paths: openArray[string]): GradientTransform =
  ## Zeros gradients for matching tree paths.
  GradientTransform(kind: gtkFreeze, freezePaths: @paths)

proc chain*(transforms: varargs[GradientTransform]): GradientTransform =
  ## Applies transformations in order.
  GradientTransform(kind: gtkChain, transforms: @transforms)

proc scaleBySchedule*(schedule: proc(step: int): Tensor {.closure.}):
    GradientTransform =
  ## Scales gradients by the tensor returned from `schedule(step)`.
  GradientTransform(kind: gtkScaleBySchedule, schedule: schedule)

proc scaleBySchedule*(base: GradientTransform;
    schedule: proc(step: int): Tensor {.closure.}): GradientTransform =
  ## Applies `schedule` before `base`.
  chain(scaleBySchedule(schedule), base)

proc partition*(parts: varargs[(string, GradientTransform)]):
    GradientTransform =
  ## Records path-prefix partitions for future fine-grained updates.
  ##
  ## v1 stores independent child state and applies the first matching prefix.
  GradientTransform(kind: gtkPartition, partitions: @parts)

proc initState*[P](tx: GradientTransform; params: P): OptimState =
  ## Initializes optimizer state for `params`.
  case tx.kind
  of gtkSgd:
    OptimState(kind: gtkSgd)
  of gtkAdamW:
    OptimState(kind: gtkAdamW, adamState: initAdamState(params))
  of gtkClipByGlobalNorm:
    OptimState(kind: gtkClipByGlobalNorm)
  of gtkFreeze:
    OptimState(kind: gtkFreeze)
  of gtkScaleBySchedule:
    OptimState(kind: gtkScaleBySchedule, step: 0)
  of gtkPartition:
    var states: seq[(string, OptimState)]
    for (prefix, childTx) in tx.partitions:
      states.add (prefix, initState(childTx, params))
    OptimState(kind: gtkPartition, partitionStates: states)
  of gtkChain:
    var states: seq[OptimState]
    for child in tx.transforms:
      states.add initState(child, params)
    OptimState(kind: gtkChain, states: states)

proc zeroLike(t: Tensor): Tensor =
  zerosLike(t)

proc maskNonParamGrads[P](grads: P; params: P): P =
  let paramLeaves = treeLeaves(params)
  let gradLeaves = treeLeaves(grads)
  if paramLeaves.len != gradLeaves.len:
    raise newException(PytreeError,
      "GradientTransform.update: leaf-count mismatch (" &
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
      "GradientTransform.update: updated leaf-count mismatch (" &
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

proc freezeGrads[P](grads: P; paths: openArray[string]): P =
  let leaves = treeLeaves(grads)
  var mapped = newSeq[Tensor](leaves.len)
  for i, leaf in leaves:
    var frozen = false
    for prefix in paths:
      if leaf.path == prefix or leaf.path.startsWith(prefix & "."):
        frozen = true
        break
    mapped[i] = if frozen: zeroLike(leaf.tensor) else: leaf.tensor
  treeUnflatten(grads, mapped)

proc update*[P](tx: GradientTransform; grads: P; state: OptimState;
    params: P): (P, OptimState) =
  ## Applies `tx` to `grads` and returns `(newParams, newState)`.
  case tx.kind
  of gtkSgd:
    let maskedGrads = maskNonParamGrads(grads, params)
    let updated = tx.sgd.step(params, maskedGrads)
    (restoreNonParamLeaves(params, updated), OptimState(kind: gtkSgd))
  of gtkAdamW:
    let maskedGrads = maskNonParamGrads(grads, params)
    let (newParams, newAdamState) =
      tx.adamw.step(params, maskedGrads, state.adamState)
    (restoreNonParamLeaves(params, newParams),
      OptimState(kind: gtkAdamW, adamState: newAdamState))
  of gtkClipByGlobalNorm:
    (clipGradNorm(grads, tx.maxNorm), OptimState(kind: gtkClipByGlobalNorm))
  of gtkFreeze:
    (freezeGrads(grads, tx.freezePaths), OptimState(kind: gtkFreeze))
  of gtkScaleBySchedule:
    let lr = tx.schedule(state.step)
    (scaleTree(grads, lr), OptimState(kind: gtkScaleBySchedule,
      step: state.step + 1))
  of gtkPartition:
    if tx.partitions.len == 0:
      return (params, state)
    let (prefix, childTx) = tx.partitions[0]
    let childState =
      if state.partitionStates.len == 0: initState(childTx, params)
      else: state.partitionStates[0][1]
    let (newParams, newChildState) =
      childTx.update(grads, childState, params)
    (newParams, OptimState(kind: gtkPartition,
      partitionStates: @[(prefix, newChildState)]))
  of gtkChain:
    var currentParams = params
    var currentGrads = grads
    var nextStates: seq[OptimState]
    for i, child in tx.transforms:
      let childState =
        if i < state.states.len: state.states[i]
        else: initState(child, currentParams)
      case child.kind
      of gtkClipByGlobalNorm, gtkFreeze, gtkScaleBySchedule:
        let (newGrads, newState) =
          child.update(currentGrads, childState, currentParams)
        currentGrads = newGrads
        nextStates.add newState
      else:
        let (newParams, newState) =
          child.update(currentGrads, childState, currentParams)
        currentParams = newParams
        nextStates.add newState
    (currentParams, OptimState(kind: gtkChain, states: nextStates))
