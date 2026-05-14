# Coherent High-Level API

This document is the alignment contract for REW's high-level API. If a future
change touches `src/rew.nim`, `src/rew/train/`, `src/rew/nn/`, `src/rew/optim/`,
`src/rew/data/`, checkpoints, metrics, or public examples, it should preserve
the language described here or update this document and the lint that guards it.

## One Sentence

The high-level API is **value state + typed steps**: plain Nim objects carry
model, batch, optimizer, runtime, and metric state, while pytrees turn those
values into compiled steps without exposing flat tensor plumbing.

This keeps REW readable like an eager framework, composable like a functional
JAX library, and explicit about devices, RNG, compilation, and state.

## Public Tiers

REW has three public import tiers.

- `import rew` is the user-level surface: tensors, devices, ops, `nn`, `optim`,
  `data`, `train`, checkpoints, typed transforms, and the coherent training
  vocabulary in this document.
- `import rew/xla` is the compiler surface: StableHLO, OpenXLA helpers, raw
  `jit`, lowering, HLO dumps, and other tools for inspecting generated programs.
- `import rew/dev` is the extension surface: VJP registration, primitive op
  extension hooks, eager dispatch internals, PJRT/plugin details, and manifest
  tooling.

New high-level APIs should enter through `import rew`. New compiler or
extension APIs should be placed under `rew/xla` or `rew/dev` instead of leaking
through the user-level import.

## Core Vocabulary

The public language is built from a small set of value types.

- `Param[T]` marks trainable leaves. For now `Param[Tensor]` is the important
  case. Optimizers, freezing, donation, checkpoint naming, and gradients all
  understand it as trainable state.
- `Buffer[T]` marks non-trainable state that still belongs to the model or
  training run. BatchNorm running statistics, RNG-derived layer state, and other
  device/checkpoint state should live here.
- Plain tensor fields may remain trainable for compatibility while the API is
  settling, but new stateful designs should prefer `Param[Tensor]` or
  `Buffer[Tensor]` when the distinction matters.
- Normal non-tensor fields are static config. They are part of the value object
  but not optimizer leaves.
- `Runtime` owns execution policy: device, mesh or sharding policy, precision,
  key stream, compile policy, and checkpoint policy. It replaces the old
  Workbench name in new public language.
- `TrainState[M, O, S]` owns model value, optimizer transform, optimizer state,
  global step, and key. It is the canonical state passed through training.
- `StepResult[S]` is the canonical compiled-step result. It returns the next
  state plus tensor metrics that can be observed or logged outside compiled
  regions.
- `Dataset[T]` yields arbitrary user batch pytrees. `DataSplits[T]` groups
  train/validation/test datasets.
- `Trainer` is an orchestration wrapper around the same `Runtime`, `TrainState`,
  and step functions users can run manually.

The litmus test: a user should be able to see `Runtime`, `TrainState`, a model,
a batch, and a typed step signature, then understand where every moving part
lives without following hidden mutable state.

## Pytrees

Pytrees are the shared composition mechanism. The same tree vocabulary must be
used by models, batches, optimizer state, metrics, device transfer, donation,
freezing, partitioning, serialization, and checkpoints.

Required concepts:

- `treeLeaves` returns tensor-like leaves in deterministic order.
- `treePaths` returns named paths for those leaves.
- `treeMap` transforms leaves while preserving structure.
- `treePartition` splits a value tree into matching subtrees by path or leaf
  predicate.

Path-based controls should be preferred over ad hoc parallel arrays. Donation,
freezing, checkpoint naming, optimizer partitioning, and sharding metadata should
all speak in tree paths.

## Typed Steps

High-level training should be expressed as typed functions, not as flat
`openArray[Tensor] -> seq[Tensor]` plumbing.

Default loss mode:

```nim
proc loss(model: Mnist; batch: Batch; ctx: CallCtx): Tensor =
  softmaxCrossEntropy(forward(model, batch.x), batch.y)

var state = initTrainState(initMnist(key), adamw(lr = 1e-3))
let step = compileTrainStep(loss, state, donate = paramsOf(state.model))
state = step(state, batch).state
```

Advanced custom-step mode:

```nim
proc trainStep(
    state: TrainState[Mnist, AdamW, AdamWState];
    batch: Batch;
    ctx: CallCtx
): StepResult[TrainState[Mnist, AdamW, AdamWState]] =
  let (loss, grads) = valueAndGrad(lossFn)(state.model, batch, ctx)
  let nextState = applyUpdates(state, grads)
  StepResult(state: nextState, metrics: {"loss": loss}.toTable)
```

`compileTrainStep` is the high-level compile boundary. It may internally use
raw `jit`, but public examples and trainer code should expose typed state and
batch arguments. No raw JitFn should appear in new high-level examples.

Raw `jit`, HLO dumping, and lowering are still important. They belong in
`rew/xla` and in lower-level implementation code.

## Optimizers

Optimizers are composable gradient transforms, not a closed enum.

Every optimizer or transform exposes:

```nim
proc initState(opt, params): OptState
proc update(opt, grads, state, params): (updates, nextState)
```

Built-in transforms should compose through `chain`, `clipByGlobalNorm`,
`scaleBySchedule`, `adamw`, `sgd`, `partition`, and `freeze`. This is the shape
that makes Lion, RMSprop, Adafactor, schedules, clipping, and path-specific
policies automatically usable by both manual loops and `Trainer`.

Frozen leaves should be represented by tree filters or optimizer transforms, not
by mutating model objects or hiding gradients.

## Runtime And Trainer

`Runtime` is the explicit owner of execution context. It should be passed or
stored where policy is needed, and it should avoid hidden global behavior.

`Trainer` is progressive disclosure over the same primitives:

- Simple users provide `loss(model, batch, ctx): Tensor`.
- Advanced users provide `trainStep(state, batch, ctx): StepResult`.
- Both modes use the same `TrainState`, `Runtime`, `StepResult`, optimizer
  transforms, datasets, metrics, and checkpoints.

Compiled steps must be cached by signature. Trainer must not create a fresh raw
`JitFunction` for every batch.

## Data, Metrics, And Checkpoints

Data, metrics, and checkpoints should preserve user structure instead of forcing
everything through framework-specific records.

- `Dataset[T]` yields arbitrary batch pytrees.
- `DataSplits[T]` groups train/validation/test streams.
- Data helpers such as `collate`, `prefetch`, `toDevice(runtime)`, and
  deterministic per-epoch shuffling should preserve `T`.
- `StepResult.metrics` contains tensor metrics produced in compiled code.
  Logging, host transfer, aggregation, and formatting happen outside compiled
  regions.
- Checkpoints save `TrainState` by named tree path, including model parameters,
  buffers, optimizer state, global step, key, and sharding metadata.

## Stateful Layers

Stateful layers should make state visible in the value tree.

- Trainable weights are `Param[Tensor]`.
- Non-trainable mutable state is `Buffer[Tensor]`.
- Calls that update state should return the updated model or updated state as
  part of the typed step result.
- Dropout receives randomness through `CallCtx` or explicit keys. It should not
  read a global RNG.
- BatchNorm running statistics live in buffers and are updated through the same
  typed step state path as the rest of training.

## Compatibility Policy

REW is unreleased. Backward compatibility is not a reason to preserve a muddled
high-level API. Prefer one coherent language over adapter layers.

When the old language conflicts with this document:

- Rename toward the new vocabulary (`Workbench` to `Runtime`, `DataPipe` to
  `DataSplits`, closed `OptimizerKind` to optimizer transforms).
- Update examples to the canonical typed-step shape.
- Keep raw compiler features available, but move them to the compiler tier.
- Update tests and lints at the same time as the API change.

## Change Checklist

Before merging a high-level API change, check:

- Does `import rew` expose user-level concepts without compiler/dev internals?
- Does the code use `Param[Tensor]`, `Buffer[Tensor]`, and pytrees where state
  semantics matter?
- Is training expressible as either a loss function or typed custom
  `trainStep`?
- Are optimizer features implemented as `GradientTransform` composition?
- Do data, metrics, and checkpoints preserve user pytrees and named paths?
- Is raw `JitFn` hidden from new high-level examples?
- Does `Trainer` reuse compiled steps by signature?
- Do docs, examples, tests, and `tools/check_high_level_api.nim` agree?

Minimum verification for this area:

```sh
nim c -r tools/check_high_level_api.nim
bau lint
```

For behavior changes, also update the coherent MNIST example and the coherent
API tests before running the broader suite.
