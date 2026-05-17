---
applyTo: "{src/rew.nim,src/rew/xla.nim,src/rew/dev.nim,src/rew/pytree.nim,src/rew/train/**,src/rew/data/**,src/rew/optim/**,src/rew/nn/**,examples/**,docs/high-level-api.md}"
---

# High-Level API Instructions

Before changing the user-facing training, data, optimizer, checkpoint, metric,
or example surface, read [docs/high-level-api.md](../../docs/high-level-api.md).
That document is the contract for the coherent high-level language.

## Rules

- Keep the language centered on value state plus typed steps:
  `Runtime`, `TrainState`, `Param`, `Buffer`, `StepResult`, `Dataset`,
  `DataSplits`, and `Trainer`.
- Make new state participate in pytrees. Use named paths for freezing,
  donation, partitioning, sharding metadata, checkpointing, and optimizer
  policies.
- Use `Param[Tensor]` for trainable model/layer leaves and `Buffer[Tensor]`
  for non-trainable model/layer state. Bare `Tensor` leaves may appear in
  batches, metrics, and temporary values, but they are not high-level model
  parameters.
- New high-level examples should not expose raw `JitFn` or
  `openArray[Tensor] -> seq[Tensor]` plumbing. Prefer `compileTrainStep` with a
  typed loss function or a typed custom `trainStep`.
- Raw `jit`, StableHLO, lowering, and HLO dumps belong in `rew/xla`.
- VJP registration, primitive op extension hooks, eager dispatch internals, and
  plugin target/manifest tooling belong in `rew/dev`. Raw PJRT C modules remain
  explicit specialist imports under `rew/pjrt/*`.
- Optimizers should be composable gradient transforms with `initState` and
  `update`, not new cases in a closed optimizer enum.
- `Trainer` should reuse compiled steps by signature; it must not create fresh
  raw JIT handles for every batch.

## Verification

When this layer changes, update the coherent MNIST example and coherent API
tests if behavior or vocabulary changed, then run:

```sh
nim c -r tools/check_high_level_api.nim
bau lint
```
