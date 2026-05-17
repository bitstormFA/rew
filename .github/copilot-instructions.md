# Repo-wide instructions for AI agents working on Rechenwerk (rew)

Rechenwerk is a Nim wrapper around OpenXLA/PJRT that delivers a PyTorch-feel
eager API and a JAX-style `jit` that share one StableHLO emitter and one vjp
registry. The full design lives in
[docs/architecture.md](../docs/architecture.md), and the high-level API
contract lives in [docs/high-level-api.md](../docs/high-level-api.md).

## The ten architectural invariants — DO NOT violate

1. The only C surface we link is the **PJRT C API**. StableHLO is **emitted
   from pure Nim** as textual MLIR in v1 (and as MLIR bytecode in a later
   phase, sharing the same IR). No MLIR/LLVM/XLA-C++ dependencies.
2. **Eager dispatch is per-op compile-and-cache.** `lazy(fn)` batches
   explicitly; no auto-flush trickery anywhere.
3. **`jit` is pure runtime tracing.** No macro-based `jit`, ever. In-graph
   control flow goes through `rew.cond` / `rew.whileLoop` / `rew.fori`.
4. **`Tensor` is a non-generic value `object`** with dynamic dtype + shape and
   a `ref BufferHandle` inside. ARC hooks live on `BufferHandle` only.
5. **One vjp registry** feeds both tape-based eager `backward()` and the
   functional `grad`/`vjp` transform. Reverse-mode only. **No
   `requires_grad` flag** — autodiff is opt-in by scope.
6. **Single-device per op, multi-plugin per process.** Cross-device ops raise;
   `.to()` is the only mover. Multiple PJRT plugins (e.g. CPU + CUDA) can be
   loaded concurrently via the `pjrt/registry`. `Device` uses a closed
   `Target` enum (`tCuda13`, `tCuda12`, `tRocm`, `tMetal`, `tTpu`, `tCpu`),
   not strings. `Tensor.sharding` carries replicated, partitioned, or manual
   sharding metadata; it must never trigger an implicit device move.
7. **Async-by-default execution; sync only on observation.** Donation via
   explicit `jit(fn, donateArgs = …)`; donated reuse raises
   `BufferDonatedError`. Host transfers are explicit only.
8. **High-level API = value state + typed steps.** `Runtime`, `TrainState`,
   `Param`, `Buffer`, `StepResult`, datasets, optimizers, metrics, and
   checkpoints compose through pytrees. No `Module` base class. No `ref object`
   under `src/rew/nn/` or `src/rew/optim/`. PRNG is explicit and threaded.
9. **Public API surface is tiered.** `import rew` is the high-level surface;
   `import rew/xla` is raw compiler/lowering/JIT; `import rew/dev` is
   extension internals and plugin target/manifest tooling. Raw PJRT C modules
   remain explicit specialist imports under `rew/pjrt/*`; other modules under
   `src/rew/<layer>/...` may change freely.
10. **Edit only the layer you came for.** If you find yourself modifying a
    different layer, stop and ask. Each layer has a per-file
    `.instructions.md` that auto-attaches when you touch it.

## Layer map (one folder per concern)

See the table in [docs/architecture.md](../docs/architecture.md). Reading
order for newcomers: pjrt → buffer/device/dtype → stablehlo → dispatch →
tensor/ops → autograd → transform → pytree → nn/optim → debug.

## Build & test commands

```
bau test               # debug + release + danger
bau testFast [filter]  # dev-profile tests for fast iteration
bau lint               # architectural lints
bau asan               # AddressSanitizer over the suite
bau fetch cpu          # download CPU PJRT plugin
bau fetch cuda12       # download CUDA 12 PJRT plugin
rew_fetch cpu          # installed plugin downloader, no source tree needed
bau buildPlugin cpu    # build from openxla/xla source
bau updateManifest     # re-resolve URLs + recompute SHA-256s
bau task doctor        # list devices for all available targets
nim c -r tests/all.nim # quick single-config run
```

Every PR must keep `bau test` and `bau lint` green.

## Style and engineering practices

Follow the Nim skills under `.agents/skills/` whenever they apply. The
ones used most often:

- [`nim-style-guide`](../.agents/skills/nim-style-guide/SKILL.md) — formatting,
  proc/func/template choice, control flow.
- [`nim-api-design`](../.agents/skills/nim-api-design/SKILL.md) — public
  surfaces, constructor naming (`initX`, `newX`, `toX`), accessor patterns.
- [`nim-error-handling`](../.agents/skills/nim-error-handling/SKILL.md) —
  exception vs `Option`/`bool`, `raises` contracts.
- [`nim-testing`](../.agents/skills/nim-testing/SKILL.md) — block-based tests,
  multi-config runs, ASan.

Layer-specific files reference more skills (c-bindings, c-wrappers,
ownership-hooks, code-organization, debugging, doc-comments).

## Conventions

- **Test files** live in `tests/` with the `t` prefix; `tests/all.nim`
  auto-discovers them.
- **Public ops** declare `{.rewOp.}` so the vjp-coverage lint can find them.
- **Doc comments** on every exported symbol (per
  [`nim-doc-comments`](../.agents/skills/nim-doc-comments/SKILL.md)).
- **High-level API changes** must follow
  [docs/high-level-api.md](../docs/high-level-api.md), update it when the
  language changes, and keep `tools/check_high_level_api.nim` green.
- **No raw `JitFn` plumbing** in new high-level examples. Use `TrainState`,
  `compileTrainStep`, and typed loss or custom `trainStep` functions.
- **No new `ref object`** outside the explicitly approved places (currently
  only `BufferHandle` in `src/rew/buffer.nim`).
- **No global mutable state** in user-visible APIs. Process-wide caches
  (compiled-executable cache, default device) are internal and explicit.
