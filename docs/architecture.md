# Rechenwerk (rew) — Architecture

A Nim deep learning framework on OpenXLA/PJRT. Provides a PyTorch-feel eager
API, full step-debuggability, and a JAX-style `jit` that lowers traced code to
one fused StableHLO program. Eager execution and `jit` tracing share **one**
StableHLO emitter and **one** VJP registry — every op has exactly one source of
truth.

Version 0.2.0. No external Nim dependencies. PJRT plugins are loaded via
`dlopen`; no MLIR/LLVM C++ in the build.

## Goals

- **One IR, one autodiff, two front-ends.** Eager and `jit` both emit the same
  StableHLO through the same `ShBuilder` and consult the same VJP registry.
  Adding an op or a VJP rule is a single-place change.
- **No magic.** No macro `jit`, no implicit host transfers, no implicit
  cross-device transfers, no global mutable state in user-visible APIs.
- **Backend-agnostic at runtime.** PJRT plugins are shared libraries resolved
  via a pinned manifest, lazy-fetched on first use, and verified with SHA-256.
- **Functional `nn`.** Layers and optimizers are plain value `object` types —
  no `Module` base class, no `ref object` under `nn/` or `optim/`.

## The ten locked invariants

1. **Backend.** PJRT C API for runtime; emit StableHLO from a pure-Nim builder
   as textual MLIR (bytecode path is partial, gated behind Phase 9). No
   MLIR/LLVM C++ in the build. Plugin binaries via pinned `pjrt_manifest.json`,
   lazy-fetched, SHA-256 verified.

2. **Eager dispatch.** Per-op compile + cache (key = op + shapes + dtypes +
   device + attrs). Every eager op compiles individually. Persistent executable
   cache under `REW_CACHE_DIR`. No auto-flush or batching trickery.

3. **`jit`.** Pure runtime tracing via dispatcher swap. Shape specialization
   with recompile on change. In-graph control flow only via explicit
   combinators: `cond`, `whileLoop`, `fori`. No macro `jit`, ever.

4. **`Tensor`.** Non-generic value `object`: `dtype`, `shape: seq[int]`,
   `device`, `sharding`, and a `ref BufferHandle`. ARC `=destroy` releases the
   PJRT buffer exactly once. Dynamic dtype + shape.

5. **Autograd.** One VJP rule registry per op. Two consumers: functional
   `grad`/`vjp` transform and tape-based eager `backward()`. Reverse-mode only
   in v1. No `requires_grad` flag — opt-in by scope (`grad(fn)` or
   `withGrad:`). `jit(grad(fn))` is the canonical training step.

6. **Devices.** Single-device per op, multi-plugin per process. Cross-device
   ops raise; `.to(device)` is the only mover. Multiple PJRT plugins can be
   loaded concurrently. Default target auto-detected via `nvcc`/`rocminfo`
   probe; `REW_TARGET` overrides.

7. **Buffers/async/transfers.** Async-by-default execute; sync only on
   observation (`echo`, `.toHost`, `.item`, control flow, `.wait()`). Explicit
   donation via `jit(fn, donateArgs = …)`; donated tensors flip to
   `bsDonated` and reuse raises `BufferDonatedError`. Host transfers explicit
   only. Refcounted aliasing, no implicit copy.

8. **nn API.** Functional layers as plain value `object` types built by
   `initLinear(...)`-style constructors. `proc forward(layer, …)` is canonical.
   Generic pytree (`treeFlatten`/`treeUnflatten` via `fieldPairs`) drives
   optimizer, `grad`, serialization, device transfer. Explicit splittable PRNG
   (`Key`, `split`, `foldIn`). Functional optimizers. No `Module` base class.

9. **Layout & build.** `nimble` package `rew`, umbrella `src/rew.nim`,
   internals under `src/rew/<layer>/...`. StableHLO op names follow pinned
   release tags. Four architectural lints under `nimble lint` (layer imports,
   VJP coverage, OpenXLA coverage, no-ref-in-nn).

10. **Agent instructions.** Per-layer `.github/instructions/*.instructions.md`
    with `applyTo:` globs; repo-wide `.github/copilot-instructions.md`; mirror
    at [AGENTS.md](../AGENTS.md). Edit only the layer you came for; keep
    `nimble test` and `nimble lint` green.

## Layer map

| # | Path | Responsibility |
|---|------|----------------|
| 0 | `src/rew/binaries/` | `Target` enum (tCpu/tCuda12/tCuda13/tRocm/tMetal/tTpu), manifest-driven plugin resolution, cache layout, SHA-256 verification, archive extraction, build-from-source shim. |
| 1 | `src/rew/pjrt/` | Raw C extern decls; `dlopen` + `GetPjrtApi`; multi-plugin thread-local registry; high-level `PjrtClient` RAII wrapper. Errors → `PjrtError`. |
| 2 | `src/rew/` | `BufferHandle` (ARC hooks, donation state), `Device`, `DType` (20 values), `Sharding` (Replicated/Partitioned/Manual + Mesh/PartitionSpec). |
| 3 | `src/rew/stablehlo/` | Pure-Nim StableHLO IR (`ShModule`, `ShFunction`, `ShBuilder`, 77+ op kinds), general value type descriptors, textual MLIR emitter, partial bytecode emitter, structural verifier. |
| 4 | `src/rew/` | Dispatcher swap (eager vs trace), per-op compile-and-cache, eager backend wired to PJRT client, persistent serialized executable cache. |
| 5 | `src/rew/tensor.nim`, `src/rew/ops/` | Public `Tensor` value type; 20 op files covering arith, unary, shape, reduce, linalg, conv, pool, normalization, compare, gather, scatter, sort, random, etc. All primitive ops marked `{.rewOp.}`. |
| 6 | `src/rew/autograd/` | One VJP registry; eager gradient tape; `grad`/`vjp`/`gradMode`; real VJP rule implementations. |
| 7 | `src/rew/transform/` | `jit` trace-and-cache, control-flow combinators (`cond`/`whileLoop`/`fori`), vectorizing map (`vmap`). |
| 8 | `src/rew/pytree.nim` | Generic flatten/unflatten via `fieldPairs`. No registration step. |
| 9 | `src/rew/nn/`, `src/rew/optim/`, `src/rew/rng.nim`, `src/rew/serialize.nim` | Functional layers, activations, losses, optimizers, learning rate schedulers, splittable PRNG (Threefry-2x32), NumPy `.npy` I/O. |
| 10 | `src/rew/data/` | Host-side data pipeline: lazy `DatasetFn[T]`, shuffle, batch, map, npy sources. |
| — | `src/rew/train/` | Two-tier training API: `Workbench` (user owns the loop) and `Trainer` (framework owns the loop), with callbacks, hooks, data pipe, optimizer config, checkpoint/earlystop/progress/logmonitor. |
| — | `src/rew/value.nim` | General OpenXLA value model: tokens, tuples, futures, resources, dynamic dimensions, quantized/complex/FP8 element types. |
| — | `src/rew/openxla/` | OpenXLA tool wrappers, custom-call registry, Tokamax kernel build metadata, pinned XLA/StableHLO/Shardy revisions. |

The umbrella module `src/rew.nim` is the **only** public surface. Everything
under `src/rew/*` is internal and may change freely.

## Build phases

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Bootstrap: package, runner, lints, agent files, CI | done |
| 1 | PJRT loader + buffer ownership | done |
| 2 | StableHLO emitter (IR, text, verifier) | done |
| 3 | Dispatcher + Tensor + first ops | done |
| 4 | Op coverage breadth (arith, unary, shape, reduce, linalg, literal) | done |
| 5 | Pytree + nn + optim + rng | done |
| 6 | Autograd (registry, tape, rules, grad/vjp) | done |
| 7 | jit + combinators (cond, whileLoop, fori) + donation + vmap | done |
| 8 | Serialization (.npy) + MNIST training step | done |
| 9 | Bytecode coverage, CUDA/ROCm/TPU smoke tests | in progress |
| 10 | Data pipeline (DatasetFn, lazy transforms, batching) | done |
| 11 | CNN ops (conv2d, maxPool2d, Conv2d layer) | done |
| 12 | Training module (Workbench + Trainer + callbacks) | done |
| 13 | PJRT binaries rework (multi-plugin registry, manifest, Target enum) | done |

## Op coverage

~83 primitive ops marked `{.rewOp.}` across 20 op files. Differentiable
primitives have real VJP rules; non-differentiable ops are registered with
`registerNoGrad`.

Composite ops (differentiate through primitives, no separate VJP): `relu`,
`sigmoid`, `gelu`, `silu`, `leakyRelu`, `softmax`, `logSoftmax`, `reduceMean`,
`mseLoss`, `softmaxCrossEntropy`, `binaryCrossEntropy`, `huberLoss`, `flatten`,
`standardize`, and more.

Trace-only primitives (used in VJP rules and combinators): `stablehlo.compare`,
`stablehlo.select`, `stablehlo.constant`, plus control-flow region builders for
`cond`/`while`/`reduce`.

## Data flow

```
User code
  │
  ├─ eager path ──────────────────────────────────────────┐
  │   Tensor op → dispatchEager() → EagerBackend           │
  │     → per-op StableHLO compile → PJRT execute          │
  │     → result Tensor with BufferHandle                  │
  │                                                        │
  ├─ trace path ───────────────────────────────────────────┤
  │   withTrace: dispatcher swaps to dmTrace               │
  │   Tensor op → emit into ShBuilder (SSA graph)          │
  │   traceReturn → ShModule ready for jit or grad         │
  │                                                        │
  └─ jit path ─────────────────────────────────────────────┤
      jit(fn): trace → ShModule → PJRT compile → cache     │
      on call: lookup cache by signature → PJRT execute    │
      recompile on shape change                            │
```

Both paths use the same `ShBuilder` ops and the same VJP registry.

## Extension guides

### Adding a new primitive op

1. **Define the op** in the appropriate file under `src/rew/ops/`. Mark it
   `{.rewOp.}` and follow the existing patterns — dispatch on `currentMode()`,
   call `dispatchEager(...)` or emit into the active `ShBuilder`.

2. **Add eager support** in `src/rew/eager.nim`. Add an entry to `EagerOpSpec`
   and implement the compile path if the op requires special handling beyond
   the generic `executeEagerOp`.

3. **Register a VJP or no-grad** in `src/rew/autograd/rules.nim`. Use
   `registerVjp(opName, rule)` for differentiable ops or `registerNoGrad(opName)`
   for non-differentiable ones.

4. **Add tests** in `tests/`. At minimum: numerical correctness, eager-vs-jit
   equivalence, and VJP correctness (if differentiable). Name the test file
   `t<feature>.nim`.

5. **Run `nimble lint`** — the VJP coverage lint will fail if the op is
   unaccounted for.

6. **Re-export** from `src/rew.nim` if the op lives in a new file (add the
   import + export pair).

### Adding a new nn layer

1. Create the file under `src/rew/nn/`. The layer must be a **value `object`**
   (no `ref object` — the `no_ref_in_nn` lint enforces this).

2. Provide an `init*` constructor that takes a `Key` for weight initialization,
   plus shape/dimension parameters.

3. Provide a `proc forward(layer: LayerType, x: Tensor): Tensor` or
   `proc forward(layer: LayerType, x: Tensor): seq[Tensor]`.

4. The layer's `forward` must be differentiable-through — it should only use
   primitive ops that have VJP rules.

5. Re-export from `src/rew/nn.nim` and verify the umbrella `src/rew.nim`
   picks it up.

### Adding a new optimizer

1. Create the file under `src/rew/optim/`. Must be a value `object` holding
   parameter state (no `ref object`).

2. Provide an `init*` constructor that stores hyperparameters as value fields.
   Learning rates that participate in traced updates should be 0-d `Tensor`s.

3. Provide a functional `step` that returns updated parameters, and updated
   optimizer state when needed: `proc step*(opt; params; grads; state): (P, S)`.
   Stateless optimizers may return just the updated params.

4. Re-export from `src/rew/optim.nim`.

### Adding a new transform

1. Create the file under `src/rew/transform/`. Transforms operate on traced
   computations and produce new `ShModule` graphs.

2. Follow the pattern of `jit.nim`, `control.nim`, or `vmap.nim`: accept a
   closure `proc`, swap the dispatcher to trace it, build the graph, and return
   a new callable.

3. If the transform introduces a new tracing context, use `withTrace` or
   `beginTrace`/`enterTrace`/`exitTrace` as appropriate.

### Adding a training callback

1. Create the file under `src/rew/train/callbacks/`. Implement the `on*` hooks
   your callback needs — see `src/rew/train/hooks.nim` for the full hook
   signature list.

2. Wrap the hook implementations in a `proc init*(...): Callback` that returns
   a `Callback` object.

3. Re-export from `src/rew/train.nim`.

### Adding a data source or transform

1. For a new data source: add a proc in `src/rew/data/source.nim` that returns
   a `DatasetFn[T]`.

2. For a new transform: add a proc in `src/rew/data/transform.nim` that takes a
   `DatasetFn[T]` and returns a `DatasetFn[U]`. Follow the lazy iterator
   closure pattern.

3. Re-export from `src/rew/data/data.nim`.

## Developer tooling

| Tool | Purpose |
|------|---------|
| `nimble test` | Full suite in debug, release, danger |
| `nimble lint` | Layer imports, VJP coverage, OpenXLA coverage, no-ref-in-nn |
| `nimble asan` | AddressSanitizer run |
| `nimble fetch <target>` | Download PJRT plugin |
| `rew_fetch <target>` | Installed standalone PJRT plugin downloader |
| `nimble buildPlugin <target>` | Build PJRT plugin from source |
| `nimble updateManifest` | Re-resolve URLs + recompute checksums |
| `nimble doctor` | List devices for all available targets |
| `nimble openxla <tool>` | Run OpenXLA CLI tools |

### Writing tests

Each `tests/t*.nim` file compiles to a standalone binary. The runner
`tests/all.nim` discovers and runs all `t*.nim` files in sequence; one failure
does not abort the rest. Test files can live in subdirectories.

Use `REW_TARGET=cpu` to force CPU-only tests when no accelerator is available.

### Import boundaries

- `src/rew/pjrt/` may import from `src/rew/binaries/` (for `Target`) but not
  the reverse.
- Only `buffer.nim`, `device.nim`, and `eager.nim` may import from
  `src/rew/pjrt/`.
- `src/rew/nn/` and `src/rew/optim/` must never use `ref object`.
- `src/rew.nim` is the only public API surface — everything else under
  `src/rew/` is internal.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `REW_TARGET` | Override auto-detected target (cpu, cuda12, cuda13, rocm, metal, tpu) |
| `REW_CACHE_DIR` | Cache root for plugins and serialized executables |
| `REW_ARCHIVE_URL` | Override PJRT plugin download URL |
| `REW_BUILD_XLA_DIR` | Path to openxla/xla checkout for source builds |
| `REW_EXECUTABLE_CACHE` | Enable/disable persistent executable cache |

## Out of scope (deliberately)

Macro `jit`, source-to-source autodiff, forward-mode autodiff (JVP rules are
registered but no public `jvp` transform yet), per-tensor `requires_grad`,
stateful `Module` base class, implicit host transfers, implicit cross-device
transfers, build-time fetching of upstream specs or plugins, `StaticTensor[T,S]`
(future sibling), `scan` combinator (v1 ships `cond`/`whileLoop`/`fori`), and
macro-style `lazy:` syntax. Lazy eager batching is available through the
runtime `lazy(fn)` API.
