# Rechenwerk (rew) — User Guide

Version 0.2.0

---

## 1. Introduction

Rechenwerk ("rew") is a Nim deep learning framework built on Google's
[OpenXLA/PJRT](https://github.com/openxla/xla) — the same compiler
ecosystem used by JAX, TensorFlow, and PyTorch/XLA. PJRT is the hardware-
independent runtime that compiles and executes ML programs on any supported
device (CPU, NVIDIA GPU, AMD GPU, or Google TPU) from a single source.

It gives you a PyTorch-like
eager API for interactive exploration, a JAX-style `jit` for fused program
compilation, and full step-by-step debuggability — all in pure Nim.

Key design decisions:

- **One IR, one autodiff, two front-ends.** Eager ops and `jit`-traced
  programs emit the same StableHLO (the ML compiler's intermediate
  representation — a graph of ~80 standardized operations) through the
  same code path. Every operation has exactly one source of truth for its
  forward and backward computation.
- **No magic.** No macro-based `jit`, no implicit host↔device transfers, no
  implicit cross-device transfers. What you write is what runs.
- **Functional from the ground up.** Layers and optimizers are plain value
  `object` types. There is no `Module` base class, no mutable global state.
  PRNG keys are explicit and splittable.
- **Coherent high-level API.** Public training code is converging on the
  value-state plus typed-step language in
  [docs/high-level-api.md](high-level-api.md): `Runtime`, `TrainState`,
  `Param`, `Buffer`, `StepResult`, `Dataset`, optimizer transforms, and
  `Trainer` all compose through pytrees.
- **Backend-agnostic.** PJRT plugins (CPU, CUDA 12/13, ROCm, TPU) are shared
  libraries resolved via a pinned manifest, lazy-fetched on first use, and
  verified with SHA-256. Plugins for multiple architectures can be loaded
  concurrently.
- **No external Nim dependencies.** Everything runs on Nim's stdlib.

> **Import tiers.** Use `import rew` for everyday tensor, model, data,
> optimizer, checkpoint, and training code. Use `import rew/xla` for raw
> `jit`, StableHLO/OpenXLA inspection, lowering, and HLO dumps. Use
> `import rew/dev` when adding primitive ops, VJP policies, VJP rules, or
> dispatch/eager internals.

---

## 2. Terminology Primer

This chapter explains the key technical terms used throughout the framework
and this guide. Skip to [section 3](#3-architecture-behind-the-scenes) if
you are already familiar with the XLA ecosystem.

### OpenXLA / XLA

**OpenXLA** (Accelerated Linear Algebra) is Google's open-source machine
learning compiler. It takes a high-level computation graph (in our case,
StableHLO) and compiles it into optimized code for the target hardware:
x86-64 instructions for CPU, PTX assembly for NVIDIA GPUs, AMDGPU code for
ROCm, or TPU instructions. Think of it as "LLVM for ML" — it fuses
operations, schedules memory, and generates kernels that would be tedious to
write by hand.

Every operation in rew — whether called eagerly or traced via `jit` — runs
through XLA's compiler and runtime.

### PJRT

**PJRT** (PJRT C API) is the runtime layer of OpenXLA. It is a C API that
lets the framework talk to XLA's device plugins — shared libraries (`.so` /
`.dylib` / `.dll`) that know how to allocate memory on a specific hardware
target, compile StableHLO programs, and execute them.

Rew loads PJRT plugins via `dlopen` at runtime. This means:
- You can target CPU, CUDA 12, CUDA 13, ROCm, Metal, or TPU by downloading
  the appropriate plugin (via `bau fetch <target>` from a checkout, or
  `rew_fetch <target>` after installing the package).
- Multiple plugins load concurrently — one process can use CPU and GPU
  simultaneously.
- No C++ build tools are required — the plugins are prebuilt binaries
  verified against SHA-256 checksums in a pinned manifest.

### StableHLO

**StableHLO** (Stable High-Level Operations) is an MLIR-based intermediate
representation (IR) for ML computations. It defines ~80 operations that
cover the common ML building blocks: add, multiply, convolution, reduce,
select, etc.

It serves as the **portable assembly** between rew and OpenXLA:
1. Rew emits StableHLO text (e.g., `stablehlo.add %0, %1 : tensor<2x3xf32>`).
2. OpenXLA reads it, optimizes it, and compiles it to device code.
3. The compiled result runs on the target hardware.

StableHLO is versioned and backward-compatible — programs written against
today's StableHLO version will compile with future XLA releases.

### VJP (Vector-Jacobian Product)

**VJP** (vector-Jacobian product) is the mathematical operation at the
heart of reverse-mode automatic differentiation. Given a function
**f**: ℝⁿ → ℝᵐ with Jacobian **J** (an m×n matrix of partial derivatives),
the VJP computes **vᵀ·J** — the product of a row vector **vᵀ** (the
*cotangent* or "upstream gradient") with the Jacobian.

Why VJP instead of the full Jacobian? The Jacobian of a neural network can
be enormous (millions × millions). But a VJP needs only matrix-vector
products, keeping memory linear in the number of parameters. Every backward
pass through a neural network computes a VJP — each layer receives the
gradient of the loss with respect to its output and produces the gradient
with respect to its inputs.

In rew's API:
- `vjp(fn, primals)` returns both the forward output and a `pullback`
  closure. Calling `pullback(cotangent)` computes the VJP.
- `grad(fn, primals)` is shorthand for `vjp(fn, primals).pullback(1.0)` —
  the special case where the cotangent is a scalar 1.0 (because the loss
  is a scalar).
- `valueAndGrad(fn, primals)` returns both the scalar output and gradients
  in one call.

**Related terms:**
- **Primal**: An input to the forward computation whose gradient we want.
- **Cotangent** (or seed): The vector that "flows backward" through the
  computation. For a scalar loss, the initial cotangent is 1.0.
- **Pullback**: The backward pass — a function that takes the output
  cotangent and returns the input cotangents.
- **Tape**: A record of every operation executed during the forward pass,
  stored so the backward pass can replay VJP rules in reverse order.

### SSA (Static Single Assignment)

**SSA** is a form of intermediate representation where every value is
assigned exactly once and never mutated. When rew traces a function inside
`jit`, it builds an SSA graph: each operation produces a new SSA value ID,
and downstream operations reference those IDs. This single-assignment
property makes it easy for XLA to analyze data dependencies, fuse
operations, and eliminate dead code.

In rew, a trace-mode `Tensor` carries a `traceId: ShValueId` — the SSA
identifier for that value in the graph being built.

### JIT (Just-In-Time Compilation)

**JIT compilation** in rew means: take a Nim procedure, trace it once to
produce a StableHLO graph, compile that graph, and cache the result. On
subsequent calls with matching argument shapes and types, the cached
executable runs directly — no re-tracing, no re-compilation.

This differs from Python JIT frameworks (like JAX or PyTorch 2) where the
tracing is driven by Python's interpreter and abstract interpretation of
Python bytecode. In rew, tracing is **pure runtime execution** — the Nim
procedure runs normally, but in a mode where every operation emits a graph
node rather than executing on the device.

### BufferHandle and ARC

A **BufferHandle** is a reference-counted wrapper around a raw PJRT device
buffer. When you create a tensor via `fromHostF32(d, data, shape)`, PJRT
allocates device memory, copies the host data in, and returns a pointer
that rew wraps in a `BufferHandle`.

**ARC** (Automatic Reference Counting) is Nim's memory management mode. ARC
automatically inserts retain/release calls when references are copied or go
out of scope. Unlike a garbage collector, ARC is deterministic and incurs
no stop-the-world pauses. When the last reference to a `BufferHandle` is
dropped, the destructor calls into PJRT to free the device buffer.

### Donation

In the `jit` API, **buffer donation** means handing a device buffer over to
the compiled executable rather than copying it. The executable writes its
output into the donated buffer's memory, avoiding an extra allocation. This
is key for training performance — instead of allocating new buffers for
updated parameters each step, you donate the old parameter buffers and
receive the updated parameters in their place.

After donation, the original tensor's buffer is marked `bsDonated` and any
attempt to read it (`.item`, `echo`, `.toHost`) raises
`BufferDonatedError`.

### LoRA and QLoRA

**LoRA** (Low-Rank Adaptation) is a technique for fine-tuning large
language models. Instead of updating all of a model's weight matrices,
LoRA trains small low-rank factor matrices (A and B) whose product is added
to the frozen pretrained weights. This reduces trainable parameters by
orders of magnitude.

**QLoRA** extends LoRA with 4-bit NormalFloat (NF4) quantization of the
base weights, making it possible to fine-tune billion-parameter models on a
single consumer GPU.

### Threefry

**Threefry** is a family of counter-based pseudo-random number generators
(PRNGs) designed by Salmon et al. Rew uses Threefry-2×32 with 4 rounds as
its PRNG primitive. Unlike traditional PRNGs that maintain mutable state,
Threefry is a pure function of `(key, counter) → output` — making it ideal
for `jit`-traced programs where mutable state would break reproducibility.

---

## 3. Architecture: Behind the Scenes

This section explains what happens when you call `forward(layer, x)` or
`jit(trainStep, ...)`. Understanding the pipeline helps you reason about
performance, debug errors, and extend the framework.

If terms like PJRT, StableHLO, or VJP are unfamiliar, read the
[Terminology Primer](#2-terminology-primer) first.

### 3.1 Layer Map

The framework is organized into twelve layers. Each layer depends only on the
ones below it:

```
src/rew/binaries/    Target enum, plugin manifest, cache layout, SHA-256
src/rew/pjrt/        Raw C PJRT bindings, dlopen, multi-plugin registry
src/rew/             BufferHandle (ARC), Device, DType, Sharding
src/rew/stablehlo/   Pure-Nim StableHLO IR builder (77+ op kinds), text emitter
src/rew/             Dispatcher swap (eager ↔ trace), per-op compile cache
src/rew/tensor.nim   Tensor (public value type), all operators
src/rew/autograd/    VJP registry, gradient tape, grad/vjp/valueAndGrad
src/rew/transform/   jit trace-and-cache, control-flow combinators, vmap
src/rew/nn/          Functional layers, activations, losses
src/rew/optim/       Functional optimizers, LR schedulers, gradient clipping
src/rew/data/        Lazy dataset pipeline (DatasetFn, transforms, batching)
src/rew/train/       Two-tier training API (Runtime + Trainer)
```

`src/rew.nim` is the umbrella module — it re-exports the entire public API.
Everything under `src/rew/` subdirectories is internal and may change.

### 3.2 Data Flow: Eager, Trace, and jit

When you call an operation like `add(a, b)`, the framework checks whether you
are in **eager mode** or **trace mode**:

```
User code
  │
  ├─ eager path ───────────────────────────────────────┐
  │   Tensor op → dispatchEager() → EagerBackend        │
  │     → per-op StableHLO compile → PJRT execute       │
  │     → result Tensor with BufferHandle               │
  │                                                     │
  ├─ trace path ────────────────────────────────────── │
  │   withTrace: dispatcher swaps to dmTrace            │
  │   Tensor op → emit into ShBuilder (SSA graph)       │
  │   traceReturn → ShModule ready for jit or grad      │
  │                                                     │
  └─ jit path ──────────────────────────────────────── │
      jit(fn): trace → ShModule → PJRT compile → cache  │
      on call: lookup cache by signature → PJRT execute │
      recompile on shape change                         │
```

**Eager mode** is the default. Every operation compiles a tiny StableHLO
program, executes it on the device immediately, and returns a `Tensor` with
a live buffer. This is interactive and debuggable — you can `echo` tensors,
inspect shapes, and step through code line by line.

**Trace mode** is activated by `jit(fn, ...)` or explicit `withTrace` blocks.
Inside a trace, operations don't execute — they emit nodes into a StableHLO
SSA (static single assignment) graph. When the traced function returns, the
graph is closed into a `ShModule` (a StableHLO module), verified, and compiled
by PJRT into a device executable. Subsequent calls with the same argument
shapes and types hit a cache in O(1) time.

Both paths use the **same** `ShBuilder` emission code and the **same** VJP
(autograd) rule registry. There is no eager-vs-jit code duplication.

### 3.3 How jit Works

When you call `jit(fn, "my_step", donateArgs = [0, 1])`:

1. **Wrap.** `fn` is stored in a `JitFunction` with an empty signature cache.
2. **Trace on first call.** On the first invocation, the dispatcher enters
   trace mode. Placeholder trace tensors matching the input shapes, dtypes,
   and device are created. `fn` runs, building a complete StableHLO SSA graph.
3. **Close and verify.** `traceReturn` patches output types and closes the
   function. The module is structurally verified.
4. **Build cache key.** The signature `f32:[100,784]@cpu:0|f32:[100,10]@cpu:0`
   is computed from the input dtypes, shapes, device, and sharding annotations.
5. **Compile.** The StableHLO text is sent to PJRT, which lowers it to the
   device's native instruction set (LLVM IR for CPU, PTX for CUDA, etc.).
6. **Cache.** The compiled module is stored in the cache, keyed by signature.
7. **Execute.** The compiled executable runs with the concrete input buffers.
   If `donateArgs` includes a buffer index, that buffer is handed over to PJRT
   and marked donated — any subsequent use raises `BufferDonatedError`.

On subsequent calls with the same signature, steps 1–5 are skipped. If a
different shape or dtype arrives, the function is re-traced and a new cache
entry is created (shape specialization).

### 3.4 How Autograd Works

Rew uses **reverse-mode automatic differentiation** via a functional VJP
(vector-Jacobian product) registry:

- **Tape recording.** When `grad(fn, primals)` or `vjp(fn, primals)` runs,
  a gradient tape records every operation. Each entry stores the op name,
  its primal inputs (as trace tensors), its output, and any integer
  attributes (e.g., axis for `reduceSum`).

- **VJP registry.** Every differentiable primitive op has a registered VJP
  rule — a function that computes cotangents for its inputs given the
  cotangent of its output. Non-differentiable ops are registered with
  `registerNoGrad` — using them inside `grad` raises a clean error.

- **Pullback replay.** When the pullback closure is called with a seed
  cotangent (typically `scalarF32(1.0)` for scalar losses), the tape is
  walked in reverse. Each VJP rule computes input cotangents, which are
  accumulated into a map keyed by SSA value ID.

- **No `requires_grad` flag.** Differentiation is opt-in by scope:
  `grad(fn, primals)` returns gradients for `primals` only. There is no
  per-tensor flag to toggle. This keeps the API surface small and predictable.

- **Composite operations.** Functions like `relu`, `sigmoid`, `softmax`,
  and `mseLoss` are built from primitives and differentiate through them
  automatically — no separate VJP rules needed.

The canonical training step is `jit(proc(args) = grad(loss, args))` — forward
and backward are traced together into a single fused StableHLO program.

### 3.5 Buffer Ownership and Donation

Every eager tensor wraps a `ref BufferHandle` that owns a PJRT buffer.
The handle uses Nim's ARC (Automatic Reference Counting):

- Copying a `Tensor` bumps the refcount. Both tensors point to the same
  device buffer — no data copy occurs.
- When the last `Tensor` referencing a buffer goes out of scope, ARC calls
  the buffer's destructor, releasing the PJRT buffer.
- **Donation** lets you hand over a buffer to a `jit`-compiled function.
  The function reuses the input buffer for its output, avoiding extra
  allocation. After donation, the buffer is marked `bsDonated` and any
  observation (`.item`, `echo`, `.toHost`) raises `BufferDonatedError`.

This model avoids garbage-collector pauses entirely. There is no GC involved
in tensor lifecycle — only deterministic reference counting.

### 3.6 Device Model

A `Device` is a `(target, ordinal)` pair:

- `Target` is a closed enum: `tCpu`, `tCuda12`, `tCuda13`, `tRocm`,
  `tMetal`, `tTpu`.
- `ordinal` selects a physical device (e.g., GPU 0 vs GPU 1).

Multiple PJRT plugins can be loaded concurrently in one process. However,
each individual op must run on a single device — cross-device operations raise
`DeviceError`. Use `.to(device)` to move tensors explicitly.

Auto-detection probes for `nvcc` and `rocminfo` at startup. Override with
the `REW_TARGET` environment variable:

```bash
REW_TARGET=cuda12 nim c -r my_training.nim
```

---

## 4. Getting Started

### 4.1 Prerequisites

- Nim ≥ 2.2.0
- A PJRT plugin for your target (download with `bau fetch` from a checkout,
  or `rew_fetch` after installing the package)
- For CUDA: NVIDIA driver and CUDA toolkit installed
- For ROCm: ROCm drivers installed

### 4.2 Installation

```bash
git clone https://github.com/rechenwerk/rew.git
cd rew

# Download the CPU PJRT plugin (works everywhere)
bau fetch cpu

# Or for CUDA 12
bau fetch cuda12

# After installing the package, use the installed downloader from any directory
rew_fetch cpu

# Run the test suite
bau test
```

### 4.3 Environment Variables

| Variable | Purpose |
|----------|---------|
| `REW_TARGET` | Override auto-detected target (`cpu`, `cuda12`, `cuda13`, `rocm`, `metal`, `tpu`) |
| `REW_CACHE_DIR` | Cache root for plugins and compiled executables |
| `REW_EXECUTABLE_CACHE` | Enable/disable persistent executable cache (default: enabled) |
| `REW_ARCHIVE_URL` | Override PJRT plugin download URL |
| `REW_BUILD_XLA_DIR` | Path to openxla/xla checkout for source builds |

### 4.4 Minimal Working Example

```nim
import rew

let d = initDevice(tCpu)
setDefaultDevice(d)
installEagerBackend()

var data = @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
let x = fromHostF32(d, data, [2, 2])
let y = mul(x, x)

echo "x = ", x.toHost(float32)
echo "x² = ", y.toHost(float32)
```

Compile and run:

```bash
REW_TARGET=cpu nim c -r minimal.nim
```

---

## 5. Core APIs

### 5.1 Devices and Backend Setup

Every program starts by setting up a device and installing the eager backend:

```nim
import rew

let d = initDevice(tCpu)       # or tCuda12, tCuda13, tRocm, tMetal, tTpu
setDefaultDevice(d)            # all subsequent tensor factories use d
installEagerBackend()          # wire PJRT execution for eager ops
```

Convenience constructors:

```nim
let cpu0 = cpu()               # Device(target: tCpu, ordinal: 0)
let gpu0 = cuda12(0)           # Device(target: tCuda12, ordinal: 0)
let gpu1 = cuda13(1)           # Device(target: tCuda13, ordinal: 1)
let macGpu = metal(0)          # Device(target: tMetal, ordinal: 0)
```

The `Target` enum:

```nim
type Target = enum
  tCuda13, tCuda12, tRocm, tMetal, tTpu, tCpu
```

Use `defaultDevice()` to get the resolved default (auto-detected or from
`REW_TARGET`).

### 5.2 DType

`DType` is a closed enum with 20 element types:

| DType | Description | Byte Size |
|-------|-------------|-----------|
| `dtFloat32` | IEEE-754 binary32 | 4 |
| `dtFloat64` | IEEE-754 binary64 | 8 |
| `dtFloat16` | IEEE-754 binary16 | 2 |
| `dtBFloat16` | bfloat16 | 2 |
| `dtInt8`, `dtInt16`, `dtInt32`, `dtInt64` | Signed integers | 1/2/4/8 |
| `dtUint8`, `dtUint16`, `dtUint32`, `dtUint64` | Unsigned integers | 1/2/4/8 |
| `dtBool` | 1-bit boolean | 1 |
| `dtComplex64` | Two float32 components | 8 |
| `dtComplex128` | Two float64 components | 16 |
| `dtInt4`, `dtUint4`, `dtNF4` | Packed 4-bit types | 1 |
| `dtFloat8E4M3Fn`, `dtFloat8E5M2` | Float8 formats | 1 |

Helper procs: `byteSize(dt)`, `bitWidth(dt)`, `name(dt)`, `isFloat(dt)`,
`isSignedInt(dt)`, `isUnsignedInt(dt)`, `isComplex(dt)`.

### 5.3 Tensors

A `Tensor` is a value `object` carrying:

```nim
type Tensor = object
  dtype: DType
  shape: seq[int]
  device: Device
  sharding: Sharding
  buffer: BufferHandle     # non-nil in eager, nil in trace
  traceId: ShValueId       # 0 in eager, valid SSA id in trace
```

#### Creating Tensors

From host data:

```nim
var data = @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]
let t = fromHostF32(d, data, [2, 2])   # shape [2, 2], float32
let s = scalarF32(d, 3.14'f32)         # 0-d scalar
let z = constantF32([2, 3], data)      # from seq[float32] (also works in trace)
```

The `f32ToDevice` helper (used in CNN example):

```nim
# Convenience: wraps fromHostF32
let t = f32ToDevice(d, data, [2, 2])
```

#### Inspecting Tensors

```nim
echo t.shape        # @[2, 2]
echo t.dtype        # dtFloat32
echo t.device       # cpu:0
echo t.numElements  # 4
echo t.isEager      # true
echo t.isTrace      # false (unless inside a trace)
```

#### Reading Data Back

```nim
# Scalar
let loss = item(t, float32)

# Full tensor to host
let host = t.toHost(float32)

# For custom shapes
var buf = newSeq[float32](t.numElements)
transferToHost(t.device, t.buffer, addr buf[0],
               buf.len * sizeof(float32))
```

#### Device Transfer

```nim
let gpu = cuda12(0)
let tGpu = t.to(gpu)   # explicit transfer, no implicit moving
```

Cross-device operations raise `DeviceError`:

```nim
let a = fromHostF32(cpu(), [1.0'f32], [1])
let b = fromHostF32(cuda12(), [2.0'f32], [1])
let c = add(a, b)   # DeviceError: cross-device operation
```

### 5.4 Operations

All operations are lazy-dispatched based on `currentMode()`. In eager mode
they execute immediately; in trace mode they emit StableHLO nodes.

#### Arithmetic

```nim
let c = add(a, b)          # c = a + b (element-wise)
let d = sub(a, b)          # d = a - b
let e = mul(a, b)          # e = a * b
let f = div(a, b)          # f = a / b
let g = matmul(a, b)       # g = a @ b (matrix multiply)
```

#### Unary Operations

```nim
let y = relu(x)            # max(0, x)
let y = sigmoid(x)         # 1 / (1 + exp(-x))
let y = gelu(x)            # Gaussian error linear unit
let y = silu(x)            # x * sigmoid(x)
let y = leakyRelu(x)       # max(0.01x, x)
let y = exp(x)
let y = log(x)
let y = sqrt(x)
let y = abs(x)
let y = neg(x)
let y = sin(x)
let y = cos(x)
let y = tanh(x)
```

#### Shape Operations

```nim
let y = reshape(x, [4, 5])
let y = broadcastTo(x, [3, 4, 5], broadcastDimensions)
let y = flatten(x)                 # collapse all dims except batch
let y = concatenate(@[a, b], axis = 0)
let y = slice(x, [0, 0], [1, 4], [1, 1])
let y = transpose(x, [1, 0])
```

#### Reductions

```nim
let y = reduceSum(x, axes = @[1])
let y = reduceMean(x, axes = @[0])
let y = reduceMax(x, axes = @[1])
let y = reduceMin(x, axes = @[])

# Composite (differentiable-through primitives):
let y = softmax(x, axis = 1)
let y = logSoftmax(x, axis = 1)
```

#### Comparison

```nim
let y = eq(a, b)            # a == b (element-wise, bool result)
let y = ne(a, b)            # a != b
let y = gt(a, b)            # a > b
let y = lt(a, b)            # a < b
let y = ge(a, b)            # a >= b
let y = le(a, b)            # a <= b
```

#### Convolution and Pooling

```nim
let y = conv2d(x, weight,
               windowStrides = [1, 1],
               padding = [[1, 1], [1, 1]],
               lhsDilation = [1, 1],
               rhsDilation = [1, 1],
               featureGroupCount = 1,
               batchGroupCount = 1)

let y = maxPool2d(x, [2, 2], [2, 2])
let y = avgPool2d(x, [2, 2], [2, 2])
```

#### Normalization

```nim
let y = layerNorm(x, weight, bias, epsilon = 1e-5'f32)
let y = batchNorm(x, scale, bias, mean, variance, epsilon = 1e-5'f32)
```

#### Factory

```nim
let y = constantF32([2, 3], seq[float32])   # trace-mode constant from seq
let y = zeros([2, 3], dtFloat32, d)         # zeros
let y = ones([2, 3], dtFloat32, d)          # ones
let y = eye(3, dtype = dtFloat32, device = d) # identity matrix
```

Factory operations require a device in eager mode but work in trace mode too.

#### Gather and Scatter

```nim
let y = gather(x, startIndices, offsetDims, collapsedSliceDims,
               startIndexMap, indexVectorDim)
let y = scatter(input, scatterIndices, updates, scatterDimensionNumbers)
```

### 5.5 PRNG (Splittable Random Keys)

Rew uses an explicit, functional PRNG system. Every random operation takes a
`Key` and splits it — there is no global random state.

```nim
type Key = object
  a: uint32
  b: uint32

let key = initKey(42'u64)      # create from 64-bit seed
let keys = split(key, n = 4)   # split into 4 independent child keys
let k2 = foldIn(key, 123'u64)  # mix data into a key (for step numbers)
let packed = toUint64(key)     # serialize to uint64
```

Why functional PRNG? Two reasons:

1. **Reproducibility.** The same key always produces the same random stream,
   even inside `jit`-traced functions.
2. **Parallelism.** Independent keys can be used in different threads without
   synchronization.

Example: initializing a two-layer MLP with independent seeds:

```nim
let key = initKey(0xBEEF'u64)
let keys = split(key, 2)

# Each layer gets its own key — weights are independent
var w1 = uniformF32(keys[0], 784 * 64, -bound, bound)
var w2 = uniformF32(keys[1], 64 * 10, -bound, bound)
```

### 5.6 Neural Network Layers

All layers are plain value `object` types — no `Module` base class, no
`ref object`. The pattern is:

```nim
type LayerType = object
  weight: Tensor
  bias: Tensor

proc initLayer(key: Key, ...): LayerType
proc forward(layer: LayerType, x: Tensor): Tensor
```

#### Linear (Fully Connected)

```nim
type Linear = object
  weight: Tensor   # [inFeatures, outFeatures]
  bias: Tensor     # [outFeatures]

let layer = initLinear(key, inFeatures = 784, outFeatures = 128)
let y = forward(layer, x)
# Equivalent to: y = x @ layer.weight + layer.bias
```

`initLinear` uses He-style uniform initialization: `bound = sqrt(1 / inFeatures)`.

#### Conv2d

```nim
type Conv2d = object
  weight: Tensor    # [outChannels, inChannels, kH, kW]
  bias: Tensor      # [outChannels]
  stride: array[2, int]
  padding: array[2, array[2, int]]   # [[padTop, padBottom], [padLeft, padRight]]
  dilation: array[2, int]

let conv = initConv2d(key,
  inChannels = 1, outChannels = 8, kernelSize = [3, 3],
  stride = [1, 1], padding = [[1, 1], [1, 1]], dilation = [1, 1])
let y = forward(conv, x)   # x shape: [N, H, W, C_in] (NHWC)
```

`initLinear` and `initConv2d` create trace-mode constants. For eager training
parameters that must already live on a CUDA device, initialize host arrays with
`uniformF32` and transfer them with `fromHostF32` or `f32ToDevice`; the MNIST
CNN tutorial below shows this pattern.

#### Embedding

```nim
let emb = initEmbedding(key, numEmbeddings = 10000, embeddingDim = 256)
let y = forward(emb, indices)   # indices: int32 tensor [N]
```

#### Normalization Layers

```nim
# Layer normalization
let ln = initLayerNorm(normalizedShape = 256, eps = 1e-5'f32)
let y = forward(ln, x)

# Batch normalization
let bn = initBatchNorm(numFeatures = 64, eps = 1e-5'f32)
let y = forward(bn, x)

# RMS normalization
let rms = initRMSNorm(normalizedShape = 256, eps = 1e-6'f32)
let y = forward(rms, x)
```

#### Dropout

```nim
let drop = initDropout(prob = 0.5)
let y = forward(drop, x, key)   # key controls which elements are dropped
```

#### Sequential

```nim
type Sequential = object
  layers: seq[...]

let model = initSequential(@[
  initLinear(key1, 784, 128),
  initLinear(key2, 128, 10),
])
let y = forward(model, x)
```

Sequential supports `treeFlatten`/`treeUnflatten` for easy integration with
optimizers and `grad`.

#### Attention Layers

```nim
# Multi-head attention
let mha = initMultiHeadAttention(key, embedDim = 512, numHeads = 8,
                                  dropout = 0.1)
let y = forward(mha, query, key, value)

# Grouped query attention
let gqa = initGroupedQueryAttention(key, embedDim = 512,
    numHeads = 8, numKvHeads = 2, dropout = 0.1)
let y = forward(gqa, query, keyVal)

# RoPE (rotary position embedding)
let rope = initRoPE(dim = 64)
let y = forward(rope, x, offset = 0)
```

#### LoRA (Low-Rank Adaptation)

```nim
type QloraLinear = object
  dequantizedWeight: Buffer[Tensor] # frozen base weight
  bias: Buffer[Tensor]              # frozen bias
  A: Param[Tensor]                  # trainable, shape [rank, inFeatures]
  B: Param[Tensor]                  # trainable, shape [outFeatures, rank]
  rank: int
  alpha: float32
  scaling: float32            # alpha / rank

proc forward(layer: QloraLinear, x: Tensor): Tensor
```

`QloraLinear` uses QLoRA-style 4-bit NF4 quantization behind the scenes.
The adapter matrices `A` and `B` are trainable while the base weight stays
frozen.

#### Loss Functions

All losses are composite operations built from primitives — they
differentiate through automatically:

```nim
let loss = softmaxCrossEntropy(logits, labels)    # multi-class
let loss = mseLoss(pred, target)                  # mean squared error
let loss = huberLoss(pred, target)                # huber loss
let loss = binaryCrossEntropy(pred, target)       # binary cross-entropy
let loss = smoothL1Loss(pred, target)             # smooth L1
```

### 5.7 Autograd

Rew provides three autograd transforms. All run inside an active trace
(either explicit `withTrace` or implicit via `jit`).

#### grad — Scalar-Output Gradient

```nim
proc grad(fn: proc(args: openArray[Tensor]): Tensor,
          primals: openArray[Tensor]): seq[Tensor]
```

Returns one gradient tensor per primal input. The function must return a
scalar (0-d tensor). Equivalent to `vjp(fn, primals).pullback(scalarF32(1'f32))`.

```nim
let lossFn = proc(params: openArray[Tensor]): Tensor =
  let l1 = Linear(weight: params[0], bias: params[1])
  softmaxCrossEntropy(forward(l1, x), y)

let grads = grad(lossFn, [layer.weight, layer.bias])
# grads[0] = dLoss/dWeight, grads[1] = dLoss/dBias
```

#### vjp — Vector-Jacobian Product (Pullback)

```nim
type VjpResult = object
  output: Tensor
  pullback: proc(cotangent: Tensor): seq[Tensor]

proc vjp(fn: proc(args: openArray[Tensor]): Tensor,
         primals: openArray[Tensor]): VjpResult
```

Returns both the forward output and a pullback closure. The pullback
accepts a cotangent (same shape as output) and returns input cotangents.

```nim
let vr = vjp(lossFn, [layer.weight, layer.bias])
let loss = vr.output
let grads = vr.pullback(scalarF32(1'f32))   # seed cotangent = 1.0
# Use grads for a custom optimizer update
```

#### valueAndGrad — Value + Gradients

```nim
type ValueAndGradResult = object
  value: Tensor      # scalar forward result
  grads: seq[Tensor] # one gradient per primal

proc valueAndGrad(fn: proc(args: openArray[Tensor]): Tensor,
                  primals: openArray[Tensor]): ValueAndGradResult
```

Returns both the scalar output and gradients in one call:

```nim
let vg = valueAndGrad(lossFn, [layer.weight, layer.bias])
echo "loss = ", item(vg.value, float32)
# vg.grads[0], vg.grads[1] are the gradients
```

### 5.8 jit Compilation

Raw `jit` is a compiler-tier API:

```nim
import rew
import rew/xla
```

The `jit` transform traces a function and cache-compiles it for a specific
input signature:

```nim
type JitFn = proc(args: openArray[Tensor]): seq[Tensor]

proc jit(fn: JitFn, funcName = "jit_fn",
         donateArgs: openArray[int] = []): JitFunction

proc call(jit: JitFunction, args: openArray[Tensor]): seq[Tensor]
```

#### Writing a jit-Compiled Training Step

The canonical pattern: trace forward + backward together:

```nim
let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
  # args = [w1, b1, w2, b2, x, y, lr]
  let lossFn = proc(p: openArray[Tensor]): Tensor =
    let l1 = Linear(weight: p[0], bias: p[1])
    let l2 = Linear(weight: p[2], bias: p[3])
    softmaxCrossEntropy(forward(l2, relu(forward(l1, args[4]))), args[5])

  let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
  let grads = vr.pullback(scalarF32(1'f32))

  # SGD update: param = param - lr * grad
  proc upd(p, g: Tensor): Tensor =
    let lrB = broadcastTo(args[6], p.shape, @[])
    sub(p, mul(lrB, g))

  @[vr.output,           # loss
    upd(args[0], grads[0]),   # w1'
    upd(args[1], grads[1]),   # b1'
    upd(args[2], grads[2]),   # w2'
    upd(args[3], grads[3])]   # b2'

let trainJ = jit(trainFn, "mnist_train_step",
                 donateArgs = [0, 1, 2, 3])

# In the training loop:
let outs = trainJ.call([layer1.weight, layer1.bias,
                        layer2.weight, layer2.bias,
                        x, y, lr])
layer1 = Linear(weight: outs[1], bias: outs[2])
layer2 = Linear(weight: outs[3], bias: outs[4])
echo "loss = ", item(outs[0], float32)
```

#### Buffer Donation

When `donateArgs = [0, 1]`, the input buffers at positions 0 and 1 are
donated to the compiled executable. PJRT may reuse their memory for outputs.
After a call, accessing those input tensors raises `BufferDonatedError`:

```nim
let outs = trainJ.call([w1, b1, w2, b2, x, y, lr])
# w1.buffer is now donated — don't use it!
let newW1 = outs[1]   # use the output tensors instead
```

#### Debugging jit Programs

```nim
# Inspect the StableHLO text
echo trainJ.text([w1, b1, w2, b2, x, y, lr])

# Dump to file
discard trainJ.dumpHlo([w1, b1, w2, b2, x, y, lr], "my_fn.hlo")

# Check cache size
echo trainJ.cacheSize()    # number of cached signatures
trainJ.clearCache()        # force re-trace on next call
```

### 5.9 Optimizers

All optimizers are functional — `step` returns updated parameters (and
optionally updated optimizer state). Nothing is mutated in-place.

#### SGD

```nim
type Sgd = object
  lr: Tensor   # 0-d float32 scalar

proc initSgd(lr: Tensor): Sgd
proc step[P](opt: Sgd, params: P, grads: P): P
```

```nim
let lr = scalarF32(d, 0.01'f32)
let opt = initSgd(lr)

# params and grads are pytree-compatible types
model = opt.step(model, grads)
```

#### Adam

```nim
type Adam = object
  lr: Tensor
  beta1, beta2: float32
  eps: float32

type AdamState = object
  m, v: seq[Tensor]   # first and second moment estimates
  t: int              # step counter

proc initAdam(lr: Tensor,
              beta1 = 0.9'f32, beta2 = 0.999'f32,
              eps = 1e-8'f32): Adam

proc initAdamState[P](model: P): AdamState

# step returns both updated params and updated state
proc step[P](opt: Adam, params: P, grads: P, state: AdamState):
  tuple[params: P, state: AdamState]
```

```nim
let opt = initAdam(lr = scalarF32(d, 1e-3'f32))
var state = initAdamState(model)

(model, state) = opt.step(model, grads, state)
```

#### AdamW

Similar to Adam but with decoupled weight decay:

```nim
type AdamW = object
  lr, weightDecay: Tensor
  beta1, beta2, eps: float32
```

#### Other Optimizers

| Optimizer | Type | Stateful? |
|-----------|------|-----------|
| `Sgd` | Vanilla SGD | No |
| `MomentumSgd` | SGD with momentum | Yes (`MomentumState`) |
| `Adam` | Adam | Yes (`AdamState`) |
| `AdamW` | AdamW (decoupled weight decay) | Yes (`AdamState`) |
| `Rmsprop` | RMSprop | Yes (`RmspropState`) |
| `NAdam` | Nesterov Adam | Yes (`NAdamState`) |
| `Lookahead` | Lookahead wrapper | Yes (`LookaheadState`) |

#### Gradient Clipping

```nim
let grads = clipGradNorm(grads, maxNorm = 1.0'f32)
let grads = clipGradValue(grads, maxValue = 1.0'f32)
```

#### Learning Rate Schedulers

```nim
let sched = initStepLR(stepSize = 10, gamma = 0.1'f32)
let sched = initCosineAnnealingLR(tMax = 100, etaMin = 1e-6'f32)
let sched = initReduceOnPlateau(factor = 0.5'f32, patience = 5)

# Apply scheduler to a 0-d lr tensor
let newLr = step(sched, currentLr, epoch)
```

### 5.10 Dataset Pipeline

The Dataset Pipeline provides lazy, chainable datasets with a tf.data-style API.

#### Core Types

```nim
type DatasetFn[T] = proc(): iterator(): T
type Dataset[T] = object
  source: DatasetFn[T]
```

#### Creating Datasets

```nim
# From an in-memory sequence
let ds = fromSeq(@[1, 2, 3, 4, 5])

# From a range
let ds = fromRange(0, 1000, step = 1)

# From NumPy files (MNIST, etc.)
let ds = fromNpy("train_images.npy", "train_labels.npy")
```

#### Transforms (Chainable)

```nim
# Reshape each sample
let ds = fromNpy("images.npy", "labels.npy")
  .map(proc(s: Sample): Sample =
    Sample(data: s.data, dataShape: @[28, 28, 1], label: s.label))

# Shuffle with a buffer
let ds = ds.shuffle(key, bufferSize = 10000)

# Batch
let ds = ds.batch(batchSize = 32)

# Prefetch with background thread (requires --threads:on)
let ds = ds.prefetch(bufferSize = 4)
```

#### Sample and Batch Types

```nim
type Sample = object
  data: seq[float32]
  dataShape: seq[int]
  label: int

type Batch = object
  data: seq[float32]
  dataShape: seq[int]        # e.g., [batchSize, 28, 28, 1]
  labels: seq[int]
  batchSize: int
```

#### Iterating and Feeding Models

```nim
let pipeline = fromNpy("train_images.npy", "train_labels.npy")
  .map(toNHWC)
  .shuffle(key, bufferSize = 10000)
  .batch(32)

let iter = pipeline.source()
while true:
  let batchSamples = iter()
  if finished(iter): break

  let b = collate(batchSamples)
  let (x, y) = toTensors(d, b, numClasses = 10)

  # x shape: [32, 28, 28, 1], y shape: [32, 10] (one-hot)
  let logits = forward(model, x)
  ...
```

Alternative direct iteration (single epoch):

```nim
for sample in pipeline:
  echo sample.dataShape
```

### 5.11 Pytree (treeFlatten / treeUnflatten / treeMap)

The pytree module provides automatic flattening of any `object`, `tuple`,
`seq`, or `array` containing `Tensor` leaves:

```nim
let leaves = treeFlatten(model)
# returns seq[Tensor] in field-declaration order

let restored = treeUnflatten(model, leaves)
# returns a copy with leaves replaced

# Apply a function to every leaf
let doubled = treeMap(model, proc(t: Tensor): Tensor = mul(t, scalarF32(2'f32)))
```

This powers the optimizer `step` implementations, `grad` over complex
parameter containers, serialization, and device transfer — without
requiring any registration step.

```nim
# Example: optimizers use treeFlatten/treeUnflatten internally
proc step[P](opt: Sgd, params: P, grads: P): P =
  let pl = treeFlatten(params)
  let gl = treeFlatten(grads)
  var updated = newSeq[Tensor](pl.len)
  for i in 0 ..< pl.len:
    updated[i] = sub(pl[i], mul(broadcastTo(opt.lr, pl[i].shape, @[]), gl[i]))
  treeUnflatten(params, updated)
```

### 5.12 Serialization

#### NumPy .npy I/O

```nim
let arr = loadNpy("train_images.npy")   # returns NpyArray
# arr.dtype, arr.shape, arr.data (seq[byte])

saveNpy("output.npy", arr)              # write back
```

Use `loadNpy` for loading MNIST-style datasets. The reader supports
`uint8`, `int64`, `float32`, and common dtypes.

#### Checkpointing with Runtime

```nim
# Save model state
runtime.save("checkpoints/epoch_5", (model, opt, optState))

# Load model state
let prototype = (model, opt, optState)
let (loadedModel, loadedOpt, loadedState) = runtime.load(
  "checkpoints/epoch_5", prototype)
```

#### Safetensors

```nim
import rew/safetensors

let tensors = loadSafeTensors("model.safetensors")
# tensors is Table[string, NpyArray]
```

### 5.13 Training API

Rew provides two training layers:

| Layer | Type | You own... | Use when... |
|-------|------|-----------|-------------|
| Runtime | `Runtime` | The loop and compile boundary | You want full control |
| Trainer | `Trainer` | A typed loss or typed custom step | You want the framework to manage the loop |

#### Runtime

`Runtime` is the explicit execution context. Use it when you want to own the
loop while still using typed training state and compiled steps.

```nim
import rew
import rew/train

let runtime = initRuntime(akCpu)
seedEverything(42)

var state = initTrainState(initModel(runtime.nextKey()), sgd(lr))
var step = compileTrainStep(loss, state, runtime,
  donate = paramsOf(state.model))

for epoch in 0 ..< 10:
  for batch in trainDs:
    state = step(state, batch).state

runtime.save("/tmp/checkpoint", state)
```

Runtime provides:

```nim
proc initRuntime(accelerator = akAuto, devices = 1, precision = prFloat32): Runtime
proc setup[T](runtime: var Runtime, model: T): T
proc setup[T](runtime: var Runtime, ds: Dataset[T]): Dataset[T]
proc computeGrads[T](runtime: Runtime,
    fn: proc(args: openArray[Tensor]): Tensor, params: T): T
proc allReduce(runtime: Runtime, t: Tensor): Tensor
proc isGlobalZero(runtime: Runtime): bool
proc nextKey(runtime: var Runtime): Key
proc save[T](runtime: Runtime, path: string, state: T)
proc load[T](runtime: Runtime, path: string, prototype: T): T
proc seedEverything(seed: int)
```

#### Trainer

The Trainer owns iteration, compiled-step caching, optimizer state, validation,
metrics, and callbacks. You provide:

1. A `TrainState[M]`
2. A typed `loss(model, batch, ctx): Tensor` or typed custom step
3. `DataSplits[B]` with train/val/test datasets
4. Optional callbacks

```nim
type MnistBatch = object
  x: Tensor
  y: Tensor

proc loss(model: Mnist; batch: MnistBatch; ctx: CallCtx): Tensor =
  discard ctx
  softmaxCrossEntropy(forward(model, batch.x), batch.y)

var state = initTrainState(initMnist(key), adamw(lr = scalarF32(d, 1e-3)))
let data = initDataSplits(trainDs, val = some(valDs))
var trainer = initTrainer(maxEpochs = 10, accelerator = akCpu)
trainer.callbacks = @[
  initCheckpoint(monitor = "val/loss").toCallback(),
  initEarlyStopping(monitor = "val/loss", patience = 3).toCallback(),
]
trainer.fit(state, data, loss)
```

Custom logic lives in a typed step when a scalar loss is not enough:

```nim
let customStep: TrainStepFn[Mnist, MnistBatch] =
  proc(state: TrainState[Mnist]; batch: MnistBatch;
      ctx: CallCtx): StepResult[TrainState[Mnist]] =
    discard ctx
    compiled.call(state, batch)

trainer.fit(state, data, customStep)
```

**Trainer fields:**

```nim
type Trainer = object
  maxEpochs: int
  maxSteps: Option[int]
  accelerator: Accelerator
  devices: int
  precision: Precision
  logEvery: int
  valInterval: Option[int]
  donateParams: bool
  callbacks: seq[Callback]
```

**Built-in callbacks:**

| Callback | Purpose |
|----------|---------|
| `Checkpoint` | Save/restore model checkpoints (top-K, last, monitored metric) |
| `EarlyStopping` | Stop training when monitored metric stops improving |
| `ProgressBar` | Terminal progress display for `Batch`-based datasets |
| `LogMonitor` | Print step-level metrics for `Batch`-based datasets |

### 5.14 Tutorial: Typed MNIST with Trainer

The compact typed Trainer example lives in `examples/mnist_trainer.nim`. It
builds a value-model, wraps trainable leaves in `Param[Tensor]`, creates a
`TrainState`, prepares `DataSplits`, and calls `trainer.fit(state, data, loss)`.

The important shape is:

```nim
type
  MnistMlp = object
    fc1: Linear
    fc2: Linear

  MnistBatch = object
    x: Tensor
    y: Tensor

proc loss(model: MnistMlp; batch: MnistBatch; ctx: CallCtx): Tensor =
  discard ctx
  softmaxCrossEntropy(forward(model, batch.x), batch.y)

let lr = scalarF32(d, 0.05'f32)
var state = initTrainState(initMnistMlp(d, initKey(42)), sgd(lr))
let data = initDataSplits(fromSeq(batches))

var trainer = initTrainer(maxEpochs = 5, accelerator = akCpu)
trainer.donateParams = true
trainer.fit(state, data, loss)
```

Run it with:

```bash
nim c -r examples/mnist_trainer.nim
```

Use `rew/xla` directly when you need raw lowering, HLO dumps, or explicit
`JitFunction` handles. High-level training examples should stay on
`TrainState`, `compileTrainStep`, and `Trainer.fit(state, data, ...)`.

### 5.15 Advanced Features

#### Control Flow Inside jit

Since `jit` traces a static computation graph, Python-level `if`/`while`
cannot control in-graph branching. Use explicit combinators instead:

```nim
# Conditional (trace mode only)
let result = cond(predicate, proc(): Tensor = thenBranch(x),
                               proc(): Tensor = elseBranch(x))

# Multi-output conditional
let results = condN(pred, proc(): seq[Tensor] = ..., proc(): seq[Tensor] = ...)

# While loop
let outputs = whileLoop(initVals,
  cond = proc(args: openArray[Tensor]): Tensor = ...,  # scalar bool
  body = proc(args: openArray[Tensor]): seq[Tensor] = ...)

# Integer-counter loop (for i in low ..< high)
let outputs = fori(low = 0, high = n, initVals,
  body = proc(i: Tensor, args: openArray[Tensor]): seq[Tensor] = ...)
```

Tape recording is paused inside control-flow bodies, so `grad` through
`cond`/`whileLoop`/`fori` is not yet supported in v0.2.0.

#### vmap — Automatic Batching

```nim
proc vmap(fn: proc(args: openArray[Tensor]): seq[Tensor],
          inAxes: openArray[int] = @[]):
  proc(args: openArray[Tensor]): seq[Tensor]
```

Wraps `fn` so it operates over a leading batch dimension via `fori`:

```nim
# fn expects scalar inputs
let fn = proc(args: openArray[Tensor]): seq[Tensor] =
  @[add(args[0], args[1])]

# batchFn expects inputs with an extra batch axis
let batchFn = vmap(fn, inAxes = @[0, 0])

let a = fromHostF32(d, [1'f32, 2'f32, 3'f32], [3])
let b = fromHostF32(d, [10'f32, 20'f32, 30'f32], [3])
let c = batchFn([a, b])   # equivalent to: for i in 0..<3: fn(a[i], b[i])
```

#### Sharding

Sharding annotations are metadata — they never trigger implicit transfers:

```nim
# Replicated (default)
let t = replicate(t)

# Partitioned across a mesh
let mesh = initMesh("dp_mp", ["data", "model"], [2, 4])
let spec = initPartitionSpec(["data", "model"])
let sharded = shard(t, mesh, spec)

# Manual sharding
let manual = manualShard(t, mesh, initPartitionSpec(["data"]))
```

Validation: `validateSharding(sharding, tensorRank)` checks that partition
specs match tensor rank and reference only axes in the mesh.

#### Hugging Face Integration

```nim
# Download a model snapshot (config, tokenizer, weights)
let modelDir = hfDownloadModel("google/gemma-4-E4B-it")

# Download a dataset snapshot
let datasetDir = hfDownloadDataset("medalpaca/medical_meadow_medical_flashcards")

# Load a tokenizer
let tokenizer = loadGemma4Tokenizer(modelDir)
let ids = tokenizer.encode("Hello, world!")
let text = tokenizer.decode(ids)
```

#### ONNX and TFLite Import

```nim
import rew/onnx
let model = loadOnnx("model.onnx")

import rew/tflite
let model = loadTflite("model.tflite")
```

#### Quantization

```nim
import rew/quantize

let quantized = quantize(model, dtype = dtNF4, groupSize = 64)
```

#### GGUF Format

```nim
import rew/gguf

let tensors = loadGGUF("model.gguf")
```

---

## 6. API Reference

This section lists all public types, procs, and templates organized by module.
All APIs are accessed through `import rew` unless otherwise noted.

### 6.1 Binaries, Device, and Target

```nim
type
  Target = enum tCuda13, tCuda12, tRocm, tMetal, tTpu, tCpu
  Device = object
    target: Target
    ordinal: int
  DeviceError = object of CatchableError

func cpu(ordinal = 0): Device
func cuda12(ordinal = 0): Device
func cuda13(ordinal = 0): Device
func rocm(ordinal = 0): Device
func metal(ordinal = 0): Device
func tpu(ordinal = 0): Device
func initDevice(target: Target, ordinal = 0): Device
func targetName(target: Target): string
func parseTarget(s: string): Target
proc parseDevice(s: string): Device
proc defaultDevice(): Device
proc setDefaultDevice(d: Device)
proc defaultTarget(): Target
proc setDefaultTarget(t: Target)
proc resetDefaultTarget()
```

### 6.2 DType

```nim
type DType = enum
  dtBool, dtInt4, dtInt8, dtInt16, dtInt32, dtInt64,
  dtUint4, dtUint8, dtUint16, dtUint32, dtUint64,
  dtFloat16, dtBFloat16, dtFloat32, dtFloat64,
  dtComplex64, dtComplex128,
  dtNF4, dtFloat8E4M3Fn, dtFloat8E5M2

func byteSize(dt: DType): int
func bitWidth(dt: DType): int
func name(dt: DType): string
func isFloat(dt: DType): bool
func isSignedInt(dt: DType): bool
func isUnsignedInt(dt: DType): bool
func isComplex(dt: DType): bool
func complexPartDType(dt: DType): DType
func complexDType(part: DType): DType
func dtypeOf(t: typedesc): DType
```

### 6.3 Tensor

```nim
type Tensor = object
  dtype: DType
  shape: seq[int]
  device: Device
  sharding: Sharding
  buffer: BufferHandle
  traceId: ShValueId

func numElements(t: Tensor): int
func isEager(t: Tensor): bool
func isTrace(t: Tensor): bool
func tensorTypeOf(t: Tensor): ShTensorType
func valueTypeOf(t: Tensor): ValueType

proc initEagerTensor(buffer: BufferHandle, dtype: DType,
                     shape: openArray[int], device: Device,
                     sharding = initReplicated()): Tensor
proc initTraceTensor(id: ShValueId, dtype: DType,
                     shape: openArray[int], device: Device,
                     sharding = initReplicated()): Tensor

proc requireEager(t: Tensor, opName: string)
proc requireTrace(t: Tensor, opName: string)
proc requireSameDevice(a, b: Tensor, opName: string)
proc requireSameMode(a, b: Tensor, opName: string)

func withSharding(t: Tensor, sharding: Sharding): Tensor
func shard(t: Tensor, mesh: Mesh, spec: PartitionSpec): Tensor
func manualShard(t: Tensor, mesh: Mesh, spec: PartitionSpec): Tensor
func replicate(t: Tensor): Tensor

proc to(t: Tensor, device: Device): Tensor
```

### 6.4 Tensor Creation & I/O

```nim
proc fromHostF32(d: Device, data: openArray[float32],
                 shape: openArray[int]): Tensor
proc fromHost[T](d: Device, data: openArray[T],
                 shape: openArray[int]): Tensor
proc fromHost[T](data: openArray[T], shape: openArray[int],
                 device: Device = defaultDevice()): Tensor
proc scalarF32(d: Device, value: float32): Tensor
proc f32ToDevice(d: Device, data: var seq[float32],
                 shape: openArray[int]): Tensor
proc constantF32(shape: openArray[int],
                 data: openArray[float32]): Tensor

proc toHost[T](t: Tensor, valueType: typedesc[T]): seq[T]
proc item(t: Tensor, T: typedesc): T
proc transferToHost(device: Device, buffer: BufferHandle,
                    dst: pointer, byteLen: int)
proc transferToDevice(device: Device, src: pointer, dtype: DType,
                      dims: openArray[int64],
                      sizeBytes: int = 0): BufferHandle
```

### 6.5 Operations

**Arithmetic:** `add`, `sub`, `mul`, `div`, `neg`, `abs`

**Unary:** `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`, `erf`,
`relu`, `sigmoid`, `gelu`, `silu`, `leakyRelu`, `softplus`, `softsign`,
`selu`, `elu`, `sign`, `ceil`, `floor`, `round`, `isFinite`

**Shape:** `reshape`, `broadcastTo`, `flatten`, `concatenate`, `slice`,
`transpose`, `pad`, `tile`, `reverse`

**Reduce:** `reduceSum`, `reduceMax`, `reduceMin`, `reduceProduct`,
`reduceAnd`, `reduceOr`, `reduceXor`, `reduceWindow`, `selectAndScatter`

**Linear Algebra:** `matmul`, `dot`, `batchNorm`, `layerNorm`

**Convolution:** `conv2d`

**Pooling:** `maxPool2d`, `avgPool2d`

**Normalization:** `softmax`, `logSoftmax`

**Comparison:** `eq`, `ne`, `gt`, `lt`, `ge`, `le`

**Gather/Scatter:** `gather`, `scatter`

**Ternary:** `select`

**Factory:** `constantF32`, `uniformF32`, `normalF32`, `zeros`, `ones`, `eye`

**Random:** `uniformF32(key, n, low, high)`, `normalF32(key, n, mean, std)`

### 6.6 PRNG

```nim
type Key = object
  a: uint32
  b: uint32

func initKey(seed: uint64): Key
func split(k: Key, n = 2): seq[Key]
func foldIn(k: Key, data: uint64): Key
func toUint64(k: Key): uint64
func `==`(a, b: Key): bool
func `$`(k: Key): string
```

### 6.7 Neural Network Layers

```nim
# Linear
type Linear = object
  weight, bias: Tensor
proc initLinear(key: Key, inFeatures, outFeatures: int): Linear
proc forward(layer: Linear, x: Tensor): Tensor

# Conv2d
type Conv2d = object
  weight, bias: Tensor
  stride: array[2, int]
  padding: array[2, array[2, int]]
  dilation: array[2, int]
proc initConv2d(key: Key, inChannels, outChannels: int,
                kernelSize: array[2, int],
                stride = [1, 1], padding = [[0, 0], [0, 0]],
                dilation = [1, 1]): Conv2d
proc forward(layer: Conv2d, x: Tensor): Tensor

# Embedding
type Embedding = object
  weight: Tensor
proc initEmbedding(key: Key, numEmbeddings, embeddingDim: int): Embedding
proc forward(layer: Embedding, indices: Tensor): Tensor

# Normalization
type LayerNorm = object
  weight, bias: Tensor
  eps: float32
type BatchNorm = object
  scale, bias, runningMean, runningVar: Tensor
  eps: float32
type RMSNorm = object
  weight: Tensor
  eps: float32

proc initLayerNorm(key: Key, normalizedShape: int,
                   eps = 1e-5'f32): LayerNorm
proc initBatchNorm(numFeatures: int,
                   eps = 1e-5'f32): BatchNorm
proc initRMSNorm(normalizedShape: int,
                 eps = 1e-6'f32): RMSNorm
proc forward(layer: LayerNorm, x: Tensor): Tensor
proc forward(layer: BatchNorm, x: Tensor): Tensor
proc forward(layer: RMSNorm, x: Tensor): Tensor

# Dropout
type Dropout = object
  prob: float32
proc initDropout(prob = 0.5'f32): Dropout
proc forward(layer: Dropout, x: Tensor, key: Key): Tensor

# Sequential
type Sequential = object
  layers: seq[...]
proc initSequential(layers: openArray[...]): Sequential
proc forward(model: Sequential, x: Tensor): Tensor

# Attention
type MultiHeadAttention = object
  ...
type GroupedQueryAttention = object
  ...

proc initMultiHeadAttention(key: Key, embedDim, numHeads: int,
    dropout = 0.0'f32, ...): MultiHeadAttention
proc initGroupedQueryAttention(key: Key, embedDim, numHeads,
    numKvHeads: int, dropout = 0.0'f32, ...): GroupedQueryAttention
proc forward(mha: MultiHeadAttention, query, key, value: Tensor): Tensor
proc forward(gqa: GroupedQueryAttention, query, keyVal: Tensor): Tensor

# RoPE (rotary position embedding)
type RoPE = object
  dim: int
  theta: float32
proc initRoPE(dim: int, theta = 10000.0'f32): RoPE
proc forward(rope: RoPE, x: Tensor, offset = 0): Tensor

# LoRA
type QloraLinear = object
  dequantizedWeight, bias: Buffer[Tensor]
  A, B: Param[Tensor]
  rank: int
  alpha, scaling: float32
type QloraConfig = object
  rank, groupSize, sequenceLength: int
  alpha, learningRate: float32
proc defaultQloraConfig(): QloraConfig
proc forward(layer: QloraLinear, x: Tensor): Tensor

# Losses
proc softmaxCrossEntropy(logits, labels: Tensor): Tensor
proc mseLoss(pred, target: Tensor): Tensor
proc huberLoss(pred, target: Tensor, delta = 1.0'f32): Tensor
proc binaryCrossEntropy(pred, target: Tensor): Tensor
proc smoothL1Loss(pred, target: Tensor, beta = 1.0'f32): Tensor
```

### 6.8 Autograd

```nim
type
  GradError = object of CatchableError
  VjpResult = object
    output: Tensor
    pullback: proc(cotangent: Tensor): seq[Tensor]
  ValueAndGradResult = object
    value: Tensor
    grads: seq[Tensor]

proc vjp(fn: proc(args: openArray[Tensor]): Tensor,
         primals: openArray[Tensor]): VjpResult
proc grad(fn: proc(args: openArray[Tensor]): Tensor,
          primals: openArray[Tensor]): seq[Tensor]
proc valueAndGrad(fn: proc(args: openArray[Tensor]): Tensor,
                  primals: openArray[Tensor]): ValueAndGradResult

# Internal (user code rarely needs these):
template gradMode(body: untyped)
proc registerVjp(opName: string, rule: VjpRule)
proc registerNoGrad(opName: string)
```

### 6.9 jit Transform

Compiler-tier import:

```nim
import rew
import rew/xla
```

```nim
type
  JitFn = proc(args: openArray[Tensor]): seq[Tensor]
  JitFunction = ref object
    fn: JitFn
    funcName: string
    donateArgs: seq[int]
    cache: TableRef[string, JitCacheEntry]
  JitError = object of CatchableError

proc jit(fn: JitFn, funcName = "jit_fn",
         donateArgs: openArray[int] = []): JitFunction
proc call(jit: JitFunction, args: openArray[Tensor]): seq[Tensor]
proc lower(jit: JitFunction, args: openArray[Tensor]): ShModule
proc text(jit: JitFunction, args: openArray[Tensor]): string
proc executableText(jit: JitFunction, args: openArray[Tensor]): string
proc dumpHlo(jit: JitFunction, args: openArray[Tensor], path: string,
             executable = false): string
proc cacheSize(jit: JitFunction): int
proc clearCache(jit: JitFunction)

func signatureOf(args: openArray[Tensor]): string
```

### 6.10 Control Flow (transform/control)

```nim
type CondError = object of CatchableError

proc cond(pred: Tensor,
          thenFn: proc(): Tensor,
          elseFn: proc(): Tensor): Tensor
proc condN(pred: Tensor,
           thenFn: proc(): seq[Tensor],
           elseFn: proc(): seq[Tensor]): seq[Tensor]
proc whileLoop(initVals: openArray[Tensor],
               cond: proc(args: openArray[Tensor]): Tensor,
               body: proc(args: openArray[Tensor]): seq[Tensor]): seq[Tensor]
proc fori(low, high: int,
          initVals: openArray[Tensor],
          body: proc(i: Tensor, args: openArray[Tensor]): seq[Tensor]): seq[Tensor]
```

### 6.11 vmap (Automatic Batching)

```nim
proc vmap(fn: proc(args: openArray[Tensor]): seq[Tensor],
          inAxes: openArray[int] = @[]):
  proc(args: openArray[Tensor]): seq[Tensor]
```

### 6.12 Optimizers

```nim
# SGD
type Sgd = object
  lr: Tensor
proc initSgd(lr: Tensor): Sgd
proc step[P](opt: Sgd, params: P, grads: P): P

# Momentum SGD
type
  MomentumSgd = object
    lr, momentum: Tensor
    dampening: float32
    nesterov: bool
  MomentumState = object
    velocity: seq[Tensor]
proc initMomentumSgd(lr: Tensor, momentum = 0.9'f32): MomentumSgd
proc initMomentumState[P](params: P): MomentumState
proc step[P](opt: MomentumSgd, params: P, grads: P,
             state: MomentumState): tuple[params: P, state: MomentumState]

# Adam
type
  Adam = object
    lr: Tensor
    beta1, beta2, eps: float32
  AdamState = object
    m, v: seq[Tensor]
    t: int
proc initAdam(lr: Tensor,
              beta1 = 0.9'f32, beta2 = 0.999'f32,
              eps = 1e-8'f32): Adam
proc initAdamState[P](params: P): AdamState
proc step[P](opt: Adam, params: P, grads: P,
             state: AdamState): tuple[params: P, state: AdamState]

# AdamW
type AdamW = object
  lr: Tensor
  weightDecay: float32
  beta1, beta2, eps: float32
proc initAdamW(lr: Tensor,
               beta1 = 0.9'f32, beta2 = 0.999'f32,
               eps = 1e-8'f32,
               weightDecay = 0.01'f32): AdamW
proc step[P](opt: AdamW, params: P, grads: P,
             state: AdamState): tuple[params: P, state: AdamState]

# RMSprop
type
  Rmsprop = object
    lr: Tensor
    alpha, eps: float32
  RmspropState = object
    squareAvg: seq[Tensor]
    momentumBuf: seq[Tensor]
    gradAvg: seq[Tensor]
proc initRmsprop(lr: Tensor,
                 alpha = 0.99'f32, eps = 1e-8'f32,
                 momentum = 0'f32, centred = false): Rmsprop
proc initRmspropState[P](params: P, momentum = 0'f32,
                         centred = false): RmspropState
proc step[P](opt: Rmsprop, params: P, grads: P,
             state: RmspropState): tuple[params: P, state: RmspropState]

# NAdam
type
  NAdam = object
    lr: Tensor
    beta1, beta2, eps: float32
  NAdamState = object
    m, v: seq[Tensor]
    t: int
proc initNAdam(lr: Tensor,
               beta1 = 0.9'f32, beta2 = 0.999'f32,
               eps = 1e-8'f32): NAdam
proc initNAdamState[P](params: P): NAdamState
proc step[P](opt: NAdam, params: P, grads: P,
             state: NAdamState): tuple[params: P, state: NAdamState]

# Gradient clipping
proc clipGradNorm[P](grads: P, maxNorm: float32): P
proc clipGradValue[P](grads: P, maxValue: float32): P

# Lookahead wrapper
type Lookahead[O, S] = object
  inner: O
  k: int
  alpha: float32
proc initLookahead[O, S, P](inner: O, alpha = 0.5'f32,
                            k = 5): Lookahead[O, S]
```

### 6.13 LR Schedulers

```nim
type
  StepLR = object
    stepSize: int
    gamma: float32
  CosineAnnealingLR = object
    tMax: int
    etaMin: float32
  ReduceOnPlateau = object
    factor: float32
    patience: int
    minLr: float32

proc initStepLR(stepSize: int, gamma = 0.1'f32): StepLR
proc step(s: StepLR, lr: Tensor, epoch: int): Tensor

proc initCosineAnnealingLR(tMax: int,
    etaMin = 1e-6'f32): CosineAnnealingLR
proc step(s: CosineAnnealingLR, lr: Tensor, epoch: int): Tensor

proc initReduceOnPlateau(factor = 0.5'f32, patience = 5,
    minLr = 1e-8'f32): ReduceOnPlateau
proc step(s: var ReduceOnPlateau, lr: Tensor, metric: float32): Tensor
```

### 6.14 Dataset Pipeline

```nim
type
  DatasetFn[T] = proc(): iterator(): T
  Dataset[T] = object
    source: DatasetFn[T]
  DataError = object of CatchableError

proc toDataset[T](fn: DatasetFn[T]): Dataset[T]
proc fromSeq[T](data: seq[T]): Dataset[T]
proc fromRange(start, stop: int, step = 1): Dataset[int]
proc fromNpy(imagesPath, labelsPath: string,
             normalise = true): Dataset[Sample]

proc map[T, U](ds: Dataset[T], fn: proc(x: T): U): Dataset[U]
proc filter[T](ds: Dataset[T], pred: proc(x: T): bool): Dataset[T]
proc batch[T](ds: Dataset[T], batchSize: int,
              dropLast = false): Dataset[seq[T]]
proc shuffle(ds: Dataset[T], key: Key,
             bufferSize: int): Dataset[T]
proc repeat[T](ds: Dataset[T], epochs = -1): Dataset[T]
proc take[T](ds: Dataset[T], n: int): Dataset[T]
proc skip[T](ds: Dataset[T], n: int): Dataset[T]
proc zip[A, B](a: Dataset[A], b: Dataset[B]): Dataset[tuple[a: A, b: B]]
proc prefetch[T](ds: Dataset[T],
                 bufferSize = 2): Dataset[T]
proc parMap[T, U](ds: Dataset[T], fn: proc(x: T): U {.gcsafe.},
                  numWorkers = 4): Dataset[U]

# Sample and Batch
type
  Sample = object
    data: seq[float32]
    dataShape: seq[int]
    label: int
  Batch = object
    data: seq[float32]
    dataShape: seq[int]
    labels: seq[int]
    batchSize: int

proc collate(samples: seq[Sample]): Batch
proc oneHotF32(labels: seq[int], numClasses: int): seq[float32]
proc f32ToDevice(d: Device, data: var seq[float32],
                 shape: openArray[int]): Tensor
proc toTensors(d: Device, b: Batch,
               numClasses: int): tuple[x, y: Tensor]
```

### 6.15 Pytree

```nim
type PytreeError = object of CatchableError

proc treeFlatten[T](v: T): seq[Tensor]
proc treeUnflatten[T](structure: T, leaves: seq[Tensor]): T
proc treeMap[T](v: T, fn: proc(x: Tensor): Tensor): T
proc treeLeafCount[T](v: T): int
```

### 6.16 Serialization

```nim
type
  NpyArray = object
    dtype: DType
    shape: seq[int]
    data: seq[byte]
  NpyError = object of CatchableError

proc loadNpy(path: string): NpyArray
proc saveNpy(path: string, arr: NpyArray)

# SafeTensors (import rew/safetensors)
proc loadSafeTensors(path: string): Table[string, NpyArray]
```

### 6.17 Training API

```nim
import rew/train

# Runtime
type
  Accelerator = enum akCpu, akCuda, akRocm, akTpu, akAuto
  Precision = enum prFloat32, prFloat16, prBFloat16, prMixedF16, prMixedBF16
  Runtime = object
    accelerator: Accelerator
    devices: int
    precision: Precision
    device: Device

proc initRuntime(accelerator = akAuto, devices = 1,
                 precision = prFloat32): Runtime
proc setup[T](runtime: var Runtime, model: T): T
proc setup[T](runtime: var Runtime, ds: Dataset[T]): Dataset[T]
proc computeGrads[T](runtime: Runtime,
    fn: proc(args: openArray[Tensor]): Tensor, params: T): T
proc allReduce(runtime: Runtime, t: Tensor): Tensor
proc isGlobalZero(runtime: Runtime): bool
proc nextKey(runtime: var Runtime): Key
proc save[T](runtime: Runtime, path: string, state: T)
proc load[T](runtime: Runtime, path: string, prototype: T): T
proc seedEverything(seed: int)

# TrainState and typed steps
type
  CallCtx = object
    runtime: Runtime
    mode: TrainMode

  TrainState[M] = object
    model: M
    opt: GradientTransform
    optState: OptimState
    step: int
    key: Key

  StepResult[S] = object
    state: S
    loss: Tensor
    metrics: seq[StepMetric]

  LossFn[M, B] = proc(model: M; batch: B; ctx: CallCtx): Tensor
  TrainStepFn[M, B] = proc(state: TrainState[M]; batch: B;
    ctx: CallCtx): StepResult[TrainState[M]]

proc initTrainState[M](model: M; opt: GradientTransform;
                       key = initKey(0)): TrainState[M]
proc compileTrainStep[M, B](loss: LossFn[M, B]; state: TrainState[M];
    runtime = initRuntime(); donate: openArray[string] = []): CompiledTrainStep[M, B]

# DataSplits
type DataSplits[T] = object
  train: Dataset[T]
  val: Option[Dataset[T]]
  test: Option[Dataset[T]]
  predict: Option[Dataset[T]]

proc initDataSplits[T](train: Dataset[T],
    val = none[Dataset[T]](),
    test = none[Dataset[T]](),
    predict = none[Dataset[T]]()): DataSplits[T]

# Trainer
type Trainer = object
  maxEpochs: int
  maxSteps: Option[int]
  accelerator: Accelerator
  devices: int
  precision: Precision
  logEvery: int
  valInterval: Option[int]
  donateParams: bool
  callbacks: seq[Callback]

proc initTrainer(maxEpochs = 10, accelerator = akAuto,
                 devices = 1, precision = prFloat32): Trainer
proc fit[M, B](trainer: var Trainer, state: var TrainState[M],
               data: DataSplits[B], loss: LossFn[M, B])
proc fit[M, B](trainer: var Trainer, state: var TrainState[M],
               data: DataSplits[B], trainStep: TrainStepFn[M, B])
proc validate[M, B](trainer: var Trainer, state: TrainState[M],
                    data: DataSplits[B], loss: LossFn[M, B]): seq[MetricEntry]

# Callbacks
proc initCheckpoint(monitor = "val/loss",
    dirPath = "checkpoints",
    saveLast = true, saveTopK = 1,
    mode = cmMin,
    filename = "ckpt-epoch={epoch}-step={step}"): Checkpoint
proc initEarlyStopping(monitor = "val/loss",
    patience = 3, minDelta = 0.0'f32,
    mode = cmMin): EarlyStopping
proc initProgressBar(refreshRate = 1): ProgressBar
proc initLogMonitor(logEvery = 50): LogMonitor
```

### 6.18 Hugging Face Integration

```nim
import rew/hf

type HfDownloadOptions = object
  revision: string
  allowPatterns: seq[string]
  ignorePatterns: seq[string]

proc hfDownloadModel(repo: string,
    options = HfDownloadOptions()): string
proc hfDownloadDataset(repo: string,
    options = HfDownloadOptions()): string
proc hfDownloadSnapshot(kind: HfRepoKind, repo: string,
    options = HfDownloadOptions()): string

# Tokenizers
type HfTokenizer = object
  vocab: Table[string, int]
  ...

proc loadHfTokenizer(path: string): HfTokenizer
proc loadGemma4Tokenizer(path: string): HfTokenizer
proc encode(t: HfTokenizer, text: string): seq[int]
proc decode(t: HfTokenizer, ids: seq[int]): string
```

### 6.19 Dispatch (Internal)

User code rarely uses these directly, but they are exported:

```nim
type
  DispatchMode = enum dmEager, dmTrace
  TraceContext = object
    builder: ShBuilder
    funcName: string
    device: Device

proc currentMode(): DispatchMode
proc currentTraceContext(): ref TraceContext
proc setEagerBackend(backend: EagerBackend)

# Low-level trace control
proc beginTrace(funcName: string, device: Device): ref TraceContext
proc enterTrace(ctx: ref TraceContext)
proc exitTrace(prev: ref TraceContext)
template withTrace(ctxVar, funcName, device, body)
proc traceInputs(ctx: ref TraceContext, dtypes: openArray[DType],
                 shapes: openArray[seq[int]]): seq[Tensor]
proc traceReturn(ctx: ref TraceContext,
                 results: openArray[Tensor])
```

---

## 7. Best Practices

This section distills conventions from the Nim standard library style guide
(NEP-1) and rew-specific rules.

### 7.1 Naming

- **Types** use PascalCase: `Linear`, `Conv2d`, `Trainer`, `JitFunction`.
- **Procs and funcs** use camelCase: `initLinear`, `treeFlatten`, `valueAndGrad`.
- **Constants** may use PascalCase or camelCase; be consistent within a file.
- **Error types** have the `Error` suffix: `TensorError`, `DeviceError`, `GradError`.
- **Enum members** in non-pure enums use a prefix: `tCpu`, `dtFloat32`, `akAuto`.
- **Exception**: `Target` members use the `t` prefix rather than full `Target`
  prefix for brevity — `tCpu`, not `ttCpu`.
- Avoid ALL_UPPERCASE except for C-API wrappers.

### 7.2 Style

- **Line length**: 80 characters.
- **Indentation**: 2 spaces, no tabs.
- **Use `let`** for variables that never change within their scope.
- **Use `func`** for pure functions with no side effects. Use `proc` when
  side effects or exceptions are expected.
- **Prefer `result`** over explicit `return` in procedures.
- **Multi-line calls** indent their parameters:

```nim
proc loss(model: Mnist; batch: MnistBatch; ctx: CallCtx): Tensor =
  discard ctx
  softmaxCrossEntropy(forward(model, batch.x), batch.y)
```

### 7.3 Tensor Management

- **Avoid holding both old and new parameters** after an optimizer step in
  a tight loop. Assign directly so ARC can release the old buffer:

```nim
# Good: old params released immediately
(model, state) = opt.step(model, grads, state)

# Bad: keeping oldModel alive unnecessarily
let newModel = opt.step(model, grads, state)
```

- **Use buffer donation** in jit-compiled training steps to avoid extra
  allocations. Donate the parameter tensors and use the step output
  tensors for the next iteration:

```nim
let trainJ = jit(trainFn, "step", donateArgs = [0, 1, 2, 3])

for step in 0 ..< nSteps:
  let outs = trainJ.call([w1, b1, w2, b2, x, y, lr])
  w1 = Linear(weight: outs[1], bias: outs[2])  # new, non-donated tensor
  w2 = Linear(weight: outs[3], bias: outs[4])
```

- **Don't mix trace and eager tensors** in the same operation. This raises
  `TensorModeError`. Use `requireEager`/`requireTrace` if you write custom
  ops.

- **Explicit device transfers only.** There is no implicit CPU→GPU movement.
  Use `.to(device)` to move tensors.

### 7.4 PRNG

- **Split, don't reuse.** After `split(key, n)`, stop using the original
  key. Each child should be used for one purpose (one layer, one data
  shard, etc.).
- **Use `foldIn` for step-indexed randomness** rather than consuming split
  slots. `foldIn(key, uint64(stepNumber))` produces deterministic,
  reproducible keys without burning through your split budget.
- **Pass keys explicitly** to all random operations. Never call
  `initKey()` inside a function that will be jit-traced — use a key
  parameter instead.

### 7.5 Training Loop Patterns

**Prefer typed compiled steps over eager training.** Tracing forward and
backward together into one fused StableHLO program eliminates the per-op
compile overhead of eager execution and enables XLA-level fusion optimizations:

```nim
# Good: single fused typed step
var state = initTrainState(model, adamw(lr))
var step = compileTrainStep(loss, state, runtime,
  donate = paramsOf(state.model))
state = step(state, batch).state

# Less ideal: per-op eager execution
for step in 0 ..< nSteps:
  let loss = forward(model, x)
  let grads = grad(lossFn, treeFlatten(model))
  model = opt.step(model, grads)
```

**Choose the right training tier:**

| Situation | Use |
|-----------|-----|
| Simple supervised learning | `Trainer` with built-in callbacks |
| Research, custom loops, debugging | `Runtime` plus `compileTrainStep` |
| GANs, RL, meta-learning | Typed custom loop over `TrainState` |

### 7.6 Error Handling

Rew errors derive from `CatchableError` (never `Defect`). Catch them with
standard Nim exception handling:

```nim
try:
  let d = initDevice(tCuda12)
  setDefaultDevice(d)
  installEagerBackend()
  discard scalarF32(d, 0'f32)  # forces the CUDA plugin to load
except CatchableError as e:
  echo "PJRT plugin not available: ", e.msg
  quit 1

try:
  let loss = item(vg.value, float32)
except TensorError as e:
  echo "Tensor error: ", e.msg
```

### 7.7 Environment Configuration

- Set `REW_TARGET` to override auto-detection rather than calling
  `setDefaultDevice` with a hardcoded target in production code.
- Set `REW_CACHE_DIR` to a persistent location to avoid re-downloading
  plugins and recompiling jit programs across runs.
- For CUDA/ROCm multi-GPU setups, use `REW_TARGET=cuda12` and select
  the ordinal via `initDevice(tCuda12, ordinal = gpuIndex)`.
- Use `bau task doctor` to list all available devices for installed targets.

### 7.8 Compilation Flags

- Use `--threads:on` when using `prefetch` or `parMap` dataset pipeline
  transforms.
- The `--d:release` flag enables optimizations. Use `--d:danger` to
  disable all runtime checks for maximum performance in production.
- AddressSanitizer: `bau asan` runs the test suite with
  `-d:addressSanitizer`.

### 7.9 Extending Rew

When adding new ops, layers, or optimizers:

1. **New primitive op:**
   - Define in `src/rew/ops/` with `{.rewOp.}` pragma.
   - Add eager support in `src/rew/eager.nim`.
   - Register VJP rule or `registerNoGrad` in `src/rew/autograd/rules.nim`.
   - Write tests (numerical correctness, eager-vs-jit equivalence, VJP).
   - Run `bau lint` — the VJP coverage lint must pass.
   - Re-export from `src/rew.nim`.

2. **New nn layer:**
   - Create file in `src/rew/nn/` as a value `object` (no `ref object`).
   - Provide `initLayer(key: Key, ...): LayerType` constructor.
   - Provide `proc forward(layer: LayerType, x: Tensor): Tensor`.
   - Use only differentiable primitives in `forward`.
   - Re-export from `src/rew/nn.nim`.

3. **New optimizer:**
   - Create file in `src/rew/optim/` as a value `object`.
   - Provide an `init*` constructor for hyperparameters.
   - Provide a functional `step` returning new params and, when needed, state.
   - Use `treeFlatten`/`treeUnflatten` for leaf-wise updates.
   - Re-export from `src/rew/optim.nim`.

### 7.10 Thread Safety

- Each thread has its own dispatch mode (`threadvar`). `jit` on one thread
  does not affect eager execution on another.
- PRNG `Key` operations are pure functions — safe to use from any thread.
- PJRT plugin state (device registry, loaded plugins) is protected by
  a process-wide lock during initialization.
- `prefetch` and `parMap` spawn native threads; they require
  `--threads:on`. `parMap` functions must be `gcsafe`.

---

## 8. Further Resources

- **Architecture reference:** `docs/architecture.md` — layered design,
  invariants, and build phases.
- **Examples directory:** `examples/` — MNIST (MLP, CNN, Runtime,
  Trainer), GAN training, QLoRA adapter training.
- **Test suite:** `tests/` — Standalone `t*.nim` files demonstrating
  every operation and subsystem.
- **Developer tooling:** `bau lint` (architectural lints), `bau test`
  (full suite), `bau task doctor` (device listing).
- **Nim style guide:** [NEP-1](https://nim-lang.org/docs/nep1.html)
