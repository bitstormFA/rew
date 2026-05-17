# Rechenwerk (rew)

Rechenwerk is an eager, debuggable Nim deep learning framework on
OpenXLA/PJRT.

It gives you a PyTorch-like tensor API, typed compiled training steps,
functional layers and optimizers, and one shared StableHLO/VJP path for eager
execution and compiler-backed transforms.

## Why Try It?

- One compiler path: eager ops and raw `jit` traces emit the same StableHLO.
- One autodiff path: VJP rules are shared by eager tape and `grad`/`vjp`.
- Plain Nim objects: layers, optimizers, configs, and pytrees are value types.
- Explicit runtime behavior: no macro `jit`, no hidden host transfers, no
  implicit cross-device copies.
- Runtime backends: CPU, CUDA 12/13, ROCm, Metal, and TPU are resolved through
  PJRT plugins and a pinned manifest.

## Install

From a published package:

```nim
requires "rew >= 0.2.0"
```

From a source checkout:

```bash
bau test
bau fetch cpu
```

Development commands are declared in `bau.toml` and require Bau 0.4.0 or newer
on `PATH`.

Installed packages also provide the standalone plugin downloader:

```bash
rew_fetch cpu
```

## Quick Start

```nim
import rew

setDefaultDevice(cpu(0))
installEagerBackend()

let x = fromHost([1.0'f32, 2.0, 3.0, 4.0], [2, 2])
let y = relu(x + x)

echo y.toHost(float32)
```

Run it after fetching a CPU plugin:

```bash
nim c -r quickstart.nim
```

## Main Workflows

- Tensor ops: arithmetic, reductions, shape ops, linalg, convolution, pooling,
  gather/scatter, sort, random, and normalization.
- Autograd: `grad`, `vjp`, `valueAndGrad`, and eager tape-style `backward`.
- Transforms: user-level `grad`, `vjp`, `cond`, `whileLoop`, `fori`, and
  `vmap`; raw `jit`, lowering, and HLO inspection live under `import rew/xla`.
- Neural nets: functional layers under `rew.nn`, explicit PRNG keys, and pytree
  flatten/unflatten for training state.
- Training: `Runtime` for user-owned loops and `Trainer` for framework-owned
  loops with callbacks.
- Data: lazy datasets, transforms, batching, image loading, and graph data
  helpers.
- Binaries: `bau fetch <target>`, `rew_fetch <target>`, `bau task doctor`,
  and manifest-driven plugin caches.

## Examples

```bash
nim c -r examples/list_devices.nim
nim c -r examples/mnist_mlp.nim
nim c -r examples/mnist_cnn.nim
nim c -r examples/mnist_trainer.nim
```

MNIST examples use synthetic data when `REW_MNIST_DIR` is not set. Set
`REW_TARGET=cpu`, `REW_TARGET=cuda12`, or another supported target to override
auto-detection.

## Development

```bash
bau lint
bau testFast       # dev-profile tests for fast iteration
bau test
bau asan
```

The full `bau test` command still runs debug, release, and danger. The test
runner defaults to parallel child test builds; set `REW_TEST_JOBS=1` to debug
serially or raise it on larger machines.

Architecture notes live in [docs/architecture.md](docs/architecture.md), the
user guide is in [docs/user-guide.md](docs/user-guide.md), and PJRT plugin
details are in [docs/binaries.md](docs/binaries.md).

The public API is tiered: `import rew` for high-level user code,
`import rew/xla` for raw compiler/lowering work, and `import rew/dev` for
extension internals.

## License

MIT. See [LICENSE](LICENSE).
