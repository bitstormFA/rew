# Rechenwerk (rew)

Rechenwerk is an eager, debuggable Nim deep learning framework on
OpenXLA/PJRT.

It gives you a PyTorch-like tensor API, a JAX-style runtime `jit`, functional
layers and optimizers, and one shared StableHLO/VJP path for eager execution and
compiled training steps.

## Why Try It?

- One compiler path: eager ops and `jit` traces emit the same StableHLO.
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
- Transforms: runtime `jit`, `cond`, `whileLoop`, `fori`, and `vmap`.
- Neural nets: functional layers under `rew.nn`, explicit PRNG keys, and pytree
  flatten/unflatten for training state.
- Training: `Workbench` for user-owned loops and `Trainer` for framework-owned
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
bau test
bau asan
```

Architecture notes live in [docs/architecture.md](docs/architecture.md), the
user guide is in [docs/user-guide.md](docs/user-guide.md), and PJRT plugin
details are in [docs/binaries.md](docs/binaries.md).

## License

MIT. See [LICENSE](LICENSE).
