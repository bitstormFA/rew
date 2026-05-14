# Benchmarks

Run the MNIST example benchmarks against equivalent PyTorch implementations:

```sh
bau bench
```

The runner compiles `benchmarks/rew_mnist.nim`, runs
`benchmarks/pytorch_mnist.py`, and prints a Markdown table with first-step
latency, steady-state step latency, throughput, and final loss. First-step
latency includes Rechenwerk tracing/PJRT compilation; steady-state timing
excludes the first step and warmup iterations.

Useful direct invocation:

```sh
python3 benchmarks/run.py --warmup=5 --iterations=20 --batch-size=32
```

PyTorch is optional. If it is not installed, or if the requested PJRT plugin
is missing, the affected rows are reported as skipped in the table.
