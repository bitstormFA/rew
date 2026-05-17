## Multi-target MNIST — demonstrates CPU + CUDA tensors in one process.
##
## Loads the CPU plugin and (if available) the CUDA12 plugin, then runs a
## simple MNIST-MLP forward pass on each. Skips targets whose plugins are
## not installed; requires at least the CPU plugin.
##
## Usage:
##   nim c -r examples/multi_target_mnist.nim

import std/[strformat, math]
import rew
import rew/xla
import rew/pjrt/[capi, client, loader]

proc f32Tensor(d: Device; shape: openArray[int];
    data: openArray[float32]): Tensor =
  var local = @data
  var dims = newSeq[int64](shape.len)
  for i, s in shape: dims[i] = int64(s)
  let h = transferToDevice(d, addr local[0], dtFloat32, dims,
    sizeBytes = local.len * sizeof(float32))
  initEagerTensor(h, dtFloat32, shape, d)

proc readBackScalar(t: Tensor): float32 =
  var v: float32
  transferToHost(t.device, t.buffer, addr v, sizeof(v))
  v

proc tryTarget(t: Target): bool =
  try:
    discard loadPlugin(t)
    true
  except PjrtError:
    false

proc forwardMlp(d: Device) =
  installEagerBackend()
  let batchSize = 4
  let inputDim = 8
  let hiddenDim = 4
  let outputDim = 2

  var xData = newSeq[float32](batchSize * inputDim)
  for i in 0 ..< xData.len: xData[i] = float32(i mod 3) * 0.1

  var w1Data = newSeq[float32](inputDim * hiddenDim)
  for i in 0 ..< w1Data.len: w1Data[i] = 0.01'f32

  var b1Data = newSeq[float32](hiddenDim)
  for i in 0 ..< b1Data.len: b1Data[i] = 0.0'f32

  var w2Data = newSeq[float32](hiddenDim * outputDim)
  for i in 0 ..< w2Data.len: w2Data[i] = 0.01'f32

  var b2Data = newSeq[float32](outputDim)
  for i in 0 ..< b2Data.len: b2Data[i] = 0.0'f32

  let x = f32Tensor(d, [batchSize, inputDim], xData)
  let w1 = f32Tensor(d, [inputDim, hiddenDim], w1Data)
  let b1 = f32Tensor(d, [hiddenDim], b1Data)
  let w2 = f32Tensor(d, [hiddenDim, outputDim], w2Data)
  let b2 = f32Tensor(d, [outputDim], b2Data)

  let h = add(matmul(x, w1), broadcastTo(b1, [batchSize, hiddenDim], [1]))
  let logits = add(matmul(h, w2), broadcastTo(b2, [batchSize, outputDim], [1]))
  let first = readBackScalar(reshape(logits, [batchSize * outputDim]))
  echo &"    forward pass logits[0] = {first:.6f}"

proc main() =
  echo "multi_target_mnist: demonstrates multi-plugin in one process"
  var ran = 0

  if tryTarget(tCpu):
    echo "  [cpu]"
    let d = cpu(0)
    setDefaultDevice(d)
    forwardMlp(d)
    inc ran
  else:
    echo "  [cpu] skipped — no plugin"

  if tryTarget(tCuda12):
    echo "  [cuda12]"
    let d = cuda12(0)
    setDefaultDevice(d)
    forwardMlp(d)
    inc ran
  else:
    echo "  [cuda12] skipped — no plugin"

  if ran == 0:
    echo "  no plugins available — install at least the CPU plugin"
    quit 1
  echo &"multi_target_mnist: completed on {ran} target(s)"

when isMainModule:
  main()
