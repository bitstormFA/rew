## MNIST MLP training with the Runtime API (Tier 1).
##
## The Runtime provides device management, PRNG keys, save/load, and
## distributed stubs — but **you own the training loop**. Compare with
## `mnist_trainer.nim` where the framework owns the loop.
##
## ## Running
##
## Point `REW_MNIST_DIR` at a directory containing `train_images.npy`
## and `train_labels.npy`, or the example runs with synthetic data:
##
## ```
## REW_MNIST_DIR=$HOME/datasets/mnist nim c -r examples/mnist_runtime.nim
## ```

import std/[math, os, strformat]
import rew
import rew/xla
import rew/train
import rew/pjrt/loader

const
  Backend = tCpu
  ImagePixels = 784
  HiddenDim = 64
  NumClasses = 10
  BatchSize = 32
  LearningRate = 0.05'f32
  Epochs = 3
  StepsPerEpoch = 5

# ---- Host helpers ---------------------------------------------------------

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

# ---- Eager model init -----------------------------------------------------

proc initModelEager(d: Device; key: Key): (Linear, Linear) =
  let keys = split(key, 2)
  let bound = sqrt(1.0'f32 / float32(ImagePixels))
  var w1 = uniformF32(keys[0], ImagePixels * HiddenDim, -bound, bound)
  var b1 = newSeq[float32](HiddenDim)
  var w2 = uniformF32(keys[1], HiddenDim * NumClasses, -bound, bound)
  var b2 = newSeq[float32](NumClasses)
  (Linear(
      weight: param(fromHostF32(d, w1, [ImagePixels, HiddenDim])),
      bias: param(fromHostF32(d, b1, [HiddenDim]))),
   Linear(
      weight: param(fromHostF32(d, w2, [HiddenDim, NumClasses])),
      bias: param(fromHostF32(d, b2, [NumClasses]))))

# ---- Forward pass ---------------------------------------------------------

proc forward(l1, l2: Linear; x: Tensor): Tensor =
  forward(l2, relu(forward(l1, x)))

# ---- Synthetic data -------------------------------------------------------

proc syntheticBatch(d: Device; key: Key): tuple[x, y: Tensor] =
  let keys = split(key, 2)
  var pixels = uniformF32(keys[0], BatchSize * ImagePixels, 0.0'f32, 1.0'f32)
  var labels = newSeq[float32](BatchSize * NumClasses)
  for i in 0 ..< BatchSize:
    labels[i * NumClasses + (i mod NumClasses)] = 1.0'f32
  (x: fromHostF32(d, pixels, [BatchSize, ImagePixels]),
   y: fromHostF32(d, labels, [BatchSize, NumClasses]))

# ---- Main -----------------------------------------------------------------

proc run() =
  echo "── PJRT plugin: ", Backend
  try:
    discard loadPlugin(Backend)
  except PjrtError as e:
    echo "  (skip) ", e.msg
    return

  let d = initDevice(Backend)
  setDefaultDevice(d)
  installEagerBackend()
  echo &"  using device: {d}"

  # ---- Runtime (you configure it) ----
  var runtime = initRuntime(akCpu)
  seedEverything(42)
  echo &"  runtime: device={runtime.device}, isGlobalZero={runtime.isGlobalZero()}"

  # ---- Model (you initialise it) ----
  var (fc1, fc2) = initModelEager(d, runtime.nextKey())
  let lr = scalarF32(d, LearningRate)

  # ---- Build a jit-compiled training step (you write it) ----
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]; let ya = args[5]; let lr = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: param(p[0]), bias: param(p[1]))
      let l2 = Linear(weight: param(p[2]), bias: param(p[3]))
      softmaxCrossEntropy(forward(l1, l2, xa), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      sub(p, mul(broadcastTo(lr, p.shape, bdims), g))
    @[vr.output, upd(args[0], grads[0]), upd(args[1], grads[1]),
      upd(args[2], grads[2]), upd(args[3], grads[3])]

  let trainJ = jit(trainFn, "mnist_runtime_step", donateArgs = [0, 1, 2, 3])

  # ---- Training loop (you own it!) ----
  echo &"  Training for {Epochs} epochs ({StepsPerEpoch} steps each)..."
  for epoch in 0 ..< Epochs:
    var epochLoss = 0.0'f32

    for step in 0 ..< StepsPerEpoch:
      let (bx, by) = syntheticBatch(d, runtime.nextKey())
      let outs = trainJ.call([
          fc1.weight.value, fc1.bias.value, fc2.weight.value, fc2.bias.value, bx, by, lr])
      fc1 = Linear(weight: param(outs[1]), bias: param(outs[2]))
      fc2 = Linear(weight: param(outs[3]), bias: param(outs[4]))
      epochLoss += readBackF32(outs[0])[0]

    echo &"  epoch {epoch}: loss = {epochLoss / float32(StepsPerEpoch):.4f}"

  # ---- Save checkpoint with the Runtime ----
  let ckptPath = "/tmp/mnist_runtime_ckpt"
  runtime.save(ckptPath, (fc1, fc2))
  echo &"  checkpoint saved to {ckptPath}"

  # ---- Load checkpoint ----
  let prototype = (fc1, fc2)
  let (loadedFc1, loadedFc2) = runtime.load(ckptPath, prototype)
  echo &"  checkpoint loaded: fc1.weight.shape = {loadedFc1.weight.shape}"

when isMainModule:
  run()
