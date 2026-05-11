## Simple GAN training with manual optimization (Tier 2 escape hatch).
##
## Demonstrates the Trainer's manual optimization mode
## (`automaticOptimization = false`). The user handles forward, backward,
## and optimizer steps for both generator and discriminator inside
## `trainingStep`. The Trainer provides infrastructure: iteration,
## hooks, logging, validation scheduling, and callbacks.
##
## This is a toy GAN: generator learns to map random noise to a target
## distribution; discriminator learns to tell real from fake.
##
## ## Running
##
## ```
## nim c -r examples/gan_trainer.nim
## ```

import std/[math, strformat]
import rew
import rew/train
import rew/pjrt/loader

const
  Backend = tCpu
  NoiseDim = 16
  DataDim = 8
  BatchSize = 64
  GLearningRate = 0.01'f32
  DLearningRate = 0.01'f32

# ---- Host helpers ---------------------------------------------------------

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

# ---- Toy GAN components ----------------------------------------------------

proc linearApply(weight, bias, x: Tensor): Tensor =
  forward(Linear(weight: weight, bias: bias), x)

proc meanAll(x: Tensor): Tensor =
  if x.shape.len == 0:
    return x
  var dims = newSeq[int](x.shape.len)
  for i in 0 ..< x.shape.len:
    dims[i] = i
  reduceMean(x, dims)

proc sgdUpdate(param, grad, lr: Tensor): Tensor =
  var bdims: seq[int] = @[]
  let lrB = broadcastTo(lr, param.shape, bdims)
  sub(param, mul(lrB, grad))

proc ganTrainStep(args: openArray[Tensor]): seq[Tensor] =
  ## One fused manual-optimization GAN step.
  ##
  ## Inputs:
  ##   0..1 generator weight/bias
  ##   2..3 discriminator weight/bias
  ##   4 real batch, 5 noise batch, 6 generator lr, 7 discriminator lr
  let gWeight = args[0]
  let gBias = args[1]
  let dWeight = args[2]
  let dBias = args[3]
  let real = args[4]
  let noise = args[5]
  let genLr = args[6]
  let discLr = args[7]

  let discLossFn = proc(params: openArray[Tensor]): Tensor =
    let fake = stopGradient(linearApply(gWeight, gBias, noise))
    let realScore = linearApply(params[0], params[1], real)
    let fakeScore = linearApply(params[0], params[1], fake)
    sub(meanAll(fakeScore), meanAll(realScore))

  let discVjp = vjp(discLossFn, [dWeight, dBias])
  let discGrads = discVjp.pullback(scalarF32(1'f32))
  let nextDWeight = sgdUpdate(dWeight, discGrads[0], discLr)
  let nextDBias = sgdUpdate(dBias, discGrads[1], discLr)

  let genLossFn = proc(params: openArray[Tensor]): Tensor =
    let fake = linearApply(params[0], params[1], noise)
    let fakeScore = linearApply(stopGradient(nextDWeight),
      stopGradient(nextDBias), fake)
    neg(meanAll(fakeScore))

  let genVjp = vjp(genLossFn, [gWeight, gBias])
  let genGrads = genVjp.pullback(scalarF32(1'f32))
  let nextGWeight = sgdUpdate(gWeight, genGrads[0], genLr)
  let nextGBias = sgdUpdate(gBias, genGrads[1], genLr)

  @[discVjp.output, genVjp.output,
    nextGWeight, nextGBias, nextDWeight, nextDBias]

# ---- Synthetic "real" data generator ---------------------------------------

proc makeRealBatch(d: Device; key: Key; n: int): Tensor =
  ## Generates a batch of "real" data from a fixed distribution.
  let keys = split(key, 2)
  var data = uniformF32(keys[0], n * DataDim, -0.5'f32, 0.5'f32)
  # Add a pattern: real data clusters around ±0.3 in even dimensions
  for i in 0 ..< n:
    for j in 0 ..< DataDim:
      if j mod 2 == 0:
        data[i * DataDim + j] = data[i * DataDim + j] * 0.3'f32 + 0.3'f32
  fromHostF32(d, data, [n, DataDim])

proc makeNoise(d: Device; key: Key; n: int): Tensor =
  let keys = split(key, 2)
  var data = uniformF32(keys[0], n * NoiseDim, -1.0'f32, 1.0'f32)
  fromHostF32(d, data, [n, NoiseDim])

# ---- Task definition -------------------------------------------------------

type
  GanTask = object
    generator: Linear
    discriminator: Linear
    genLr: Tensor
    discLr: Tensor
    trainJ: JitFunction
    realKey: Key
    noiseKey: Key
    d: Device
    stepCount: int
    log: seq[string]

proc initGanTask(d: Device; key: Key; genLr, discLr: Tensor;
    trainJ: JitFunction): GanTask =
  let keys = split(key, 4)
  let bound = sqrt(1.0'f32 / float32(NoiseDim))
  # Generator: NoiseDim -> DataDim
  var gw = uniformF32(keys[0], NoiseDim * DataDim, -bound, bound)
  var gb = newSeq[float32](DataDim)
  # Discriminator: DataDim -> 1
  var dw = uniformF32(keys[2], DataDim * 1, -bound, bound)
  var db = newSeq[float32](1)
  var task = GanTask(
    generator:    Linear(weight: fromHostF32(d, gw, [NoiseDim, DataDim]),
                          bias:   fromHostF32(d, gb, [DataDim])),
    discriminator: Linear(weight: fromHostF32(d, dw, [DataDim, 1]),
                          bias:   fromHostF32(d, db, [1])),
    genLr: genLr, discLr: discLr, trainJ: trainJ,
    realKey: keys[1], noiseKey: keys[3], d: d,
  )
  task

# ---- Required hooks --------------------------------------------------------

proc trainingStep(t: var GanTask; batch: Batch; batchIdx: int;
    ctx: var TrainContext): Tensor =
  ## Manual optimization mode. We handle both generator and discriminator
  ## updates inside this hook.
  discard batch
  discard batchIdx

  let real = makeRealBatch(t.d, t.realKey, BatchSize)
  t.realKey = foldIn(t.realKey, uint64(t.stepCount))

  let noise = makeNoise(t.d, t.noiseKey, BatchSize)
  t.noiseKey = foldIn(t.noiseKey, uint64(t.stepCount))

  let outs = t.trainJ.call([
    t.generator.weight, t.generator.bias,
    t.discriminator.weight, t.discriminator.bias,
    real, noise, t.genLr, t.discLr,
  ])
  t.generator = Linear(weight: outs[2], bias: outs[3])
  t.discriminator = Linear(weight: outs[4], bias: outs[5])

  let dLossVal = readBackF32(outs[0])[0]
  let gLossVal = readBackF32(outs[1])[0]

  ctx.log("d_loss", dLossVal, onStep = true, onEpoch = true)
  ctx.log("g_loss", gLossVal, onStep = true, onEpoch = true, progBar = true)
  t.stepCount += 1
  outs[1]

proc configureOptimizers(t: GanTask): OptimizerConfig =
  ## In manual mode, configureOptimizers tells the Trainer about
  ## scheduler configs. The actual optimizer steps are handled manually.
  initOptimizerConfig(OptimizerKind(kind: otSgd, sgd: initSgd(t.genLr)))

# ---- Optional hooks --------------------------------------------------------

proc onFitStart(t: var GanTask; ctx: var TrainContext) =
  echo "GAN training started (manual optimization mode)"

proc onTrainEpochStart(t: var GanTask; ctx: var TrainContext) =
  echo &"\n── Epoch {ctx.epoch} ──"

proc onTrainEpochEnd(t: var GanTask; ctx: var TrainContext) =
  for m in ctx.epochMetrics:
    if m.name == "d_loss" or m.name == "g_loss":
      echo &"  avg {m.name} = {m.value:.4f}"

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

  let trainJ = jit(ganTrainStep, "gan_manual_train_step",
    donateArgs = [0, 1, 2, 3])
  var task = initGanTask(d, initKey(0x6A),
      scalarF32(d, GLearningRate),
      scalarF32(d, DLearningRate),
      trainJ)

  # Synthetic batch data (just placeholders for the Trainer's iteration)
  var batches = newSeq[Batch](15)  # 3 epochs * 5 batches
  var key = initKey(999)
  for i in 0 ..< batches.len:
    key = foldIn(key, uint64(i))
    batches[i] = Batch(data: @[], dataShape: @[0], labels: @[],
        batchSize: BatchSize)
  let data = initDataPipe(fromSeq(batches))

  # Create Trainer in manual mode
  var trainer = initTrainer(maxEpochs = 3, accelerator = akCpu,
      automaticOptimization = false)
  trainer.logEvery = 1
  trainer.callbacks = @[
    initProgressBar(refreshRate = 1).toCallback(),
  ]

  trainer.fit(task, data)

  echo &"\nGAN training done. {task.stepCount} steps total."

when isMainModule:
  run()
