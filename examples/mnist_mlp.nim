## MNIST MLP forward-pass skeleton.
##
## Demonstrates the Phase 8 wiring end-to-end:
##  - `loadNpy` reads MNIST-shaped image and label arrays from `.npy`
##    files (uint8 pixels, int64 labels).
##  - `transferToDevice` ships the host buffers to the active PJRT
##    device.
##  - `Linear` / `relu` / `softmaxCrossEntropy` (composite ops) run
##    through the eager dispatcher.
##  - One full SGD step (loss + grad + parameter update) is traced as
##    a single `jit`-compiled StableHLO function with donated param
##    buffers, then iterated for a short training loop. Per step we
##    print mean cross-entropy and batch accuracy.
##
## ## Running
##
## With real data, point `REW_MNIST_DIR` at a directory containing
## `train_images.npy` (uint8, shape `[N, 784]` or `[N, 28, 28]`) and
## `train_labels.npy` (uint8 or int64, shape `[N]`):
##
## ```
## REW_MNIST_DIR=$HOME/datasets/mnist nim c -r examples/mnist_mlp.nim
## ```
##
## Without `REW_MNIST_DIR`, a small synthetic batch is generated so the
## pipeline still exercises end-to-end (load/transfer/forward/loss).
##
## Skips with a one-line diagnostic when no PJRT plugin is installed.

import std/[math, os, strformat]
import rew
import rew/xla
import rew/pjrt/loader

const
  Backend = tCpu
  ImageSide = 28
  ImagePixels = ImageSide * ImageSide
  HiddenDim = 64
  NumClasses = 10
  SyntheticBatch = 32

# ---- host-bytes -> eager tensor ----------------------------------------

proc shapeDims(shape: openArray[int]): seq[int64] =
  result = newSeq[int64](shape.len)
  for i, v in shape: result[i] = int64(v)

proc f32EagerTensor(d: Device; shape: openArray[int];
    data: var seq[float32]): Tensor =
  ## Wraps `data` (taken by `var` so we can take its address) as a
  ## device-resident float32 tensor on `d`.
  var n = 1
  for s in shape: n *= s
  doAssert data.len == n,
    "f32EagerTensor: shape product " & $n & " != data length " & $data.len
  let dims = shapeDims(shape)
  let bytes = data.len * sizeof(float32)
  let ptrIn = if data.len == 0: nil else: addr data[0]
  let h = transferToDevice(d, ptrIn, dtFloat32, dims, bytes)
  initEagerTensor(h, dtFloat32, shape, d)

proc readBackF32(t: Tensor): seq[float32] =
  var n = 1
  for s in t.shape: n *= s
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

# ---- init helpers (eager, mirror of nn.initLinear) ---------------------

proc initLinearEager(d: Device; key: Key; inFeat, outFeat: int): Linear =
  let keys = split(key, 2)
  let bound = sqrt(1.0f32 / float32(inFeat))
  var w = uniformF32(keys[0], inFeat * outFeat, -bound, bound)
  var b = newSeq[float32](outFeat)
  Linear(
    weight: param(f32EagerTensor(d, [inFeat, outFeat], w)),
    bias: param(f32EagerTensor(d, [outFeat], b)),
  )

# ---- data loading ------------------------------------------------------

proc loadImages(d: Device; path: string): Tensor =
  ## Loads MNIST images from `path` (uint8, shape `[N,784]` or
  ## `[N,28,28]`), normalises to `[N, 784]` float32 in `[0, 1]` on `d`.
  let arr = loadNpy(path)
  if arr.dtype != dtUint8:
    raise newException(NpyError,
      "loadImages: expected uint8 pixels, got " & $arr.dtype)
  let n =
    if arr.shape.len == 2 and arr.shape[1] == ImagePixels: arr.shape[0]
    elif arr.shape.len == 3 and arr.shape[1] == ImageSide and
        arr.shape[2] == ImageSide: arr.shape[0]
    else:
      raise newException(NpyError,
        "loadImages: expected [N,784] or [N,28,28], got " & $arr.shape)
  var pixels = newSeq[float32](n * ImagePixels)
  for i in 0 ..< pixels.len:
    pixels[i] = float32(arr.data[i]) / 255.0'f32
  f32EagerTensor(d, [n, ImagePixels], pixels)

proc loadLabelsOneHot(d: Device; path: string): Tensor =
  ## Loads MNIST labels from `path` (uint8 or int64, shape `[N]`) and
  ## returns a one-hot float32 `[N, NumClasses]` on `d`.
  let arr = loadNpy(path)
  if arr.shape.len != 1:
    raise newException(NpyError,
      "loadLabelsOneHot: expected rank-1 labels, got " & $arr.shape)
  let n = arr.shape[0]
  proc labelAt(i: int): int =
    case arr.dtype
    of dtUint8: int(arr.data[i])
    of dtInt64:
      var v: int64
      copyMem(addr v, unsafeAddr arr.data[i * 8], 8)
      int(v)
    else:
      raise newException(NpyError,
        "loadLabelsOneHot: unsupported label dtype " & $arr.dtype)
  var onehot = newSeq[float32](n * NumClasses)
  for i in 0 ..< n:
    let c = labelAt(i)
    if c < 0 or c >= NumClasses:
      raise newException(NpyError,
        "loadLabelsOneHot: label " & $c & " out of range at row " & $i)
    onehot[i * NumClasses + c] = 1.0'f32
  f32EagerTensor(d, [n, NumClasses], onehot)

# ---- synthetic fallback ------------------------------------------------

proc syntheticBatch(d: Device; key: Key): tuple[x, y: Tensor] =
  let keys = split(key, 2)
  var pixels = uniformF32(keys[0], SyntheticBatch * ImagePixels, 0.0'f32,
      1.0'f32)
  var labels = newSeq[float32](SyntheticBatch * NumClasses)
  # Cycle classes 0..9 so every output gets some target mass.
  for i in 0 ..< SyntheticBatch:
    labels[i * NumClasses + (i mod NumClasses)] = 1.0'f32
  result = (
    x: f32EagerTensor(d, [SyntheticBatch, ImagePixels], pixels),
    y: f32EagerTensor(d, [SyntheticBatch, NumClasses], labels),
  )

# ---- main --------------------------------------------------------------

proc forward(l1, l2: Linear; x: Tensor): Tensor =
  ## Two-layer MLP with `relu` between layers.
  let h = relu(forward(l1, x))
  forward(l2, h)

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

  let key = initKey(0xC0FFEE'u64)
  let keys = split(key, 3)

  # Load data (real if REW_MNIST_DIR is set, otherwise synthetic).
  let dir = getEnv("REW_MNIST_DIR")
  var x, y: Tensor
  if dir.len > 0:
    let imgPath = dir / "train_images.npy"
    let lblPath = dir / "train_labels.npy"
    if not fileExists(imgPath) or not fileExists(lblPath):
      echo "  (skip) REW_MNIST_DIR=", dir,
        " is missing train_images.npy / train_labels.npy"
      return
    echo "  loading real MNIST from ", dir
    x = loadImages(d, imgPath)
    y = loadLabelsOneHot(d, lblPath)
  else:
    echo "  REW_MNIST_DIR unset — using a synthetic ", SyntheticBatch,
        "-sample batch"
    let (sx, sy) = syntheticBatch(d, keys[0])
    x = sx
    y = sy

  echo &"  x.shape = {x.shape}, y.shape = {y.shape}"
  let batchSize = x.shape[0]

  proc accuracy(logits, oneHot: Tensor): float32 =
    ## Host-side argmax accuracy on a `[N, NumClasses]` minibatch.
    let lh = readBackF32(logits)
    let yh = readBackF32(oneHot)
    var correct = 0
    let n = logits.shape[0]
    for i in 0 ..< n:
      var bestLogit = -Inf.float32
      var bestL = 0
      var bestY = 0
      var bestYV = -1.0'f32
      for c in 0 ..< NumClasses:
        let lv = lh[i * NumClasses + c]
        if lv > bestLogit:
          bestLogit = lv; bestL = c
        let yv = yh[i * NumClasses + c]
        if yv > bestYV:
          bestYV = yv; bestY = c
      if bestL == bestY: inc correct
    float32(correct) / float32(n)

  # Build a tiny MLP eagerly: 784 -> 64 -> 10.
  var layer1 = initLinearEager(d, keys[1], ImagePixels, HiddenDim)
  var layer2 = initLinearEager(d, keys[2], HiddenDim, NumClasses)

  let logits = forward(layer1, layer2, x)
  echo &"  logits.shape = {logits.shape}"
  let initLoss = readBackF32(softmaxCrossEntropy(logits, y))[0]
  let initAcc = accuracy(logits, y)
  echo &"  initial loss = {initLoss:.4f}, accuracy = {initAcc * 100:.1f}%"

  # ---- one full SGD training step via jit(vjp) -----------------------
  #
  # We trace `loss + sgd_update` as a single StableHLO function that
  # consumes `[w1, b1, w2, b2, x, y, lr]` and returns
  # `[loss, w1', b1', w2', b2']`. Donating the param slots lets PJRT
  # reuse the input buffers for the updates.
  let trainFn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let xa = args[4]
    let ya = args[5]
    let lr = args[6]
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let l1 = Linear(weight: param(p[0]), bias: param(p[1]))
      let l2 = Linear(weight: param(p[2]), bias: param(p[3]))
      softmaxCrossEntropy(forward(l2, relu(forward(l1, xa))), ya)
    let vr = vjp(lossFn, [args[0], args[1], args[2], args[3]])
    let grads = vr.pullback(scalarF32(1'f32))
    proc upd(p, g: Tensor): Tensor =
      var bdims: seq[int] = @[]
      let lrB = broadcastTo(lr, p.shape, bdims)
      sub(p, mul(lrB, g))
    @[vr.output,
      upd(args[0], grads[0]),
      upd(args[1], grads[1]),
      upd(args[2], grads[2]),
      upd(args[3], grads[3])]

  let trainJ = jit(trainFn, "mnist_train_step", donateArgs = [0, 1, 2, 3])

  var lrHost = @[0.01'f32]
  let lr = f32EagerTensor(d, [], lrHost)

  const Steps = 5
  for step in 1 .. Steps:
    let outs = trainJ.call([layer1.weight.value, layer1.bias.value,
                            layer2.weight.value, layer2.bias.value,
                            x, y, lr])
    layer1 = Linear(weight: param(outs[1]), bias: param(outs[2]))
    layer2 = Linear(weight: param(outs[3]), bias: param(outs[4]))
    let l = readBackF32(outs[0])[0]
    let acc = accuracy(forward(layer1, layer2, x), y)
    echo &"  step {step}: loss = {l:.4f}, accuracy = {acc * 100:.1f}% (batch={batchSize})"

when isMainModule:
  run()
