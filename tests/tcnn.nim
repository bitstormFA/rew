## End-to-end CNN op tests \u2014 eager and jit numerical correctness.
##
## Runs `conv2d` and `maxPool2d` through the PJRT eager backend and the
## `jit` transform and asserts equality. Skips cleanly when no PJRT
## plugin is installed.

import std/[math, strutils]
import rew
import rew/xla
import rew/pjrt/loader
import rew/binaries/target

let TestDevice = cpu(0)

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "tcnn: skipped \u2014 ", e.msg
    false

proc readBack(t: Tensor; n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

proc f32Buffer(d: Device; shape: openArray[int];
    data: openArray[float32]): Tensor =
  var local = @data
  var dims = newSeq[int64](shape.len)
  for i, s in shape: dims[i] = int64(s)
  let h = transferToDevice(d, addr local[0], dtFloat32, dims,
    sizeBytes = local.len * sizeof(float32))
  initEagerTensor(h, dtFloat32, shape, d)

block conv2d_eager_or_skip:
  if not canLoadCpu(): break conv2d_eager_or_skip
  let d = cpu(0)
  setDefaultDevice(d)
  installEagerBackend()

  # Single-channel 3x3 input, 1x1 identity-like kernel of size 1x1,
  # padding=0, stride=1 \u2014 result equals the input.
  let x = f32Buffer(d, [1, 3, 3, 1],
    [1.0'f32, 2, 3, 4, 5, 6, 7, 8, 9])
  let k = f32Buffer(d, [1, 1, 1, 1], [1.0'f32])
  let y = conv2d(x, k, [1, 1], [[0, 0], [0, 0]])
  doAssert y.shape == @[1, 3, 3, 1]
  let got = readBack(y, 9)
  doAssert got == @[1.0'f32, 2, 3, 4, 5, 6, 7, 8, 9]

  # 3x3 box filter with padding=1 against a 4x4 input. We compare the
  # eager result with a hand-computed reference.
  let xs = [
    1.0'f32, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1]
  let x2 = f32Buffer(d, [1, 4, 4, 1], xs)
  let kBox = f32Buffer(d, [1, 1, 3, 3],
    [1.0'f32, 1, 1,
     1, 1, 1,
     1, 1, 1])
  let y2 = conv2d(x2, kBox, [1, 1], [[1, 1], [1, 1]])
  doAssert y2.shape == @[1, 4, 4, 1]
  let got2 = readBack(y2, 16)
  # Centre cells see all 9 ones, edge cells see 6, corners see 4.
  let expected2 = @[
    4.0'f32, 6, 6, 4,
    6, 9, 9, 6,
    6, 9, 9, 6,
    4, 6, 6, 4]
  for i, v in got2:
    doAssert abs(v - expected2[i]) < 1e-4'f32

  echo "tcnn: conv2d eager OK"

block maxpool2d_eager_or_skip:
  if not canLoadCpu(): break maxpool2d_eager_or_skip
  let d = cpu(0)

  # 4x4 input pooled to 2x2 with kernel 2 and stride 2.
  let xs = [
    1.0'f32, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16]
  let x = f32Buffer(d, [1, 4, 4, 1], xs)
  let y = maxPool2d(x, [2, 2], [2, 2])
  doAssert y.shape == @[1, 2, 2, 1]
  let got = readBack(y, 4)
  doAssert got == @[6.0'f32, 8, 14, 16]

  echo "tcnn: maxPool2d eager OK"

block cnn_full_trace:
  ## Trace a full CNN forward + softmax-CE loss + grad against the
  ## parameters. Verifies the entire CNN pipeline (Conv2d, maxPool2d,
  ## flatten, Linear, ReLU, softmaxCrossEntropy) lowers to valid
  ## StableHLO and that grads have the expected shapes. No PJRT
  ## execution required.
  withTrace ctx, "main", TestDevice:
    let dts = @[dtFloat32, dtFloat32, dtFloat32, dtFloat32,
                dtFloat32, dtFloat32, dtFloat32, dtFloat32,
                dtFloat32, dtFloat32]
    let shapes = @[
      @[8, 1, 3, 3],     # conv1.weight
      @[8],              # conv1.bias
      @[16, 8, 3, 3],    # conv2.weight
      @[16],             # conv2.bias
      @[16 * 7 * 7, 32], # fc1.weight
      @[32],             # fc1.bias
      @[32, 10],         # fc2.weight
      @[10],             # fc2.bias
      @[2, 28, 28, 1],   # x
      @[2, 10],          # y (one-hot)
    ]
    let inputs = ctx.traceInputs(dts, shapes)
    let lossFn = proc(p: openArray[Tensor]): Tensor =
      let c1 = Conv2d(weight: param(p[0]), bias: param(p[1]),
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let c2 = Conv2d(weight: param(p[2]), bias: param(p[3]),
        stride: [1, 1], padding: [[1, 1], [1, 1]], dilation: [1, 1])
      let l1 = Linear(weight: param(p[4]), bias: param(p[5]))
      let l2 = Linear(weight: param(p[6]), bias: param(p[7]))
      let xa = inputs[8]
      let ya = inputs[9]
      let h1 = relu(forward(c1, xa))
      let pp1 = maxPool2d(h1, [2, 2], [2, 2])
      let h2 = relu(forward(c2, pp1))
      let pp2 = maxPool2d(h2, [2, 2], [2, 2])
      let flat = flatten(pp2)
      let h3 = relu(forward(l1, flat))
      let logits = forward(l2, h3)
      softmaxCrossEntropy(logits, ya)
    let grads = grad(lossFn, [
      inputs[0], inputs[1], inputs[2], inputs[3],
      inputs[4], inputs[5], inputs[6], inputs[7]])
    doAssert grads.len == 8
    for i, expected in shapes[0 ..< 8]:
      doAssert grads[i].shape == expected,
        "grad #" & $i & " shape mismatch: got " & $grads[i].shape &
          ", expected " & $expected
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.convolution" in text
  doAssert "stablehlo.reduce_window" in text
  echo "tcnn: full CNN trace + grad OK"

block conv2d_jit_vs_eager_or_skip:
  if not canLoadCpu(): break conv2d_jit_vs_eager_or_skip
  let d = cpu(0)

  let xs = [
    0.5'f32, 1, 1.5, 2,
    2.5, 3, 3.5, 4,
    4.5, 5, 5.5, 6,
    6.5, 7, 7.5, 8]
  let kData = [0.1'f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  let x = f32Buffer(d, [1, 4, 4, 1], xs)
  let k = f32Buffer(d, [1, 1, 3, 3], kData)

  let yEager = conv2d(x, k, [1, 1], [[1, 1], [1, 1]])

  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[conv2d(args[0], args[1], [1, 1], [[1, 1], [1, 1]])]
  let j = jit(f, "conv2dJit")
  let outs = j.call([x, k])

  let gotEager = readBack(yEager, 16)
  let gotJit = readBack(outs[0], 16)
  doAssert gotEager.len == gotJit.len
  for i, ve in gotEager:
    doAssert abs(ve - gotJit[i]) < 1e-4'f32

  echo "tcnn: conv2d jit-vs-eager OK"
