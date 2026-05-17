## End-to-end eager dispatch via the PJRT backend.

import std/math
import rew
import rew/xla
import rew/eager
import rew/pjrt/loader
import rew/binaries/target

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "teager_dispatch: skipped — ", e.msg
    false

proc readBack(t: Tensor; n: int): seq[float32] =
  result = newSeq[float32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(float32))

proc readBackU32(t: Tensor; n: int): seq[uint32] =
  result = newSeq[uint32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(uint32))

proc readBackI32(t: Tensor; n: int): seq[int32] =
  result = newSeq[int32](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(int32))

proc readBackBool(t: Tensor; n: int): seq[bool] =
  result = newSeq[bool](n)
  if n > 0:
    transferToHost(t.device, t.buffer, addr result[0], n * sizeof(bool))

proc f32Tensor(d: Device; data: openArray[float32]): Tensor =
  var local = @data
  let dims: array[1, int64] = [int64(local.len)]
  let h = transferToDevice(d, addr local[0], dtFloat32, dims,
                           sizeBytes = local.len * sizeof(float32))
  initEagerTensor(h, dtFloat32, [local.len], d)

proc u32Tensor(d: Device; data: openArray[uint32]): Tensor =
  var local = @data
  let dims: array[1, int64] = [int64(local.len)]
  let h = transferToDevice(d, addr local[0], dtUint32, dims,
                           sizeBytes = local.len * sizeof(uint32))
  initEagerTensor(h, dtUint32, [local.len], d)

proc i32Tensor(d: Device; data: openArray[int32]): Tensor =
  var local = @data
  let dims: array[1, int64] = [int64(local.len)]
  let h = transferToDevice(d, addr local[0], dtInt32, dims,
                           sizeBytes = local.len * sizeof(int32))
  initEagerTensor(h, dtInt32, [local.len], d)

proc i32Scalar(d: Device; value: int32): Tensor =
  var local = @[value]
  let dims: seq[int64] = @[]
  let h = transferToDevice(d, addr local[0], dtInt32, dims,
                           sizeBytes = sizeof(int32))
  initEagerTensor(h, dtInt32, [], d)

proc i32Vector(d: Device; data: openArray[int32]): Tensor =
  var local = @data
  let dims: array[1, int64] = [int64(local.len)]
  let h = transferToDevice(d, addr local[0], dtInt32, dims,
                           sizeBytes = local.len * sizeof(int32))
  initEagerTensor(h, dtInt32, [local.len], d)

proc checkClose(got, want: openArray[float32]; tol = 1e-4'f32) =
  doAssert got.len == want.len
  for i in 0 ..< got.len:
    doAssert abs(got[i] - want[i]) < tol

block eager_dispatch_or_skip:
  if not canLoadCpu(): break eager_dispatch_or_skip
  let d = cpu(0)
  setDefaultDevice(d)
  installEagerBackend()

  let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
  let b = f32Tensor(d, [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32])

  block literalAndFactoryCheck:
    let c = constantF32([4], [5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32], d)
    checkClose(readBack(c, 4), @[5.0'f32, 6.0'f32, 7.0'f32, 8.0'f32])
    let z = zeros([4], dtFloat32, d)
    let o = ones([4], dtFloat32, d)
    let f = full([4], 3.5'f32, dtFloat32, d)
    checkClose(readBack(z, 4), @[0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32])
    checkClose(readBack(o, 4), @[1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    checkClose(readBack(f, 4), @[3.5'f32, 3.5'f32, 3.5'f32, 3.5'f32])

  block nearestInterpolateCheck:
    let img = reshape(f32Tensor(d,
      [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32]),
      [1, 2, 3, 1])
    let resized = interpolate(img, [3, 2], ipNearest)
    doAssert resized.shape == @[1, 3, 2, 1]
    checkClose(readBack(resized, 6),
      @[1.0'f32, 2.0'f32, 1.0'f32, 2.0'f32, 4.0'f32, 5.0'f32])

  block linearAndCubicInterpolateCheck:
    let vec = f32Tensor(d, [0.0'f32, 10.0'f32, 20.0'f32])
    let linear = interpolate(vec, [5], ipBilinear)
    doAssert linear.shape == @[5]
    checkClose(readBack(linear, 5),
      @[0.0'f32, 6.0'f32, 12.0'f32, 18.0'f32, 20.0'f32])

    let cubic = interpolate(vec, [5], ipBicubic)
    doAssert cubic.shape == @[5]
    checkClose(readBack(cubic, 5),
      @[0.0'f32, 5.52'f32, 12.16'f32, 18.64'f32, 20.72'f32])

  block spatialBilinearAndBicubicInterpolateCheck:
    let img = reshape(f32Tensor(d,
      [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]), [1, 2, 2, 1])

    let linear = interpolate(img, [3, 3], ipBilinear)
    doAssert linear.shape == @[1, 3, 3, 1]
    checkClose(readBack(linear, 9), @[
      1.0'f32, 1.6666666'f32, 2.0'f32,
      2.3333333'f32, 3.0'f32, 3.3333333'f32,
      3.0'f32, 3.6666667'f32, 4.0'f32,
    ])

    let cubic = interpolate(img, [3, 3], ipBicubic)
    doAssert cubic.shape == @[1, 3, 3, 1]
    checkClose(readBack(cubic, 9), @[
      1.0'f32, 1.7037037'f32, 2.074074'f32,
      2.4074074'f32, 3.1111112'f32, 3.4814816'f32,
      3.1481481'f32, 3.851852'f32, 4.2222223'f32,
    ])

  block foldAndUnfoldCheck:
    let img = reshape(f32Tensor(d, [
      1.0'f32, 2.0'f32, 3.0'f32,
      4.0'f32, 5.0'f32, 6.0'f32,
      7.0'f32, 8.0'f32, 9.0'f32,
    ]), [1, 1, 3, 3])
    let patches = unfold(img, [2, 2])
    doAssert patches.shape == @[1, 4, 4]
    checkClose(readBack(patches, 16), @[
      1.0'f32, 2.0'f32, 4.0'f32, 5.0'f32,
      2.0'f32, 3.0'f32, 5.0'f32, 6.0'f32,
      4.0'f32, 5.0'f32, 7.0'f32, 8.0'f32,
      5.0'f32, 6.0'f32, 8.0'f32, 9.0'f32,
    ])

    let folded = fold(patches, [3, 3], [2, 2])
    doAssert folded.shape == @[1, 1, 3, 3]
    checkClose(readBack(folded, 9), @[
      1.0'f32, 4.0'f32, 3.0'f32,
      8.0'f32, 20.0'f32, 12.0'f32,
      7.0'f32, 16.0'f32, 9.0'f32,
    ])

    let padded = unfold(reshape(f32Tensor(d, [
      1.0'f32, 2.0'f32,
      3.0'f32, 4.0'f32,
    ]), [1, 1, 2, 2]), [2, 2], padding = [1, 1])
    doAssert padded.shape == @[1, 4, 9]
    checkClose(readBack(padded, 36), @[
      0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32, 1.0'f32,
      2.0'f32, 0.0'f32, 3.0'f32, 4.0'f32,
      0.0'f32, 0.0'f32, 0.0'f32, 1.0'f32, 2.0'f32,
      0.0'f32, 3.0'f32, 4.0'f32, 0.0'f32,
      0.0'f32, 1.0'f32, 2.0'f32, 0.0'f32, 3.0'f32,
      4.0'f32, 0.0'f32, 0.0'f32, 0.0'f32,
      1.0'f32, 2.0'f32, 0.0'f32, 3.0'f32, 4.0'f32,
      0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32,
    ])

  block gridSampleCheck:
    let img = reshape(f32Tensor(d, [
      1.0'f32, 2.0'f32,
      3.0'f32, 4.0'f32,
    ]), [1, 1, 2, 2])
    let grid = constantF32([1, 2, 2, 2], [
      -1.0'f32, -1.0'f32,
       1.0'f32, -1.0'f32,
      -1.0'f32,  1.0'f32,
       0.0'f32,  0.0'f32,
    ], d)

    let linear = gridSample(img, grid, gsBilinear, alignCorners = true)
    doAssert linear.shape == @[1, 1, 2, 2]
    checkClose(readBack(linear, 4),
      @[1.0'f32, 2.0'f32, 3.0'f32, 2.5'f32])

    let nearest = gridSample(img, grid, gsNearest, alignCorners = true)
    doAssert nearest.shape == @[1, 1, 2, 2]
    checkClose(readBack(nearest, 4),
      @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])

  block maxPoolWithIndicesAndUnpoolCheck:
    let img = reshape(f32Tensor(d, [
      1.0'f32, 2.0'f32, 3.0'f32,
      4.0'f32, 5.0'f32, 6.0'f32,
      7.0'f32, 8.0'f32, 9.0'f32,
    ]), [1, 3, 3, 1])
    let pooled = maxPool2dWithIndices(img, [2, 2], [1, 1])
    doAssert pooled.values.shape == @[1, 2, 2, 1]
    doAssert pooled.indices.shape == @[1, 2, 2, 1]
    checkClose(readBack(pooled.values, 4),
      @[5.0'f32, 6.0'f32, 8.0'f32, 9.0'f32])
    doAssert readBackI32(pooled.indices, 4) == @[4'i32, 5'i32, 7'i32, 8'i32]

    let unpooled = maxUnpool2d(pooled.values, pooled.indices, [3, 3])
    doAssert unpooled.shape == @[1, 3, 3, 1]
    checkClose(readBack(unpooled, 9), @[
      0.0'f32, 0.0'f32, 0.0'f32,
      0.0'f32, 5.0'f32, 6.0'f32,
      0.0'f32, 8.0'f32, 9.0'f32,
    ])

  block ctcLossCheck:
    let logProbs = constantF32([2, 1, 2], [
      ln(0.6'f32), ln(0.4'f32),
      ln(0.3'f32), ln(0.7'f32),
    ], d)
    let targets = reshape(i32Tensor(d, [1'i32]), [1, 1])
    let inputLengths = i32Tensor(d, [2'i32])
    let targetLengths = i32Tensor(d, [1'i32])
    let loss = ctcLoss(logProbs, targets, inputLengths, targetLengths)
    doAssert loss.shape.len == 0
    checkClose(readBack(loss, 1), @[-ln(0.82'f32)])

  block tensorProductCheck:
    let tp = TensorProduct(
      weights: param(constantF32([1, 1], [2.0'f32], d)),
      cgCoeffs: buffer(constantF32([1, 1, 1, 1], [1.0'f32], d)),
      inIrreps: @[0],
      outIrreps: @[0],
      sharedIrreps: @[0],
      totalChannels: 1,
      outChannels: 1,
    )
    let x1 = reshape(f32Tensor(d, [3.0'f32, 4.0'f32]), [2, 1])
    let x2 = reshape(f32Tensor(d, [5.0'f32, 6.0'f32]), [2, 1])
    let y = tp.forward(x1, x2)
    doAssert y.shape == @[2, 1]
    checkClose(readBack(y, 2), @[30.0'f32, 48.0'f32])

  block qloraInitializationCheck:
    let qhost = initQuantizedLoraLinearFromF32(initKey(7u64),
      [1.0'f32, 0.0'f32, 0.0'f32, 1.0'f32], 2, 2,
      rank = 1, groupSize = 3)
    doAssert qhost.qweight.len == 2
    doAssert qhost.qscales.len == 2
    doAssert qhost.scaling == 1.0'f32

    let layer = initQloraLinearFromF32(initKey(8u64),
      [1.0'f32, 0.0'f32, 0.0'f32, 1.0'f32], 2, 2,
      bias = [0.5'f32, -1.0'f32], rank = 1, alpha = 1.0'f32,
      groupSize = 3)
    let y = layer.forward(reshape(f32Tensor(d, [2.0'f32, 3.0'f32]), [1, 2]))
    doAssert y.shape == @[1, 2]
    checkClose(readBack(y, 2), @[2.5'f32, 2.0'f32])

  block addCheck:
    let r = add(a, b)
    let got = readBack(r, 4)
    doAssert got == @[11.0'f32, 22.0'f32, 33.0'f32, 44.0'f32]

  block mulCheck:
    let r = mul(a, b)
    let got = readBack(r, 4)
    doAssert got == @[10.0'f32, 40.0'f32, 90.0'f32, 160.0'f32]

  block clampScalarBoundsCheck:
    let low = scalarF32(d, 1.5'f32)
    let high = scalarF32(d, 3.5'f32)
    let r = clamp(low, a, high)
    doAssert r.shape == @[4]
    checkClose(readBack(r, 4), @[1.5'f32, 2.0'f32, 3.0'f32, 3.5'f32])

  block negCheck:
    let r = neg(a)
    let got = readBack(r, 4)
    doAssert got == @[-1.0'f32, -2.0'f32, -3.0'f32, -4.0'f32]

  block expCheck:
    let r = exp(a)
    let got = readBack(r, 4)
    for i, v in got:
      let want = exp(float32(i + 1))
      doAssert abs(v - want) < 1e-5'f32

  block openxlaBinaryCheck:
    let gotAtan2 = readBack(atan2(a, b), 4)
    var wantAtan2 = newSeq[float32](4)
    for i in 0 ..< wantAtan2.len:
      wantAtan2[i] = arctan2(float32(i + 1), float32((i + 1) * 10))
    checkClose(gotAtan2, wantAtan2)

    let base = f32Tensor(d, [1.5'f32, 2.0'f32, 2.5'f32, 3.0'f32])
    let exponent = f32Tensor(d, [2.0'f32, 3.0'f32, 2.0'f32, 2.0'f32])
    let gotPower = readBack(power(base, exponent), 4)
    var wantPower = newSeq[float32](4)
    let baseVals = [1.5'f32, 2.0'f32, 2.5'f32, 3.0'f32]
    let exponentVals = [2.0'f32, 3.0'f32, 2.0'f32, 2.0'f32]
    for i in 0 ..< wantPower.len:
      wantPower[i] = pow(baseVals[i], exponentVals[i])
    checkClose(gotPower, wantPower)

    let dividend = f32Tensor(d, [5.5'f32, 7.25'f32, 9.0'f32, 10.5'f32])
    let divisor = f32Tensor(d, [2.0'f32, 3.0'f32, 4.0'f32, 4.0'f32])
    let gotRemainder = readBack(remainder(dividend, divisor), 4)
    checkClose(gotRemainder, @[1.5'f32, 1.25'f32, 1.0'f32, 2.5'f32])

  block openxlaUnaryCheck:
    let frac = f32Tensor(d, [1.2'f32, -1.2'f32, 2.8'f32, -2.8'f32])
    doAssert readBack(ceil(frac), 4) == @[2.0'f32, -1.0'f32, 3.0'f32, -2.0'f32]
    doAssert readBack(floor(frac), 4) == @[1.0'f32, -2.0'f32, 2.0'f32, -3.0'f32]

    let gotCbrt = readBack(cbrt(a), 4)
    var wantCbrt = newSeq[float32](4)
    for i in 0 ..< wantCbrt.len:
      wantCbrt[i] = pow(float64(i + 1), 1.0 / 3.0).float32
    checkClose(gotCbrt, wantCbrt)

    let gotExpm1 = readBack(expm1(a), 4)
    var wantExpm1 = newSeq[float32](4)
    for i in 0 ..< wantExpm1.len:
      wantExpm1[i] = exp(float32(i + 1)) - 1'f32
    checkClose(gotExpm1, wantExpm1)

    let gotLog1p = readBack(log1p(a), 4)
    var wantLog1p = newSeq[float32](4)
    for i in 0 ..< wantLog1p.len:
      wantLog1p[i] = ln(float32(i + 2))
    checkClose(gotLog1p, wantLog1p)

    let gotLogistic = readBack(logistic(a), 4)
    var wantLogistic = newSeq[float32](4)
    for i in 0 ..< wantLogistic.len:
      let x = float32(i + 1)
      wantLogistic[i] = 1'f32 / (1'f32 + exp(-x))
    checkClose(gotLogistic, wantLogistic)

    let gotTan = readBack(tan(a), 4)
    var wantTan = newSeq[float32](4)
    for i in 0 ..< wantTan.len:
      wantTan[i] = tan(float32(i + 1))
    checkClose(gotTan, wantTan)

    let signed = f32Tensor(d, [-2.0'f32, 0.0'f32, 3.5'f32, -4.5'f32])
    checkClose(readBack(sign(signed), 4),
      @[-1.0'f32, 0.0'f32, 1.0'f32, -1.0'f32])

    let roundVals = f32Tensor(d, [1.5'f32, 2.5'f32, -1.5'f32, -2.5'f32])
    checkClose(readBack(roundNearestAfz(roundVals), 4),
      @[2.0'f32, 3.0'f32, -2.0'f32, -3.0'f32])
    checkClose(readBack(roundNearestEven(roundVals), 4),
      @[2.0'f32, 2.0'f32, -2.0'f32, -2.0'f32])

  block openxlaBitwiseCheck:
    let x = u32Tensor(d, [0x0Fu32, 0xF0u32, 0xAAu32, 0x55u32])
    let y = u32Tensor(d, [0x33u32, 0x33u32, 0x0Fu32, 0xF0u32])
    doAssert readBackU32(bitwiseAnd(x, y), 4) ==
      @[0x03u32, 0x30u32, 0x0Au32, 0x50u32]
    doAssert readBackU32(bitwiseOr(x, y), 4) ==
      @[0x3Fu32, 0xF3u32, 0xAFu32, 0xF5u32]
    doAssert readBackU32(bitwiseXor(x, y), 4) ==
      @[0x3Cu32, 0xC3u32, 0xA5u32, 0xA5u32]
    doAssert readBackU32(bitwiseNot(x), 4) ==
      @[not 0x0Fu32, not 0xF0u32, not 0xAAu32, not 0x55u32]

  block openxlaShiftCheck:
    let shifts = u32Tensor(d, [1u32, 2u32, 3u32, 4u32])
    let leftVals = u32Tensor(d, [1u32, 2u32, 4u32, 8u32])
    doAssert readBackU32(shiftLeft(leftVals, shifts), 4) ==
      @[2u32, 8u32, 32u32, 128u32]

    let logicalVals = u32Tensor(d, [0x80000000u32, 16u32, 255u32, 1024u32])
    let logicalShifts = u32Tensor(d, [31u32, 2u32, 4u32, 10u32])
    doAssert readBackU32(shiftRightLogical(logicalVals, logicalShifts), 4) ==
      @[1u32, 4u32, 15u32, 1u32]

    let arithmeticVals = i32Tensor(d, [-8'i32, -16'i32, 32'i32, 64'i32])
    let arithmeticShifts = i32Tensor(d, [1'i32, 2'i32, 3'i32, 4'i32])
    doAssert readBackI32(shiftRightArithmetic(arithmeticVals, arithmeticShifts),
      4) == @[-4'i32, -4'i32, 4'i32, 4'i32]

  block openxlaIntegerBitCountCheck:
    let clzVals = u32Tensor(d, [0u32, 1u32, 0x80000000u32, 0x00F00000u32])
    doAssert readBackU32(countLeadingZeros(clzVals), 4) ==
      @[32u32, 31u32, 0u32, 8u32]

    let popVals = u32Tensor(d, [0u32, 1u32, 3u32, 15u32])
    doAssert readBackU32(popcnt(popVals), 4) == @[0u32, 1u32, 2u32, 4u32]

  block optimizationBarrierCheck:
    doAssert readBack(optimizationBarrier(a), 4) ==
      @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

  block openxlaTypeChangingUnaryCheck:
    let finiteVals = f32Tensor(d,
      [NegInf.float32, Inf.float32, NaN.float32, -10.0'f32, 0.0'f32])
    doAssert isFinite(finiteVals).dtype == dtBool
    doAssert readBackBool(isFinite(finiteVals), 5) ==
      @[false, false, false, true, true]

    let ints = i32Tensor(d, [1'i32, -2'i32, 3'i32, 4'i32])
    let asFloat = astype(ints, dtFloat32)
    doAssert asFloat.dtype == dtFloat32
    doAssert readBack(asFloat, 4) == @[1.0'f32, -2.0'f32, 3.0'f32, 4.0'f32]

    let rawFloats = u32Tensor(d,
      [0x3f800000'u32, 0x40000000'u32, 0x40400000'u32, 0x40800000'u32])
    let bitcast = bitcastConvert(rawFloats, dtFloat32, [4])
    doAssert bitcast.dtype == dtFloat32
    doAssert readBack(bitcast, 4) ==
      @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

  block openxlaReducePrecisionCheck:
    let r = reducePrecision(a, 8, 23)
    doAssert r.dtype == dtFloat32
    doAssert readBack(r, 4) ==
      @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

  block openxlaBatchNormInferenceCheck:
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let offset = f32Tensor(d, [0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32])
    let mean = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let variance = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let r = batchNormInference(a, scale, offset, mean, variance, 0.0'f32, 0)
    doAssert r.dtype == dtFloat32
    doAssert readBack(r, 4) == @[0.0'f32, 1.0'f32, 2.0'f32, 3.0'f32]

  block openxlaBatchNormTrainingCheck:
    let x = reshape(a, [2, 2])
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32])
    let offset = f32Tensor(d, [0.0'f32, 0.0'f32])
    let outs = batchNormTraining(x, scale, offset, 0.0'f32, 1)
    doAssert outs.output.shape == @[2, 2]
    doAssert outs.batchMean.shape == @[2]
    doAssert outs.batchVar.shape == @[2]
    checkClose(readBack(outs.output, 4),
      @[-1.0'f32, -1.0'f32, 1.0'f32, 1.0'f32])
    doAssert readBack(outs.batchMean, 2) == @[2.0'f32, 3.0'f32]
    doAssert readBack(outs.batchVar, 2) == @[1.0'f32, 1.0'f32]

  block openxlaBatchNormGradCheck:
    let x = reshape(a, [2, 2])
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32])
    let mean = f32Tensor(d, [2.0'f32, 3.0'f32])
    let variance = f32Tensor(d, [1.0'f32, 1.0'f32])
    let gradOutput = reshape(
      f32Tensor(d, [0.1'f32, 0.1'f32, 0.1'f32, 0.1'f32]), [2, 2])
    let outs = batchNormGrad(x, scale, mean, variance, gradOutput,
      0.0'f32, 1)
    doAssert outs.gradOperand.shape == @[2, 2]
    doAssert outs.gradScale.shape == @[2]
    doAssert outs.gradOffset.shape == @[2]
    checkClose(readBack(outs.gradOperand, 4),
      @[0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32])
    checkClose(readBack(outs.gradScale, 2), @[0.0'f32, 0.0'f32])
    checkClose(readBack(outs.gradOffset, 2), @[0.2'f32, 0.2'f32])

  block openxlaCholeskyGetDimensionSizeAndPadCheck:
    let spd = reshape(f32Tensor(d, [4.0'f32, 2.0'f32, 2.0'f32, 3.0'f32]),
      [2, 2])
    let chol = cholesky(spd)
    doAssert chol.shape == @[2, 2]
    checkClose(readBack(chol, 4), @[2.0'f32, 0.0'f32, 1.0'f32, sqrt(2.0'f32)])

    let dim = getDimensionSize(spd, 0)
    doAssert dim.dtype == dtInt32
    doAssert dim.shape == newSeq[int]()
    doAssert readBackI32(dim, 1) == @[2'i32]

    let padded = pad(spd, scalarF32(d, 0.0'f32), [1, 1], [1, 1], [0, 0])
    doAssert padded.shape == @[4, 4]
    doAssert readBack(padded, 16) == @[
      0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32,
      0.0'f32, 4.0'f32, 2.0'f32, 0.0'f32,
      0.0'f32, 2.0'f32, 3.0'f32, 0.0'f32,
      0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32]

  block openxlaBroadcastAndDynamicSliceCheck:
    let x = reshape(a, [2, 2])
    let start0 = i32Scalar(d, 0'i32)
    let start1 = i32Scalar(d, 1'i32)
    let broad = broadcast(x, [3])
    doAssert broad.shape == @[3, 2, 2]
    doAssert readBack(broad, 12) == @[
      1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32,
      1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32,
      1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

    let sliced = dynamicSlice(x, [start0, start1], [2, 1])
    doAssert sliced.shape == @[2, 1]
    doAssert readBack(sliced, 2) == @[2.0'f32, 4.0'f32]

    let update = reshape(f32Tensor(d, [9.0'f32]), [1, 1])
    let updated = dynamicUpdateSlice(x, update, [start0, start1])
    doAssert updated.shape == @[2, 2]
    doAssert readBack(updated, 4) == @[1.0'f32, 9.0'f32, 3.0'f32, 4.0'f32]

  block openxlaIotaCheck:
    let r = iota(dtInt32, [2, 3], 1, d)
    doAssert r.dtype == dtInt32
    doAssert r.shape == @[2, 3]
    doAssert readBackI32(r, 6) == @[0'i32, 1'i32, 2'i32, 0'i32, 1'i32, 2'i32]

  block openxlaReplicaAndPartitionIdCheck:
    let rid = replicaId(d)
    let pid = partitionId(d)
    doAssert rid.dtype == dtUint32
    doAssert pid.dtype == dtUint32
    doAssert readBackU32(rid, 1) == @[0'u32]
    doAssert readBackU32(pid, 1) == @[0'u32]

  block openxlaDynamicShapeOpsCheck:
    let x = reshape(a, [2, 2])
    let sized = setDimensionSize(x, i32Scalar(d, 2'i32), 0)
    doAssert sized.shape == @[2, 2]
    doAssert readBack(sized, 4) == readBack(x, 4)

    let reshaped = dynamicReshape(x, i32Vector(d, [4'i32]), [4])
    doAssert reshaped.shape == @[4]
    doAssert readBack(reshaped, 4) == readBack(a, 4)

  block openxlaDynamicIotaCheck:
    let dynIota = dynamicIota(dtInt32, i32Vector(d, [2'i32, 3'i32]),
      [2, 3], 1)
    doAssert dynIota.shape == @[2, 3]
    doAssert readBackI32(dynIota, 6) ==
      @[0'i32, 1'i32, 2'i32, 0'i32, 1'i32, 2'i32]

  block cacheReuse:
    let r1 = add(a, b)
    let r2 = add(a, b)
    let g1 = readBack(r1, 4)
    let g2 = readBack(r2, 4)
    doAssert g1 == g2

  block reshapeCheck:
    let r = reshape(a, [2, 2])
    doAssert r.shape == @[2, 2]
    let got = readBack(r, 4)
    doAssert got == @[1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32]

  block transposeCheck:
    let m = reshape(a, [2, 2])
    let r = transpose(m, [1, 0])
    doAssert r.shape == @[2, 2]
    let got = readBack(r, 4)
    doAssert got == @[1.0'f32, 3.0'f32, 2.0'f32, 4.0'f32]

  block reverseCheck:
    let r = reverse(a, [0])
    doAssert r.shape == @[4]
    let got = readBack(r, 4)
    doAssert got == @[4.0'f32, 3.0'f32, 2.0'f32, 1.0'f32]

  block reduceSumCheck:
    let r = reduceSum(a, [0])
    doAssert r.shape.len == 0
    let got = readBack(r, 1)
    doAssert got == @[10.0'f32]

  block reduceProdCheck:
    let r = reduceProd(reshape(a, [2, 2]), [1])
    doAssert r.shape == @[2]
    let got = readBack(r, 2)
    doAssert got == @[2.0'f32, 12.0'f32]

  block boolReductionCheck:
    let flags = fromHost(d,
      [true, true, false, false, false, true], [2, 3])
    let allRows = all(flags, [1])
    let anyCols = any(flags, [0], keepdims = true)
    doAssert allRows.shape == @[2]
    doAssert anyCols.shape == @[1, 3]
    doAssert readBackBool(allRows, 2) == @[false, false]
    doAssert readBackBool(anyCols, 3) == @[true, true, true]

  block matmulCheck:
    let m1 = reshape(a, [2, 2])
    let m2 = reshape(b, [2, 2])
    let r = matmul(m1, m2)
    doAssert r.shape == @[2, 2]
    let got = readBack(r, 4)
    doAssert got == @[70.0'f32, 100.0'f32, 150.0'f32, 220.0'f32]

  block dotCheck:
    let m1 = reshape(a, [2, 2])
    let m2 = reshape(b, [2, 2])
    let r = dot(m1, m2)
    doAssert r.shape == @[2, 2]
    let got = readBack(r, 4)
    doAssert got == @[70.0'f32, 100.0'f32, 150.0'f32, 220.0'f32]

  block broadcastCheck:
    let s = f32Tensor(d, [5.0'f32])
    let r = broadcastTo(s, [3], [0])
    doAssert r.shape == @[3]
    let got = readBack(r, 3)
    doAssert got == @[5.0'f32, 5.0'f32, 5.0'f32]

  block clearCacheCheck:
    clearEagerCache()
    let r = add(a, b)
    let got = readBack(r, 4)
    doAssert got == @[11.0'f32, 22.0'f32, 33.0'f32, 44.0'f32]

  echo "teager_dispatch: eager elementwise + reshape + transpose + " &
    "reduce + matmul + broadcast OK on CPU plugin"
