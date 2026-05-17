## End-to-end `jit` execution via the PJRT eager backend.

import std/math
import rew
import rew/xla
import rew/pjrt/loader
import rew/binaries/target

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "tjit_execute: skipped — ", e.msg
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

block jit_execute_or_skip:
  if not canLoadCpu(): break jit_execute_or_skip
  let d = cpu(0)
  setDefaultDevice(d)
  installEagerBackend()

  block addPair:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[add(args[0], args[1])]
    let j = jit(f, "addPair")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let b = f32Tensor(d, [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32])
    let outs = j.call([a, b])
    doAssert outs.len == 1
    doAssert outs[0].shape == @[4]
    doAssert readBack(outs[0], 4) == @[11.0'f32, 22.0'f32, 33.0'f32, 44.0'f32]
    let outs2 = j.call([a, b])
    doAssert readBack(outs2[0], 4) == @[11.0'f32, 22.0'f32, 33.0'f32, 44.0'f32]
    doAssert j.cacheSize == 1

  block lazyBatchExecute:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let y = add(args[0], args[1])
      @[mul(y, y)]
    let lz = lazy(f, "lazyBatchExecute")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let b = f32Tensor(d, [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32])
    let outs = lz.call([a, b])
    doAssert outs.len == 1
    doAssert outs[0].shape == @[4]
    checkClose(readBack(outs[0], 4),
      @[121.0'f32, 484.0'f32, 1089.0'f32, 1936.0'f32])
    doAssert lz.cacheSize == 1

  block openxlaUnaryEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = args[0]
      @[cbrt(x), ceil(x), expm1(x), floor(x), log1p(x), logistic(x),
        tan(x), sign(x), roundNearestAfz(x), roundNearestEven(x)]
    let j = jit(f, "openxlaUnary")
    let a = f32Tensor(d, [0.25'f32, 0.5'f32, 1.0'f32, 1.5'f32])
    let eagerOuts = f([a])
    let jitOuts = j.call([a])
    doAssert jitOuts.len == eagerOuts.len
    for i in 0 ..< jitOuts.len:
      checkClose(readBack(jitOuts[i], 4), readBack(eagerOuts[i], 4))

  block openxlaBinaryEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let a = args[0]
      let b = args[1]
      @[atan2(a, b), power(a, b), remainder(a, b)]
    let j = jit(f, "openxlaBinary")
    let a = f32Tensor(d, [1.5'f32, 2.0'f32, 2.5'f32, 3.0'f32])
    let b = f32Tensor(d, [2.0'f32, 3.0'f32, 2.0'f32, 2.0'f32])
    let eagerOuts = f([a, b])
    let jitOuts = j.call([a, b])
    doAssert jitOuts.len == eagerOuts.len
    for i in 0 ..< jitOuts.len:
      checkClose(readBack(jitOuts[i], 4), readBack(eagerOuts[i], 4))

  block clampScalarBoundsEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[clamp(args[1], args[0], args[2])]
    let j = jit(f, "clampScalarBounds")
    let x = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let low = scalarF32(d, 1.5'f32)
    let high = scalarF32(d, 3.5'f32)
    let inputs = [x, low, high]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts[0].shape == @[4]
    checkClose(readBack(jitOuts[0], 4), readBack(eagerOuts[0], 4))

  block nearestInterpolateEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[interpolate(reshape(args[0], [1, 2, 3, 1]), [3, 2], ipNearest)]
    let j = jit(f, "nearestInterpolate")
    let x = f32Tensor(d,
      [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts[0].shape == @[1, 3, 2, 1]
    checkClose(readBack(jitOuts[0], 6), readBack(eagerOuts[0], 6))

  block bilinearInterpolateEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[interpolate(reshape(args[0], [1, 2, 2, 1]), [3, 3], ipBilinear)]
    let j = jit(f, "bilinearInterpolate")
    let x = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts[0].shape == @[1, 3, 3, 1]
    checkClose(readBack(jitOuts[0], 9), readBack(eagerOuts[0], 9))

  block bicubicInterpolateEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[interpolate(reshape(args[0], [1, 2, 2, 1]), [3, 3], ipBicubic)]
    let j = jit(f, "bicubicInterpolate")
    let x = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts[0].shape == @[1, 3, 3, 1]
    checkClose(readBack(jitOuts[0], 9), readBack(eagerOuts[0], 9))

  block foldAndUnfoldEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let img = reshape(args[0], [1, 1, 3, 3])
      let patches = unfold(img, [2, 2])
      @[patches, fold(patches, [3, 3], [2, 2])]
    let j = jit(f, "foldAndUnfold")
    let x = f32Tensor(d, [
      1.0'f32, 2.0'f32, 3.0'f32,
      4.0'f32, 5.0'f32, 6.0'f32,
      7.0'f32, 8.0'f32, 9.0'f32,
    ])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts.len == 2
    doAssert jitOuts[0].shape == @[1, 4, 4]
    doAssert jitOuts[1].shape == @[1, 1, 3, 3]
    checkClose(readBack(jitOuts[0], 16), readBack(eagerOuts[0], 16))
    checkClose(readBack(jitOuts[1], 9), readBack(eagerOuts[1], 9))

  block gridSampleEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let img = reshape(args[0], [1, 1, 2, 2])
      let grid = reshape(args[1], [1, 2, 2, 2])
      @[gridSample(img, grid, gsBilinear, alignCorners = true),
        gridSample(img, grid, gsNearest, alignCorners = true)]
    let j = jit(f, "gridSample")
    let x = f32Tensor(d, [
      1.0'f32, 2.0'f32,
      3.0'f32, 4.0'f32,
    ])
    let grid = f32Tensor(d, [
      -1.0'f32, -1.0'f32,
       1.0'f32, -1.0'f32,
      -1.0'f32,  1.0'f32,
       0.0'f32,  0.0'f32,
    ])
    let eagerOuts = f([x, grid])
    let jitOuts = j.call([x, grid])
    doAssert jitOuts.len == 2
    doAssert jitOuts[0].shape == @[1, 1, 2, 2]
    doAssert jitOuts[1].shape == @[1, 1, 2, 2]
    checkClose(readBack(jitOuts[0], 4), readBack(eagerOuts[0], 4))
    checkClose(readBack(jitOuts[1], 4), readBack(eagerOuts[1], 4))

  block maxPoolIndicesAndUnpoolEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let img = reshape(args[0], [1, 3, 3, 1])
      let pooled = maxPool2dWithIndices(img, [2, 2], [1, 1])
      @[pooled.values, pooled.indices,
        maxUnpool2d(pooled.values, pooled.indices, [3, 3])]
    let j = jit(f, "maxPoolIndicesAndUnpool")
    let x = f32Tensor(d, [
      1.0'f32, 2.0'f32, 3.0'f32,
      4.0'f32, 5.0'f32, 6.0'f32,
      7.0'f32, 8.0'f32, 9.0'f32,
    ])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts.len == 3
    doAssert jitOuts[0].shape == @[1, 2, 2, 1]
    doAssert jitOuts[1].shape == @[1, 2, 2, 1]
    doAssert jitOuts[2].shape == @[1, 3, 3, 1]
    checkClose(readBack(jitOuts[0], 4), readBack(eagerOuts[0], 4))
    doAssert readBackI32(jitOuts[1], 4) == readBackI32(eagerOuts[1], 4)
    checkClose(readBack(jitOuts[2], 9), readBack(eagerOuts[2], 9))

  block ctcLossEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let logProbs = reshape(args[0], [2, 1, 2])
      let targets = reshape(args[1], [1, 1])
      @[ctcLoss(logProbs, targets, args[2], args[3])]
    let j = jit(f, "ctcLoss")
    let logProbs = f32Tensor(d, [
      ln(0.6'f32), ln(0.4'f32),
      ln(0.3'f32), ln(0.7'f32),
    ])
    let targets = i32Tensor(d, [1'i32])
    let inputLengths = i32Tensor(d, [2'i32])
    let targetLengths = i32Tensor(d, [1'i32])
    let inputs = [logProbs, targets, inputLengths, targetLengths]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts.len == 1
    doAssert jitOuts[0].shape.len == 0
    checkClose(readBack(jitOuts[0], 1), readBack(eagerOuts[0], 1))

  block tensorProductEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let tp = TensorProduct(
        weights: param(constantF32([1, 1], [2.0'f32])),
        cgCoeffs: buffer(constantF32([1, 1, 1, 1], [1.0'f32])),
        inIrreps: @[0],
        outIrreps: @[0],
        sharedIrreps: @[0],
        totalChannels: 1,
        outChannels: 1,
      )
      @[tp.forward(reshape(args[0], [2, 1]), reshape(args[1], [2, 1]))]
    let j = jit(f, "tensorProduct")
    let x1 = f32Tensor(d, [3.0'f32, 4.0'f32])
    let x2 = f32Tensor(d, [5.0'f32, 6.0'f32])
    let eagerOuts = f([x1, x2])
    let jitOuts = j.call([x1, x2])
    doAssert jitOuts.len == 1
    doAssert jitOuts[0].shape == @[2, 1]
    checkClose(readBack(jitOuts[0], 2), readBack(eagerOuts[0], 2))

  block qloraForwardEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let layer = initQloraLinearFromF32(initKey(8u64),
        [1.0'f32, 0.0'f32, 0.0'f32, 1.0'f32], 2, 2,
        bias = [0.5'f32, -1.0'f32], rank = 1, alpha = 1.0'f32,
        groupSize = 3)
      @[layer.forward(reshape(args[0], [1, 2]))]
    let j = jit(f, "qloraForward")
    let x = f32Tensor(d, [2.0'f32, 3.0'f32])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts.len == 1
    doAssert jitOuts[0].shape == @[1, 2]
    checkClose(readBack(jitOuts[0], 2), readBack(eagerOuts[0], 2))

  block dotEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[dot(reshape(args[0], [2, 2]), reshape(args[1], [2, 2]))]
    let j = jit(f, "dot")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let b = f32Tensor(d, [10.0'f32, 20.0'f32, 30.0'f32, 40.0'f32])
    let eagerOuts = f([a, b])
    let jitOuts = j.call([a, b])
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)

  block openxlaBitwiseEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = args[0]
      let y = args[1]
      @[bitwiseAnd(x, y), bitwiseOr(x, y), bitwiseXor(x, y), bitwiseNot(x)]
    let j = jit(f, "openxlaBitwise")
    let x = u32Tensor(d, [0x0Fu32, 0xF0u32, 0xAAu32, 0x55u32])
    let y = u32Tensor(d, [0x33u32, 0x33u32, 0x0Fu32, 0xF0u32])
    let eagerOuts = f([x, y])
    let jitOuts = j.call([x, y])
    doAssert jitOuts.len == eagerOuts.len
    for i in 0 ..< jitOuts.len:
      doAssert readBackU32(jitOuts[i], 4) == readBackU32(eagerOuts[i], 4)

  block openxlaShiftEagerVsJit:
    let shiftUnsigned = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = args[0]
      let s = args[1]
      @[shiftLeft(x, s), shiftRightLogical(x, s)]
    let ju = jit(shiftUnsigned, "openxlaShiftUnsigned")
    let x = u32Tensor(d, [1u32, 16u32, 255u32, 0x80000000u32])
    let s = u32Tensor(d, [1u32, 2u32, 4u32, 31u32])
    let eagerUnsigned = shiftUnsigned([x, s])
    let jitUnsigned = ju.call([x, s])
    for i in 0 ..< jitUnsigned.len:
      doAssert readBackU32(jitUnsigned[i], 4) == readBackU32(eagerUnsigned[i], 4)

    let shiftSigned = proc(args: openArray[Tensor]): seq[Tensor] =
      @[shiftRightArithmetic(args[0], args[1])]
    let js = jit(shiftSigned, "openxlaShiftSigned")
    let sx = i32Tensor(d, [-8'i32, -16'i32, 32'i32, 64'i32])
    let ss = i32Tensor(d, [1'i32, 2'i32, 3'i32, 4'i32])
    let eagerSigned = shiftSigned([sx, ss])
    let jitSigned = js.call([sx, ss])
    doAssert readBackI32(jitSigned[0], 4) == readBackI32(eagerSigned[0], 4)

  block openxlaIntegerBitCountsEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[countLeadingZeros(args[0]), popcnt(args[0])]
    let j = jit(f, "openxlaIntegerBitCounts")
    let x = u32Tensor(d, [0u32, 1u32, 0x80000000u32, 15u32])
    let eagerOuts = f([x])
    let jitOuts = j.call([x])
    doAssert jitOuts.len == eagerOuts.len
    for i in 0 ..< jitOuts.len:
      doAssert readBackU32(jitOuts[i], 4) == readBackU32(eagerOuts[i], 4)

  block reverseEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[reverse(args[0], [0])]
    let j = jit(f, "reverse")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([a])
    let jitOuts = j.call([a])
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)

  block optimizationBarrierEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[optimizationBarrier(args[0])]
    let j = jit(f, "optimizationBarrier")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([a])
    let jitOuts = j.call([a])
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)

  block reductionsEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[reduceProd(reshape(args[0], [2, 2]), [1])]
    let j = jit(f, "reduceProd")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([a])
    let jitOuts = j.call([a])
    doAssert readBack(jitOuts[0], 2) == readBack(eagerOuts[0], 2)

  block boolReductionsEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[all(args[0], [1]), any(args[0], [0], keepdims = true)]
    let j = jit(f, "boolReductions")
    let flags = fromHost(d,
      [true, true, false, false, false, true], [2, 3])
    let eagerOuts = f([flags])
    let jitOuts = j.call([flags])
    doAssert readBackBool(jitOuts[0], 2) == readBackBool(eagerOuts[0], 2)
    doAssert readBackBool(jitOuts[1], 3) == readBackBool(eagerOuts[1], 3)

  block openxlaTypeChangingUnaryEagerVsJit:
    let finiteFn = proc(args: openArray[Tensor]): seq[Tensor] =
      @[isFinite(args[0])]
    let jf = jit(finiteFn, "isFinite")
    let finiteVals = f32Tensor(d,
      [NegInf.float32, Inf.float32, NaN.float32, -10.0'f32])
    let finiteEager = finiteFn([finiteVals])
    let finiteJit = jf.call([finiteVals])
    doAssert readBackBool(finiteJit[0], 4) == readBackBool(finiteEager[0], 4)

    let convertFn = proc(args: openArray[Tensor]): seq[Tensor] =
      @[astype(args[0], dtFloat32)]
    let jc = jit(convertFn, "astypeI32F32")
    let ints = i32Tensor(d, [1'i32, -2'i32, 3'i32, 4'i32])
    let convertEager = convertFn([ints])
    let convertJit = jc.call([ints])
    doAssert readBack(convertJit[0], 4) == readBack(convertEager[0], 4)

    let bitcastFn = proc(args: openArray[Tensor]): seq[Tensor] =
      @[bitcastConvert(args[0], dtFloat32, [4])]
    let jb = jit(bitcastFn, "bitcastConvertU32F32")
    let rawFloats = u32Tensor(d,
      [0x3f800000'u32, 0x40000000'u32, 0x40400000'u32, 0x40800000'u32])
    let bitcastEager = bitcastFn([rawFloats])
    let bitcastJit = jb.call([rawFloats])
    doAssert readBack(bitcastJit[0], 4) == readBack(bitcastEager[0], 4)

  block reducePrecisionEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[reducePrecision(args[0], 8, 23)]
    let j = jit(f, "reducePrecision")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let eagerOuts = f([a])
    let jitOuts = j.call([a])
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)

  block batchNormInferenceEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[batchNormInference(args[0], args[1], args[2],
        args[3], args[4], 0.0'f32, 0)]
    let j = jit(f, "batchNormInference")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let offset = f32Tensor(d, [0.0'f32, 0.0'f32, 0.0'f32, 0.0'f32])
    let mean = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let variance = f32Tensor(d, [1.0'f32, 1.0'f32, 1.0'f32, 1.0'f32])
    let inputs = [a, scale, offset, mean, variance]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)

  block batchNormTrainingEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let outs = batchNormTraining(reshape(args[0], [2, 2]),
        args[1], args[2], 0.0'f32, 1)
      @[outs.output, outs.batchMean, outs.batchVar]
    let j = jit(f, "batchNormTraining")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32])
    let offset = f32Tensor(d, [0.0'f32, 0.0'f32])
    let inputs = [a, scale, offset]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts.len == 3
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)
    doAssert readBack(jitOuts[1], 2) == readBack(eagerOuts[1], 2)
    doAssert readBack(jitOuts[2], 2) == readBack(eagerOuts[2], 2)

  block batchNormGradEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let outs = batchNormGrad(reshape(args[0], [2, 2]), args[1],
        args[2], args[3], reshape(args[4], [2, 2]), 0.0'f32, 1)
      @[outs.gradOperand, outs.gradScale, outs.gradOffset]
    let j = jit(f, "batchNormGrad")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let scale = f32Tensor(d, [1.0'f32, 1.0'f32])
    let mean = f32Tensor(d, [2.0'f32, 3.0'f32])
    let variance = f32Tensor(d, [1.0'f32, 1.0'f32])
    let gradOutput = f32Tensor(d, [0.1'f32, 0.1'f32, 0.1'f32, 0.1'f32])
    let inputs = [a, scale, mean, variance, gradOutput]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts.len == 3
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)
    doAssert readBack(jitOuts[1], 2) == readBack(eagerOuts[1], 2)
    doAssert readBack(jitOuts[2], 2) == readBack(eagerOuts[2], 2)

  block choleskyGetDimensionSizePadEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = reshape(args[0], [2, 2])
      @[cholesky(x), getDimensionSize(x, 0),
        pad(x, args[1], [1, 1], [1, 1], [0, 0])]
    let j = jit(f, "choleskyGetDimensionSizePad")
    let spd = f32Tensor(d, [4.0'f32, 2.0'f32, 2.0'f32, 3.0'f32])
    let zero = scalarF32(d, 0.0'f32)
    let inputs = [spd, zero]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts.len == 3
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)
    doAssert readBackI32(jitOuts[1], 1) == readBackI32(eagerOuts[1], 1)
    doAssert readBack(jitOuts[2], 16) == readBack(eagerOuts[2], 16)

  block broadcastDynamicSliceEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = reshape(args[0], [2, 2])
      let update = reshape(args[1], [1, 1])
      @[broadcast(x, [3]),
        dynamicSlice(x, [args[2], args[3]], [2, 1]),
        dynamicUpdateSlice(x, update, [args[2], args[3]])]
    let j = jit(f, "broadcastDynamicSlice")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let update = f32Tensor(d, [9.0'f32])
    let start0 = i32Scalar(d, 0'i32)
    let start1 = i32Scalar(d, 1'i32)
    let inputs = [a, update, start0, start1]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert jitOuts.len == 3
    doAssert readBack(jitOuts[0], 12) == readBack(eagerOuts[0], 12)
    doAssert readBack(jitOuts[1], 2) == readBack(eagerOuts[1], 2)
    doAssert readBack(jitOuts[2], 4) == readBack(eagerOuts[2], 4)

  block iotaEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[iota(dtInt32, [2, 3], 1)]
    let j = jit(f, "iota")
    let dummy = f32Tensor(d, [0.0'f32])
    let eagerOuts = @[iota(dtInt32, [2, 3], 1, d)]
    let jitOuts = j.call([dummy])
    doAssert readBackI32(jitOuts[0], 6) == readBackI32(eagerOuts[0], 6)

  block replicaPartitionIdEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[replicaId(), partitionId()]
    let j = jit(f, "replicaPartitionId")
    let dummy = f32Tensor(d, [0.0'f32])
    let jitOuts = j.call([dummy])
    doAssert readBackU32(jitOuts[0], 1) == @[0'u32]
    doAssert readBackU32(jitOuts[1], 1) == @[0'u32]

  block dynamicShapeOpsEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      let x = reshape(args[0], [2, 2])
      @[setDimensionSize(x, args[1], 0),
        dynamicReshape(x, args[2], [4])]
    let j = jit(f, "dynamicShapeOps")
    let a = f32Tensor(d, [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32])
    let size = i32Scalar(d, 2'i32)
    let shape4 = i32Vector(d, [4'i32])
    let inputs = [a, size, shape4]
    let eagerOuts = f(inputs)
    let jitOuts = j.call(inputs)
    doAssert readBack(jitOuts[0], 4) == readBack(eagerOuts[0], 4)
    doAssert readBack(jitOuts[1], 4) == readBack(eagerOuts[1], 4)

  block dynamicIotaEagerVsJit:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[dynamicIota(dtInt32, args[0], [2, 3], 1)]
    let j = jit(f, "dynamicIota")
    let shape = i32Vector(d, [2'i32, 3'i32])
    let inputs = [shape]
    let eagerIota = dynamicIota(dtInt32, shape, [2, 3], 1)
    let jitOuts = j.call(inputs)
    doAssert readBackI32(jitOuts[0], 6) == readBackI32(eagerIota, 6)

  block donation:
    let f = proc(args: openArray[Tensor]): seq[Tensor] =
      @[mul(args[0], args[1])]
    let j = jit(f, "mulDonate", donateArgs = [0])
    let a = f32Tensor(d, [2.0'f32, 3.0'f32])
    let b = f32Tensor(d, [5.0'f32, 7.0'f32])
    let outs = j.call([a, b])
    doAssert readBack(outs[0], 2) == @[10.0'f32, 21.0'f32]
    var dst: array[2, float32]
    var raised = false
    try:
      transferToHost(a.device, a.buffer, addr dst[0], 2 * sizeof(float32))
    except BufferDonatedError:
      raised = true
    doAssert raised

  echo "tjit_execute: jit + donation OK on CPU plugin"
