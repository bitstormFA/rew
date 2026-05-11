## Device-side RNG ops: `rng`, `rng_bit_generator`.

import ../tensor
import ../dispatch
import ../stablehlo/[ir, ops as shops]
import ../autograd/tape
import ./marker

import ../dtype
import ./literal
import ./compare
import ./linalg
import ./arith
import ./unary
import ./ternary
proc rng*(a, bound, shape: Tensor; distribution: RngDistribution;
    resultShape: openArray[int]): Tensor {.rewOp.} =
  ## Generates random numbers with the given `distribution` in the range
  ## [`a`, `bound`). `a` and `bound` must be 0-rank scalar tensors of the
  ## same dtype. `shape` is a 1-D integer tensor specifying the runtime
  ## output dimensions; `resultShape` provides the compile-time shape.
  case currentMode()
  of dmTrace:
    requireTrace(a, "rng")
    requireTrace(bound, "rng")
    requireTrace(shape, "rng")
    let ctx = currentTraceContext()
    let id = shops.rng(ctx.builder, a.traceId, bound.traceId,
      shape.traceId, distribution, resultShape)
    result = initTraceTensor(id, a.dtype, resultShape, a.device,
      a.sharding)
    recordTraceOp("rng", [a, bound, shape], result)
  of dmEager:
    raise newException(TensorError,
      "rng: only supported in trace/jit mode")

proc rngBitGenerator*(initialState: Tensor; algorithm: RngAlgorithm;
    outputType: ShTensorType): (Tensor, Tensor) {.rewOp.} =
  ## Advances a PRNG state and returns `(newState, randomOutput)`.
  case currentMode()
  of dmTrace:
    requireTrace(initialState, "rngBitGenerator")
    let ctx = currentTraceContext()
    let ids = shops.rngBitGenerator(ctx.builder, initialState.traceId,
      algorithm, outputType)
    let outState = initTraceTensor(ids[0], initialState.dtype,
      initialState.shape, initialState.device, initialState.sharding)
    let output = initTraceTensor(ids[1], outputType.dtype,
      outputType.shape, initialState.device, initialState.sharding)
    result = (outState, output)
    recordTraceOp("rngBitGenerator", [initialState], outState)
  of dmEager:
    raise newException(TensorError,
      "rngBitGenerator: only supported in trace/jit mode")

# ---- random convenience API -----------------------------------------------

proc rand*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Generate uniform random samples in [0, 1). Trace mode only.
  let a = scalarF32(0'f32)
  let bound = scalarF32(1'f32)
  var shapeData = newSeq[int32](shape.len)
  for i, s in shape:
    shapeData[i] = int32(s)
  let shapeTensor = constant(dtInt32, [shape.len], i32Bytes(shapeData))
  rng(a, bound, shapeTensor, rdUniform, @shape)

proc randn*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Generate samples from a standard normal distribution.
  ## Trace mode only.
  let a = scalarF32(0'f32)
  let bound = scalarF32(1'f32)
  var shapeData = newSeq[int32](shape.len)
  for i, s in shape:
    shapeData[i] = int32(s)
  let shapeTensor = constant(dtInt32, [shape.len], i32Bytes(shapeData))
  rng(a, bound, shapeTensor, rdNormal, @shape)

proc randint*(low, high: int; shape: openArray[int]): Tensor =
  ## Generate uniform random integers in [low, high). Trace mode only.
  if low >= high:
    raise newException(TensorError,
      "randint: low must be < high")
  let a = scalarI32(int32(low))
  let bound = scalarI32(int32(high))
  var shapeData = newSeq[int32](shape.len)
  for i, s in shape:
    shapeData[i] = int32(s)
  let shapeTensor = constant(dtInt32, [shape.len], i32Bytes(shapeData))
  rng(a, bound, shapeTensor, rdUniform, @shape)

proc bernoulli*(p: float32; shape: openArray[int]): Tensor =
  ## Generate Bernoulli samples: 1 with probability `p`, 0 otherwise.
  ## Trace mode only.
  let uniform = rand(shape, dtFloat32)
  let pScalar = scalarF32(p)
  var zeroDims: seq[int] = @[]
  let pBroad = broadcastTo(pScalar, @shape, zeroDims)
  compare(uniform, pBroad, "LT")

# ---- distribution composites (trace-mode, built on rand/randn) -------------

proc exponential*(shape: openArray[int]; rate: float32 = 1.0'f32;
    dtype: DType = dtFloat32): Tensor =
  ## Exponential distribution samples with the given `rate`.
  ## Uses inverse-CDF: `-log(rand) / rate`. Trace mode only.
  let u = rand(shape, dtype)
  let eps = broadcastTo(scalarF32(1e-10'f32), @shape, @[])
  let safe = maximum(u, eps)
  var res = neg(log(safe))
  if rate != 1.0'f32:
    let rateS = broadcastTo(scalarF32(rate), @shape, @[])
    res = divide(res, rateS)
  res

proc gumbel*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Gumbel(0, 1) samples via inverse-CDF.
  ## `-log(-log(rand))`. Trace mode only.
  let u = rand(shape, dtype)
  let eps = broadcastTo(scalarF32(1e-10'f32), @shape, @[])
  let safe = maximum(u, eps)
  neg(log(neg(log(safe))))

proc cauchy*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Cauchy(0, 1) samples as the ratio of two independent standard
  ## normals. Trace mode only.
  divide(randn(shape, dtype), randn(shape, dtype))

proc laplace*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Laplace(0, 1) samples via inverse-CDF.
  ## Trace mode only.
  let u = rand(shape, dtype)
  let half = broadcastTo(scalarF32(0.5'f32), @shape, @[])
  let two = broadcastTo(scalarF32(2.0'f32), @shape, @[])
  let one = broadcastTo(scalarF32(1.0'f32), @shape, @[])
  let left = log(mul(two, u))
  let right = neg(log(mul(two, sub(one, u))))
  select(compare(u, half, "LT"), left, right)

proc logistic*(shape: openArray[int]; dtype: DType = dtFloat32): Tensor =
  ## Logistic(0, 1) samples via inverse-CDF.
  ## `log(u / (1 - u))`. Trace mode only.
  let u = rand(shape, dtype)
  let eps = broadcastTo(scalarF32(1e-10'f32), @shape, @[])
  let safe = maximum(u, eps)
  let one = broadcastTo(scalarF32(1.0'f32), @shape, @[])
  let oneMinus = sub(one, safe)
  let oneMinusSafe = maximum(oneMinus, eps)
  sub(log(safe), log(oneMinusSafe))
