## Explicit probabilistic modeling and MCMC helpers.

import std/[algorithm, math]
import ./dataframe
import ./device
import ./tensor
import ./eager
import ./ops/[arith, literal, unary]
import ./autograd/transform
import ./transform/jit

type
  ProbError* = object of CatchableError
    ## Raised by probabilistic modeling helpers.

  DistributionKind* = enum
    pkNormal,
    pkHalfNormal,
    pkBernoulli,
    pkExponential

  Prior* = object
    name*: string
    kind*: DistributionKind
    a*: float64
    b*: float64

  LogProbFn* = proc(params: openArray[float64]): float64 {.closure.}
  TensorLogProbFn* = proc(params: openArray[Tensor]): Tensor {.closure.}

  ProbModel* = object
    priors*: seq[Prior]
    logLikelihood*: LogProbFn

  TensorProbModel* = object
    ## Tensor-backed probabilistic model. The NUTS control flow remains on the
    ## host, while log-probability and gradients are evaluated with rew tensor
    ## operations and autograd.
    names*: seq[string]
    device*: Device
    logProbFn*: TensorLogProbFn

  McmcConfig* = object
    draws*: int
    warmup*: int
    stepSize*: float64
    maxTreeDepth*: int
    seed*: uint64

  NutsSampler* = object
    config*: McmcConfig

  McmcTrace* = object
    names*: seq[string]
    samples*: seq[seq[float64]]
    acceptRate*: float64
    divergences*: int

  PosteriorSummary* = object
    name*: string
    mean*: float64
    sd*: float64
    q05*: float64
    q50*: float64
    q95*: float64

func normal*(name: string; mu, sigma: float64): Prior =
  if sigma <= 0:
    raise newException(ProbError, "normal sigma must be positive")
  Prior(name: name, kind: pkNormal, a: mu, b: sigma)

func halfNormal*(name: string; sigma: float64): Prior =
  if sigma <= 0:
    raise newException(ProbError, "halfNormal sigma must be positive")
  Prior(name: name, kind: pkHalfNormal, a: 0, b: sigma)

func bernoulli*(name: string; p: float64): Prior =
  if p <= 0 or p >= 1:
    raise newException(ProbError, "bernoulli p must be in (0, 1)")
  Prior(name: name, kind: pkBernoulli, a: p, b: 0)

func exponential*(name: string; rate: float64): Prior =
  if rate <= 0:
    raise newException(ProbError, "exponential rate must be positive")
  Prior(name: name, kind: pkExponential, a: rate, b: 0)

func initProbModel*(priors: openArray[Prior];
    logLikelihood: LogProbFn): ProbModel =
  if priors.len == 0:
    raise newException(ProbError, "ProbModel requires at least one prior")
  ProbModel(priors: @priors, logLikelihood: logLikelihood)

proc initTensorProbModel*(names: openArray[string]; logProb: TensorLogProbFn;
    device: Device = defaultDevice()): TensorProbModel =
  ## Creates a tensor-backed probabilistic model with explicit parameter names.
  if names.len == 0:
    raise newException(ProbError, "TensorProbModel requires at least one name")
  if logProb == nil:
    raise newException(ProbError, "TensorProbModel requires a logProb function")
  for name in names:
    if name.len == 0:
      raise newException(ProbError,
        "TensorProbModel parameter names must not be empty")
  TensorProbModel(names: @names, device: device, logProbFn: logProb)

func initMcmcConfig*(draws = 1000; warmup = 500; stepSize = 0.1;
    maxTreeDepth = 6; seed: uint64 = 0): McmcConfig =
  if draws <= 0:
    raise newException(ProbError, "MCMC draws must be positive")
  if warmup < 0:
    raise newException(ProbError, "MCMC warmup must be non-negative")
  if stepSize <= 0:
    raise newException(ProbError, "MCMC stepSize must be positive")
  if maxTreeDepth <= 0:
    raise newException(ProbError, "MCMC maxTreeDepth must be positive")
  McmcConfig(draws: draws, warmup: warmup, stepSize: stepSize,
    maxTreeDepth: maxTreeDepth, seed: seed)

func initNutsSampler*(config: McmcConfig = initMcmcConfig()):
    NutsSampler =
  NutsSampler(config: config)

func logNormalPdf*(x, mu, sigma: float64): float64 =
  -0.5 * ln(2.0 * PI) - ln(sigma) -
    0.5 * ((x - mu) / sigma) * ((x - mu) / sigma)

proc tensorScalarLike*(x: Tensor; value: float32): Tensor =
  ## Constant tensor matching `x.shape` and `x.device`.
  var data = newSeq[float32](x.numElements)
  for i in 0 ..< data.len:
    data[i] = value
  constantF32(x.shape, data, x.device)

proc tensorLogNormalPdf*(x, mu, sigma: Tensor): Tensor =
  ## Elementwise normal log-density using rew tensor operations.
  let half = tensorScalarLike(x, 0.5'f32)
  let log2Pi = tensorScalarLike(x, float32(ln(2.0 * PI)))
  let z = divide(sub(x, mu), sigma)
  neg(add(add(mul(half, log2Pi), log(sigma)), mul(half, mul(z, z))))

proc tensorNormalLogLikelihood*(values: openArray[float32]; mu, sigma: Tensor):
    Tensor =
  ## Scalar normal log-likelihood for host observations and tensor params.
  result = tensorScalarLike(mu, 0'f32)
  for value in values:
    result = add(result, tensorLogNormalPdf(tensorScalarLike(mu, value), mu,
      sigma))

proc tensorNormalLogLikelihood*(values: openArray[float64]; mu, sigma: Tensor):
    Tensor =
  result = tensorScalarLike(mu, 0'f32)
  for value in values:
    result = add(result, tensorLogNormalPdf(tensorScalarLike(mu,
      float32(value)), mu, sigma))

func normalLogLikelihood*(values: openArray[float64]; mu, sigma: float64):
    float64 =
  for value in values:
    result += logNormalPdf(value, mu, sigma)

func logPrior(prior: Prior; value: float64): float64 =
  case prior.kind
  of pkNormal:
    logNormalPdf(value, prior.a, prior.b)
  of pkHalfNormal:
    if value < 0:
      NegInf
    else:
      ln(2.0) + logNormalPdf(value, 0, prior.b)
  of pkBernoulli:
    if abs(value) < 0.5:
      ln(1.0 - prior.a)
    elif abs(value - 1.0) < 0.5:
      ln(prior.a)
    else:
      NegInf
  of pkExponential:
    if value < 0:
      NegInf
    else:
      ln(prior.a) - prior.a * value

proc logProb*(model: ProbModel; params: openArray[float64]): float64 =
  if params.len != model.priors.len:
    raise newException(ProbError,
      "logProb parameter count does not match priors")
  for i, prior in model.priors:
    result += prior.logPrior(params[i])
  if model.logLikelihood != nil:
    result += model.logLikelihood(params)

type RngState = object
  state: uint64

func initRng(seed: uint64): RngState =
  RngState(state: if seed == 0: 0x9E3779B97F4A7C15'u64 else: seed)

func nextU64(rng: var RngState): uint64 =
  rng.state = rng.state * 6364136223846793005'u64 + 1442695040888963407'u64
  rng.state

func uniform(rng: var RngState): float64 =
  let bits = rng.nextU64() shr 11
  max(1e-12, float64(bits) / float64(1'u64 shl 53))

func normal01(rng: var RngState): float64 =
  let u1 = rng.uniform()
  let u2 = rng.uniform()
  sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)

func dot(a, b: seq[float64]): float64 =
  for i in 0 ..< a.len:
    result += a[i] * b[i]

proc gradLogProb(model: ProbModel; theta: seq[float64]): seq[float64] =
  result = newSeq[float64](theta.len)
  for i in 0 ..< theta.len:
    var left = theta
    var right = theta
    let eps = 1e-5 * max(1.0, abs(theta[i]))
    left[i] -= eps
    right[i] += eps
    result[i] = (model.logProb(right) - model.logProb(left)) / (2.0 * eps)

func kinetic(r: seq[float64]): float64 =
  for v in r:
    result += v * v
  result *= 0.5

proc hamiltonian(model: ProbModel; theta, r: seq[float64]): float64 =
  -model.logProb(theta) + kinetic(r)

proc leapfrog(model: ProbModel; theta, r: seq[float64]; step: float64):
    tuple[theta, r: seq[float64]] =
  var momentum = r
  var position = theta
  let g0 = model.gradLogProb(position)
  for i in 0 ..< momentum.len:
    momentum[i] += 0.5 * step * g0[i]
    position[i] += step * momentum[i]
  let g1 = model.gradLogProb(position)
  for i in 0 ..< momentum.len:
    momentum[i] += 0.5 * step * g1[i]
  (theta: position, r: momentum)

func noUTurn(thetaMinus, thetaPlus, rMinus, rPlus: seq[float64]): bool =
  var delta = newSeq[float64](thetaMinus.len)
  for i in 0 ..< delta.len:
    delta[i] = thetaPlus[i] - thetaMinus[i]
  dot(delta, rMinus) < 0 or dot(delta, rPlus) < 0

func finite(x: float64): bool =
  classify(x) notin {fcNan, fcInf, fcNegInf}

proc nutsTransition(model: ProbModel; theta: seq[float64]; rng: var RngState;
    cfg: McmcConfig): tuple[theta: seq[float64], accepted: bool,
    divergent: bool] =
  var r0 = newSeq[float64](theta.len)
  for i in 0 ..< r0.len:
    r0[i] = rng.normal01()
  let startH = model.hamiltonian(theta, r0)
  let logSlice = -startH + ln(rng.uniform())

  var thetaMinus = theta
  var thetaPlus = theta
  var rMinus = r0
  var rPlus = r0
  var proposal = theta
  var accepted = false
  var n = 1
  var keepGoing = true

  for depth in 0 ..< cfg.maxTreeDepth:
    if not keepGoing:
      break
    let direction = if rng.uniform() < 0.5: -1.0 else: 1.0
    let steps = 1 shl depth
    var candidate = proposal
    var candidateCount = 0
    for _ in 0 ..< steps:
      if direction < 0:
        let moved = model.leapfrog(thetaMinus, rMinus,
          -abs(direction) * cfg.stepSize)
        thetaMinus = moved.theta
        rMinus = moved.r
        candidate = thetaMinus
      else:
        let moved = model.leapfrog(thetaPlus, rPlus, cfg.stepSize)
        thetaPlus = moved.theta
        rPlus = moved.r
        candidate = thetaPlus
      let h = model.hamiltonian(candidate, if direction < 0: rMinus else: rPlus)
      if not h.finite or h - startH > 1000:
        return (theta: theta, accepted: false, divergent: true)
      if logSlice <= -h:
        inc candidateCount
        if rng.uniform() < float64(candidateCount) / float64(max(1, n + candidateCount)):
          proposal = candidate
          accepted = true
      keepGoing = keepGoing and not noUTurn(thetaMinus, thetaPlus,
        rMinus, rPlus)
      if not keepGoing:
        break
    n += candidateCount
  (theta: proposal, accepted: accepted, divergent: false)

proc sample*(sampler: NutsSampler; model: ProbModel;
    initial: openArray[float64]): McmcTrace =
  if initial.len != model.priors.len:
    raise newException(ProbError, "initial parameter count does not match model")
  var rng = initRng(sampler.config.seed)
  var theta = @initial
  result.names = newSeq[string](model.priors.len)
  for i, prior in model.priors:
    result.names[i] = prior.name

  let total = sampler.config.warmup + sampler.config.draws
  var accepted = 0
  for iter in 0 ..< total:
    let next = nutsTransition(model, theta, rng, sampler.config)
    if next.divergent:
      inc result.divergences
    if next.accepted:
      inc accepted
      theta = next.theta
    if iter >= sampler.config.warmup:
      result.samples.add theta
  result.acceptRate = float64(accepted) / float64(total)

type
  TensorProbEvaluator = object
    model: TensorProbModel
    compiled: JitFunction

proc initTensorProbEvaluator(model: TensorProbModel): TensorProbEvaluator =
  let logProbFn = model.logProbFn
  let fn: JitFn = proc(args: openArray[Tensor]): seq[Tensor] =
    let value = logProbFn(args)
    result = @[value]
    result.add grad(logProbFn, args)
  result.model = model
  result.compiled = jit(fn, "rew_prob_logprob_grad")

proc evaluate(eval: var TensorProbEvaluator; theta: seq[float64]): tuple[
    logp: float64, grad: seq[float64]] =
  var args: seq[Tensor]
  for value in theta:
    args.add fromHostF32(eval.model.device, [float32(value)], [])
  let outs = eval.compiled.call(args)
  result.logp = float64(outs[0].toHost(float32)[0])
  for i in 0 ..< theta.len:
    result.grad.add float64(outs[i + 1].toHost(float32)[0])

proc hamiltonian(eval: var TensorProbEvaluator; theta, r: seq[float64]):
    float64 =
  -eval.evaluate(theta).logp + kinetic(r)

proc leapfrog(eval: var TensorProbEvaluator; theta, r: seq[float64];
    step: float64): tuple[theta, r: seq[float64]] =
  var momentum = r
  var position = theta
  let g0 = eval.evaluate(position).grad
  for i in 0 ..< momentum.len:
    momentum[i] += 0.5 * step * g0[i]
    position[i] += step * momentum[i]
  let g1 = eval.evaluate(position).grad
  for i in 0 ..< momentum.len:
    momentum[i] += 0.5 * step * g1[i]
  (theta: position, r: momentum)

proc nutsTransition(eval: var TensorProbEvaluator; theta: seq[float64];
    rng: var RngState; cfg: McmcConfig): tuple[theta: seq[float64],
    accepted: bool, divergent: bool] =
  var r0 = newSeq[float64](theta.len)
  for i in 0 ..< r0.len:
    r0[i] = rng.normal01()
  let startH = eval.hamiltonian(theta, r0)
  let logSlice = -startH + ln(rng.uniform())

  var thetaMinus = theta
  var thetaPlus = theta
  var rMinus = r0
  var rPlus = r0
  var proposal = theta
  var accepted = false
  var n = 1
  var keepGoing = true

  for depth in 0 ..< cfg.maxTreeDepth:
    if not keepGoing:
      break
    let direction = if rng.uniform() < 0.5: -1.0 else: 1.0
    let steps = 1 shl depth
    var candidate = proposal
    var candidateCount = 0
    for _ in 0 ..< steps:
      if direction < 0:
        let moved = eval.leapfrog(thetaMinus, rMinus,
          -abs(direction) * cfg.stepSize)
        thetaMinus = moved.theta
        rMinus = moved.r
        candidate = thetaMinus
      else:
        let moved = eval.leapfrog(thetaPlus, rPlus, cfg.stepSize)
        thetaPlus = moved.theta
        rPlus = moved.r
        candidate = thetaPlus
      let h = eval.hamiltonian(candidate, if direction < 0: rMinus else: rPlus)
      if not h.finite or h - startH > 1000:
        return (theta: theta, accepted: false, divergent: true)
      if logSlice <= -h:
        inc candidateCount
        if rng.uniform() < float64(candidateCount) /
            float64(max(1, n + candidateCount)):
          proposal = candidate
          accepted = true
      keepGoing = keepGoing and not noUTurn(thetaMinus, thetaPlus,
        rMinus, rPlus)
      if not keepGoing:
        break
    n += candidateCount
  (theta: proposal, accepted: accepted, divergent: false)

proc sample*(sampler: NutsSampler; model: TensorProbModel;
    initial: openArray[float64]): McmcTrace =
  ## Runs NUTS with tensor/autograd log-probability and gradient evaluation.
  if initial.len != model.names.len:
    raise newException(ProbError, "initial parameter count does not match model")
  var eval = initTensorProbEvaluator(model)
  var rng = initRng(sampler.config.seed)
  var theta = @initial
  result.names = model.names

  let total = sampler.config.warmup + sampler.config.draws
  var accepted = 0
  for iter in 0 ..< total:
    let next = eval.nutsTransition(theta, rng, sampler.config)
    if next.divergent:
      inc result.divergences
    if next.accepted:
      inc accepted
      theta = next.theta
    if iter >= sampler.config.warmup:
      result.samples.add theta
  result.acceptRate = float64(accepted) / float64(total)

proc summary*(trace: McmcTrace): seq[PosteriorSummary] =
  if trace.samples.len == 0:
    raise newException(ProbError, "summary requires at least one sample")
  for j, name in trace.names:
    var values = newSeq[float64](trace.samples.len)
    for i, row in trace.samples:
      values[i] = row[j]
    values.sort()
    var avg = 0.0
    for v in values:
      avg += v
    avg /= float64(values.len)
    var variance = 0.0
    for v in values:
      variance += (v - avg) * (v - avg)
    variance /= float64(max(1, values.len - 1))
    result.add PosteriorSummary(
      name: name,
      mean: avg,
      sd: sqrt(variance),
      q05: values[int(floor(0.05 * float64(values.high)))],
      q50: values[int(floor(0.50 * float64(values.high)))],
      q95: values[int(floor(0.95 * float64(values.high)))],
    )

proc observedColumn*(df: DataFrame; column: string): seq[float64] =
  ## Collects a numeric observed column from a DataFrame for host-side
  ## probabilistic models.
  let rows = df.collect()
  let col = rows.columns[rows.requireColumn(column)]
  for value in col.values:
    case value.kind
    of dfvInt:
      result.add float64(value.intVal)
    of dfvFloat:
      result.add value.floatVal
    else:
      raise newException(ProbError,
        "observedColumn expects numeric non-null values: " & column)

proc posteriorPredictiveNormal*(trace: McmcTrace; paramName: string;
    sigma: float64; seed: uint64 = 0): seq[float64] =
  var index = -1
  for i, name in trace.names:
    if name == paramName:
      index = i
      break
  if index < 0:
    raise newException(ProbError, "unknown posterior parameter: " & paramName)
  var rng = initRng(seed)
  for row in trace.samples:
    result.add row[index] + sigma * rng.normal01()
