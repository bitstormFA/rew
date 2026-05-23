## Probabilistic modeling and MCMC tests.

import rew
import rew/dataframe

proc close(a, b: float64; eps: float64): bool =
  abs(a - b) <= eps

proc setupCpu(d: var Device): bool =
  d = cpu()
  setDefaultDevice(d)
  installEagerBackend()
  try:
    discard scalarF32(d, 0'f32)
    true
  except EagerError as e:
    echo "  (skip) no CPU plugin - skipping tensor probabilistic test: ", e.msg
    false

block priors_and_log_prob_are_explicit:
  let model = initProbModel(
    [normal("mu", 0, 1)],
    proc(params: openArray[float64]): float64 =
      normalLogLikelihood([1.0, 1.2, 0.8], params[0], 0.5))

  doAssert model.priors[0].name == "mu"
  doAssert model.logProb([1.0]) > model.logProb([5.0])

block observed_data_can_come_from_dataframe:
  let values = observedColumn(sql("""
      select * from (values
        (1.0::double),
        (2.0::double),
        (3.0::double)
      ) as t(y)
    """), "y")

  doAssert values == @[1.0, 2.0, 3.0]

block nuts_sampler_produces_trace_and_summary:
  let obs = @[0.9, 1.0, 1.1, 1.2]
  let model = initProbModel(
    [normal("mu", 0, 2)],
    proc(params: openArray[float64]): float64 =
      normalLogLikelihood(obs, params[0], 0.2))
  let sampler = initNutsSampler(initMcmcConfig(
    draws = 40, warmup = 20, stepSize = 0.05, maxTreeDepth = 4, seed = 42))
  let trace = sampler.sample(model, [0.0])
  let stats = trace.summary()

  doAssert trace.samples.len == 40
  doAssert trace.names == @["mu"]
  doAssert trace.acceptRate > 0.0
  doAssert stats.len == 1
  doAssert stats[0].name == "mu"
  doAssert stats[0].mean.close(1.05, 0.5)

block tensor_backed_sampler_uses_rew_gradients:
  var d: Device
  if setupCpu(d):
    let obs = @[0.9'f32, 1.0'f32, 1.1'f32]
    let model = initTensorProbModel(["mu"],
      proc(params: openArray[Tensor]): Tensor =
        let mu = params[0]
        let prior = tensorLogNormalPdf(mu, tensorScalarLike(mu, 0'f32),
          tensorScalarLike(mu, 2'f32))
        let likelihood = tensorNormalLogLikelihood(obs, mu,
          tensorScalarLike(mu, 0.2'f32))
        add(prior, likelihood),
      d)
    let sampler = initNutsSampler(initMcmcConfig(
      draws = 8, warmup = 4, stepSize = 0.02, maxTreeDepth = 2, seed = 7))
    let trace = sampler.sample(model, [0.0])

    doAssert trace.samples.len == 8
    doAssert trace.names == @["mu"]
    doAssert trace.acceptRate >= 0.0

block posterior_predictive_uses_named_parameter:
  let trace = McmcTrace(names: @["mu"], samples: @[@[1.0], @[2.0]])
  let draws = posteriorPredictiveNormal(trace, "mu", sigma = 0.1, seed = 1)

  doAssert draws.len == 2
