## Diffusion sampling loops — DDPM and DDIM reverse diffusion.
##
## These are host-side orchestration functions that call a jitted
## denoising model in a loop. The model must be jitted separately:
##   let denoiseFn = jit(proc(args: openArray[Tensor]): seq[Tensor] =
##     @[model.forward(args[0], args[1])])
##
## The full reverse-diffusion step (including noise addition) should
## also be jitted since `randn` is only available in trace mode:
##   let ddpmStepFn = jit(proc(args: openArray[Tensor]): seq[Tensor] =
##     @[ddpmStep(schedule, args[0], args[1], args[2])])
## where `schedule` constants are captured by closure.
##
## This module provides the step formulas; users wire them into jit
## and host-side loops themselves.

import std/math
import ../../tensor
import ../../ops/literal
import ../../ops/arith
import ../../ops/linalg
import ../../ops/random
import ./schedulers

# ---- DDPM step helper (to be jitted) ----------------------------------------

proc ddpmStepPrediction*(schedule: NoiseSchedule; step: int;
    x, predictedNoise: Tensor): Tensor =
  ## DDPM reverse step from the predicted noise.
  ##
  ## Computes x_{t-1} = 1/sqrt(alpha_t) * (x - (1-alpha_t)/sqrt(1-alphaBar_t) * eps)
  ##                  + sqrt(beta_tilde) * z  (for t > 0)
  ##
  ## `schedule`: precomputed noise schedule constants.
  ## `step`: current timestep index (0-indexed).
  ## `x`: noisy input at timestep `step`.
  ## `predictedNoise`: model's noise prediction.
  ## Returns: x at timestep `step - 1` (or clean output at step 0).
  let alpha = schedule.alphas[step]
  let alphaBar = schedule.alphaBars[step]
  let oneMinusAlpha = 1'f32 - alpha
  let sqrtAlpha = sqrt(alpha.float64).float32
  let sqrtOneMinusAlphaBar = sqrt(1'f32.float64 - alphaBar.float64).float32
  let coeff = divide(
    broadcastTo(scalarF32(oneMinusAlpha), x.shape, @[]),
    broadcastTo(scalarF32(sqrtOneMinusAlphaBar), x.shape, @[]))
  let xInner = sub(x, mul(coeff, predictedNoise))
  var xPrev = divide(xInner, broadcastTo(scalarF32(sqrtAlpha), x.shape, @[]))
  if step > 0:
    # sigma = sqrt(beta_tilde), where
    # beta_tilde = (1 - alphaBar_{t-1}) / (1 - alphaBar_t) * beta_t
    let beta = schedule.betas[step]
    let prevAlphaBar = schedule.alphaBars[step - 1]
    let betaTilde = ((1'f32 - prevAlphaBar) / (1'f32 - alphaBar)) * beta
    let sigma = sqrt(betaTilde.float64).float32
    let z = randn(x.shape, x.dtype)
    xPrev = add(xPrev, mul(broadcastTo(scalarF32(sigma), x.shape, @[]), z))
  xPrev

# ---- DDIM step helper (to be jitted) ----------------------------------------

proc ddimStepPrediction*(schedule: DDIMScheduler; stepIdx: int;
    x, predictedNoise: Tensor; eta: float32 = 0'f32): Tensor =
  ## DDIM reverse step from the predicted noise.
  ##
  ## `schedule`: DDIM subsampled schedule.
  ## `stepIdx`: index into `schedule.timesteps`.
  ## `x`: noisy input at current timestep.
  ## `predictedNoise`: model's noise prediction.
  ## `eta`: 0 = deterministic, 1 = DDPM-style stochastic.
  ## Returns: x at the previous timestep.
  let step = schedule.timesteps[stepIdx]
  let prevStep = if stepIdx < schedule.timesteps.len - 1:
    schedule.timesteps[stepIdx + 1] else: 0
  let sched = schedule.schedule
  let alphaBar = sched.alphaBars[step]
  let prevAlphaBar = sched.alphaBars[prevStep]
  # Predict x0.
  let sqrtAlphaBar = sqrt(alphaBar.float64).float32
  let sqrtOneMinusAlphaBar = sqrt(1'f32.float64 - alphaBar.float64).float32
  let predX0 = divide(
    sub(x, mul(broadcastTo(scalarF32(sqrtOneMinusAlphaBar), x.shape, @[]),
      predictedNoise)),
    broadcastTo(scalarF32(sqrtAlphaBar), x.shape, @[]))
  # Direction to x_t.
  let sqrtPrevAlphaBar = sqrt(prevAlphaBar.float64).float32
  let sqrtOneMinusPrevAlphaBar = sqrt(
    1'f32.float64 - prevAlphaBar.float64).float32
  let dirXt = mul(
    broadcastTo(scalarF32(sqrtOneMinusPrevAlphaBar), x.shape, @[]),
    predictedNoise)
  var xPrev = add(
    mul(broadcastTo(scalarF32(sqrtPrevAlphaBar), x.shape, @[]), predX0),
    dirXt)
  # Stochastic noise.
  if eta > 0'f32 and prevStep > 0:
    let sigma = eta * sqrt(
      ((1'f32 - prevAlphaBar) / (1'f32 - alphaBar)).float64 *
      (1'f32 - alphaBar / prevAlphaBar).float64)
    let z = randn(x.shape, x.dtype)
    xPrev = add(xPrev,
      mul(broadcastTo(scalarF32(sigma), x.shape, @[]), z))
  xPrev
