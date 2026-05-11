## Diffusion noise schedulers — DDPM (linear), cosine, and DDIM.
##
## These are host-side utilities that precompute alpha / beta arrays
## for the diffusion process. The arrays are loaded into the model
## graph as `constantF32` tensors.

import std/math

type
  NoiseSchedule* = object
    ## Precomputed diffusion schedule.
    betas*: seq[float32]     ## Beta values per timestep
    alphas*: seq[float32]    ## Alpha = 1 - beta per timestep
    alphaBars*: seq[float32] ## Cumulative product of alphas
    numSteps*: int

proc linearNoiseSchedule*(numSteps: int;
    betaStart: float32 = 1e-4'f32;
    betaEnd: float32 = 0.02'f32): NoiseSchedule =
  ## Linear beta schedule (Ho et al., 2020 / DDPM).
  ##
  ## `betaStart` and `betaEnd` control the noise level at the start
  ## and end of the diffusion process. Defaults are the DDPM paper
  ## values for image generation.
  if numSteps <= 0:
    raise newException(ValueError,
      "linearNoiseSchedule: numSteps must be positive, got " & $numSteps)
  var betas = newSeq[float32](numSteps)
  var alphas = newSeq[float32](numSteps)
  var alphaBars = newSeq[float32](numSteps)
  for i in 0 ..< numSteps:
    betas[i] = betaStart + float32(i) * (betaEnd - betaStart) /
      float32(numSteps - 1)
    alphas[i] = 1'f32 - betas[i]
    if i == 0:
      alphaBars[i] = alphas[0]
    else:
      alphaBars[i] = alphaBars[i - 1] * alphas[i]
  NoiseSchedule(betas: betas, alphas: alphas, alphaBars: alphaBars,
    numSteps: numSteps)

proc cosineNoiseSchedule*(numSteps: int;
    offset: float32 = 0.008'f32): NoiseSchedule =
  ## Cosine beta schedule (Nichol & Dhariwal, 2021 / Improved DDPM).
  ##
  ## Provides better noise distribution than the linear schedule,
  ## especially for high-resolution images. `offset` prevents the
  ## signal-to-noise ratio from becoming too small near t=0.
  if numSteps <= 0:
    raise newException(ValueError,
      "cosineNoiseSchedule: numSteps must be positive, got " & $numSteps)
  let maxBeta = 0.999'f32
  var betas = newSeq[float32](numSteps)
  var alphas = newSeq[float32](numSteps)
  var alphaBars = newSeq[float32](numSteps)
  for i in 0 ..< numSteps:
    let t = float32(i) / float32(numSteps)
    let ft = cos((t + offset) / (1'f32 + offset) * PI.float32 / 2'f32)
    let alphaBar = ft * ft
    # Clamp to prevent beta from going above maxBeta.
    var prevAlphaBar = 1'f32
    if i > 0:
      prevAlphaBar = alphaBars[i - 1]
    var beta = min(1'f32 - alphaBar / prevAlphaBar, maxBeta)
    if i == 0:
      beta = max(beta, 0'f32)
    betas[i] = beta
    alphas[i] = 1'f32 - beta
    alphaBars[i] = alphaBar
  NoiseSchedule(betas: betas, alphas: alphas, alphaBars: alphaBars,
    numSteps: numSteps)

type
  DDIMScheduler* = object
    ## DDIM (Song et al., 2021) deterministic / fast sampling schedule.
    ## Uses a subset of timesteps from the base DDPM schedule.
    timesteps*: seq[int]       ## Subsampled timestep indices
    schedule*: NoiseSchedule   ## The underlying DDPM schedule

proc initDDIMScheduler*(schedule: NoiseSchedule;
    ddimSteps: int): DDIMScheduler =
  ## Create a DDIM scheduler that samples `ddimSteps` equally-spaced
  ## timesteps from the `schedule`'s `numSteps`.
  ##
  ## `ddimSteps` must be <= `schedule.numSteps`.
  if ddimSteps <= 0 or ddimSteps > schedule.numSteps:
    raise newException(ValueError,
      "initDDIMScheduler: ddimSteps must be in [1, " &
        $schedule.numSteps & "], got " & $ddimSteps)
  var timesteps = newSeq[int](ddimSteps)
  let stepSize = schedule.numSteps div ddimSteps
  for i in 0 ..< ddimSteps:
    timesteps[i] = schedule.numSteps - 1 - i * stepSize
  DDIMScheduler(timesteps: timesteps, schedule: schedule)

proc scaledLinearNoiseSchedule*(numSteps: int;
    betaStart: float32 = 1e-4'f32;
    betaEnd: float32 = 0.02'f32): NoiseSchedule =
  ## Linear schedule scaled to [0, 1] for step-indexed use.
  ## Convenience wrapper around `linearNoiseSchedule`.
  linearNoiseSchedule(numSteps, betaStart, betaEnd)
