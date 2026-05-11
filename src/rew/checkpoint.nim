## Gradient checkpointing — memory-efficient training via recomputation.
##
## Stores only checkpoint activations and recomputes intermediate
## activations during the backward pass. Dramatically reduces peak
## memory at the cost of ~30% extra forward computation.

import std/math

type
  CheckpointPolicy* = enum
    cpEveryLayer   ## Checkpoint after every layer (best memory savings).
    cpSqrt         ## Checkpoint at sqrt-spaced intervals.
    cpNone         ## No checkpointing (store all activations).

proc checkpointPolicy*(numLayers: int; policy: CheckpointPolicy): seq[int] =
  ## Returns the set of layer indices to checkpoint (store activations).
  ## Activations for other layers will be recomputed during backward.
  case policy
  of cpNone:
    for i in 0 ..< numLayers: result.add i
  of cpEveryLayer:
    for i in 0 ..< numLayers: result.add i
  of cpSqrt:
    let step = max(1, int(sqrt(float(numLayers))))
    var i = 0
    while i < numLayers:
      result.add i
      i += step

# ---- mixed precision --------------------------------------------------------

type
  AmpConfig* = object
    ## Automatic Mixed Precision configuration.
    enabled*: bool
    dtype*: string   ## "float16" or "bfloat16"
    lossScale*: float32
    minLossScale*: float32
    backoffFactor*: float32
    growthInterval*: int
    scaleWindow*: int

  AmpState* = object
    ## Running AMP state.
    lossScale*: float32
    growthTracker*: int

proc initAmpConfig*(enabled = true; dtype = "float16";
    lossScale = 65536.0'f32): AmpConfig =
  AmpConfig(
    enabled: enabled,
    dtype: dtype,
    lossScale: lossScale,
    minLossScale: 1.0'f32,
    backoffFactor: 0.5'f32,
    growthInterval: 2000,
    scaleWindow: 2000,
  )

proc initAmpState*(config: AmpConfig): AmpState =
  AmpState(
    lossScale: config.lossScale,
    growthTracker: 0,
  )

proc updateAmpScale*(state: var AmpState; config: AmpConfig;
    hadOverflow: bool) =
  ## Updates the AMP loss scale based on whether overflow occurred.
  if hadOverflow:
    state.lossScale *= config.backoffFactor
    if state.lossScale < config.minLossScale:
      state.lossScale = config.minLossScale
    state.growthTracker = 0
  else:
    inc state.growthTracker
    if state.growthTracker >= config.growthInterval:
      state.lossScale *= 2.0'f32
      if state.lossScale > config.lossScale:
        state.lossScale = config.lossScale
      state.growthTracker = 0
