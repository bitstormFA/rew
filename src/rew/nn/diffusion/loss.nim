## Diffusion-specific loss helpers.
##
## Standard noise-prediction loss (DDPM objective) and x0-prediction
## loss (v-prediction variant).

import ../../tensor
import ../../ops/arith
import ../../ops/reduce
proc diffusionNoiseLoss*(predictedNoise, targetNoise: Tensor): Tensor =
  ## Mean-squared error between predicted and target noise.
  ##
  ## This is the standard DDPM training objective: the model predicts
  ## the noise that was added to the input, and we minimize MSE.
  ## Both inputs must have matching shapes. Result is a 0-d scalar.
  if predictedNoise.shape != targetNoise.shape:
    raise newException(TensorError,
      "diffusionNoiseLoss: shape mismatch (" &
        $predictedNoise.shape & " vs " & $targetNoise.shape & ")")
  let diff = sub(predictedNoise, targetNoise)
  let sq = mul(diff, diff)
  var dims = newSeq[int](predictedNoise.shape.len)
  for i in 0 ..< predictedNoise.shape.len: dims[i] = i
  reduceMean(sq, dims)

proc diffusionX0Loss*(predictedX0, targetX0: Tensor): Tensor =
  ## Mean-squared error between predicted and target clean data.
  ##
  ## Used in x0-prediction / v-prediction variants. Both inputs must
  ## have matching shapes. Result is a 0-d scalar.
  if predictedX0.shape != targetX0.shape:
    raise newException(TensorError,
      "diffusionX0Loss: shape mismatch (" &
        $predictedX0.shape & " vs " & $targetX0.shape & ")")
  let diff = sub(predictedX0, targetX0)
  let sq = mul(diff, diff)
  var dims = newSeq[int](predictedX0.shape.len)
  for i in 0 ..< predictedX0.shape.len: dims[i] = i
  reduceMean(sq, dims)
