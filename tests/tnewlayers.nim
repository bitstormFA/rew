## Compile-time smoke tests for new layers added in PR (rnn cells, rnn,
## bidirectional, pool, pixelshuffle, upsample, depthwise, pad, spectralnorm,
## bilinear, identity, dropout2d, optimizers, losses, ops).

import rew

proc main() =
  echo "Compile-time smoke test for new layers: PASS"

  # Verify types are exported
  var _: LSTMCell
  var _: LSTMState
  var _: GRUCell
  var _: GRUState
  var _: LSTM
  var _: LSTMStates
  var _: GRU
  var _: GRUStates
  var _: Bidirectional[Identity, int]
  var _: BidirectionalState[int]
  var _: DepthwiseConv2d
  var _: SeparableConv2d
  var _: ReflectionPad2d
  var _: Upsample
  var _: Identity
  var _: Dropout2d
  var _: Bilinear
  var _: SpectralNorm[Identity]

  # Verify optimizers
  var _: Adagrad
  var _: AdagradState
  var _: Adadelta
  var _: AdadeltaState
  var _: Adamax
  var _: AdamaxState
  var _: Lion
  var _: LionState

  # Verify loss/pool procs are accessible
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalAvgPool1d
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalAvgPool2d
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalAvgPool3d
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalMaxPool1d
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalMaxPool2d
  var _: proc(x: Tensor): Tensor {.nimcall.} = globalMaxPool3d
  var _: proc(x: Tensor; outputSize: int): Tensor {.nimcall.} = adaptiveAvgPool1d
  var _: proc(x: Tensor; outputSize: int): Tensor {.nimcall.} = adaptiveMaxPool1d
  var _: proc(x: Tensor; upscaleFactor: int): Tensor {.nimcall.} = pixelShuffle
  var _: proc(x: Tensor; downscaleFactor: int): Tensor {.nimcall.} = pixelUnshuffle
  var _: proc(x: Tensor; padding: array[4, int]): Tensor {.nimcall.} = reflectionPad2d
  var _: proc(x: Tensor; scaleFactor: array[2, int]): Tensor {.nimcall.} = upsampleNearest2d

  # Verify loss procs
  var _: proc(pred, target: Tensor; beta: float32): Tensor {.nimcall.} = smoothL1Loss
  var _: proc(logits, target: Tensor; alpha, gamma: float32): Tensor {.nimcall.} = sigmoidFocalLoss
  var _: proc(logits, target: Tensor; gamma: float32): Tensor {.nimcall.} = softmaxFocalLoss
  var _: proc(x1, x2: Tensor; y: Tensor; margin: float32): Tensor {.nimcall.} = cosineEmbeddingLoss
  var _: proc(anchor, positive, negative: Tensor; margin, p: float32): Tensor {.nimcall.} = tripletMarginLoss

  # Verify interpolate op
  var _: proc(x: Tensor; size: openArray[int]; mode: InterpolationMode): Tensor {.nimcall.} = interpolate

main()
