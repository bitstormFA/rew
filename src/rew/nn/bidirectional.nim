## Bidirectional wrapper for recurrent modules.
##
## Wraps any forward-only recurrent module and runs it in both directions,
## concatenating outputs along the feature axis.

import ../tensor
import ../ops/shape
import ../ops/concat

type
  Bidirectional*[L, S] = object
    ## Generic bidirectional wrapper. `forwardLayer` and `backwardLayer` are
    ## separate instances run in opposite directions over the time axis
    ## (axis=1 in NTC layout). Outputs are concatenated along the feature
    ## axis (axis=2).
    forwardLayer*: L
    backwardLayer*: L

  BidirectionalState*[S] = object
    ## Holds forward and backward states.
    forwardState*: S
    backwardState*: S

proc initBidirectional*[L, S](forwardLayer, backwardLayer: L;
    forwardState: S; backwardState: S): Bidirectional[L, S] =
  ## Constructs a bidirectional wrapper from two layer instances and their
  ## initial states.
  Bidirectional[L, S](
    forwardLayer: forwardLayer,
    backwardLayer: backwardLayer,
  )

proc forward*[L, S](layer: Bidirectional[L, S]; x: Tensor;
    state: BidirectionalState[S]): (Tensor, BidirectionalState[S]) =
  ## Runs the forward layer in the forward direction and the backward layer
  ## in the reverse direction over axis=1. Concatenates outputs along axis=2.
  ##
  ## Each layer's `forward` must accept `(x: Tensor, state: S)` and return
  ## `(Tensor, S)`.
  if x.shape.len < 2:
    raise newException(TensorError,
      "Bidirectional.forward: input must have at least 2 dims (NTC layout)")
  let seqLen = x.shape[1]
  # Forward direction
  let (fwdOut, fwdState) = layer.forwardLayer.forward(x, state.forwardState)
  # Backward direction: reverse along axis=1, run, reverse back
  let reversed = reverse(x, [1])
  let (bwdOut, bwdState) = layer.backwardLayer.forward(reversed, state.backwardState)
  let bwdOutFixed = reverse(bwdOut, [1])
  # Concatenate along feature axis
  let output = concat(@[fwdOut, bwdOutFixed], 2)
  (output, BidirectionalState[S](forwardState: fwdState, backwardState: bwdState))
