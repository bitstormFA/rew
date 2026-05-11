## `Residual` — wrapper that adds a skip connection around any layer.
##
## `forward(layer, x)` returns `x + layer.forward(x)`. Shapes must match.

import ../tensor
import ../ops/arith

type
  Residual*[L] = object
    ## Wraps a layer `L` with a residual (skip) connection:
    ## `output = x + forward(layer, x)`.
    inner*: L

proc initResidual*[L](layer: L): Residual[L] =
  Residual[L](inner: layer)

proc forward*[L](r: Residual[L]; x: Tensor): Tensor =
  ## `forward` is not defined generically — layers have their own
  ## `forward` proc. See `residualForward` template below.
  raise newException(TensorError,
    "forward not available for Residual. Use `residualForward` template instead.")

template residualForward*(r: untyped; x: Tensor): Tensor =
  add(x, forward(r.inner, x))
