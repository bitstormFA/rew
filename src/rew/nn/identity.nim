## Identity layer — pass-through no-op.
##
## Useful for model surgery, ablation studies, and skip-connection gating.

import ../tensor

type
  Identity* = object
    ## Identity layer. `forward` returns the input unchanged. Carries no
    ## parameters. Useful for programmatic model construction where a
    ## placeholder layer is needed.

proc initIdentity*(): Identity =
  Identity()

proc forward*(layer: Identity; x: Tensor): Tensor =
  ## Returns `x` unchanged.
  x
