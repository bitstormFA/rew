## `Sequential` — chains layers (and any callable) in order.
##
## Pure value type following rew's functional nn invariant. The layers
## are stored as a flat `seq[Tensor]` via pytree: each layer's
## parameters are flattened into the leaf list so optimizers and
## `grad` can reach them without knowing the internal structure.
##
## `forward` applies each layer in sequence, threading the tensor
## through.

import ../tensor
import ../pytree

type
  Sequential* = object
    ## A linear chain of callable objects. Each forward is stored as a
    ## closure `proc(x: Tensor): Tensor`. Use the `add` template to add
    ## layers that support `forward(layer, x)`.
    forwards*: seq[proc(x: Tensor): Tensor {.closure.}]
    leaves*: seq[Tensor]

proc initSequential*(): Sequential =
  Sequential(forwards: @[], leaves: @[])

template add*(s: var Sequential; layer: untyped) =
  ## Adds `layer` to the chain. The layer must support
  ## `forward(layer, x: Tensor): Tensor` or be directly callable as
  ## `layer(x: Tensor): Tensor`.
  let fwd = proc(x: Tensor): Tensor =
    when compiles(forward(layer, x)):
      forward(layer, x)
    else:
      layer(x)
  s.forwards.add(fwd)
  when compiles(treeFlatten(layer)):
    let layerLeaves = treeFlatten(layer)
    for leaf in layerLeaves:
      s.leaves.add(leaf)

proc forward*(s: Sequential; x: Tensor): Tensor =
  var y = x
  for f in s.forwards:
    y = f(y)
  y

proc treeFlatten*(s: Sequential): seq[Tensor] =
  s.leaves

proc treeUnflatten*(s: Sequential; leaves: seq[Tensor]): Sequential =
  result = s
  result.leaves = leaves
