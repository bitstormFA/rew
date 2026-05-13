## Extension and development surface.
##
## Import this when adding primitive ops, VJP rules, or debugging dispatch.
## PJRT C internals remain in `rew/pjrt/*` to preserve the layer boundary.

import ./xla
export xla

import ./autograd/registry
export registry

import ./eager
export eager

import ./binaries/[target, manifest]
export target, manifest
