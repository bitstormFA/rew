## Extension and development surface.
##
## Import this when adding primitive ops, VJP rules, or debugging dispatch.
## Raw PJRT C internals remain specialist imports under `rew/pjrt/*` to
## preserve the layer boundary.

import ./xla
export xla

import ./autograd/registry
export registry

import ./autograd/tape
export tape

import ./ops/marker
export marker

import ./eager
export eager

import ./binaries/[target, manifest]
export target, manifest
