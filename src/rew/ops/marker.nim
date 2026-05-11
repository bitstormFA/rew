## Shared marker pragma for `{.rewOp.}`.
##
## Every public eager-or-trace op declared under `src/rew/ops/*.nim` (and
## `src/rew/tensor.nim`, when methods land there) annotates its proc
## declaration with `{.rewOp.}`. The pragma carries no codegen meaning;
## it is consumed by the architectural lint
## `tools/check_vjp_coverage.nim`, which scans for it and asserts that
## every match has a matching `registerVjp("name")` in
## `src/rew/autograd/registry.nim`.

template rewOp*() {.pragma.}
  ## Marker pragma for `rewOp`; see module doc.
