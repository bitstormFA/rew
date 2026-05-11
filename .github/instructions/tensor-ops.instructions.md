---
applyTo: "src/rew/{tensor.nim,ops/**}"
---

# Layer 5 — Tensor + op surface

The user-facing API. Every change here is API-visible.

## Hard rules

- Every public op is a `proc` (or `func` for pure shape/dtype helpers)
  marked with `{.rewOp.}` so the `check_vjp_coverage` lint can find it.
- Every `{.rewOp.}` requires:
  1. A forward emitter that calls into `src/rew/dispatch.nim` (never PJRT or
     StableHLO directly).
  2. A vjp rule registered in `src/rew/autograd/registry.nim`.
  3. A numerical-correctness test in `tests/`.
  4. A `jit`-vs-eager equivalence test in `tests/`.
  Lint #2 and the test runner enforce items 2–4.
- No implicit dtype promotion that hides perf cliffs; promote explicitly via
  `astype(...)` at the call site or raise a clear error.
- No implicit cross-device transfer. `.to(device)` is the only mover.
- Indexing returns `Tensor`, never a scalar. `.item` is the only scalar
  path and is only valid on 0-d tensors.
- No user-visible eager in-place ops. The small explicit set tied to
  donation lives behind a clearly named `*InPlace` suffix.

## Skills to follow

- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for op
  signatures, overloading, and `openArray` use.
- [`nim-style-guide`](../../.agents/skills/nim-style-guide/SKILL.md) for
  naming and formatting.
- [`nim-doc-comments`](../../.agents/skills/nim-doc-comments/SKILL.md) for
  every exported op.
