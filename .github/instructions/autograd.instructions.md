---
applyTo: "src/rew/autograd/**"
---

# Layer 6 — Autograd (vjp registry, eager tape, transform)

One vjp registry. Two consumers (eager `backward()` and functional
`grad`/`vjp`). Reverse-mode only in v1.

## Hard rules

- The registry in `registry.nim` is the single source of truth. There is no
  second backward implementation anywhere else.
- A vjp rule is expressed in the same op surface from `tensor.nim`/`ops/` —
  rules emit StableHLO via the public op API, not via the dispatcher
  directly. This guarantees rules also work under `jit`.
- The eager tape in `tape.nim` records to a thread-local stack inside a
  `gradMode:` block. Outside such a block, ops do not record. The tape is
  not a global flag and there is no `requires_grad` field on `Tensor`.
- `transform.nim` implements `grad(fn, argnums = …)` and `vjp(fn, args)`
  by swapping the dispatcher to "trace" and replaying through the registry.
- Forward-mode (`jvp`) is **deferred**; do not add it without an explicit
  design review.

## Skills to follow

- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for the
  registry's public shape.
- [`nim-error-handling`](../../.agents/skills/nim-error-handling/SKILL.md)
  for missing-rule and shape-mismatch errors.
