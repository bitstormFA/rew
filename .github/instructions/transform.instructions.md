---
applyTo: "src/rew/transform/**"
---

# Layer 7 — `jit` and combinators

The compile-the-graph entry points. Reuses the StableHLO emitter, the
dispatcher, and the vjp registry — does not duplicate any of them.

## Hard rules

- `jit(fn, donateArgs = …, staticArgs = …)` returns a wrapped proc. On first
  call with a given input signature it traces, compiles, caches; subsequent
  calls hit the cache. The cache is keyed on `(fn id, input shapes, dtypes,
  devices, static-arg values)`.
- Donated `Tensor`s are flipped to `Donated` state (see Layer 2). Any reuse
  raises `BufferDonatedError` with a message naming the consuming `jit`
  call.
- The combinators `cond` / `whileLoop` / `fori` lower to StableHLO `cond`
  and `while` when the trace dispatcher is active, and behave eagerly
  otherwise. They are the **only** in-graph control-flow surface; do not
  add others. `lazy(fn)` is a runtime eager-batching wrapper over `jit`, not
  a macro syntax. `scan` remains reserved for a follow-up.
- **Never** add a macro-based `jit` here or anywhere else.

## Skills to follow

- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for the
  `jit` parameter shape.
- [`nim-error-handling`](../../.agents/skills/nim-error-handling/SKILL.md)
  for donation and observation errors.
