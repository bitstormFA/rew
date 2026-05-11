---
applyTo: "src/rew/{nn/**,optim/**,pytree.nim,rng.nim,serialize.nim}"
---

# Layer 8/9 — Pytree, nn layers, optimizers, PRNG, serialize

All pure functional. Plain value types only.

## Hard rules

- **No `ref object` declarations** under `src/rew/nn/` or `src/rew/optim/`.
  The `check_no_ref_in_nn` lint enforces this. Holding an interior `Tensor`
  (which itself contains a `ref BufferHandle`) is fine — declaring new
  `ref` types here is not.
- Layers are plain `object` types built by `initLinear(...)`-style
  constructors. Forward is a `proc forward(layer, …)`; the call site is
  `model.forward(x)`. The `()` sugar is opt-in via the `Callable` concept.
- One pytree mechanism (`treeFlatten`/`treeUnflatten` in `pytree.nim`)
  drives optimizer, `grad`, serialization, and device transfer. Anything
  satisfying the `Pytree` concept becomes a module — there is **no
  `Module` base class**.
- PRNG is explicit: `Key` values are passed in and split. **No global
  RNG.** Layers that need randomness (dropout, init) take a `Key` argument.
- Optimizers return new state, never mutate inputs:
  `let (newParams, newOptState) = sgd.step(params, grads, optState)`.
- `serialize.nim` walks pytrees structurally — it does not register types.

## Skills to follow

- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for
  layer/optimizer constructors and accessors.
- [`nim-style-guide`](../../.agents/skills/nim-style-guide/SKILL.md) for
  the `proc` vs `func` choice on pure helpers.
- [`nim-doc-comments`](../../.agents/skills/nim-doc-comments/SKILL.md) for
  every exported layer / optimizer / helper.
