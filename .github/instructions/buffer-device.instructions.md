---
applyTo: "src/rew/{buffer,device,dtype,sharding}.nim"
---

# Layer 2 — Buffer ownership, device, dtype, sharding

The thin layer between the raw C API and everything above.

## Hard rules

- `BufferHandle` is the **only** `ref object` allowed in this layer (and one
  of only a few in the entire codebase). Its `=destroy` hook is the single
  point that calls into PJRT to release a buffer; it must be exception-safe
  and idempotent (donated/already-released handles are a no-op).
- A `BufferHandle` carries a state field with at least `Live` and `Donated`.
  Any operation on a `Donated` handle must raise `BufferDonatedError` with a
  message pointing at the originating `jit` call.
- `Device` is a value type (`(target, ordinal)`). The default-target
  resolution happens once per thread (cuda13 > cuda12 > rocm > cpu) and is
  overridable via `REW_TARGET` and `setDefaultDevice`.
- `Sharding` carries replicated, partitioned, or manual annotations. Keep it
  metadata-first: validation is allowed, but sharding must not introduce
  implicit host or cross-device transfers.
- `DType` is a closed enum; conversions to/from native Nim scalar types are
  exhaustive `case` matches.

## Skills to follow

- [`nim-ownership-hooks`](../../.agents/skills/nim-ownership-hooks/SKILL.md)
  for `=destroy`/`=copy`/`=sink` semantics on `BufferHandle`.
- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for the
  public surface (`initDevice`, `defaultDevice`, etc.).
- [`nim-error-handling`](../../.agents/skills/nim-error-handling/SKILL.md)
  for `BufferDonatedError` and other failure modes.
