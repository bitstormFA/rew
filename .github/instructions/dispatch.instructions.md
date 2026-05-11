---
applyTo: "src/rew/{dispatch,eager}.nim"
---

# Layer 4 — Dispatcher (eager / trace) + compile cache

This is the **only** place that decides whether an op runs eagerly or is
recorded into an in-progress trace. `dispatch.nim` owns the dispatcher
state machine and the eager compile cache; `eager.nim` wires the eager
backend to the PJRT client.

## Hard rules

- `tensor.nim` and every file under `src/rew/ops/` must call into this layer
  for execution; they must never construct a StableHLO program directly nor
  call PJRT directly.
- The compile cache key includes: op identity (or full traced program
  fingerprint), input shapes, dtypes, device, and op attributes. Add new
  cache-key components only with a written justification in a doc comment.
- The "current dispatcher" is a thread-local. It is set by `jit`'s tracer
  and by `lazy(fn)` through the same runtime tracing path; it is never set
  globally outside those entry points.
- Eager dispatch sees one op at a time; trace dispatch appends to an
  in-progress program. Both produce identical observable results for the
  same input.

## Skills to follow

- [`nim-code-organization`](../../.agents/skills/nim-code-organization/SKILL.md)
  for the dispatcher state machine.
- [`nim-error-handling`](../../.agents/skills/nim-error-handling/SKILL.md)
  for compile failures (translate StableHLO/PJRT errors into the user's call
  site).
