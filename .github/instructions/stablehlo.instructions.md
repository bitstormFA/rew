---
applyTo: "src/rew/stablehlo/**"
---

# Layer 3 — StableHLO IR + emitter (textual now, bytecode later)

Pure Nim. This layer **never** imports from `src/rew/pjrt/`.

## Hard rules

- v1 emits **textual MLIR** (the `module { func.func @main(...) { ... } }`
  form). Bytecode emission is deferred to Phase 9 and must reuse the same
  IR + verifier. The MLIR-bytecode primitives in `mlirbc.nim` are kept
  unit-tested ahead of time so the later bytecode emitter can build on a
  validated foundation.
- StableHLO op names and attribute spellings follow a pinned StableHLO
  release tag. New ops require a corresponding builder entry in
  `ops.nim`, a verifier rule in `verify.nim`, and an emitter case in
  `text.nim`.
- The verifier (`verify.nim`) runs **before** emission and produces
  Nim exceptions whose message references the user-facing op name and
  argument shapes — never the internal IR node id.
- The builder API in `ops.nim` is the single surface used by both the eager
  dispatcher and the trace dispatcher. Do not introduce a parallel
  emit-path elsewhere.
- IR types in `ir.nim` are plain value `object`s. The graph is built by
  appending nodes, not by linking refs.

## Skills to follow

- [`nim-code-organization`](../../.agents/skills/nim-code-organization/SKILL.md)
  for the multi-step builder/verifier/serializer split.
- [`nim-api-design`](../../.agents/skills/nim-api-design/SKILL.md) for the
  builder's public shape.
