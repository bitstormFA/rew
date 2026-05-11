---
applyTo: "src/rew/pjrt/**"
---

# Layer 1 — PJRT C bindings, loader, and multi-plugin registry

This folder is the **only** place in the codebase that touches the PJRT C API
directly.

## Hard rules

- Files here may import stdlib modules, other files inside this directory, and
  modules from `src/rew/binaries/` (for the `Target` enum and resolver).
  Outside `src/rew/pjrt/`, only `src/rew/buffer.nim`, `src/rew/device.nim`,
  and `src/rew/eager.nim` are allowed to import from `pjrt/`. The
  `check_layer_imports` lint enforces both directions.
- `src/rew/binaries/` must NOT import from `src/rew/pjrt/` (one-directional
  dependency: `pjrt/ -> binaries/`).
- All PJRT error codes are translated into one Nim exception type
  `PjrtError` (`CatchableError`); never let a raw integer status escape.
- No allocation of higher-level types here (no `Tensor`, no `BufferHandle`).
  This layer is purely C structs, integer codes, function pointers, a
  loader, and a registry.
- Pointer lifetime contracts (who frees what) live in doc comments on each
  extern declaration.

## Skills to follow

- [`nim-c-bindings`](../../.agents/skills/nim-c-bindings/SKILL.md) for
  declaring the C surface, headers, and cross-platform shared-library load.
- [`nim-c-wrappers`](../../.agents/skills/nim-c-wrappers/SKILL.md) for
  shaping the thin Nim-side wrapper procs around the raw extern decls.

## Files

- `capi.nim` — extern decls, structs, status codes, function-pointer
  typedefs from the PJRT C API.
- `loader.nim` — resolves plugin path via `binaries/resolver`, then `dlopen`
  + `GetPjrtApi` symbol resolution. Entry point: `loadPlugin(t: Target)`.
- `registry.nim` — multi-plugin API table and per-`(Target, ordinal)` client
  cache. Provides `releaseBuffer(target, raw)` used by `BufferHandle`'s
  destructor.
- `client.nim` — high-level Nim wrapper around the raw PJRT client and
  loaded-executable types.
