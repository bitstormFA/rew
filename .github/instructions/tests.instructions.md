---
applyTo: "tests/**"
---

# Tests

## Hard rules

- Files use the `t` prefix (`tbasic.nim`, `tjit.nim`, …); `tests/all.nim`
  auto-discovers them.
- Use `block` + `doAssert`. **Do not** use `std/unittest`. **Do not** use
  bare `assert` (it is compiled out under `-d:danger`).
- Catch `Defect` subclasses (e.g. `AssertionDefect`, `OverflowDefect`) by
  their specific type; bare `except:` and `except CatchableError` do not
  catch them.
- Tests must pass under all three configurations: default, `-d:release`,
  `-d:danger`. Use `when defined(danger):` guards for overflow-dependent
  tests.
- For every public op (`{.rewOp.}`):
  - One numerical-correctness test against a hand-computed reference.
  - One `jit`-vs-eager equivalence test (same inputs, same outputs).
  - One vjp test (numerical gradient vs registered rule).

## Skill to follow

- [`nim-testing`](../../.agents/skills/nim-testing/SKILL.md) — the canonical
  reference for everything above, plus the AddressSanitizer setup that
  `nimble asan` invokes.
