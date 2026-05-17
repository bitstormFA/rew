# AGENTS.md

Mirror of [.github/copilot-instructions.md](.github/copilot-instructions.md)
for non-Copilot agents (Claude Code, Cursor, Aider, etc.).

For the full architecture and the ten invariants, see
[docs/architecture.md](docs/architecture.md). For the high-level API contract,
see [docs/high-level-api.md](docs/high-level-api.md). For per-layer rules, see
the files under [.github/instructions/](.github/instructions/) — open the one
matching the layer you are about to edit before making changes.

## Quick rules

- Edit only the layer you came for.
- Keep `bau test` and `bau lint` green.
- New ops require: forward emitter + vjp registration + numerical test +
  `jit`-vs-eager equivalence test.
- No `ref object` under `src/rew/nn/` or `src/rew/optim/`.
- No imports out of `src/rew/pjrt/` from anywhere except `buffer.nim`,
  `device.nim`, and `eager.nim`. `pjrt/` may import from `binaries/`
  (for the `Target` enum) but not the reverse.
- No macro `jit`, no `requires_grad` flag, no implicit host or cross-device
  transfers.
- Public API is tiered: `import rew` for high-level user code, `import rew/xla`
  for raw compiler/lowering/JIT, and `import rew/dev` for extension internals
  plus plugin target/manifest tooling. Raw PJRT C modules stay under
  `rew/pjrt/*`.
- High-level API changes must follow `docs/high-level-api.md`; new high-level
  examples should use `TrainState` and `compileTrainStep`, not raw `JitFn`
  plumbing.
- `Device` uses a closed `Target` enum (`tCpu`, `tCuda12`, `tCuda13`,
  `tRocm`, `tMetal`, `tTpu`), not strings.
- Environment variables use the `REW_` prefix (`REW_TARGET`, `REW_CACHE_DIR`,
  `REW_BUILD`, etc.).

## Build / test

```
bau test               # debug + release + danger
bau testFast [filter]  # dev-profile tests for fast iteration
bau lint               # architectural lints
bau asan               # AddressSanitizer
bau fetch cpu          # download CPU PJRT plugin
bau fetch cuda12       # download CUDA 12 PJRT plugin
rew_fetch cpu          # installed plugin downloader, no source tree needed
bau buildPlugin cpu    # build from openxla/xla source
bau updateManifest     # re-resolve URLs + recompute SHA-256s
bau task doctor        # list devices for all available targets
```

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- If the graphify MCP server is active, utilize tools like `query_graph`, `get_node`, `god_nodes`, and `shortest_path` for precise architecture navigation
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
