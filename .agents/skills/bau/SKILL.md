---
name: bau
description: Use Bau for this Nim project. Use when building, testing, running, managing dependencies, working with tasks/cache/docs, inspecting metadata, or connecting coding agents through MCP.
---

# Bau

Use Bau as the orchestration layer for this Nim project.

## First Moves

1. Read `bau.toml` before changing build behavior.
2. Prefer Bau commands over raw `nim`, `nimble`, or `atlas` commands.
3. Use `bau metadata --json` when you need project structure.
4. Preserve `bau.lock`; use `bau deps sync --locked` for normal setup.
5. Use `bau deps sync` only when intentionally refreshing dependency state.

## Common Commands

```sh
bau check
bau test
bau ci
bau run -- --help
bau deps sync --locked
bau deps verify
bau doc
bau affected list --since origin/main
bau affected test --since origin/main
```

## Agent Integration

Bau exposes the same operations through CLI and MCP. Start the MCP server with:

```sh
bau mcp
```

Use MCP tools for build, run, test, check, lint, CI, docs, dependency, task,
cache, metadata, graph, query, affected-work, package, publish dry-run, and
setup operations when they are available.
