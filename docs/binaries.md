# PJRT Plugin Binaries

Rew loads PJRT plugin shared libraries at runtime via `dlopen`. This document
covers the target matrix, environment variables, manifest format, and
build-from-source recipe.

## Target matrix

| Slot key | Source | Platform |
|----------|--------|----------|
| `x86_64-linux-gnu-cpu` | vendor (gomlx) | Linux x86_64 |
| `aarch64-linux-gnu-cpu` | vendor (gomlx) | Linux ARM64 |
| `aarch64-darwin-cpu` | vendor (gomlx) | macOS ARM64 |
| `x86_64-windows-cpu` | vendor (gomlx) | Windows x86_64 |
| `x86_64-linux-gnu-cuda12` | vendor (PyPI) | Linux x86_64 + CUDA 12 |
| `aarch64-linux-gnu-cuda12` | vendor (PyPI) | Linux ARM64 + CUDA 12 |
| `x86_64-linux-gnu-cuda13` | vendor (PyPI) | Linux x86_64 + CUDA 13 |
| `aarch64-linux-gnu-cuda13` | vendor (PyPI) | Linux ARM64 + CUDA 13 |
| `x86_64-linux-gnu-rocm` | vendor (PyPI) | Linux x86_64 + ROCm 7 |
| `x86_64-darwin-metal` | vendor (PyPI) | macOS x86_64 + Metal |
| `aarch64-darwin-metal` | vendor (PyPI) | macOS ARM64 + Metal |
| `x86_64-linux-gnu-tpu` | vendor (PyPI) | Linux x86_64 + TPU |

## Target enum

```nim
type Target* = enum
  tCuda13, tCuda12, tRocm, tMetal, tTpu, tCpu
```

Constructors: `cpu(0)`, `cuda12(0)`, `cuda13(0)`, `rocm(0)`, `metal(0)`,
`tpu(0)`.

## Environment variables

All variables use the `REW_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| `REW_TARGET` | auto-detected | Override the default target (cpu, cuda12, cuda13, rocm, metal, tpu) |
| `REW_TARGET_PLATFORM` | auto-detected | Override host triplet (e.g. `x86_64-linux-gnu`) |
| `REW_CACHE_DIR` | `~/.cache/rew` | Root of the plugin cache directory |
| `REW_EXECUTABLE_CACHE` | enabled | Set to `0`, `false`, `off`, or `no` to disable serialized PJRT executable caching |
| `REW_ARCHIVE_URL` | — | Override the manifest URL for downloads |
| `REW_ARCHIVE_PATH` | — | Use a local archive instead of downloading |
| `REW_PJRT_PLUGIN_PATH` | — | Colon-separated directories to search for plugin `.so`/`.dylib` |
| `REW_BUILD` | — | Set to `true` to use build-from-source path |
| `REW_BUILD_XLA_DIR` | — | Path to an openxla/xla checkout (required for `REW_BUILD=true`) |

Optional OpenXLA tool wrappers use the same prefix. Set these only when you
want rew to invoke external tools:

| Variable | Tool |
|----------|------|
| `REW_HLO_OPT` | `hlo-opt` |
| `REW_ISOLATE_HLO` | `isolate_hlo` |
| `REW_MULTIHOST_HLO_RUNNER` | `multihost_hlo_runner` |
| `REW_MPMD_OPT` | `mpmd_opt` |
| `REW_PTX_OPT` | `ptx-opt` |
| `REW_RUN_HLO_MODULE` | `run_hlo_module` |
| `REW_SDY_OPT` | `sdy_opt` |
| `REW_SDY_TRANSLATE` | `sdy_translate` |
| `REW_XPROF` | `xprof` |
| `REW_TOKAMAX` | `tokamax` |

## Default target auto-detection

When `REW_TARGET` is not set, rew probes the system in priority order:

1. `nvcc --version` → `tCuda13` (release 13.x) or `tCuda12` (release 12.x)
2. `rocminfo` or `hipcc` on PATH → `tRocm`
3. Fallback → `tCpu`

The result is cached for the process lifetime.

## Cache layout

```
<REW_CACHE_DIR>/<rew_version>/
  download/     # archives from manifest URLs
  external/     # archives from REW_ARCHIVE_URL
  build/        # locally built plugins
  plugins/      # extracted plugin shared libraries
  executables/  # serialized PJRT executables when plugins support it
```

The version-qualified subdirectory ensures different rew releases never share
cached plugins or serialized executables that might be ABI-incompatible.

## Manifest format (`pjrt_manifest.json`)

```json
{
  "rew_version": "0.2.0",
  "openxla_xla_rev": "<sha>",
  "expected_pjrt_api_version": { "major": 0, "minor": 107 },
  "slots": {
    "<slot-key>": {
      "source": "vendor | vendor_wheel | rew",
      "url": "<archive URL>",
      "archive_member": "<path inside archive>",
      "sha256": "<hex digest>"
    }
  }
}
```

The manifest is committed at the repo root and embedded at compile time via
`staticRead`. Run `nimble updateManifest` to re-resolve URLs and recompute
checksums.

## Fetching plugins

When using rew from a source checkout, use the Nimble task:

```bash
nimble fetch cpu
```

Installed packages also provide a standalone downloader:

```bash
rew_fetch cpu      # or cuda12, cuda13, rocm, metal, tpu
```

Both commands use the same embedded manifest and write to the same cache
layout. They honor `REW_CACHE_DIR`, `REW_ARCHIVE_URL`, and
`REW_ARCHIVE_PATH`.

## Build from source

For platforms not in the manifest, build from an openxla/xla checkout:

```bash
export REW_BUILD=true
export REW_BUILD_XLA_DIR=/path/to/openxla/xla
nimble buildPlugin cpu     # or cuda12, cuda13, rocm, tpu
```

Requirements: Bazel v7.7+, Clang, and platform-specific SDKs (CUDA, cuDNN,
ROCm, etc.). Metal is distributed as a binary `jax-metal` plugin and is not
supported by the `buildPlugin` source-build shim.

## Nimble tasks

| Task | Description |
|------|-------------|
| `nimble fetch <target>` | Download + cache a PJRT plugin |
| `rew_fetch <target>` | Installed standalone downloader for the same plugins |
| `nimble buildPlugin <target>` | Build a plugin from source |
| `nimble updateManifest` | Re-resolve URLs + recompute SHA-256s |
| `nimble doctor` | List devices for all available targets |
| `nimble openxla list` | Show optional OpenXLA tool paths and availability |
| `nimble openxla <tool> [args...]` | Run `run_hlo_module`, Shardy, XProf, Tokamax, or another configured tool |
