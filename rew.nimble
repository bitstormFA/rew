# Package

version       = "0.2.0"
author        = "Rechenwerk contributors"
description   = "Rechenwerk (rew) — eager, debuggable Nim deep learning framework on OpenXLA/PJRT"
license       = "MIT"
srcDir        = "src"
installExt    = @["nim"]
bin           = @["rew_fetch", "rewrun"]

# Dependencies

requires "nim >= 2.2.0"

# Tasks

task test, "Run the full test suite (debug + release + danger)":
  exec "nim c -r tests/all.nim"
  exec "nim c -d:release -r tests/all.nim"
  exec "nim c -d:danger -r tests/all.nim"

task asan, "Run the test suite under AddressSanitizer":
  exec "nim c -d:addressSanitizer -r tests/all.nim"

task lint, "Run architectural lints (layer imports, vjp coverage, no ref in nn/optim, manifest)":
  exec "nim c -r tools/check_layer_imports.nim"
  exec "nim c -r tools/check_vjp_coverage.nim"
  exec "nim c -r tools/check_openxla_coverage.nim"
  exec "nim c -r tools/check_no_ref_in_nn.nim"

task bench, "Run Rechenwerk/PyTorch MNIST example benchmarks":
  exec "python3 benchmarks/run.py"

task fetch, "Download a PJRT plugin for a target into the cache":
  ## Usage: nimble fetch <target>      (target in: cpu, cuda12, cuda13, rocm, metal, tpu)
  ##
  ## Downloads the PJRT plugin for the given target using the manifest
  ## (pjrt_manifest.json) URLs and verifies SHA-256 integrity.
  ## Installed packages also provide `rew_fetch <target>`, which does not
  ## require running from the rew source tree.
  ## Override the cache directory with `REW_CACHE_DIR`.
  ## Override the archive URL with `REW_ARCHIVE_URL`.
  var cmd = "nim c -r --hints:off tools/fetch.nim"
  for a in commandLineParams:
    if a != "fetch":
      cmd.add " " & a
  exec cmd

task hfFetch, "Download Hugging Face model or dataset assets into the HF cache":
  ## Usage:
  ##   nimble hfFetch model google/gemma-4-E4B-it
  ##   nimble hfFetch dataset NousResearch/hermes-function-calling-v1 func_calling_singleturn
  var cmd = "nim c -r --hints:off tools/hf_fetch.nim"
  for a in commandLineParams:
    if a != "hfFetch":
      cmd.add " " & a
  exec cmd

task buildPlugin, "Build a PJRT plugin from an openxla/xla checkout":
  ## Usage: nimble buildPlugin <target>
  ## Requires REW_BUILD_XLA_DIR and bazel on PATH.
  let args = commandLineParams
  var target = ""
  for a in args:
    if a.len > 0 and a[0] != '-' and a notin ["buildPlugin"]:
      target = a
  if target.len == 0:
    echo "Usage: nimble buildPlugin <target>   (cpu | cuda12 | cuda13 | rocm | tpu)"
    quit 1
  exec "nim c -r --hints:off tools/build_plugin.nim " & target

task updateManifest, "Re-resolve vendor URLs and recompute manifest checksums":
  exec "nim c -r --hints:off tools/update_manifest.nim"

task doctor, "Verify installed PJRT plugins by listing devices":
  exec "nim c -r --hints:off examples/list_devices.nim"

task openxla, "Run an optional OpenXLA tool through rew's wrapper":
  ## Usage: nimble openxla list
  ##        nimble openxla <tool> [args...]
  let args = commandLineParams
  var cmd = "nim c -r --hints:off tools/openxla_tool.nim"
  var seenTool = false
  for a in args:
    if not seenTool:
      if a == "openxla":
        continue
      if a.len > 0 and a[0] == '-':
        continue
      seenTool = true
    if a.len > 0:
      cmd.add " " & a
  exec cmd
