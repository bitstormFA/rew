# Compatibility metadata for the Nimble registry.
# Build orchestration is declared in bau.toml and executed with Bau.

# Package

version       = "0.2.0"
author        = "Rechenwerk contributors"
description   = "Rechenwerk (rew) - eager, debuggable Nim deep learning framework on OpenXLA/PJRT"
license       = "MIT"
srcDir        = "src"
installExt    = @["nim"]
bin           = @["rew_fetch", "rewrun"]

# Dependencies

requires "nim >= 2.2.0"

# Tasks

task test, "Run the full Bau test matrix":
  exec "bau test"

task asan, "Run the test suite under AddressSanitizer":
  exec "bau asan"

task lint, "Run architectural lints":
  exec "bau lint"

task bench, "Run Rechenwerk/PyTorch MNIST example benchmarks":
  exec "bau bench"

task fetch, "Download a PJRT plugin for a target into the cache":
  ## Source checkout usage:
  ##   bau fetch <target>
  ##
  ## Nimble compatibility usage:
  ##   nimble fetch <target>      (target in: cpu, cuda12, cuda13, rocm, metal, tpu)
  var cmd = "bau fetch"
  for a in commandLineParams:
    if a != "fetch":
      cmd.add " " & a
  exec cmd

task hfFetch, "Download Hugging Face model or dataset assets into the HF cache":
  ## Source checkout usage:
  ##   bau hfFetch model google/gemma-4-E4B-it
  ##
  ## Nimble compatibility usage:
  ##   nimble hfFetch model google/gemma-4-E4B-it
  ##   nimble hfFetch dataset NousResearch/hermes-function-calling-v1 func_calling_singleturn
  var cmd = "bau hfFetch"
  for a in commandLineParams:
    if a != "hfFetch":
      cmd.add " " & a
  exec cmd

task buildPlugin, "Build a PJRT plugin from an openxla/xla checkout":
  ## Source checkout usage:
  ##   bau buildPlugin <target>
  ##
  ## Nimble compatibility usage:
  ##   nimble buildPlugin <target>
  var cmd = "bau buildPlugin"
  for a in commandLineParams:
    if a != "buildPlugin":
      cmd.add " " & a
  exec cmd

task updateManifest, "Re-resolve vendor URLs and recompute manifest checksums":
  exec "bau updateManifest"

task doctor, "Verify installed PJRT plugins by listing devices":
  exec "bau task doctor"

task openxla, "Run an optional OpenXLA tool through rew's wrapper":
  ## Source checkout usage:
  ##   bau openxla list
  ##
  ## Nimble compatibility usage:
  ##   nimble openxla list
  ##   nimble openxla <tool> [args...]
  var cmd = "bau openxla"
  for a in commandLineParams:
    if a != "openxla":
      cmd.add " " & a
  exec cmd
