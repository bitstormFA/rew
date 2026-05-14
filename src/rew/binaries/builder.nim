## Builder — minimal build-from-source path for PJRT plugins.
##
## When `REW_BUILD=true`, invokes `bazel build` against a user-provided
## openxla/xla checkout (`REW_BUILD_XLA_DIR`) and copies the resulting
## plugin into the rew cache. No hermetic config management — users are
## responsible for their toolchain (Bazel, Clang, CUDA SDK, etc.).

import std/[os, osproc, sequtils, strformat, strutils]
import ./target
import ./cache

const
  BuildXlaDirEnvVar* = "REW_BUILD_XLA_DIR"

type
  BuildError* = object of CatchableError
    ## Raised on build failures or missing prerequisites.

proc bazelTarget(t: Target): string =
  ## Returns the Bazel target that produces the PJRT plugin for `t`.
  case t
  of tCpu: "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
  of tCuda12, tCuda13: "//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
  of tTpu: "//xla/pjrt/c:pjrt_c_api_tpu_plugin.so"
  of tRocm: "//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
  of tMetal:
    raise newException(BuildError,
      "building the Metal PJRT plugin from openxla/xla is not supported; " &
        "use `bau fetch metal` to install the jax-metal binary")

proc bazelBinPath(t: Target): string =
  ## Relative path under `bazel-bin/` where the plugin lands.
  case t
  of tCpu: "xla/pjrt/c/pjrt_c_api_cpu_plugin.so"
  of tCuda12, tCuda13: "xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
  of tTpu: "xla/pjrt/c/pjrt_c_api_tpu_plugin.so"
  of tRocm: "xla/pjrt/c/pjrt_c_api_gpu_plugin.so"
  of tMetal:
    raise newException(BuildError,
      "building the Metal PJRT plugin from openxla/xla is not supported; " &
        "use `bau fetch metal` to install the jax-metal binary")

proc buildPlugin*(t: Target) =
  ## Builds the PJRT plugin for `t` from source and copies it into the
  ## rew cache. Requires `REW_BUILD_XLA_DIR` pointing to an openxla/xla
  ## checkout and `bazel` on PATH.
  let xlaDir = getEnv(BuildXlaDirEnvVar)
  if xlaDir.len == 0:
    raise newException(BuildError,
      BuildXlaDirEnvVar & " must be set to an openxla/xla checkout path " &
      "when building from source")
  if not dirExists(xlaDir):
    raise newException(BuildError,
      BuildXlaDirEnvVar & " directory does not exist: " & xlaDir)
  if findExe("bazel").len == 0:
    raise newException(BuildError,
      "bazel not found on PATH. Install Bazel v7.7+ to build PJRT plugins")

  let target = bazelTarget(t)
  var flags = @["build", "-c", "opt"]

  case t
  of tCuda12:
    flags.add "--config=cuda"
    flags.add "--repo_env=HERMETIC_CUDA_VERSION=\"12.9.1\""
    flags.add "--repo_env=HERMETIC_CUDNN_VERSION=\"9.8.0\""
  of tCuda13:
    flags.add "--config=cuda"
    flags.add "--repo_env=HERMETIC_CUDA_VERSION=\"13.0.0\""
    flags.add "--repo_env=HERMETIC_CUDNN_VERSION=\"9.12.0\""
  of tRocm:
    flags.add "--config=rocm"
  of tCpu, tTpu, tMetal:
    discard

  flags.add target

  echo &"  building {targetName(t)} plugin in {xlaDir}"
  echo &"  bazel {flags.join(\" \")}"
  let cmd = "cd " & quoteShell(xlaDir) & " && bazel " &
    flags.mapIt(quoteShell(it)).join(" ")
  let rc = execCmd(cmd)
  if rc != 0:
    raise newException(BuildError,
      &"bazel build failed for {targetName(t)} (exit {rc})")

  let srcPath = xlaDir / "bazel-bin" / bazelBinPath(t)
  if not fileExists(srcPath):
    raise newException(BuildError,
      &"build output not found at {srcPath}")

  ensureDirs()
  let destPath = pluginPathForTarget(t)
  copyFile(srcPath, destPath)
  setFilePermissions(destPath, {fpUserRead, fpUserWrite, fpUserExec,
                                fpGroupRead, fpGroupExec,
                                fpOthersRead, fpOthersExec})
  echo &"  copied plugin → {destPath}"
