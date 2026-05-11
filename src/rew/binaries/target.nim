## Target — closed enum for PJRT plugin backend selection.
##
## Replaces the old `Device.plugin: string` with a compile-time-exhaustive
## set of supported accelerator targets. Each `Target` maps to exactly one
## PJRT plugin shared library and one slot in the manifest.
##
## Host-triplet detection, `REW_TARGET` / `REW_TARGET_PLATFORM` env-var
## parsing, and the lazy `nvcc` / `rocminfo` probe all live here so the
## rest of the codebase never touches raw platform strings.

import std/[os, osproc, strutils]
import ./manifest

type
  Target* = enum
    ## Accelerator backend. Use `ProbeOrder` for auto-detection priority.
    tCuda13  ## CUDA >= 13.0, cuDNN >= 9.12.
    tCuda12  ## CUDA >= 12.1, cuDNN >= 9.8.
    tRocm    ## AMD ROCm via the ROCm PJRT plugin.
    tMetal   ## Apple Metal via the jax-metal PJRT plugin.
    tTpu     ## Google TPU (via libtpu).
    tCpu     ## Host CPU fallback.

  Arch* = enum
    aX86_64
    aAarch64

  HostOs* = enum
    osLinux
    osDarwin
    osWindows

  HostAbi* = enum
    abiGnu
    abiNone

  HostTriplet* = object
    ## Detected or overridden (arch, os, abi) triple.
    arch*: Arch
    os*: HostOs
    abi*: HostAbi

  TargetError* = object of CatchableError
    ## Raised on invalid target strings, unsupported platforms, or probe
    ## failures.

const
  TargetEnvVar* = "REW_TARGET"
  TargetPlatformEnvVar* = "REW_TARGET_PLATFORM"

  ProbeOrder* = [tCuda13, tCuda12, tRocm, tCpu]
    ## Auto-detectable target priority when `REW_TARGET` is not set.

func targetName*(t: Target): string =
  ## Canonical short name used in env vars, CLI, and manifest keys.
  case t
  of tCpu: "cpu"
  of tCuda12: "cuda12"
  of tCuda13: "cuda13"
  of tRocm: "rocm"
  of tMetal: "metal"
  of tTpu: "tpu"

func `$`*(t: Target): string = targetName(t)

func parseTarget*(s: string): Target =
  ## Parses a target short name. Raises `TargetError` on unknown values.
  case s.strip().toLowerAscii()
  of "cpu": tCpu
  of "cuda12": tCuda12
  of "cuda13": tCuda13
  of "rocm": tRocm
  of "metal": tMetal
  of "tpu": tTpu
  else:
    raise newException(TargetError,
      "unknown target '" & s &
        "'. Valid: cpu, cuda12, cuda13, rocm, metal, tpu")

func archString*(a: Arch): string =
  case a
  of aX86_64: "x86_64"
  of aAarch64: "aarch64"

func osString*(o: HostOs): string =
  case o
  of osLinux: "linux"
  of osDarwin: "darwin"
  of osWindows: "windows"

func abiString*(a: HostAbi): string =
  case a
  of abiGnu: "gnu"
  of abiNone: ""

func `$`*(t: HostTriplet): string =
  let a = abiString(t.abi)
  if a.len > 0:
    archString(t.arch) & "-" & osString(t.os) & "-" & a
  else:
    archString(t.arch) & "-" & osString(t.os)

proc detectHostTriplet*(): HostTriplet =
  ## Detects the current host architecture and OS at runtime. Can be
  ## overridden via `REW_TARGET_PLATFORM`.
  let envOverride = getEnv(TargetPlatformEnvVar)
  if envOverride.len > 0:
    let parts = envOverride.strip().split('-')
    var arch: Arch
    var hostOs: HostOs
    var abi = abiNone
    case parts[0]
    of "x86_64", "amd64": arch = aX86_64
    of "aarch64", "arm64": arch = aAarch64
    else:
      raise newException(TargetError,
        "unknown arch in " & TargetPlatformEnvVar & ": " & parts[0])
    if parts.len < 2:
      raise newException(TargetError,
        TargetPlatformEnvVar & " must be ARCH-OS or ARCH-OS-ABI, got: " &
        envOverride)
    case parts[1]
    of "linux": hostOs = osLinux
    of "darwin", "macos": hostOs = osDarwin
    of "windows", "win": hostOs = osWindows
    else:
      raise newException(TargetError,
        "unknown OS in " & TargetPlatformEnvVar & ": " & parts[1])
    if parts.len >= 3:
      case parts[2]
      of "gnu": abi = abiGnu
      else: abi = abiNone
    elif hostOs == osLinux:
      abi = abiGnu
    return HostTriplet(arch: arch, os: hostOs, abi: abi)

  # Auto-detect from compile-time platform and runtime arch
  when defined(linux):
    let hostOs = osLinux
    let abi = abiGnu
  elif defined(macosx):
    let hostOs = osDarwin
    let abi = abiNone
  elif defined(windows):
    let hostOs = osWindows
    let abi = abiNone
  else:
    let hostOs = osLinux
    let abi = abiGnu

  when defined(amd64) or defined(x86_64):
    let arch = aX86_64
  elif defined(arm64) or defined(aarch64):
    let arch = aAarch64
  else:
    let arch = aX86_64

  HostTriplet(arch: arch, os: hostOs, abi: abi)

proc slotKeyFor*(triplet: HostTriplet; target: Target): string =
  ## Builds the manifest slot key for a (triplet, target) pair.
  slotKey(archString(triplet.arch), osString(triplet.os),
          abiString(triplet.abi), targetName(target))

proc pluginFileName*(target: Target): string =
  ## Returns the canonical PJRT plugin filename for `target` on the
  ## current platform.
  let ext =
    when defined(windows): ".dll"
    elif defined(macosx): ".dylib"
    else: ".so"
  "pjrt_c_api_" & targetName(target) & "_plugin" & ext

# ----- Lazy probe: nvcc / rocminfo -----------------------------------------

proc probeNvcc(): Target =
  ## Probes `nvcc --version` and returns `tCuda13` or `tCuda12` if a
  ## matching CUDA installation is found. Raises `TargetError` on failure.
  let nvcc = findExe("nvcc")
  if nvcc.len == 0:
    raise newException(TargetError, "nvcc not found on PATH")
  let (output, exitCode) = execCmdEx(nvcc & " --version")
  if exitCode != 0:
    raise newException(TargetError, "nvcc --version failed (exit " &
      $exitCode & ")")
  if "release 13." in output: return tCuda13
  if "release 12." in output: return tCuda12
  raise newException(TargetError,
    "nvcc found but CUDA version not recognized: " & output.strip())

proc probeRocm(): bool =
  ## Returns true if `rocminfo` or `hipcc` is on PATH.
  findExe("rocminfo").len > 0 or findExe("hipcc").len > 0

proc probeTarget(): Target =
  ## Walks `ProbeOrder` and returns the first available target. Falls back
  ## to `tCpu` unconditionally (CPU plugin may still be absent, but that
  ## error surfaces at load time, not probe time).
  try:
    let cuda = probeNvcc()
    return cuda
  except TargetError:
    discard
  if probeRocm():
    return tRocm
  tCpu

var cachedDefaultTarget: Target
var defaultTargetResolved: bool

proc defaultTarget*(): Target =
  ## Returns the default target for this process. Resolution order:
  ## `REW_TARGET` env var -> lazy probe (cuda13 -> cuda12 -> rocm -> cpu).
  ## Cached after first call.
  if defaultTargetResolved:
    return cachedDefaultTarget
  let env = getEnv(TargetEnvVar)
  if env.len > 0:
    cachedDefaultTarget = parseTarget(env)
  else:
    cachedDefaultTarget = probeTarget()
  defaultTargetResolved = true
  cachedDefaultTarget

proc setDefaultTarget*(t: Target) =
  ## Explicitly sets the default target. Overrides any previously cached
  ## value or env-var-derived target.
  cachedDefaultTarget = t
  defaultTargetResolved = true

proc resetDefaultTarget*() =
  ## Clears the cached default target so the next `defaultTarget()` call
  ## re-probes. Used by tests.
  defaultTargetResolved = false
