## Device — value-typed identifier for a (target, ordinal) pair.
##
## A `Device` carries a `Target` (closed enum: cpu, cuda12, cuda13, rocm,
## metal, tpu) and an integer ordinal selecting the physical device within that
## target's addressable set.
##
## **Default-device resolution** runs at first observation, using the lazy
## probe in `binaries/target` (cuda13 -> cuda12 -> rocm -> cpu via
## `nvcc --version` / `rocminfo`). The environment variable `REW_TARGET`
## overrides auto-detection.

import std/strutils
import ./binaries/target

export target.Target, target.targetName, target.parseTarget
export target.defaultTarget, target.setDefaultTarget, target.resetDefaultTarget

type
  Device* = object
    ## (target, ordinal) pair. Two devices are equal iff both fields match.
    target*: Target
    ordinal*: int

  DeviceError* = object of CatchableError
    ## Raised by cross-device ops and by failed default-device resolution.

func cpu*(ordinal: int = 0): Device =
  ## Constructs a CPU device.
  Device(target: tCpu, ordinal: ordinal)

func cuda12*(ordinal: int = 0): Device =
  ## Constructs a CUDA 12 device.
  Device(target: tCuda12, ordinal: ordinal)

func cuda13*(ordinal: int = 0): Device =
  ## Constructs a CUDA 13 device.
  Device(target: tCuda13, ordinal: ordinal)

func rocm*(ordinal: int = 0): Device =
  ## Constructs a ROCm device.
  Device(target: tRocm, ordinal: ordinal)

func metal*(ordinal: int = 0): Device =
  ## Constructs a Metal device.
  Device(target: tMetal, ordinal: ordinal)

func tpu*(ordinal: int = 0): Device =
  ## Constructs a TPU device.
  Device(target: tTpu, ordinal: ordinal)

func initDevice*(target: Target; ordinal: int = 0): Device =
  ## Generic constructor from a `Target` enum value.
  Device(target: target, ordinal: ordinal)

func `==`*(a, b: Device): bool =
  ## Structural equality on (target, ordinal).
  a.target == b.target and a.ordinal == b.ordinal

func `$`*(d: Device): string =
  ## `"cpu:0"`-style string. Matches the syntax accepted by `parseDevice`.
  targetName(d.target) & ":" & $d.ordinal

proc parseDevice*(s: string): Device =
  ## Parses a `"target[:ordinal]"` string. Raises `DeviceError` on malformed
  ## input.
  let trimmed = s.strip()
  if trimmed.len == 0:
    raise newException(DeviceError, "empty device string")
  let parts = trimmed.split(':')
  if parts.len == 1:
    try:
      return initDevice(parseTarget(parts[0]), 0)
    except TargetError as e:
      raise newException(DeviceError, e.msg)
  if parts.len == 2:
    try:
      let t = parseTarget(parts[0])
      let ord = parseInt(parts[1])
      return initDevice(t, ord)
    except TargetError as e:
      raise newException(DeviceError, e.msg)
    except ValueError:
      raise newException(DeviceError,
        "invalid device ordinal in '" & s & "'")
  raise newException(DeviceError,
    "device string must be 'target' or 'target:ordinal', got '" & s & "'")

var defaultDeviceCache {.threadvar.}: Device
var defaultDeviceResolved {.threadvar.}: bool

proc setDefaultDevice*(d: Device) =
  ## Overrides the default device for subsequent tensor creation on this
  ## thread. Does not eagerly load the plugin.
  defaultDeviceCache = d
  defaultDeviceResolved = true

proc defaultDevice*(): Device =
  ## Returns the current default device, resolving it if necessary.
  ##
  ## Resolution order: previously-set value -> `REW_TARGET` env var ->
  ## lazy probe (cuda13 -> cuda12 -> rocm -> cpu). Caches the result on
  ## the current thread.
  if defaultDeviceResolved:
    return defaultDeviceCache
  let d = initDevice(defaultTarget(), 0)
  setDefaultDevice(d)
  d

proc requireSameDevice*(a, b: Device; opName: string) =
  ## Enforces the "no implicit cross-device transfer" rule. Raises
  ## `DeviceError` with the offending op name and both device strings.
  if a != b:
    raise newException(DeviceError,
      opName & ": cross-device op forbidden (got " & $a & " vs " & $b &
      "). Use `.to(device)` to move tensors explicitly.")
