## Optional OpenXLA tool wrappers.
##
## These helpers deliberately shell out to pinned external tools instead of
## linking MLIR/XLA C++ into the core rew package.

import std/[os, osproc, strutils]

type
  OpenXlaToolKind* = enum
    otHloOpt
    otIsolateHlo
    otMultihostHloRunner
    otMpmdOpt
    otPtxOpt
    otRunHloModule
    otSdyOpt
    otSdyTranslate
    otXprof
    otTokamax

  OpenXlaToolError* = object of CatchableError
    ## Raised when an optional OpenXLA tool is missing or fails.

  ToolResult* = object
    ## Captured result of an OpenXLA tool invocation.
    command*: string
    output*: string
    exitCode*: int

  XprofProfileSpec* = object
    ## Metadata for an XProf command such as `capture` or `view`.
    action*: string
    args*: seq[string]
    metadata*: seq[(string, string)]

func toolName*(kind: OpenXlaToolKind): string =
  ## Default executable name for an OpenXLA tool.
  case kind
  of otHloOpt: "hlo-opt"
  of otIsolateHlo: "isolate_hlo"
  of otMultihostHloRunner: "multihost_hlo_runner"
  of otMpmdOpt: "mpmd_opt"
  of otPtxOpt: "ptx-opt"
  of otRunHloModule: "run_hlo_module"
  of otSdyOpt: "sdy_opt"
  of otSdyTranslate: "sdy_translate"
  of otXprof: "xprof"
  of otTokamax: "tokamax"

func toolEnvVar*(kind: OpenXlaToolKind): string =
  ## Environment override for a tool path.
  case kind
  of otHloOpt: "REW_HLO_OPT"
  of otIsolateHlo: "REW_ISOLATE_HLO"
  of otMultihostHloRunner: "REW_MULTIHOST_HLO_RUNNER"
  of otMpmdOpt: "REW_MPMD_OPT"
  of otPtxOpt: "REW_PTX_OPT"
  of otRunHloModule: "REW_RUN_HLO_MODULE"
  of otSdyOpt: "REW_SDY_OPT"
  of otSdyTranslate: "REW_SDY_TRANSLATE"
  of otXprof: "REW_XPROF"
  of otTokamax: "REW_TOKAMAX"

proc toolPath*(kind: OpenXlaToolKind): string =
  ## Resolves a tool path using the `REW_*` override first, then `PATH`.
  let env = getEnv(toolEnvVar(kind))
  if env.len > 0:
    return env
  findExe(toolName(kind))

func isExplicitToolPath(path: string): bool =
  for ch in path:
    if ch == '/' or ch == '\\':
      return true
  false

proc executableExists(path: string): bool =
  if path.len == 0:
    return false
  if fileExists(path):
    return true
  if path.isExplicitToolPath:
    return false
  findExe(path).len > 0

proc toolAvailable*(kind: OpenXlaToolKind): bool =
  ## True when the tool is discoverable or explicitly configured.
  let path = toolPath(kind)
  path.executableExists

proc requireTool*(kind: OpenXlaToolKind): string =
  ## Returns a resolved tool path or raises a clear optional-tool error.
  var resolved = toolPath(kind)
  if resolved.len == 0:
    raise newException(OpenXlaToolError,
      "OpenXLA tool '" & toolName(kind) & "' not found. Set " &
      toolEnvVar(kind) & " to an executable path.")
  if not resolved.executableExists:
    let msg =
      "OpenXLA tool '" & toolName(kind) & "' path does not exist: " & resolved
    resolved.setLen(0)
    raise newException(OpenXlaToolError,
      msg)
  result = resolved

proc runTool*(kind: OpenXlaToolKind; args: openArray[string] = []):
    ToolResult =
  ## Runs an optional OpenXLA tool and captures combined stdout/stderr.
  let path = requireTool(kind)
  var cmd = quoteShell(path)
  for arg in args:
    cmd.add ' '
    cmd.add quoteShell(arg)
  let (output, exitCode) = execCmdEx(cmd)
  result = ToolResult(command: cmd, output: output, exitCode: exitCode)
  if exitCode != 0:
    raise newException(OpenXlaToolError,
      "OpenXLA tool failed (" & $exitCode & "): " & cmd & "\n" &
      output.strip())

proc runHloModule*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for `run_hlo_module`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otRunHloModule, allArgs)

proc hloOpt*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for `hlo-opt`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otHloOpt, allArgs)

proc ptxOpt*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for `ptx-opt`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otPtxOpt, allArgs)

proc isolateHlo*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for `isolate_hlo`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otIsolateHlo, allArgs)

proc multihostHloRunner*(path: string; args: openArray[string] = []):
    ToolResult =
  ## Convenience wrapper for `multihost_hlo_runner`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otMultihostHloRunner, allArgs)

proc sdyOpt*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for Shardy's `sdy_opt`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otSdyOpt, allArgs)

proc sdyTranslate*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for Shardy's `sdy_translate`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otSdyTranslate, allArgs)

proc mpmdOpt*(path: string; args: openArray[string] = []): ToolResult =
  ## Convenience wrapper for Shardy's `mpmd_opt`.
  var allArgs = @[path]
  for arg in args: allArgs.add arg
  runTool(otMpmdOpt, allArgs)

proc xprof*(args: openArray[string] = []): ToolResult =
  ## Runs `xprof` with caller-supplied arguments.
  runTool(otXprof, args)

proc initXprofProfile*(action: string; args: openArray[string] = [];
    metadata: openArray[(string, string)] = []): XprofProfileSpec =
  ## Creates metadata for an XProf command without invoking the tool.
  XprofProfileSpec(action: action, args: @args, metadata: @metadata)

func xprofArgs*(spec: XprofProfileSpec): seq[string] =
  ## Converts XProf metadata into command-line arguments.
  result = @[]
  if spec.action.len > 0:
    result.add spec.action
  for arg in spec.args:
    result.add arg

proc runXprof*(spec: XprofProfileSpec): ToolResult =
  ## Runs the optional XProf tool for `spec`.
  xprof(xprofArgs(spec))

proc tokamax*(args: openArray[string] = []): ToolResult =
  ## Runs `tokamax` with caller-supplied arguments.
  runTool(otTokamax, args)
