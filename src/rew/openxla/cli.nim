## Small command-line dispatcher for optional OpenXLA tools.

import std/strutils
import ./tools

type
  OpenXlaCliError* = object of CatchableError
    ## Raised for malformed OpenXLA CLI invocations.

func normalizedToolName(name: string): string =
  name.strip().toLowerAscii().replace("-", "_")

func parseOpenXlaToolName*(name: string): OpenXlaToolKind =
  ## Parses command-line spellings such as `hlo-opt` or `hlo_opt`.
  case normalizedToolName(name)
  of "hlo_opt": otHloOpt
  of "isolate_hlo": otIsolateHlo
  of "multihost_hlo_runner": otMultihostHloRunner
  of "mpmd_opt": otMpmdOpt
  of "ptx_opt": otPtxOpt
  of "run_hlo_module": otRunHloModule
  of "sdy_opt": otSdyOpt
  of "sdy_translate": otSdyTranslate
  of "xprof": otXprof
  of "tokamax": otTokamax
  else:
    raise newException(OpenXlaCliError,
      "unknown OpenXLA tool '" & name & "'")

func openXlaToolChoices*(): seq[string] =
  ## Canonical tool names accepted by the CLI.
  @[
    "hlo-opt",
    "isolate_hlo",
    "multihost_hlo_runner",
    "mpmd_opt",
    "ptx-opt",
    "run_hlo_module",
    "sdy_opt",
    "sdy_translate",
    "xprof",
    "tokamax",
  ]

func openXlaCliUsage*(): string =
  ## Usage text for `tools/openxla_tool.nim` and installed wrappers.
  "Usage:\n" &
    "  rew-openxla list\n" &
    "  rew-openxla <tool> [args...]\n\n" &
    "Tools: " & openXlaToolChoices().join(", ") & "\n" &
    "Tool paths are resolved from REW_* environment variables first."

proc runOpenXlaCli*(args: openArray[string]): int =
  ## Runs the OpenXLA tool CLI. Returns a process exit code.
  var argv: seq[string] = @[]
  for arg in args:
    if argv.len == 0 and arg == "--":
      discard
    else:
      argv.add arg

  if argv.len == 0 or argv[0] in ["-h", "--help", "help"]:
    echo openXlaCliUsage()
    return 0

  if argv[0] == "list":
    for name in openXlaToolChoices():
      let kind = parseOpenXlaToolName(name)
      let path = toolPath(kind)
      let status =
        if path.len == 0: "missing"
        elif toolAvailable(kind): path
        else: path & " (missing)"
      echo name & "\t" & toolEnvVar(kind) & "\t" & status
    return 0

  let kind =
    try: parseOpenXlaToolName(argv[0])
    except OpenXlaCliError as e:
      stderr.writeLine(e.msg)
      stderr.writeLine(openXlaCliUsage())
      return 2

  var forwarded: seq[string] = @[]
  for i in 1 ..< argv.len:
    forwarded.add argv[i]

  try:
    let res = runTool(kind, forwarded)
    stdout.write(res.output)
    result = res.exitCode
  except OpenXlaToolError as e:
    stderr.writeLine(e.msg)
    result = 2
