## rewrun — local distributed worker launcher.
##
## This binary intentionally uses only Nim stdlib. It starts one worker process
## per local rank, exports `REW_DIST_*`, and leaves remote orchestration to SSH
## wrappers or manually started workers.

import std/[os, osproc, strutils, times]

proc usage() =
  echo "Usage: rewrun --nproc N [--nnodes N] [--node-rank R] [--host H] [--port P] -- command [args...]"

proc parseIntArg(args: seq[string]; i: var int; name: string): int =
  if i + 1 >= args.len:
    quit "rewrun: missing value for " & name, 2
  inc i
  try:
    parseInt(args[i])
  except ValueError:
    quit "rewrun: invalid integer for " & name & ": " & args[i], 2

proc parseStrArg(args: seq[string]; i: var int; name: string): string =
  if i + 1 >= args.len:
    quit "rewrun: missing value for " & name, 2
  inc i
  args[i]

when isMainModule:
  let args = commandLineParams()
  var
    nproc = 1
    nnodes = 1
    nodeRank = 0
    host = "127.0.0.1"
    port = 29500
    runId = "rew-" & $epochTime().int
    commandStart = -1
  var i = 0
  while i < args.len:
    case args[i]
    of "--":
      commandStart = i + 1
      break
    of "--help", "-h":
      usage()
      quit 0
    of "--nproc", "--nproc-per-node":
      nproc = parseIntArg(args, i, args[i])
    of "--nnodes":
      nnodes = parseIntArg(args, i, args[i])
    of "--node-rank":
      nodeRank = parseIntArg(args, i, args[i])
    of "--host":
      host = parseStrArg(args, i, args[i])
    of "--port":
      port = parseIntArg(args, i, args[i])
    of "--run-id":
      runId = parseStrArg(args, i, args[i])
    else:
      if args[i].startsWith("-"):
        quit "rewrun: unknown option " & args[i], 2
      commandStart = i
      break
    inc i

  if commandStart < 0 or commandStart >= args.len:
    usage()
    quit 2
  if nproc <= 0 or nnodes <= 0:
    quit "rewrun: --nproc and --nnodes must be positive", 2
  if nodeRank < 0 or nodeRank >= nnodes:
    quit "rewrun: --node-rank must be in [0, nnodes)", 2

  let cmd = args[commandStart]
  let cmdArgs =
    if commandStart + 1 < args.len: args[(commandStart + 1) .. ^1]
    else: @[]
  let worldSize = nproc * nnodes
  let coordinator = host & ":" & $port
  var procs: seq[Process] = @[]

  for localRank in 0 ..< nproc:
    let rank = nodeRank * nproc + localRank
    putEnv("REW_DIST_RANK", $rank)
    putEnv("REW_DIST_WORLD_SIZE", $worldSize)
    putEnv("REW_DIST_LOCAL_RANK", $localRank)
    putEnv("REW_DIST_LOCAL_SIZE", $nproc)
    putEnv("REW_DIST_PROCESS_INDEX", $rank)
    putEnv("REW_DIST_PROCESS_COUNT", $worldSize)
    putEnv("REW_DIST_HOST", host)
    putEnv("REW_DIST_PORT", $port)
    putEnv("REW_DIST_COORDINATOR", coordinator)
    putEnv("REW_DIST_RUN_ID", runId)
    procs.add startProcess(cmd, args = cmdArgs,
      options = {poUsePath, poParentStreams})

  var exitCode = 0
  for p in procs.mitems:
    let code = waitForExit(p)
    if code != 0 and exitCode == 0:
      exitCode = code
    close(p)
  quit exitCode
