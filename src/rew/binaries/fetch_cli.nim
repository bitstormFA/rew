## Command-line frontend for downloading PJRT plugins into the rew cache.

import std/[os, strutils, tables]
import ./target
import ./manifest
import ./fetcher
import ./cache

const TargetHelp = "cpu | cuda12 | cuda13 | rocm | metal | tpu"

proc printUsage(programName: string) =
  echo "Usage: ", programName, " <target>"
  echo "  target: ", TargetHelp
  echo ""
  echo "Downloads a PJRT plugin shared library into the rew cache."
  echo "Override the destination by setting REW_CACHE_DIR."
  echo "Override the archive URL with REW_ARCHIVE_URL."
  echo "Use a local archive with REW_ARCHIVE_PATH."

proc runFetchCli*(programName = "rew_fetch") =
  ## Runs the plugin-fetch CLI. Intended for both the installed `rew_fetch`
  ## executable and the source-tree `nimble fetch` task.
  if paramCount() < 1:
    printUsage(programName)
    quit 1

  let rawTarget = paramStr(1)
  if rawTarget in ["-h", "--help", "help"]:
    printUsage(programName)
    quit 0

  let t = try:
    parseTarget(rawTarget)
  except ValueError as e:
    echo "ERROR: ", e.msg
    echo ""
    printUsage(programName)
    quit 1

  let triplet = detectHostTriplet()
  let key = slotKeyFor(triplet, t)
  echo "Fetching '", targetName(t), "' plugin for ", $triplet
  echo "  slot key: ", key

  let m = manifest()
  if key notin m.slots:
    echo "  ERROR: no manifest slot for '", key, "'"
    echo "  Available: ", m.allSlotKeys().join(", ")
    quit 1

  try:
    fetchSlot(key, triplet, t)
  except CatchableError as e:
    echo "  ERROR: ", e.msg
    quit 1

  echo "  plugin ready at: ", pluginPathForTarget(t)
