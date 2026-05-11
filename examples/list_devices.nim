## Lists PJRT devices visible to `rew` for one or more targets.
##
## Usage:
##   nim c -r examples/list_devices.nim          # tries the default targets
##   nim c -r examples/list_devices.nim cpu cuda12
##
## Each target is probed independently — a missing plugin reports a
## one-line diagnostic and execution moves on to the next target.

import std/[os, strformat]
import rew
import rew/pjrt/loader
import rew/pjrt/capi
import rew/pjrt/client

const DefaultTargets = [tCpu, tCuda12]

proc targetsFromArgs(): seq[Target] =
  result = @[]
  for i in 1 .. paramCount():
    try:
      result.add parseTarget(paramStr(i))
    except TargetError as e:
      echo "  warning: ", e.msg
  if result.len == 0:
    for t in DefaultTargets: result.add t

proc listOne(t: Target) =
  echo "── target: ", t
  let api =
    try: loadPlugin(t)
    except PjrtError as e:
      echo "  (skip) ", e.msg
      return
  let (major, minor) = apiVersion(api)
  echo &"  PJRT API version: {major}.{minor}"
  let c = newPjrtClient(api)
  let devices = c.addressableDevices()
  echo &"  addressable devices: {devices.len}"
  for i, d in devices:
    let kind = c.deviceKind(d)
    let id = c.deviceId(d)
    echo &"    [{i}] kind={kind} id={id}"

proc main() =
  let targets = targetsFromArgs()
  echo "rew v", RewVersion, " — PJRT device enumeration"
  echo "  default target: ", defaultTarget()
  for t in targets:
    listOne(t)

when isMainModule:
  main()
