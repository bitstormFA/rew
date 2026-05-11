## Architectural lint: OpenXLA pinned coverage.
##
## This lint makes the "full OpenXLA coverage" roadmap executable. It checks
## that `openxla_manifest.json` is internally consistent and that the
## StableHLO ops recorded as supported match the textual emitter's op table.

import std/[algorithm, json, os, re, sequtils, sets, strutils]

const
  ManifestFile = "openxla_manifest.json"
  TextEmitterFile = "src/rew/stablehlo/text.nim"

let opMappingRe = re"""of\s+ok[A-Za-z0-9_]+:\s*"stablehlo\.([a-zA-Z0-9_-]+)""""

proc collectStringSet(node: JsonNode): HashSet[string] =
  result = initHashSet[string]()
  for item in node.items:
    result.incl item.getStr()

proc collectEmitterOps(path: string): HashSet[string] =
  result = initHashSet[string]()
  let src = readFile(path)
  for m in src.findAll(opMappingRe):
    var captures: array[1, string]
    if m.match(opMappingRe, captures):
      result.incl captures[0]

proc showSet(label: string; s: HashSet[string]) =
  if s.len == 0: return
  echo "  " & label & ": " & toSeq(s.items).sorted().join(", ")

if not fileExists(ManifestFile):
  quit "check_openxla_coverage: missing " & ManifestFile, 1
if not fileExists(TextEmitterFile):
  quit "check_openxla_coverage: missing " & TextEmitterFile, 1

let manifest = parseJson(readFile(ManifestFile))
let stablehlo = manifest["stablehlo"]
let pinnedOps = collectStringSet(stablehlo["ops"])
let supportedOps = collectStringSet(stablehlo["supported_ops"])
let emittedOps = collectEmitterOps(TextEmitterFile)
let declaredCount = stablehlo["op_count"].getInt()

var failures: seq[string] = @[]
if pinnedOps.len != declaredCount:
  failures.add "stablehlo.op_count=" & $declaredCount &
    " but ops list has " & $pinnedOps.len

let supportedUnknown = supportedOps - pinnedOps
if supportedUnknown.len > 0:
  failures.add "supported_ops contains names not in pinned ops"
let supportedNotEmitted = supportedOps - emittedOps
if supportedNotEmitted.len > 0:
  failures.add "supported_ops contains names the emitter does not expose"
let emittedNotSupported = emittedOps - supportedOps
if emittedNotSupported.len > 0:
  failures.add "emitter exposes StableHLO names missing from supported_ops"

if failures.len > 0:
  echo "check_openxla_coverage: FAIL"
  for msg in failures:
    echo "  " & msg
  showSet("supported_unknown", supportedUnknown)
  showSet("supported_not_emitted", supportedNotEmitted)
  showSet("emitted_not_supported", emittedNotSupported)
  quit 1

let missing = pinnedOps - supportedOps
echo "check_openxla_coverage: OK (" & $supportedOps.len & "/" &
  $pinnedOps.len & " StableHLO op(s), " & $missing.len & " pending)"
