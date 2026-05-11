## Typed view of `openxla_manifest.json`.
##
## The manifest pins the upstream OpenXLA ecosystem revisions and records the
## StableHLO op inventory used by rew's coverage lint. Tool binaries are
## optional: the core library remains pure Nim/PJRT.

import std/[json, tables]

type
  OpenXlaApiVersion* = object
    ## Pinned PJRT API version for the selected XLA revision.
    major*: int
    minor*: int

  OpenXlaComponent* = object
    ## One upstream component pin.
    repo*: string
    rev*: string
    pjrtApi*: OpenXlaApiVersion

  OpenXlaTool* = object
    ## Optional external tool metadata.
    binary*: string
    required*: bool

  StableHloInventory* = object
    ## Pinned StableHLO operation inventory and current rew coverage.
    opsSource*: string
    opCount*: int
    ops*: seq[string]
    supportedOps*: seq[string]

  OpenXlaManifest* = object
    ## Top-level OpenXLA ecosystem manifest.
    schemaVersion*: int
    generatedAt*: string
    components*: Table[string, OpenXlaComponent]
    stablehlo*: StableHloInventory
    tools*: Table[string, OpenXlaTool]

const ManifestJson = staticRead("../../../openxla_manifest.json")

proc parseApiVersion(j: JsonNode): OpenXlaApiVersion =
  if j.isNil:
    return OpenXlaApiVersion()
  OpenXlaApiVersion(
    major: j{"major"}.getInt(),
    minor: j{"minor"}.getInt())

proc stringSeq(j: JsonNode): seq[string] =
  result = @[]
  for item in j.items:
    result.add item.getStr()

proc parseOpenXlaManifest*(raw: string): OpenXlaManifest =
  ## Parses an OpenXLA manifest JSON document.
  let j = parseJson(raw)
  result.schemaVersion = j["schema_version"].getInt()
  result.generatedAt = j["generated_at"].getStr()
  result.components = initTable[string, OpenXlaComponent]()
  for name, node in j["components"].pairs:
    result.components[name] = OpenXlaComponent(
      repo: node["repo"].getStr(),
      rev: node["rev"].getStr(),
      pjrtApi: parseApiVersion(node{"pjrt_api"}))
  let stablehlo = j["stablehlo"]
  result.stablehlo = StableHloInventory(
    opsSource: stablehlo["ops_source"].getStr(),
    opCount: stablehlo["op_count"].getInt(),
    ops: stringSeq(stablehlo["ops"]),
    supportedOps: stringSeq(stablehlo["supported_ops"]))
  result.tools = initTable[string, OpenXlaTool]()
  for name, node in j["tools"].pairs:
    result.tools[name] = OpenXlaTool(
      binary: node["binary"].getStr(),
      required: node["required"].getBool())

var manifestCache: OpenXlaManifest
var manifestLoaded: bool

proc openXlaManifest*(): OpenXlaManifest =
  ## Returns the parsed OpenXLA manifest singleton.
  if not manifestLoaded:
    manifestCache = parseOpenXlaManifest(ManifestJson)
    manifestLoaded = true
  manifestCache

proc missingStableHloOps*(m: OpenXlaManifest): seq[string] =
  ## StableHLO ops pinned upstream but not yet implemented by rew.
  for op in m.stablehlo.ops:
    if op notin m.stablehlo.supportedOps:
      result.add op

proc stableHloCoverage*(m: OpenXlaManifest): tuple[supported, total: int] =
  ## Returns `(supported, total)` for the pinned StableHLO inventory.
  (m.stablehlo.supportedOps.len, m.stablehlo.ops.len)
