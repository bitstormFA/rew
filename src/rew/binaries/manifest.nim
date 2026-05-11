## Typed view of the `pjrt_manifest.json` file committed at the repo root.
##
## The manifest is embedded at compile time via `staticRead` and parsed into
## Nim objects at module init. It pins exact upstream URLs and SHA-256
## checksums for every supported (arch x os x target) slot, enabling
## reproducible binary resolution and integrity verification.

import std/[json, tables, strutils, sequtils]

type
  SlotSource* = enum
    ## How the plugin binary is obtained for a given slot.
    ssVendor       ## Downloaded from a third-party vendor release (`.tar.gz`).
    ssVendorWheel  ## Extracted from a PyPI wheel (`.whl` / `.zip`).
    ssRew          ## Built and published by rew CI on GitHub Releases.

  PjrtApiVersionPin* = object
    ## Expected PJRT API version for compatibility checking.
    major*: int
    minor*: int

  SlotInfo* = object
    ## Per-slot metadata: where to fetch and how to verify the plugin.
    source*: SlotSource
    url*: string
    archiveMember*: string
    sha256*: string

  Manifest* = object
    ## Top-level manifest model.
    rewVersion*: string
    openxlaXlaRev*: string
    expectedApiVersion*: PjrtApiVersionPin
    slots*: Table[string, SlotInfo]

const ManifestJson = staticRead("../../../pjrt_manifest.json")
  ## Raw JSON embedded at compile time from the repo root.

proc parseSlotSource(s: string): SlotSource =
  case s
  of "vendor": ssVendor
  of "vendor_wheel": ssVendorWheel
  of "rew": ssRew
  else: raise newException(ValueError, "unknown slot source: " & s)

proc parseManifest*(raw: string): Manifest =
  ## Parses the manifest JSON into the typed model.
  let j = parseJson(raw)
  result.rewVersion = j["rew_version"].getStr()
  result.openxlaXlaRev = j["openxla_xla_rev"].getStr()
  let ver = j["expected_pjrt_api_version"]
  result.expectedApiVersion = PjrtApiVersionPin(
    major: ver["major"].getInt(),
    minor: ver["minor"].getInt())
  result.slots = initTable[string, SlotInfo]()
  for key, val in j["slots"].pairs:
    result.slots[key] = SlotInfo(
      source: parseSlotSource(val["source"].getStr()),
      url: val["url"].getStr(),
      archiveMember: val["archive_member"].getStr(),
      sha256: val["sha256"].getStr())

var manifestCache: Manifest
var manifestLoaded: bool

proc manifest*(): Manifest =
  ## Returns the parsed manifest singleton. Parsed once on first call.
  if not manifestLoaded:
    manifestCache = parseManifest(ManifestJson)
    manifestLoaded = true
  manifestCache

proc slotKey*(arch, os, abi, target: string): string =
  ## Builds the canonical slot key used to index into the manifest.
  ## Example: `slotKey("x86_64", "linux", "gnu", "cpu")` -> `"x86_64-linux-gnu-cpu"`.
  if abi.len > 0:
    arch & "-" & os & "-" & abi & "-" & target
  else:
    arch & "-" & os & "-" & target

proc lookupSlot*(m: Manifest; key: string): SlotInfo =
  ## Returns the `SlotInfo` for `key`, raising `KeyError` if the slot is
  ## not in the manifest.
  if key notin m.slots:
    raise newException(KeyError,
      "no manifest entry for slot '" & key & "'. " &
      "Available: " & toSeq(m.slots.keys).join(", "))
  m.slots[key]

proc allSlotKeys*(m: Manifest): seq[string] =
  ## Returns all slot keys in the manifest.
  result = @[]
  for k in m.slots.keys:
    result.add k
