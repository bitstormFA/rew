## Resolver — given a `Target`, returns the path to a cached PJRT plugin.
##
## This is the single entry point consumed by `pjrt/loader.nim`. It
## orchestrates: env-var overrides (`REW_PJRT_PLUGIN_PATH`, `REW_BUILD`) →
## cached plugin check → lazy download + extraction → final path.

import std/[os, strutils, tables]
import ./target
import ./cache
import ./fetcher
import ./extract
import ./manifest

const
  PluginPathEnvVar* = "REW_PJRT_PLUGIN_PATH"
  BuildEnvVar* = "REW_BUILD"

type
  ResolveError* = object of CatchableError
    ## Raised when a plugin cannot be resolved for the requested target.

proc resolvePluginPath*(t: Target): string =
  ## Returns an absolute path to the PJRT plugin shared library for `t`.
  ## Lazy-fetches the archive from the manifest if no cached copy exists.
  ## Raises `ResolveError` on failure.

  let fname = pluginFileName(t)

  # 1. Explicit plugin path override (colon-separated directories)
  let explicitPath = getEnv(PluginPathEnvVar)
  if explicitPath.len > 0:
    for dir in explicitPath.split(PathSep):
      let d = dir.strip()
      if d.len > 0:
        let candidate = d / fname
        if fileExists(candidate):
          return candidate

  # 2. Build-from-source path
  let buildFlag = getEnv(BuildEnvVar)
  if buildFlag in ["1", "true"]:
    let built = buildDir() / fname
    if fileExists(built):
      return built
    raise newException(ResolveError,
      "REW_BUILD=true but no built plugin found at " & built &
      ". Run `bau buildPlugin " & targetName(t) & "` first.")

  # 3. Check for already-extracted plugin in cache
  let cached = pluginPathForTarget(t)
  if fileExists(cached):
    return cached

  # 4. Lazy fetch: determine the slot, download, extract
  let triplet = detectHostTriplet()
  let key = slotKeyFor(triplet, t)
  let m = manifest()
  if key notin m.slots:
    raise newException(ResolveError,
      "no manifest slot for '" & key & "' on this platform. " &
      "Available slots: " & m.allSlotKeys().join(", ") & ". " &
      "Use REW_BUILD=true to build from source or " &
      "REW_ARCHIVE_PATH to provide a prebuilt archive.")
  try:
    fetchSlot(key, triplet, t)
  except FetchError as e:
    raise newException(ResolveError,
      "failed to fetch plugin for " & targetName(t) & ": " & e.msg)
  except ExtractError as e:
    raise newException(ResolveError,
      "failed to extract plugin for " & targetName(t) & ": " & e.msg)

  if fileExists(cached):
    return cached

  raise newException(ResolveError,
    "plugin for " & targetName(t) & " not found after fetch. " &
    "Expected at: " & cached)
