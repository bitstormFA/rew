## Cache — on-disk layout for downloaded/built PJRT plugin archives.
##
## Layout mirrors elixir-nx/xla:
##   `<cache_dir>/<rew_version>/{download,external,build}/<filename>`
##
## The version-qualified subdirectory ensures different rew releases never
## share cached plugins that might be ABI-incompatible.

import std/[os, strutils, tables]
import ./manifest
import ./target

const
  CacheDirEnvVar* = "REW_CACHE_DIR"

proc baseCacheDir*(): string =
  ## Returns the root cache directory. Respects `REW_CACHE_DIR`; falls
  ## back to the OS-standard user cache location.
  let env = getEnv(CacheDirEnvVar)
  if env.len > 0:
    return env.expandTilde()
  when defined(macosx):
    getHomeDir() / "Library" / "Caches" / "rew"
  elif defined(windows):
    let appData = getEnv("LOCALAPPDATA", getHomeDir() / "AppData" / "Local")
    appData / "rew" / "cache"
  else:
    let xdg = getEnv("XDG_CACHE_HOME")
    if xdg.len > 0: xdg / "rew"
    else: getHomeDir() / ".cache" / "rew"

proc versionedCacheDir*(): string =
  ## Returns the version-qualified cache subdirectory.
  baseCacheDir() / manifest().rewVersion

proc downloadDir*(): string =
  ## `<cache>/<ver>/download/` — for archives fetched from manifest URLs.
  versionedCacheDir() / "download"

proc externalDir*(): string =
  ## `<cache>/<ver>/external/` — for archives from `REW_ARCHIVE_URL`.
  versionedCacheDir() / "external"

proc buildDir*(): string =
  ## `<cache>/<ver>/build/` — for locally built plugins (`REW_BUILD`).
  versionedCacheDir() / "build"

proc pluginsDir*(): string =
  ## `<cache>/<ver>/plugins/` — extracted plugin shared libraries.
  versionedCacheDir() / "plugins"

proc executablesDir*(): string =
  ## `<cache>/<ver>/executables/` — serialized PJRT executables.
  versionedCacheDir() / "executables"

proc archivePathForSlot*(key: string): string =
  ## Returns the expected archive path for a manifest slot download.
  let url = manifest().slots[key].url
  let filename = url.split('/')[^1]
  downloadDir() / filename

proc pluginPathForTarget*(t: Target): string =
  ## Returns the expected final path of the extracted plugin `.so`/`.dylib`.
  pluginsDir() / pluginFileName(t)

proc ensureDirs*() =
  ## Creates all cache subdirectories if they don't exist.
  createDir(downloadDir())
  createDir(externalDir())
  createDir(buildDir())
  createDir(pluginsDir())
  createDir(executablesDir())
