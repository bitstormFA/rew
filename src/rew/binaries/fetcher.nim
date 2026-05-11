## Fetcher — HTTPS download with retries and integrity verification.
##
## Downloads plugin archives from manifest URLs (or `REW_ARCHIVE_URL` /
## `REW_ARCHIVE_PATH` overrides) into the cache, verifies SHA-256, and
## extracts the plugin shared library.

import std/[os, osproc, strformat, strutils]
when defined(ssl):
  import std/httpclient
import ./manifest
import ./target
import ./cache
import ./checksum
import ./extract

const
  ArchiveUrlEnvVar* = "REW_ARCHIVE_URL"
  ArchivePathEnvVar* = "REW_ARCHIVE_PATH"
  MaxRetries = 3
  RetryDelayMs = 2000

type
  FetchError* = object of CatchableError
    ## Raised on download failures, checksum mismatches, or extraction errors.

proc tryDownloadWithCurl(url, destPath: string): tuple[ok: bool, msg: string] =
  let curl = findExe("curl")
  if curl.len == 0:
    return (false, "curl not found on PATH")
  let args = @[curl, "-fL", "--retry", $MaxRetries, "--retry-delay",
    $(RetryDelayMs div 1000), "-o", destPath, url]
  let (output, exitCode) = execCmdEx(quoteShellCommand(args))
  if exitCode == 0:
    return (true, "")
  (false, output.strip())

proc tryDownloadWithWget(url, destPath: string): tuple[ok: bool, msg: string] =
  let wget = findExe("wget")
  if wget.len == 0:
    return (false, "wget not found on PATH")
  let args = @[wget, "--tries=" & $MaxRetries,
    "--waitretry=" & $(RetryDelayMs div 1000), "-O", destPath, url]
  let (output, exitCode) = execCmdEx(quoteShellCommand(args))
  if exitCode == 0:
    return (true, "")
  (false, output.strip())

proc downloadWithExternalTool(url, destPath: string): tuple[ok: bool, msg: string] =
  let curl = tryDownloadWithCurl(url, destPath)
  if curl.ok:
    return curl
  let wget = tryDownloadWithWget(url, destPath)
  if wget.ok:
    return wget
  (false, "curl: " & curl.msg & "; wget: " & wget.msg)

proc downloadFile(url, destPath: string) =
  ## Downloads `url` to `destPath` with retry logic.
  createDir(parentDir(destPath))
  var lastErr = ""
  when defined(ssl):
    for attempt in 1 .. MaxRetries:
      try:
        var client = newHttpClient(timeout = 60_000,
          headers = newHttpHeaders({
            "User-Agent": "rew-fetcher/" & manifest().rewVersion}))
        defer: client.close()
        client.downloadFile(url, destPath)
        return
      except CatchableError as e:
        lastErr = e.msg
        if attempt < MaxRetries:
          sleep(RetryDelayMs)
  else:
    lastErr = "Nim SSL support is not enabled"

  let external = downloadWithExternalTool(url, destPath)
  if external.ok:
    return
  if lastErr.len > 0:
    lastErr.add "; "
  lastErr.add external.msg
  raise newException(FetchError,
    &"download failed after {MaxRetries} attempts: {url} — {lastErr}")

proc verifyChecksum(path, expected: string) =
  ## Verifies the SHA-256 of `path` against `expected`. Raises `FetchError`
  ## on mismatch. Skips verification when expected is all zeros (placeholder).
  if expected.allCharsInSet({'0'}):
    return
  let actual = sha256File(path)
  if actual != expected:
    raise newException(FetchError,
      &"checksum mismatch for {path}: expected {expected}, got {actual}")

func archiveSuffix(url: string): string =
  ## Returns the extractor-relevant suffix from a URL or path.
  let clean = url.split({'?', '#'})[0].toLowerAscii()
  if clean.endsWith(".tar.gz"):
    ".tar.gz"
  elif clean.endsWith(".tgz"):
    ".tgz"
  elif clean.endsWith(".whl"):
    ".whl"
  elif clean.endsWith(".zip"):
    ".zip"
  else:
    ".tar.gz"

proc fetchSlot*(key: string; triplet: HostTriplet; t: Target) =
  ## Downloads the archive for manifest slot `key`, verifies its SHA-256,
  ## and extracts the plugin into the plugins directory. No-op if the
  ## extracted plugin already exists.
  let pluginDest = pluginPathForTarget(t)
  if fileExists(pluginDest):
    return

  ensureDirs()

  # Check for env-var overrides first
  let archivePath = getEnv(ArchivePathEnvVar)
  if archivePath.len > 0:
    if not fileExists(archivePath):
      raise newException(FetchError,
        ArchivePathEnvVar & " points to non-existent file: " & archivePath)
    let slot = manifest().lookupSlot(key)
    extractMember(archivePath, slot.archiveMember, pluginDest)
    echo "  extracted plugin from ", ArchivePathEnvVar, " → ", pluginDest
    return

  let archiveUrl = getEnv(ArchiveUrlEnvVar)
  if archiveUrl.len > 0:
    let hash = sha256Hex(cast[seq[byte]](archiveUrl))
    let filename = "rew_external_" & hash[0 ..< 16] & archiveSuffix(archiveUrl)
    let dest = externalDir() / filename
    if not fileExists(dest):
      echo &"  downloading from {ArchiveUrlEnvVar}: {archiveUrl}"
      downloadFile(archiveUrl, dest)
    let slot = manifest().lookupSlot(key)
    extractMember(dest, slot.archiveMember, pluginDest)
    echo "  extracted plugin → ", pluginDest
    return

  # Normal manifest-driven download
  let slot = manifest().lookupSlot(key)
  let dest = archivePathForSlot(key)
  if not fileExists(dest):
    echo &"  downloading {targetName(t)} plugin: {slot.url}"
    downloadFile(slot.url, dest)
  verifyChecksum(dest, slot.sha256)
  extractMember(dest, slot.archiveMember, pluginDest)
  echo "  extracted plugin → ", pluginDest
