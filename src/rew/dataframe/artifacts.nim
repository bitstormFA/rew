## Pinned DuckDB artifact resolution.
##
## The DataFrame backend prefers a known DuckDB release artifact over whatever
## happens to be installed on the host. This keeps the C ABI and SQL behavior
## stable across machines.

import std/[os, osproc, strformat, strutils]
import ../binaries/[cache, checksum, extract]

const
  DuckDbVersion* = "1.5.3"
  DuckDbVersionTag* = "v" & DuckDbVersion
  DuckDbEnvLib* = "REW_DUCKDB_LIB"
  DuckDbEnvArchivePath* = "REW_DUCKDB_ARCHIVE_PATH"
  DuckDbEnvArchiveUrl* = "REW_DUCKDB_ARCHIVE_URL"
  DuckDbEnvNoSystemFallback* = "REW_DUCKDB_NO_SYSTEM_FALLBACK"
  MaxRetries = 3

type
  DuckDbArtifactError* = object of CatchableError
    ## Raised when the pinned DuckDB artifact cannot be resolved.

  DuckDbArtifactSpec* = object
    ## Pinned DuckDB release artifact metadata for one OS/architecture pair.
    url*: string
    sha256*: string
    member*: string
    libraryName*: string
    platformName*: string

proc duckDbArtifactSpec*(osName, archName: string): DuckDbArtifactSpec =
  ## Returns pinned DuckDB artifact metadata for a supported platform.
  ##
  ## `osName` accepts `linux`, `macosx`/`darwin`/`osx`, and `windows`.
  ## `archName` accepts `amd64`/`x86_64` and `arm64`/`aarch64` where the
  ## upstream release publishes distinct archives. macOS uses DuckDB's
  ## universal artifact for both Intel and Apple Silicon.
  let base = "https://github.com/duckdb/duckdb/releases/download/" &
    DuckDbVersionTag & "/"
  let os = osName.toLowerAscii()
  let arch = archName.toLowerAscii()

  case os
  of "linux":
    case arch
    of "amd64", "x86_64":
      DuckDbArtifactSpec(
        url: base & "libduckdb-linux-amd64.zip",
        sha256: "0a926eba5bce0abc0010f4b9109133e4440cb74e97bd10fd2d0fc2a721621b05",
        member: "libduckdb.so",
        libraryName: "libduckdb.so",
        platformName: "linux-amd64")
    of "arm64", "aarch64":
      DuckDbArtifactSpec(
        url: base & "libduckdb-linux-arm64.zip",
        sha256: "162806d591c0431d031d9bdf43dbecc5f00755da01a2064df68f9a69a6f50a10",
        member: "libduckdb.so",
        libraryName: "libduckdb.so",
        platformName: "linux-arm64")
    else:
      raise newException(DuckDbArtifactError,
        "unsupported Linux architecture for DuckDB artifact: " & archName)
  of "macosx", "darwin", "osx", "macos":
    DuckDbArtifactSpec(
      url: base & "libduckdb-osx-universal.zip",
      sha256: "386f8e8b3b4bc8d128762327121e22065ce45f2ee55ef1b1f412ce11e0e6c51f",
      member: "libduckdb.dylib",
      libraryName: "libduckdb.dylib",
      platformName: "osx-universal")
  of "windows", "win32":
    case arch
    of "amd64", "x86_64":
      DuckDbArtifactSpec(
        url: base & "libduckdb-windows-amd64.zip",
        sha256: "11842aca19ec7a415ffbb732ec4818a1562111fb4151fd59d1b3a40b551db26e",
        member: "duckdb.dll",
        libraryName: "duckdb.dll",
        platformName: "windows-amd64")
    of "arm64", "aarch64":
      DuckDbArtifactSpec(
        url: base & "libduckdb-windows-arm64.zip",
        sha256: "e3edbaffc815e87918c0c0450996b48e0c61627e92682dab9ed9742649ce5586",
        member: "duckdb.dll",
        libraryName: "duckdb.dll",
        platformName: "windows-arm64")
    else:
      raise newException(DuckDbArtifactError,
        "unsupported Windows architecture for DuckDB artifact: " & archName)
  else:
    raise newException(DuckDbArtifactError,
      "unsupported platform for DuckDB artifact: " & osName)

proc specForHost(): DuckDbArtifactSpec =
  when defined(linux):
    duckDbArtifactSpec("linux", hostCPU)
  elif defined(macosx):
    duckDbArtifactSpec("macosx", hostCPU)
  elif defined(windows):
    duckDbArtifactSpec("windows", hostCPU)
  else:
    raise newException(DuckDbArtifactError,
      "unsupported platform for DuckDB artifact")

proc duckDbRoot(): string =
  versionedCacheDir() / "duckdb" / DuckDbVersionTag

proc duckDbArchivePath(): string =
  let spec = specForHost()
  duckDbRoot() / "download" / spec.platformName / spec.url.split('/')[^1]

proc duckDbLibraryPath*(): string =
  let spec = specForHost()
  duckDbRoot() / "lib" / spec.platformName / spec.libraryName

func duckDbPlatformName*(): string =
  specForHost().platformName

func duckDbArtifactUrl*(): string =
  specForHost().url

func duckDbArtifactSha256*(): string =
  specForHost().sha256

proc tryDownload(url, destPath: string): tuple[ok: bool, msg: string] =
  let curl = findExe("curl")
  if curl.len > 0:
    let args = @[curl, "-fL", "--retry", $MaxRetries, "-o", destPath, url]
    let (outp, code) = execCmdEx(quoteShellCommand(args))
    if code == 0:
      return (true, "")
    return (false, outp.strip())

  let wget = findExe("wget")
  if wget.len > 0:
    let args = @[wget, "--tries=" & $MaxRetries, "-O", destPath, url]
    let (outp, code) = execCmdEx(quoteShellCommand(args))
    if code == 0:
      return (true, "")
    return (false, outp.strip())

  (false, "neither curl nor wget is available")

proc verifyArchive(path, expected: string) =
  let actual = sha256File(path)
  if actual != expected:
    removeFile(path)
    raise newException(DuckDbArtifactError,
      &"DuckDB archive checksum mismatch: expected {expected}, got {actual}")

proc ensureDuckDbArtifact*(): string =
  ## Ensures the pinned DuckDB shared library is present and returns its path.
  let libPath = duckDbLibraryPath()
  if fileExists(libPath):
    return libPath

  let spec = specForHost()
  createDir(parentDir(libPath))
  createDir(parentDir(duckDbArchivePath()))

  let overrideArchive = getEnv(DuckDbEnvArchivePath)
  let archivePath =
    if overrideArchive.len > 0:
      if not fileExists(overrideArchive):
        raise newException(DuckDbArtifactError,
          DuckDbEnvArchivePath & " points to a missing file: " &
            overrideArchive)
      overrideArchive
    else:
      let dest = duckDbArchivePath()
      let url = getEnv(DuckDbEnvArchiveUrl, spec.url)
      if not fileExists(dest):
        echo &"  downloading DuckDB {DuckDbVersionTag}: {url}"
        let dl = tryDownload(url, dest)
        if not dl.ok:
          raise newException(DuckDbArtifactError,
            "failed to download pinned DuckDB artifact: " & dl.msg)
      if url == spec.url:
        verifyArchive(dest, spec.sha256)
      dest

  extractMember(archivePath, spec.member, libPath)
  result = libPath

proc resolveDuckDbLibraryPath*(): string =
  ## Returns the library path to load. Pinned artifacts are preferred over
  ## host installs; explicit/system fallbacks still have their runtime version
  ## checked by the loader.
  try:
    return ensureDuckDbArtifact()
  except CatchableError as e:
    let explicit = getEnv(DuckDbEnvLib)
    if explicit.len > 0:
      if not fileExists(explicit):
        raise newException(DuckDbArtifactError,
          DuckDbEnvLib & " points to a missing file after pinned artifact " &
            "resolution failed: " & explicit & " (" & e.msg & ")")
      return explicit
    if getEnv(DuckDbEnvNoSystemFallback).len == 0:
      when defined(macosx):
        return "libduckdb.dylib"
      elif defined(windows):
        return "duckdb.dll"
      else:
        return "libduckdb.so"
    raise e
