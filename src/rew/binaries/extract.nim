## Extract — archive extraction for `.tar.gz` and `.whl`/`.zip` files.
##
## Uses system `tar` for tarballs (universally available on Linux/macOS)
## and `python3 -c` + `zipfile` for `.whl`/`.zip` (same as the old
## `fetch_plugin.nim` but cleaned up). Keeps rew dependency-free.

import std/[os, osproc, strformat, strutils]

type
  ExtractError* = object of CatchableError
    ## Raised when extraction of an archive member fails.

proc requireCmd(cmd: string) =
  if findExe(cmd).len == 0:
    raise newException(ExtractError,
      "required command not found on PATH: '" & cmd & "'")

proc extractTarGzMember*(archivePath, member, destFile: string) =
  ## Extracts a single member from a `.tar.gz` and writes it to `destFile`.
  ## Uses `python3` + `tarfile` for portable behavior across platforms.
  requireCmd("python3")
  let script =
    "import sys, tarfile, shutil, os\n" &
    "tf = tarfile.open(sys.argv[1], 'r:*')\n" &
    "wanted = sys.argv[2]\n" &
    "match = None\n" &
    "for m in tf.getmembers():\n" &
    "    if not m.isreg(): continue\n" &
    "    if m.name == wanted or m.name.endswith('/'+wanted) or " &
        "os.path.basename(m.name)==os.path.basename(wanted):\n" &
    "        match = m; break\n" &
    "if match is None: raise SystemExit('member not found: '+wanted)\n" &
    "src = tf.extractfile(match)\n" &
    "with open(sys.argv[3], 'wb') as dst: shutil.copyfileobj(src, dst)\n" &
    "os.chmod(sys.argv[3], 0o755)\n"
  let cmd = quoteShellCommand(@["python3", "-c", script, archivePath,
                                 member, destFile])
  let rc = execCmd(cmd)
  if rc != 0:
    raise newException(ExtractError,
      &"failed to extract '{member}' from '{archivePath}'")

proc extractZipMember*(archivePath, member, destFile: string) =
  ## Extracts a single member from a `.whl`/`.zip` and writes it to
  ## `destFile`. Uses `python3` + `zipfile`.
  requireCmd("python3")
  let script =
    "import sys, zipfile, shutil, os\n" &
    "z = zipfile.ZipFile(sys.argv[1])\n" &
    "wanted = sys.argv[2]\n" &
    "names = [n for n in z.namelist() if n == wanted or " &
        "n.endswith('/'+wanted) or " &
        "os.path.basename(n)==os.path.basename(wanted)]\n" &
    "if not names: raise SystemExit('member not found: '+wanted)\n" &
    "with z.open(names[0]) as src, open(sys.argv[3],'wb') as dst:\n" &
    "    shutil.copyfileobj(src, dst)\n" &
    "os.chmod(sys.argv[3], 0o755)\n"
  let cmd = quoteShellCommand(@["python3", "-c", script, archivePath,
                                 member, destFile])
  let rc = execCmd(cmd)
  if rc != 0:
    raise newException(ExtractError,
      &"failed to extract '{member}' from '{archivePath}'")

proc extractMember*(archivePath, member, destFile: string) =
  ## Auto-dispatches between tar.gz and zip extraction based on the
  ## archive file extension.
  createDir(parentDir(destFile))
  let lower = archivePath.toLowerAscii()
  if lower.endsWith(".tar.gz") or lower.endsWith(".tgz"):
    extractTarGzMember(archivePath, member, destFile)
  elif lower.endsWith(".whl") or lower.endsWith(".zip"):
    extractZipMember(archivePath, member, destFile)
  else:
    raise newException(ExtractError,
      "unsupported archive format: " & archivePath &
      " (expected .tar.gz, .tgz, .whl, or .zip)")
