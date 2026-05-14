## Architectural lint #1 — Layer import boundaries.
##
## Rules enforced:
##  - Files under `src/rew/pjrt/` may import stdlib, other pjrt/ files, and
##    `../binaries/target` (for the Target enum).
##  - Files under `src/rew/binaries/` may import only stdlib and other
##    binaries/ files. They must NOT import from pjrt/.
##  - Outside `src/rew/pjrt/`, only the allow-listed adapter modules may
##    import from `src/rew/pjrt/`.
##
## Exits non-zero on any violation. Run via `bau lint`.

import std/[os, strutils, sequtils]

const
  PjrtDir = "src/rew/pjrt"
  BinariesDir = "src/rew/binaries"
  AllowedPjrtImporters = [
    "src/rew/buffer.nim",
    "src/rew/device.nim",
    "src/rew/eager.nim",
  ]

type ImportKind = enum
  ikStdlib
  ikInPjrt
  ikInBinaries
  ikPjrt       ## pjrt/ imported from outside pjrt/
  ikBinaries   ## binaries/ imported from outside binaries/
  ikOther

proc classify(line: string; fromFile: string): seq[ImportKind] =
  result = @[]
  var s = line.strip()
  if s.startsWith("from "):
    s = s[5 .. ^1]
    let cut = s.find(" import")
    if cut >= 0: s = s[0 ..< cut]
  elif s.startsWith("import "):
    s = s[7 .. ^1]
  else:
    return

  let hash = s.find('#')
  if hash >= 0: s = s[0 ..< hash]
  s = s.strip()

  var parts: seq[string] = @[]
  var depth = 0
  var cur = ""
  for ch in s:
    case ch
    of '[': inc depth; cur.add ch
    of ']': dec depth; cur.add ch
    of ',':
      if depth == 0:
        parts.add cur
        cur = ""
      else:
        cur.add ch
    else: cur.add ch
  if cur.len > 0: parts.add cur

  for raw in parts:
    let p = raw.strip().strip(chars = {';'})
    if p.len == 0: continue
    let lb = p.find('[')
    var modules: seq[string] = @[]
    if lb >= 0 and p.endsWith("]"):
      let prefix = p[0 ..< lb]
      let inner = p[lb + 1 ..< p.len - 1]
      for m in inner.split(','):
        let mm = m.strip()
        if mm.len > 0:
          modules.add prefix & mm
    else:
      modules.add p

    for m in modules:
      if m.startsWith("std/") or m == "std" or
          m.startsWith("system/") or m == "system":
        result.add ikStdlib
      elif m.startsWith("./") or m.startsWith("../"):
        let dir = parentDir(fromFile)
        let abs = absolutePath(dir / m).normalizedPath
        let pjrtAbs = absolutePath(PjrtDir).normalizedPath
        let binAbs = absolutePath(BinariesDir).normalizedPath
        if abs.startsWith(pjrtAbs):
          result.add ikInPjrt
        elif abs.startsWith(binAbs):
          result.add ikInBinaries
        else:
          result.add ikOther
      elif m.startsWith("rew/pjrt") or m.startsWith("pjrt/") or m == "pjrt":
        result.add ikPjrt
      elif m.startsWith("rew/binaries") or m.startsWith("binaries/"):
        result.add ikBinaries
      else:
        result.add ikOther

proc importsOf(path: string): seq[ImportKind] =
  result = @[]
  if not fileExists(path): return
  for line in lines(path):
    result.add classify(line, path)

var failures: seq[string] = @[]

# pjrt/ may import: stdlib, other pjrt/ files, and binaries/ (for Target)
if dirExists(PjrtDir):
  for f in walkDirRec(PjrtDir):
    if not f.endsWith(".nim"): continue
    for k in importsOf(f):
      if k notin {ikStdlib, ikInPjrt, ikInBinaries, ikBinaries}:
        failures.add f & ": pjrt module has a forbidden import (kind " &
          $k & ")"

# binaries/ may import: stdlib, other binaries/ files. NOT pjrt/.
if dirExists(BinariesDir):
  for f in walkDirRec(BinariesDir):
    if not f.endsWith(".nim"): continue
    for k in importsOf(f):
      if k notin {ikStdlib, ikInBinaries}:
        failures.add f & ": binaries module has a forbidden import (kind " &
          $k & ")"

# Outside pjrt/ and binaries/, only allow-listed files may import pjrt/
if dirExists("src/rew"):
  for f in walkDirRec("src/rew"):
    if not f.endsWith(".nim"): continue
    if f.startsWith(PjrtDir): continue
    if f.startsWith(BinariesDir): continue
    let rel = f.replace('\\', '/')
    if rel in AllowedPjrtImporters: continue
    if anyIt(importsOf(f), it == ikPjrt or it == ikInPjrt):
      failures.add rel & ": only " & $AllowedPjrtImporters &
        " may import pjrt/*"

if failures.len > 0:
  echo "check_layer_imports: FAIL"
  for msg in failures:
    echo "  " & msg
  quit 1
echo "check_layer_imports: OK"
