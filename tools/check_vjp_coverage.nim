## Architectural lint #2 — VJP coverage.
##
## Every op exported by `src/rew/tensor.nim` or any `src/rew/ops/*.nim` module
## (procs/funcs marked with the `{.rewOp.}` pragma) must have a matching entry
## in `src/rew/autograd/registry.nim` (a call to `registerVjp("opName", ...)`
## or `registerNoGrad("opName", ...)`).
##
## Until those modules exist, this lint is a no-op that prints OK so Phase 0
## can pass `bau lint`. It begins enforcing as soon as `tensor.nim` lands.

import std/[os, strutils, re, sets]

const
  RegistryFile = "src/rew/autograd/registry.nim"
  OpsDir = "src/rew/ops"
  TensorFile = "src/rew/tensor.nim"

let registerRe = re"""register(?:Vjp|NoGrad)\(\s*"([^"]+)"\s*"""

proc declName(line: string): string =
  let s = line.strip()
  let offset =
    if s.startsWith("proc "): 5
    elif s.startsWith("func "): 5
    else: return ""

  var i = offset
  while i < s.len and s[i].isSpaceAscii:
    inc i
  let start = i
  while i < s.len and (s[i].isAlphaNumeric or s[i] == '_'):
    inc i
  if i == start: return ""
  s[start ..< i]

proc collectOps(path: string, ops: var HashSet[string]) =
  if not fileExists(path):
    return
  var current = ""
  for raw in readFile(path).splitLines:
    let line = raw.strip()
    if line.len == 0 or line.startsWith("#"):
      continue
    let name = declName(line)
    if name.len > 0:
      current = name
    if current.len > 0 and "{.rewOp.}" in line:
      ops.incl current
    if current.len > 0 and line.endsWith("="):
      current = ""

proc collectRegistered(path: string, regs: var HashSet[string]) =
  if not fileExists(path):
    return
  for raw in readFile(path).splitLines:
    let line = raw.strip()
    if line.len == 0 or line.startsWith("#"):
      continue
    for m in line.findAll(registerRe):
      var captures: array[1, string]
      if m.match(registerRe, captures):
        regs.incl captures[0]

var ops, regs: HashSet[string]
collectOps(TensorFile, ops)
if dirExists(OpsDir):
  for f in walkDirRec(OpsDir):
    if f.endsWith(".nim"):
      collectOps(f, ops)
collectRegistered(RegistryFile, regs)

let missing = ops - regs
if missing.len > 0:
  echo "check_vjp_coverage: FAIL"
  for op in missing:
    echo "  op '" & op & "' has no registerVjp/registerNoGrad entry in " &
      RegistryFile
  quit 1

let extra = regs - ops
if extra.len > 0:
  echo "check_vjp_coverage: FAIL"
  for op in extra:
    echo "  registry entry '" & op & "' does not match a {.rewOp.} proc"
  quit 1

if ops.len == 0 and not fileExists(TensorFile):
  echo "check_vjp_coverage: OK (no ops yet — pre-Phase 3)"
else:
  echo "check_vjp_coverage: OK (" & $ops.len & " op(s) covered)"
