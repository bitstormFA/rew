## Architectural lint #3 — No `ref object` declarations under `nn/` or `optim/`.
##
## The functional nn/optim contract requires layers and optimizers to be plain
## value `object` types. Any `ref object` declaration there is a violation.
##
## Imports of `ref` types from elsewhere are allowed (e.g. an inner field
## holding a `ref BufferHandle` indirectly via a `Tensor` value); only
## *declaring* a `ref object` here is forbidden.

import std/[os, strutils, re]

const Watched = ["src/rew/nn", "src/rew/optim"]

let refObjRe = re"=\s*ref\s+object\b"

var failures: seq[string] = @[]
for dir in Watched:
  if not dirExists(dir): continue
  for f in walkDirRec(dir):
    if not f.endsWith(".nim"): continue
    let src = readFile(f)
    if src.contains(refObjRe):
      failures.add f & ": declares a `ref object` (forbidden under nn/ and optim/)"

if failures.len > 0:
  echo "check_no_ref_in_nn: FAIL"
  for msg in failures:
    echo "  " & msg
  quit 1
echo "check_no_ref_in_nn: OK"
