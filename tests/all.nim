## Central test runner.
##
## Auto-discovers every `t*.nim` file in `tests/` and its subdirectories
## (except this runner) and runs each as its own compiled binary so a
## failure in one test file does not abort the rest.
##
## Run from the project root:
##
## ```
## nim c -r tests/all.nim
## ```

import std/[algorithm, os, strutils]

proc exec(cmd: string): bool =
  echo "Running: " & cmd
  if execShellCmd(cmd) != 0:
    echo "FAILURE: " & cmd
    return false
  true

proc childSwitches(): string =
  when defined(danger):
    result.add " -d:danger"
  elif defined(release):
    result.add " -d:release"

  when defined(addressSanitizer):
    result.add " -d:addressSanitizer"

let testDir = getCurrentDir() / "tests"
var files: seq[string]
for f in walkDirRec(testDir):
  if not f.endsWith(".nim"): continue
  let name = f.extractFilename
  if not name.startsWith("t"): continue
  if name in ["all.nim", "thelper.nim"]: continue
  files.add f

files.sort()

var failures: seq[string]
for f in files:
  if not exec("nim c" & childSwitches() & " -r " & quoteShell(f)):
    failures.add f

if failures.len > 0:
  echo "Test failures (" & $failures.len & " of " & $files.len & "):"
  for f in failures:
    echo "  " & f
  quit "FAILURE: test suite failed", 1

echo "All test files completed (" & $files.len & " file(s))."
