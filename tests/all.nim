## Central test runner.
##
## Auto-discovers every `t*.nim` file in `tests/` and its subdirectories
## (except this runner) and runs each as its own compiled binary so a
## failure in one test file does not abort the rest.
##
## The child test binaries run concurrently. Set `REW_TEST_JOBS=1` to force the
## old serial behavior, or raise it when the machine has enough memory.
##
## Run from the project root:
##
## ```
## nim c -r tests/all.nim
## ```

import std/[algorithm, os, osproc, strutils]

const MaxDefaultJobs = 8

type TestProc = object
  file: string
  command: string
  process: Process

proc quotedCommand(args: openArray[string]): string =
  result = "nim"
  for arg in args:
    result.add " " & quoteShell(arg)

proc childSwitches(): seq[string] =
  when defined(danger):
    result.add "-d:danger"
  elif defined(release):
    result.add "-d:release"

  when defined(addressSanitizer):
    result.add "-d:addressSanitizer"

proc parseTestJobs(): int =
  let envJobs = getEnv("REW_TEST_JOBS")
  if envJobs.len > 0:
    try:
      result = parseInt(envJobs)
    except ValueError:
      echo "Ignoring invalid REW_TEST_JOBS value: " & envJobs
      result = 0

  if result <= 0:
    result = countProcessors()
    if result <= 0:
      result = 1
    result = min(result, MaxDefaultJobs)

proc matchesFilter(path: string; filters: openArray[string]): bool =
  if filters.len == 0:
    return true

  let normalized = path.replace('\\', '/').toLowerAscii()
  let name = path.extractFilename.toLowerAscii()
  for filter in filters:
    let needle = filter.toLowerAscii()
    if needle.len == 0:
      continue
    if needle in normalized or needle in name:
      return true

proc testArgs(file: string): seq[string] =
  result = @["c"]
  result.add childSwitches()
  result.add "--hints:off"
  result.add "-r"
  result.add file

proc startTest(file: string): TestProc =
  let args = testArgs(file)
  let command = quotedCommand(args)
  echo "Running: " & command
  result = TestProc(
    file: file,
    command: command,
    process: startProcess("nim", args = args, options = {poUsePath, poParentStreams}))

proc reapFinished(active: var seq[TestProc]; failures: var seq[string]): bool =
  for i in countdown(active.high, 0):
    let code = peekExitCode(active[i].process)
    if code == -1:
      continue

    result = true
    if code == 0:
      echo "OK: " & active[i].file
    else:
      echo "FAILURE (" & $code & "): " & active[i].command
      failures.add active[i].file

    close(active[i].process)
    active.delete i

let testDir = getCurrentDir() / "tests"
var filters: seq[string]
for arg in commandLineParams():
  if arg != "--":
    filters.add arg
var files: seq[string]
for f in walkDirRec(testDir):
  if not f.endsWith(".nim"): continue
  let name = f.extractFilename
  if not name.startsWith("t"): continue
  if name in ["all.nim", "thelper.nim"]: continue
  if not matchesFilter(f, filters): continue
  files.add f

files.sort()
if files.len == 0:
  quit "No test files matched.", 1

let jobs = parseTestJobs()
echo "Running " & $files.len & " test file(s) with " & $jobs & " job(s)."

var failures: seq[string]
var active: seq[TestProc]
var nextFile = 0

while nextFile < files.len or active.len > 0:
  while nextFile < files.len and active.len < jobs:
    active.add startTest(files[nextFile])
    inc nextFile

  if active.len == 0:
    continue

  while not reapFinished(active, failures):
    sleep 50

if failures.len > 0:
  echo "Test failures (" & $failures.len & " of " & $files.len & "):"
  for f in failures:
    echo "  " & f
  quit "FAILURE: test suite failed", 1

echo "All test files completed (" & $files.len & " file(s))."
