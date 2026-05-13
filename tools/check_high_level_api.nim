## Architectural lint - coherent high-level API.
##
## Keeps the new public language from drifting back into raw JIT plumbing.

import std/[os, strutils]

var failures: seq[string]

proc requireFile(path: string) =
  if not fileExists(path):
    failures.add "missing required high-level API file: " & path

proc requireContains(path, needle, message: string) =
  if not fileExists(path):
    failures.add "cannot inspect missing file: " & path
    return
  let src = readFile(path)
  if needle notin src:
    failures.add message

requireFile("src/rew/xla.nim")
requireFile("src/rew/dev.nim")
requireFile("src/rew/train/state.nim")
requireFile("examples/mnist_coherent_api.nim")
requireFile("docs/high-level-api.md")
requireFile(".github/instructions/high-level-api.instructions.md")

if fileExists("src/rew/train/state.nim"):
  let src = readFile("src/rew/train/state.nim")
  if "compileTrainStep*" notin src:
    failures.add "src/rew/train/state.nim must export compileTrainStep"
  if "TrainState*" notin src:
    failures.add "src/rew/train/state.nim must export TrainState"

if fileExists("examples/mnist_coherent_api.nim"):
  let src = readFile("examples/mnist_coherent_api.nim")
  if "JitFn" in src:
    failures.add "coherent example must not expose raw JitFn"
  if "compileTrainStep" notin src:
    failures.add "coherent example should demonstrate compileTrainStep"
  if "initTrainState" notin src:
    failures.add "coherent example should demonstrate TrainState"

for needle in [
  "value state + typed steps",
  "Param[Tensor]",
  "Buffer[Tensor]",
  "Runtime",
  "TrainState",
  "StepResult",
  "GradientTransform",
  "import rew/xla",
  "import rew/dev",
  "No raw JitFn"
]:
  requireContains(
    "docs/high-level-api.md",
    needle,
    "docs/high-level-api.md must define the coherent API term: " & needle)

for path in [
  "docs/architecture.md",
  "docs/user-guide.md",
  ".github/copilot-instructions.md",
  "AGENTS.md",
  ".github/instructions/high-level-api.instructions.md"
]:
  requireContains(
    path,
    "docs/high-level-api.md",
    path & " must point future agents at docs/high-level-api.md")

if failures.len > 0:
  echo "check_high_level_api: FAIL"
  for msg in failures:
    echo "  " & msg
  quit 1

echo "check_high_level_api: OK"
