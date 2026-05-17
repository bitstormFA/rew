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

proc requireNotContains(path, needle, message: string) =
  if not fileExists(path):
    failures.add "cannot inspect missing file: " & path
    return
  let src = readFile(path)
  if needle in src:
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
  if "compiled*: JitFunction" in src:
    failures.add "CompiledTrainStep must not expose raw JitFunction"

if fileExists("examples/mnist_coherent_api.nim"):
  let src = readFile("examples/mnist_coherent_api.nim")
  if "JitFn" in src:
    failures.add "coherent example must not expose raw JitFn"
  for forbiddenImport in ["import rew/xla", "import rew/dev", "import rew/pjrt"]:
    if forbiddenImport in src:
      failures.add "coherent example must stay on the user surface: " &
        forbiddenImport
  if "compileTrainStep" notin src:
    failures.add "coherent example should demonstrate compileTrainStep"
  if "initTrainState" notin src:
    failures.add "coherent example should demonstrate TrainState"

for needle in [
  "value state + typed steps",
  "Param[Tensor]",
  "Buffer[Tensor]",
  "Runtime",
  "TrainState[M]",
  "StepResult",
  "Dataset Pipeline",
  "DataSplits",
  "GradientTransform",
  "Bare `Tensor` leaves",
  "raw JIT handles",
  "import rew/xla",
  "import rew/dev",
  "No raw JitFn"
]:
  requireContains(
    "docs/high-level-api.md",
    needle,
    "docs/high-level-api.md must define the coherent API term: " & needle)

for forbidden in [
  "TrainState[M, O, S]",
  "AdamWState",
  "applyUpdates(state, grads)",
  "Plain tensor fields may remain trainable",
  "automaticOptimization",
  "configureOptimizers",
  "trainingStep",
]:
  requireNotContains(
    "docs/high-level-api.md",
    forbidden,
    "docs/high-level-api.md must use canonical TrainState[M] language, not: " &
      forbidden)

for path in [
  "docs/high-level-api.md",
  "docs/user-guide.md",
  "src/rew/train.nim",
  "src/rew/train/datasplits.nim",
  "src/rew/train/runtime.nim",
  "src/rew/train/trainer.nim",
  "examples/mnist_coherent_api.nim"
]:
  if fileExists(path):
    let src = readFile(path)
    for forbidden in [
      "Workbench",
      "initWorkbench",
      "DataPipe",
      "initDataPipe",
      "Data Pipe",
      "OptimizerKind",
      "OptimizerConfig",
      "initOptimizerConfig",
      "automaticOptimization",
      "configureOptimizers",
      "trainingStep",
      "src/rew/train/hooks.nim",
    ]:
      if forbidden in src:
        failures.add path & " must not use legacy high-level term: " & forbidden

if fileExists("src/rew/train/hooks.nim"):
  failures.add "src/rew/train/hooks.nim must not exist; Trainer uses typed steps"

for root in ["src/rew/nn", "src/rew/models"]:
  if dirExists(root):
    for path in walkDirRec(root):
      if path.endsWith(".nim"):
        var lineNo = 0
        for line in lines(path):
          inc lineNo
          if "*: Tensor" in line:
            failures.add path & ":" & $lineNo &
              " model/layer tensor state must be Param[Tensor] or Buffer[Tensor]"

for path in ["src/rew/multimodal/vit.nim"]:
  if fileExists(path):
    var lineNo = 0
    for line in lines(path):
      inc lineNo
      if "*: Tensor" in line:
        failures.add path & ":" & $lineNo &
          " model/layer tensor state must be Param[Tensor] or Buffer[Tensor]"

for needle in [
  "import ./rew/openxla",
  "import ./rew/stablehlo",
  "import ./rew/[tensor, dispatch]",
  "export dispatch",
  "import ./rew/autograd/registry",
  "export registry",
  "import ./rew/transform\nexport transform",
  "import ./rew/eager\nexport eager",
  "import ./rew/pjrt",
  "import ./rew/value\nexport value",
  "import ./rew/onnx\nexport onnx",
  "import ./rew/tflite\nexport tflite",
  "import ./rew/distributed",
]:
  requireNotContains(
    "src/rew.nim",
    needle,
    "src/rew.nim must not leak compiler/dev tier: " & needle)

for needle in [
  "import ./openxla",
  "import ./stablehlo",
  "import ./dispatch",
  "import ./transform",
]:
  requireContains(
    "src/rew/xla.nim",
    needle,
    "src/rew/xla.nim must expose compiler-tier module: " & needle)

for needle in [
  "import ./autograd/registry",
  "import ./autograd/tape",
  "import ./ops/marker",
  "import ./eager",
]:
  requireContains(
    "src/rew/dev.nim",
    needle,
    "src/rew/dev.nim must expose extension-tier module: " & needle)

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
