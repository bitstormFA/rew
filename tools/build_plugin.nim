## CLI entry point for `bau buildPlugin <target>`.
## Builds a PJRT plugin from an openxla/xla checkout.

import std/os
import ../src/rew/binaries/[target, builder]

proc main() =
  if paramCount() < 1:
    echo "Usage: bau buildPlugin <target>"
    echo "  target: cpu | cuda12 | cuda13 | rocm | tpu"
    echo ""
    echo "Builds a PJRT plugin from source. Requires:"
    echo "  REW_BUILD_XLA_DIR — path to an openxla/xla checkout"
    echo "  bazel v7.7+ on PATH"
    quit 1
  let t = parseTarget(paramStr(1))
  echo "Building '", targetName(t), "' plugin from source"
  buildPlugin(t)

when isMainModule:
  main()
