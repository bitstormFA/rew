## CLI entry point for optional OpenXLA tools.

import std/os
import ../src/rew/openxla/cli

when isMainModule:
  quit runOpenXlaCli(commandLineParams())
