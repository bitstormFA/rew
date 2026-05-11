## Source-tree wrapper for `nimble fetch <target>`.

import ../src/rew/binaries/fetch_cli

when isMainModule:
  runFetchCli("nimble fetch")
