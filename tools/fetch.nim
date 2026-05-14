## Source-tree wrapper for `bau fetch <target>`.

import ../src/rew/binaries/fetch_cli

when isMainModule:
  runFetchCli("bau fetch")
