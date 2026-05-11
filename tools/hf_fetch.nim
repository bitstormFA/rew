## Download Hugging Face model or dataset assets into the rew/HF cache.
##
## Usage:
##   nim c -r tools/hf_fetch.nim model google/gemma-4-E4B-it
##   nim c -r tools/hf_fetch.nim dataset NousResearch/hermes-function-calling-v1 func_calling_singleturn

import std/[os, strutils]
import ../src/rew/hf

proc usage() =
  echo "Usage:"
  echo "  hf_fetch model <repo-id> [revision]"
  echo "  hf_fetch dataset <repo-id> [config] [revision]"
  quit 1

when isMainModule:
  if paramCount() < 2:
    usage()
  let kind = paramStr(1).toLowerAscii()
  let repoId = paramStr(2)
  case kind
  of "model":
    let revision = if paramCount() >= 3: paramStr(3) else: "main"
    let path = hfDownloadModel(repoId, revision = revision)
    echo path
  of "dataset":
    let config = if paramCount() >= 3: paramStr(3) else: ""
    let revision = if paramCount() >= 4: paramStr(4) else: "main"
    let path = hfDownloadDataset(repoId, config = config, revision = revision)
    echo path
  else:
    usage()
