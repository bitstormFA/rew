## Re-resolves vendor URLs and recomputes SHA-256 checksums in the manifest.
##
## For each slot in `pjrt_manifest.json`, downloads the archive (if not
## already cached), computes its SHA-256, and rewrites the manifest file
## with updated checksums.

import std/[json, os, strutils, tables]
import ../src/rew/binaries/[manifest, target, cache, checksum, fetcher]

proc main() =
  echo "Updating pjrt_manifest.json checksums..."
  let m = manifest()
  let manifestPath = getCurrentDir() / "pjrt_manifest.json"
  let j = parseJson(readFile(manifestPath))

  for key, slot in m.slots:
    echo "  slot: ", key
    let dest = archivePathForSlot(key)
    if not fileExists(dest):
      echo "    downloading: ", slot.url
      ensureDirs()
      # Use a temporary download via httpclient
      import std/httpclient
      var client = newHttpClient(timeout = 60_000)
      defer: client.close()
      createDir(parentDir(dest))
      client.downloadFile(slot.url, dest)
    let hash = sha256File(dest)
    echo "    sha256: ", hash
    j["slots"][key]["sha256"] = newJString(hash)

  writeFile(manifestPath, j.pretty(2) & "\n")
  echo "Manifest updated: ", manifestPath

when isMainModule:
  main()
