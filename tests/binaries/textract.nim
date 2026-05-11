## Archive extraction round-trips on small fixtures (no network).

import std/[os, osproc]
import rew/binaries/extract

let tmpDir = getTempDir() / "rew_extract_test"
createDir(tmpDir)

block tar_gz_round_trip:
  let content = "hello from tar"
  let dataFile = tmpDir / "testfile.txt"
  writeFile(dataFile, content)
  let archivePath = tmpDir / "test.tar.gz"
  let rc = execCmd("cd " & quoteShell(tmpDir) &
    " && tar czf test.tar.gz testfile.txt")
  doAssert rc == 0, "tar creation failed"
  let destFile = tmpDir / "extracted_tar.txt"
  extractTarGzMember(archivePath, "testfile.txt", destFile)
  let got = readFile(destFile)
  doAssert got == content, "tar.gz extraction mismatch: '" & got & "'"
  echo "textract: tar.gz round-trip OK"

block zip_round_trip:
  let content = "hello from zip"
  let dataFile = tmpDir / "zipfile.txt"
  writeFile(dataFile, content)
  let archivePath = tmpDir / "test.zip"
  let rc = execCmd("cd " & quoteShell(tmpDir) &
    " && python3 -c \"import zipfile; z = zipfile.ZipFile('test.zip', 'w'); z.write('zipfile.txt'); z.close()\"")
  if rc != 0:
    echo "textract: zip test skipped (python3 not available)"
  else:
    let destFile = tmpDir / "extracted_zip.txt"
    extractZipMember(archivePath, "zipfile.txt", destFile)
    let got = readFile(destFile)
    doAssert got == content, "zip extraction mismatch: '" & got & "'"
    echo "textract: zip round-trip OK"

removeDir(tmpDir)
echo "textract: OK"
