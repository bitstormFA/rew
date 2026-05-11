## Pure-Nim SHA-256 vs known NIST test vectors.

import rew/binaries/checksum
import std/os

block sha256_empty:
  let hash = sha256Hex(newSeq[byte](0))
  doAssert hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "empty-input SHA-256 mismatch: " & hash

block sha256_abc:
  let data = @[byte('a'), byte('b'), byte('c')]
  let hash = sha256Hex(data)
  doAssert hash == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    "'abc' SHA-256 mismatch: " & hash

block sha256_file_round_trip:
  let tmp = getTempDir() / "rew_sha256_test.bin"
  writeFile(tmp, "hello world")
  defer: removeFile(tmp)
  let hash = sha256File(tmp)
  doAssert hash == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
    "'hello world' file SHA-256 mismatch: " & hash

block sha256_larger:
  var data = newSeq[byte](1000)
  for i in 0 ..< data.len:
    data[i] = byte(i mod 256)
  let hash = sha256Hex(data)
  doAssert hash.len == 64
  doAssert hash != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

echo "tchecksum: OK"
