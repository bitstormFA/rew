## Phase 5a — splittable PRNG.

import rew
import std/strutils

block init_key_distinct_seeds_distinct:
  let k1 = initKey(0)
  let k2 = initKey(1)
  let k3 = initKey(0xDEADBEEF'u64)
  doAssert k1 != k2
  doAssert k1 != k3
  doAssert k2 != k3

block split_is_deterministic:
  let k = initKey(42)
  let a = split(k, 4)
  let b = split(k, 4)
  doAssert a.len == 4
  doAssert b.len == 4
  for i in 0 ..< 4:
    doAssert a[i] == b[i]

block split_children_are_distinct:
  let k = initKey(7)
  let kids = split(k, 8)
  for i in 0 ..< kids.len:
    for j in i + 1 ..< kids.len:
      doAssert kids[i] != kids[j],
        "child keys " & $i & " and " & $j & " collided: " & $kids[i]
    # Children should not equal the parent either.
    doAssert kids[i] != k

block split_zero_returns_empty:
  let k = initKey(99)
  doAssert split(k, 0).len == 0

block fold_in_changes_key:
  let k = initKey(11)
  let k2 = foldIn(k, 5)
  let k3 = foldIn(k, 6)
  doAssert k2 != k
  doAssert k2 != k3

block fold_in_is_deterministic:
  let k = initKey(11)
  doAssert foldIn(k, 5) == foldIn(k, 5)

block to_uint64_round_trip_is_lossless:
  let k = initKey(0xCAFEBABE12345678'u64)
  let bits = toUint64(k)
  let k2 = Key(a: uint32(bits and 0xFFFFFFFF'u64),
               b: uint32(bits shr 32))
  doAssert k == k2

block stringify_includes_both_halves:
  let s = $initKey(0)
  doAssert s.len > 0
  doAssert s.startsWith("Key(")
