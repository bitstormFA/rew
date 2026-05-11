## Functional, splittable PRNG.
##
## ## Why
## Invariant #8 forbids global RNG state. Every layer or transform that
## needs randomness takes an explicit `Key` argument and is responsible
## for splitting it before calling further random procs. This keeps
## `jit`-traced computations deterministic and reproducible — the same
## key always produces the same stream.
##
## ## Algorithm
## We use a small Threefry-2x32 round (4 rounds) as the splitter / hash
## primitive. It is not cryptographic, but it is statistically uniform
## and trivially threadable. The key is a 64-bit value carried as two
## 32-bit halves; `split(k, n)` produces `n` independent keys, and
## `foldIn(k, data)` mixes a 64-bit datum into a key without consuming
## entropy from the caller's split count.
##
## Sampling procs (`uniform`, `normal`, …) live in `nn/` and `ops/`
## modules — those need tensor ops, which the rng layer deliberately does
## not depend on.

import std/strutils

type
  Key* = object
    ## Splittable PRNG key. Pass by value; copy is cheap (16 bytes).
    a*: uint32
    b*: uint32

# ---- Threefry-2x32 (4 rounds) --------------------------------------------

const
  ThreefryRotations: array[4, uint32] = [13'u32, 15'u32, 26'u32, 6'u32]
  ThreefryParity: uint32 = 0x1BD11BDA'u32

func rotl32(x: uint32; r: uint32): uint32 {.inline.} =
  (x shl r) or (x shr (32'u32 - r))

func threefry2x32(key: Key; ctr: Key): Key =
  ## Four-round Threefry-2x32. Returns a new `Key` that is a
  ## near-bijection of the input pair `(key, ctr)`.
  let
    k0 = key.a
    k1 = key.b
    k2 = k0 xor k1 xor ThreefryParity
  var
    x0 = ctr.a + k0
    x1 = ctr.b + k1
  for r in 0 ..< 4:
    x0 = x0 + x1
    x1 = rotl32(x1, ThreefryRotations[r])
    x1 = x1 xor x0
  x0 = x0 + k1
  x1 = x1 + k2 + 1'u32
  Key(a: x0, b: x1)

# ---- public API -----------------------------------------------------------

func initKey*(seed: uint64): Key =
  ## Constructs a `Key` from a 64-bit seed. Two seeds that differ in any
  ## bit produce uncorrelated streams.
  Key(a: uint32(seed and 0xFFFFFFFF'u64),
      b: uint32(seed shr 32))

func split*(k: Key; n: int = 2): seq[Key] =
  ## Splits `k` into `n` independent child keys. The original key should
  ## not be reused after splitting.
  if n <= 0:
    return @[]
  result = newSeq[Key](n)
  for i in 0 ..< n:
    let ctr = Key(a: uint32(i), b: 0'u32)
    result[i] = threefry2x32(k, ctr)

func foldIn*(k: Key; data: uint64): Key =
  ## Mixes a 64-bit datum into `k`. Useful for keying off a step number,
  ## sample index, or layer name hash without consuming a split slot.
  let ctr = Key(a: uint32(data and 0xFFFFFFFF'u64),
                b: uint32(data shr 32))
  threefry2x32(k, ctr)

func toUint64*(k: Key): uint64 =
  ## Packs `k` into a 64-bit integer. Mostly for serialization /
  ## hashing; do not use as a sample.
  (uint64(k.b) shl 32) or uint64(k.a)

func `==`*(a, b: Key): bool {.inline.} = a.a == b.a and a.b == b.b

func `$`*(k: Key): string =
  ## Hex representation; first half = low 32 bits.
  result = "Key("
  result.add toHex(k.a)
  result.add ":"
  result.add toHex(k.b)
  result.add ")"
