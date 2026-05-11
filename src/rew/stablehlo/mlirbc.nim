## MLIR bytecode primitives — varints, magic header, byte writer.
##
## This is **container-level only**: the MLIR bytecode framing that wraps
## any dialect's payload. The StableHLO-specific encoding lives in
## `bytecode.nim` (Phase 2c). Splitting the two means we can unit-test
## the framing exhaustively before any dialect lowering exists.
##
## Reference: `mlir/lib/Bytecode/Encoding.h` in LLVM. The varint format is
## **not** LEB128: the byte count is encoded in the count of trailing zero
## bits of the first byte (1 = 1 byte, 01 = 2 bytes, …, 00000000 = 9
## bytes with the value in the next 8 LE bytes).
##
## Pure Nim. No PJRT, no IR imports.

type
  ByteWriter* = object
    ## Append-only byte sink. The underlying `data` is a `seq[byte]`
    ## owned by the writer; consume with `take`.
    data*: seq[byte]

  BytecodeError* = object of CatchableError
    ## Raised on encoding/decoding errors at the MLIR-bytecode layer.
    ## Distinct from `StableHloError` (verifier) and `StableHloEmitError`
    ## (top-level dialect lowering).

const
  MlirBytecodeMagic* = [byte 0x4D, 0x4C, 0xEF, 0x52]  ## ASCII "ML\xefR"

# ---- writer ---------------------------------------------------------------

func initByteWriter*(initialCap = 256): ByteWriter =
  ## Create a fresh writer with `initialCap` bytes preallocated.
  ByteWriter(data: newSeqOfCap[byte](initialCap))

func len*(w: ByteWriter): int {.inline.} = w.data.len
  ## Number of bytes written so far.

func writeByte*(w: var ByteWriter; b: byte) {.inline.} =
  ## Append one raw byte.
  w.data.add b

func writeBytes*(w: var ByteWriter; bs: openArray[byte]) =
  ## Append a raw byte sequence verbatim.
  for b in bs: w.data.add b

proc padToAlignment*(w: var ByteWriter; alignment: int) =
  ## Pad with zero bytes until `len` is a multiple of `alignment`.
  ## `alignment` must be a positive power of two; raises otherwise.
  if alignment <= 0 or (alignment and (alignment - 1)) != 0:
    raise newException(BytecodeError,
      "padToAlignment: alignment must be a positive power of two, got " &
      $alignment)
  while (w.data.len and (alignment - 1)) != 0:
    w.data.add 0'u8

func take*(w: var ByteWriter): seq[byte] =
  ## Move the accumulated bytes out, leaving the writer empty.
  result = move(w.data)

# ---- MLIR varint ----------------------------------------------------------
#
# Encoding rules for an unsigned 64-bit value V:
#
#   - Pick the smallest n in 1..8 such that V fits in 7*n bits.
#   - First byte = (V << n) | (1 shl (n-1))   (low n bits encode length)
#   - Bytes 2..n carry the remaining bits, little-endian.
#
# If V needs more than 56 bits:
#
#   - First byte = 0
#   - Bytes 2..9 = V as little-endian uint64.
#
# This packing means very small values (< 128) take exactly 1 byte —
# the dominant case for op kinds, lengths, etc.

func varintByteCount*(v: uint64): int =
  ## Returns the number of bytes the MLIR varint encoding of `v` would
  ## occupy. Always in 1..9.
  if v < (1'u64 shl 7): 1
  elif v < (1'u64 shl 14): 2
  elif v < (1'u64 shl 21): 3
  elif v < (1'u64 shl 28): 4
  elif v < (1'u64 shl 35): 5
  elif v < (1'u64 shl 42): 6
  elif v < (1'u64 shl 49): 7
  elif v < (1'u64 shl 56): 8
  else: 9

proc writeVarint*(w: var ByteWriter; v: uint64) =
  ## Append `v` in MLIR-bytecode varint encoding.
  let n = varintByteCount(v)
  if n <= 8:
    # First byte holds the length tag in its low n bits and the bottom
    # (8 - n) bits of the value in its top bits, then n-1 more bytes
    # carry the rest little-endian.
    let tag = 1'u64 shl (n - 1)
    let packed = (v shl n) or tag
    var rem = packed
    for i in 0 ..< n:
      w.data.add byte(rem and 0xFF)
      rem = rem shr 8
  else:
    # 9-byte form: tag byte 0, then the raw 64-bit value little-endian.
    w.data.add 0'u8
    var rem = v
    for i in 0 ..< 8:
      w.data.add byte(rem and 0xFF)
      rem = rem shr 8

proc readVarint*(data: openArray[byte]; pos: var int): uint64 =
  ## Decode one varint starting at `data[pos]`. Advances `pos` past the
  ## consumed bytes. Raises `BytecodeError` on truncation.
  if pos >= data.len:
    raise newException(BytecodeError, "readVarint: input truncated")
  let first = data[pos]
  if first == 0'u8:
    # 9-byte form.
    if pos + 9 > data.len:
      raise newException(BytecodeError,
        "readVarint: 9-byte form truncated at offset " & $pos)
    var v: uint64 = 0
    for i in 0 ..< 8:
      v = v or (uint64(data[pos + 1 + i]) shl (8 * i))
    pos += 9
    return v
  # Count trailing zero bits of the first byte → byte count - 1.
  var n = 1
  var probe = first
  while (probe and 1'u8) == 0:
    inc n
    probe = probe shr 1
  if pos + n > data.len:
    raise newException(BytecodeError,
      "readVarint: " & $n & "-byte form truncated at offset " & $pos)
  # Reassemble the packed value.
  var packed: uint64 = 0
  for i in 0 ..< n:
    packed = packed or (uint64(data[pos + i]) shl (8 * i))
  pos += n
  result = packed shr n

# ---- signed (zigzag) ------------------------------------------------------

func zigzagEncode*(v: int64): uint64 {.inline.} =
  ## Standard zigzag mapping: 0 → 0, -1 → 1, 1 → 2, -2 → 3, …
  cast[uint64]((v shl 1) xor (v shr 63))

func zigzagDecode*(v: uint64): int64 {.inline.} =
  ## Inverse of `zigzagEncode`.
  let u = (v shr 1) xor (0'u64 - (v and 1'u64))
  cast[int64](u)

proc writeSignedVarint*(w: var ByteWriter; v: int64) =
  ## Convenience wrapper combining zigzag + varint.
  writeVarint(w, zigzagEncode(v))

proc readSignedVarint*(data: openArray[byte]; pos: var int): int64 =
  ## Convenience wrapper combining varint + zigzag.
  zigzagDecode(readVarint(data, pos))

# ---- header ---------------------------------------------------------------

proc writeMagic*(w: var ByteWriter) =
  ## Append the 4-byte `ML\xefR` magic that opens every MLIR bytecode file.
  writeBytes(w, MlirBytecodeMagic)

proc writeVersion*(w: var ByteWriter; version: uint64) =
  ## Append the producer version varint that follows the magic.
  writeVarint(w, version)

proc writeProducer*(w: var ByteWriter; producer: string) =
  ## Append a length-prefixed (varint) producer string. Bytes are written
  ## verbatim — no encoding conversion.
  writeVarint(w, uint64(producer.len))
  for ch in producer:
    w.data.add byte(ch)

proc readMagicAndVersion*(data: openArray[byte]; pos: var int):
    tuple[version: uint64, producer: string] =
  ## Companion of `writeMagic`+`writeVersion`+`writeProducer` for tests.
  ## Raises `BytecodeError` if the magic doesn't match.
  if data.len < 4 or data[0] != MlirBytecodeMagic[0] or
      data[1] != MlirBytecodeMagic[1] or
      data[2] != MlirBytecodeMagic[2] or
      data[3] != MlirBytecodeMagic[3]:
    raise newException(BytecodeError,
      "readMagicAndVersion: wrong magic; not an MLIR bytecode stream")
  pos = 4
  let version = readVarint(data, pos)
  let plen = readVarint(data, pos)
  if pos + int(plen) > data.len:
    raise newException(BytecodeError,
      "readMagicAndVersion: producer string truncated")
  var producer = newStringOfCap(int(plen))
  for i in 0 ..< int(plen):
    producer.add char(data[pos + i])
  pos += int(plen)
  (version, producer)
