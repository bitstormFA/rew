## Phase 2b — MLIR bytecode primitives: varint, header, writer.
##
## We exhaustively roundtrip the varint at each byte-count boundary plus
## a handful of extreme values, since getting this wrong silently corrupts
## every later byte of any module we emit.

import rew/stablehlo/mlirbc

block writer_basic:
  var w = initByteWriter()
  doAssert w.len == 0
  w.writeByte 0xAB'u8
  w.writeBytes [byte 0x01, 0x02, 0x03]
  doAssert w.len == 4
  let bs = w.take()
  doAssert bs == @[byte 0xAB, 0x01, 0x02, 0x03]
  doAssert w.len == 0

block padToAlignment_pads_zero_bytes:
  var w = initByteWriter()
  w.writeBytes [byte 0xFF, 0xFF, 0xFF]
  w.padToAlignment 8
  doAssert w.len == 8
  let bs = w.take()
  doAssert bs == @[byte 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0]

block padToAlignment_no_op_when_aligned:
  var w = initByteWriter()
  w.writeBytes [byte 1, 2, 3, 4, 5, 6, 7, 8]
  w.padToAlignment 8
  doAssert w.len == 8

block padToAlignment_rejects_non_power_of_two:
  var w = initByteWriter()
  var raised = false
  try:
    w.padToAlignment 6
  except BytecodeError:
    raised = true
  doAssert raised

# ---- varint byte-count boundaries ---------------------------------------

block varint_byte_count_table:
  doAssert varintByteCount(0) == 1
  doAssert varintByteCount(127) == 1
  doAssert varintByteCount(128) == 2
  doAssert varintByteCount((1'u64 shl 14) - 1) == 2
  doAssert varintByteCount(1'u64 shl 14) == 3
  doAssert varintByteCount(1'u64 shl 21) == 4
  doAssert varintByteCount(1'u64 shl 28) == 5
  doAssert varintByteCount(1'u64 shl 35) == 6
  doAssert varintByteCount(1'u64 shl 42) == 7
  doAssert varintByteCount(1'u64 shl 49) == 8
  doAssert varintByteCount((1'u64 shl 56) - 1) == 8
  doAssert varintByteCount(1'u64 shl 56) == 9
  doAssert varintByteCount(high(uint64)) == 9

block varint_one_byte_layout:
  ## A 1-byte varint encodes V as `(V shl 1) or 1`. Spot-check the bit
  ## pattern so a future regression in the packer is caught visibly.
  var w = initByteWriter()
  writeVarint(w, 0)
  writeVarint(w, 1)
  writeVarint(w, 127)
  let bs = w.take()
  doAssert bs == @[byte 0x01, 0x03, 0xFF]

block varint_two_byte_layout:
  ## 2-byte varint: low two bits = 0b10, top 14 bits = value.
  var w = initByteWriter()
  writeVarint(w, 128)  # smallest 2-byte value
  let bs = w.take()
  doAssert bs.len == 2
  # 128 << 2 | 0b10 = 0x202 → little-endian bytes 0x02 0x02
  doAssert bs == @[byte 0x02, 0x02]

block varint_nine_byte_layout:
  var w = initByteWriter()
  writeVarint(w, 1'u64 shl 56)
  let bs = w.take()
  doAssert bs.len == 9
  doAssert bs[0] == 0x00'u8
  # next 8 bytes are 1<<56 in LE = 0,0,0,0,0,0,0,1
  doAssert bs[1 .. 7] == @[byte 0, 0, 0, 0, 0, 0, 0]
  doAssert bs[8] == 0x01'u8

block varint_roundtrip_boundaries:
  let cases = @[
    0'u64, 1, 2, 126, 127, 128, 129, 255, 256,
    (1'u64 shl 14) - 1, 1'u64 shl 14,
    (1'u64 shl 21) - 1, 1'u64 shl 21,
    (1'u64 shl 28) - 1, 1'u64 shl 28,
    (1'u64 shl 35) - 1, 1'u64 shl 35,
    (1'u64 shl 42) - 1, 1'u64 shl 42,
    (1'u64 shl 49) - 1, 1'u64 shl 49,
    (1'u64 shl 56) - 1, 1'u64 shl 56,
    1'u64 shl 63, high(uint64),
  ]
  for v in cases:
    var w = initByteWriter()
    writeVarint(w, v)
    doAssert w.len == varintByteCount(v),
      "size mismatch for " & $v & ": wrote " & $w.len &
      " expected " & $varintByteCount(v)
    let bs = w.take()
    var pos = 0
    let got = readVarint(bs, pos)
    doAssert got == v, "roundtrip mismatch: wrote " & $v & " read " & $got
    doAssert pos == bs.len, "consumed " & $pos & " of " & $bs.len & " bytes"

block varint_consecutive_decode:
  ## Pack three values back-to-back; the decoder must advance correctly
  ## through all of them.
  var w = initByteWriter()
  writeVarint(w, 7)
  writeVarint(w, 200)
  writeVarint(w, 1'u64 shl 40)
  let bs = w.take()
  var pos = 0
  doAssert readVarint(bs, pos) == 7
  doAssert readVarint(bs, pos) == 200
  doAssert readVarint(bs, pos) == 1'u64 shl 40
  doAssert pos == bs.len

block varint_truncation_raises:
  let bs = @[byte 0x02]  # claims 2-byte form but only 1 byte present
  var pos = 0
  var raised = false
  try:
    discard readVarint(bs, pos)
  except BytecodeError:
    raised = true
  doAssert raised

block varint_nine_byte_truncation_raises:
  let bs = @[byte 0x00, 0x01]  # 9-byte form, only 2 bytes
  var pos = 0
  var raised = false
  try:
    discard readVarint(bs, pos)
  except BytecodeError:
    raised = true
  doAssert raised

# ---- zigzag --------------------------------------------------------------

block zigzag_known_values:
  doAssert zigzagEncode(0) == 0'u64
  doAssert zigzagEncode(-1) == 1'u64
  doAssert zigzagEncode(1) == 2'u64
  doAssert zigzagEncode(-2) == 3'u64
  doAssert zigzagEncode(2) == 4'u64
  doAssert zigzagEncode(high(int64)) == cast[uint64](high(int64)) * 2'u64
  doAssert zigzagDecode(0) == 0
  doAssert zigzagDecode(1) == -1
  doAssert zigzagDecode(2) == 1

block signed_varint_roundtrip:
  let cases = @[
    int64 0, 1, -1, 2, -2, 63, -64, 64, -65, 127, -128,
    high(int64), low(int64), low(int64) + 1,
  ]
  for v in cases:
    var w = initByteWriter()
    writeSignedVarint(w, v)
    let bs = w.take()
    var pos = 0
    let got = readSignedVarint(bs, pos)
    doAssert got == v, "signed roundtrip mismatch: " & $v & " → " & $got
    doAssert pos == bs.len

# ---- header --------------------------------------------------------------

block magic_constant_value:
  doAssert MlirBytecodeMagic == [byte 0x4D, 0x4C, 0xEF, 0x52]  # ML\xefR

block header_roundtrip:
  var w = initByteWriter()
  writeMagic(w)
  writeVersion(w, 6)
  writeProducer(w, "rew/0.1.0")
  let bs = w.take()
  doAssert bs[0 .. 3] == @[byte 0x4D, 0x4C, 0xEF, 0x52]
  var pos = 0
  let (version, producer) = readMagicAndVersion(bs, pos)
  doAssert version == 6
  doAssert producer == "rew/0.1.0"
  doAssert pos == bs.len

block header_wrong_magic_raises:
  let bs = @[byte 0x4D, 0x4C, 0x58, 0x52, 0x01]  # 'M','L','X','R',0x01
  var pos = 0
  var raised = false
  try:
    discard readMagicAndVersion(bs, pos)
  except BytecodeError:
    raised = true
  doAssert raised
