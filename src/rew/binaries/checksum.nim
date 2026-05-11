## Checksum — self-contained SHA-256 implementation for archive integrity.
##
## Zero external dependencies. Implements FIPS 180-4 SHA-256 from scratch
## so rew stays dependency-free. Operates on files via streaming reads
## (64 KiB chunks).

import std/[strutils]

type
  Sha256Ctx = object
    state: array[8, uint32]
    count: uint64
    buffer: array[64, byte]
    bufLen: int

const
  K256: array[64, uint32] = [
    0x428a2f98'u32, 0x71374491'u32, 0xb5c0fbcf'u32, 0xe9b5dba5'u32,
    0x3956c25b'u32, 0x59f111f1'u32, 0x923f82a4'u32, 0xab1c5ed5'u32,
    0xd807aa98'u32, 0x12835b01'u32, 0x243185be'u32, 0x550c7dc3'u32,
    0x72be5d74'u32, 0x80deb1fe'u32, 0x9bdc06a7'u32, 0xc19bf174'u32,
    0xe49b69c1'u32, 0xefbe4786'u32, 0x0fc19dc6'u32, 0x240ca1cc'u32,
    0x2de92c6f'u32, 0x4a7484aa'u32, 0x5cb0a9dc'u32, 0x76f988da'u32,
    0x983e5152'u32, 0xa831c66d'u32, 0xb00327c8'u32, 0xbf597fc7'u32,
    0xc6e00bf3'u32, 0xd5a79147'u32, 0x06ca6351'u32, 0x14292967'u32,
    0x27b70a85'u32, 0x2e1b2138'u32, 0x4d2c6dfc'u32, 0x53380d13'u32,
    0x650a7354'u32, 0x766a0abb'u32, 0x81c2c92e'u32, 0x92722c85'u32,
    0xa2bfe8a1'u32, 0xa81a664b'u32, 0xc24b8b70'u32, 0xc76c51a3'u32,
    0xd192e819'u32, 0xd6990624'u32, 0xf40e3585'u32, 0x106aa070'u32,
    0x19a4c116'u32, 0x1e376c08'u32, 0x2748774c'u32, 0x34b0bcb5'u32,
    0x391c0cb3'u32, 0x4ed8aa4a'u32, 0x5b9cca4f'u32, 0x682e6ff3'u32,
    0x748f82ee'u32, 0x78a5636f'u32, 0x84c87814'u32, 0x8cc70208'u32,
    0x90befffa'u32, 0xa4506ceb'u32, 0xbef9a3f7'u32, 0xc67178f2'u32,
  ]

func rotr(x: uint32; n: int): uint32 {.inline.} =
  (x shr n) or (x shl (32 - n))

func ch(x, y, z: uint32): uint32 {.inline.} =
  (x and y) xor ((not x) and z)

func maj(x, y, z: uint32): uint32 {.inline.} =
  (x and y) xor (x and z) xor (y and z)

func sigma0(x: uint32): uint32 {.inline.} =
  rotr(x, 2) xor rotr(x, 13) xor rotr(x, 22)

func sigma1(x: uint32): uint32 {.inline.} =
  rotr(x, 6) xor rotr(x, 11) xor rotr(x, 25)

func gamma0(x: uint32): uint32 {.inline.} =
  rotr(x, 7) xor rotr(x, 18) xor (x shr 3)

func gamma1(x: uint32): uint32 {.inline.} =
  rotr(x, 17) xor rotr(x, 19) xor (x shr 10)

proc transform(ctx: var Sha256Ctx; blk: openArray[byte]) =
  var w: array[64, uint32]
  for i in 0 ..< 16:
    w[i] = (uint32(blk[i * 4]) shl 24) or (uint32(blk[i * 4 + 1]) shl 16) or
            (uint32(blk[i * 4 + 2]) shl 8) or uint32(blk[i * 4 + 3])
  for i in 16 ..< 64:
    w[i] = gamma1(w[i - 2]) + w[i - 7] + gamma0(w[i - 15]) + w[i - 16]
  var a = ctx.state[0]
  var b = ctx.state[1]
  var c = ctx.state[2]
  var d = ctx.state[3]
  var e = ctx.state[4]
  var f = ctx.state[5]
  var g = ctx.state[6]
  var h = ctx.state[7]
  for i in 0 ..< 64:
    let t1 = h + sigma1(e) + ch(e, f, g) + K256[i] + w[i]
    let t2 = sigma0(a) + maj(a, b, c)
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2
  ctx.state[0] += a
  ctx.state[1] += b
  ctx.state[2] += c
  ctx.state[3] += d
  ctx.state[4] += e
  ctx.state[5] += f
  ctx.state[6] += g
  ctx.state[7] += h

proc initSha256(): Sha256Ctx =
  result.state = [
    0x6a09e667'u32, 0xbb67ae85'u32, 0x3c6ef372'u32, 0xa54ff53a'u32,
    0x510e527f'u32, 0x9b05688c'u32, 0x1f83d9ab'u32, 0x5be0cd19'u32,
  ]
  result.count = 0
  result.bufLen = 0

proc update(ctx: var Sha256Ctx; data: openArray[byte]) =
  var offset = 0
  var remaining = data.len
  ctx.count += uint64(remaining)
  if ctx.bufLen > 0:
    let space = 64 - ctx.bufLen
    let take = min(space, remaining)
    copyMem(addr ctx.buffer[ctx.bufLen], unsafeAddr data[offset], take)
    ctx.bufLen += take
    offset += take
    remaining -= take
    if ctx.bufLen == 64:
      ctx.transform(ctx.buffer)
      ctx.bufLen = 0
  while remaining >= 64:
    ctx.transform(data.toOpenArray(offset, offset + 63))
    offset += 64
    remaining -= 64
  if remaining > 0:
    copyMem(addr ctx.buffer[0], unsafeAddr data[offset], remaining)
    ctx.bufLen = remaining

proc finalize(ctx: var Sha256Ctx): array[32, byte] =
  let bitLen = ctx.count * 8
  var pad: array[1, byte] = [0x80'u8]
  ctx.update(pad)
  while ctx.bufLen != 56:
    var zero: array[1, byte] = [0x00'u8]
    ctx.update(zero)
  var lenBuf: array[8, byte]
  for i in 0 ..< 8:
    lenBuf[7 - i] = byte((bitLen shr (i * 8)) and 0xFF)
  ctx.update(lenBuf)
  for i in 0 ..< 8:
    result[i * 4] = byte((ctx.state[i] shr 24) and 0xFF)
    result[i * 4 + 1] = byte((ctx.state[i] shr 16) and 0xFF)
    result[i * 4 + 2] = byte((ctx.state[i] shr 8) and 0xFF)
    result[i * 4 + 3] = byte(ctx.state[i] and 0xFF)

proc sha256Hex*(data: openArray[byte]): string =
  ## Returns the lowercase hex SHA-256 digest of `data`.
  var ctx = initSha256()
  ctx.update(data)
  let digest = ctx.finalize()
  result = newStringOfCap(64)
  for b in digest:
    result.add toLowerAscii(toHex(int(b), 2))

proc sha256File*(path: string): string =
  ## Returns the lowercase hex SHA-256 digest of the file at `path`.
  ## Reads in 64 KiB chunks to keep memory usage bounded.
  var ctx = initSha256()
  var f = open(path, fmRead)
  defer: f.close()
  var buf: array[65536, byte]
  while true:
    let n = f.readBytes(buf, 0, buf.len)
    if n == 0: break
    ctx.update(buf.toOpenArray(0, n - 1))
  let digest = ctx.finalize()
  result = newStringOfCap(64)
  for b in digest:
    result.add toLowerAscii(toHex(int(b), 2))
