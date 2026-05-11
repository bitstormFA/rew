## NumPy `.npy` v1.0 reader and writer for host byte buffers.
##
## Scope: just enough I/O to bootstrap MNIST and other example datasets
## without pulling Python into the loop. The header is a tiny Python-dict
## literal — we parse the three keys we care about (`descr`, `fortran_order`,
## `shape`) and reject anything fancier (object dtype, structured arrays,
## byte order other than little-endian or native, Fortran order).
##
## v1 reads/writes the **raw byte buffer** plus dtype + shape; lifting that
## into a `Tensor` is the eager backend's job (`transferToDevice` after
## `loadNpy`).
##
## See the spec at
## https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html.

import std/[streams, strutils, json, os]
import ./dtype
import ./tensor
import ./pytree
import ./eager
import ./device

type
  NpyError* = object of CatchableError
    ## Raised on malformed `.npy` data or unsupported dtype/layout.

  NpyArray* = object
    ## Result of `loadNpy`. `data` holds the raw host bytes in C
    ## (row-major) order; its length equals `dt.byteSize * prod(shape)`.
    dtype*: DType
    shape*: seq[int]
    data*: seq[byte]

const
  NpyMagic = "\x93NUMPY"
  HeaderAlignment = 64

# ---- dtype <-> numpy descriptor mapping --------------------------------

func descrOf(dt: DType): string =
  ## Returns the little-endian numpy descriptor string for `dt`. Raises
  ## `NpyError` for dtypes we don't yet write.
  case dt
  of dtBool:    "|b1"
  of dtInt8:    "|i1"
  of dtUint8:   "|u1"
  of dtInt16:   "<i2"
  of dtUint16:  "<u2"
  of dtFloat16: "<f2"
  of dtInt32:   "<i4"
  of dtUint32:  "<u4"
  of dtFloat32: "<f4"
  of dtInt64:   "<i8"
  of dtUint64:  "<u8"
  of dtFloat64: "<f8"
  of dtComplex64: "<c8"
  of dtComplex128: "<c16"
  of dtBFloat16:
    raise newException(NpyError,
      "loadNpy/saveNpy: bfloat16 is not representable in numpy v1.0")
  of dtInt4, dtUint4, dtNF4, dtFloat8E4M3Fn, dtFloat8E5M2:
    raise newException(NpyError,
      "loadNpy/saveNpy: " & dt.name & " is not representable in numpy v1.0")

func dtypeOfDescr(descr: string): DType =
  ## Parses a numpy dtype descriptor (e.g. `"<f4"`, `"|u1"`). Native byte
  ## order (`"="`) is accepted on little-endian hosts only.
  if descr.len < 2:
    raise newException(NpyError,
      "loadNpy: dtype descriptor too short: '" & descr & "'")
  let endian = descr[0]
  case endian
  of '<', '|':
    discard
  of '=':
    when system.cpuEndian != littleEndian:
      raise newException(NpyError,
        "loadNpy: native-endian descriptor on a big-endian host is " &
          "not supported")
  of '>':
    raise newException(NpyError,
      "loadNpy: big-endian arrays are not supported (descriptor '" &
        descr & "')")
  else:
    raise newException(NpyError,
      "loadNpy: unknown byte-order char '" & $endian & "'")
  let kind = descr[1]
  let widthStr = descr[2 .. ^1]
  let width =
    try: parseInt(widthStr)
    except ValueError:
      raise newException(NpyError,
        "loadNpy: bad width in descriptor '" & descr & "'")
  case kind
  of 'b':
    if width == 1: return dtBool
  of 'i':
    case width
    of 1: return dtInt8
    of 2: return dtInt16
    of 4: return dtInt32
    of 8: return dtInt64
    else: discard
  of 'u':
    case width
    of 1: return dtUint8
    of 2: return dtUint16
    of 4: return dtUint32
    of 8: return dtUint64
    else: discard
  of 'f':
    case width
    of 2: return dtFloat16
    of 4: return dtFloat32
    of 8: return dtFloat64
    else: discard
  of 'c':
    case width
    of 8: return dtComplex64
    of 16: return dtComplex128
    else: discard
  else:
    discard
  raise newException(NpyError,
    "loadNpy: unsupported dtype descriptor '" & descr & "'")

# ---- header parsing ----------------------------------------------------

proc fieldValue(header: string; key: string): string =
  ## Extracts the value associated with `'key': ...` from a Python-dict
  ## literal. Returns the substring up to the next top-level comma (depth
  ## tracked across `()`, `[]`, `{}`).
  let needle = "'" & key & "':"
  let kIdx = header.find(needle)
  if kIdx < 0:
    raise newException(NpyError,
      "loadNpy: header missing key '" & key & "'")
  var i = kIdx + needle.len
  while i < header.len and header[i] == ' ': inc i
  var depth = 0
  let start = i
  while i < header.len:
    let c = header[i]
    if depth == 0 and (c == ',' or c == '}'): break
    case c
    of '(', '[', '{': inc depth
    of ')', ']', '}': dec depth
    else: discard
    inc i
  result = header[start ..< i].strip()

func parseShape(s: string): seq[int] =
  ## Parses a Python tuple literal of non-negative ints into a `seq`.
  ## Accepts trailing commas (`(3,)`).
  var t = s.strip()
  if t.len < 2 or t[0] != '(' or t[^1] != ')':
    raise newException(NpyError,
      "loadNpy: shape is not a tuple: '" & s & "'")
  t = t[1 ..< t.len - 1]
  result = @[]
  for part in t.split(','):
    let p = part.strip()
    if p.len == 0: continue
    let v =
      try: parseInt(p)
      except ValueError:
        raise newException(NpyError,
          "loadNpy: bad integer in shape '" & s & "'")
    if v < 0:
      raise newException(NpyError,
        "loadNpy: negative shape element in '" & s & "'")
    result.add v

func parseFortranOrder(s: string): bool =
  case s.strip()
  of "True":  true
  of "False": false
  else:
    raise newException(NpyError,
      "loadNpy: fortran_order must be True or False, got '" & s & "'")

func unquote(s: string): string =
  let t = s.strip()
  if t.len < 2 or (t[0] != '\'' and t[0] != '"') or t[^1] != t[0]:
    raise newException(NpyError,
      "loadNpy: expected quoted string, got '" & s & "'")
  t[1 ..< t.len - 1]

# ---- public API --------------------------------------------------------

proc readNpyHeader(s: Stream): tuple[dtype: DType; shape: seq[int];
    headerLen: int] {.raises: [NpyError, IOError, OSError].} =
  ## Reads and validates the magic + v1.0 header. Returns the dtype,
  ## shape, and total bytes consumed (magic + version + len + header).
  ## The stream is positioned at the first data byte on success.
  var magic = newString(NpyMagic.len)
  if s.readData(addr magic[0], magic.len) != magic.len:
    raise newException(NpyError, "loadNpy: short magic")
  if magic != NpyMagic:
    raise newException(NpyError, "loadNpy: bad magic bytes")
  let major = s.readUint8()
  let minor = s.readUint8()
  if major != 1'u8:
    raise newException(NpyError,
      "loadNpy: only v1.x supported, got v" & $major & "." & $minor)
  let headerLen = int s.readUint16()
  var header = newString(headerLen)
  if headerLen > 0 and s.readData(addr header[0], headerLen) != headerLen:
    raise newException(NpyError, "loadNpy: truncated header")
  let descr = unquote(fieldValue(header, "descr"))
  let dt = dtypeOfDescr(descr)
  if parseFortranOrder(fieldValue(header, "fortran_order")):
    raise newException(NpyError,
      "loadNpy: fortran_order=True is not supported")
  let shape = parseShape(fieldValue(header, "shape"))
  result = (dtype: dt, shape: shape,
            headerLen: NpyMagic.len + 2 + 2 + headerLen)

func elementCount(shape: openArray[int]): int =
  result = 1
  for d in shape: result *= d

proc loadNpy*(s: Stream): NpyArray
    {.raises: [NpyError, IOError, OSError].} =
  ## Reads one `.npy` v1.0 array from `s`. The full data buffer is
  ## materialised in `result.data`.
  let (dt, shape, _) = readNpyHeader(s)
  let n = elementCount(shape)
  let bytes = n * dt.byteSize
  var data = newSeq[byte](bytes)
  if bytes > 0:
    let got = s.readData(addr data[0], bytes)
    if got != bytes:
      raise newException(NpyError,
        "loadNpy: truncated data: expected " & $bytes & " bytes, got " & $got)
  result = NpyArray(dtype: dt, shape: shape, data: data)

proc loadNpy*(path: string): NpyArray
    {.raises: [NpyError, IOError, OSError].} =
  ## Convenience wrapper that opens `path` for reading.
  let s = newFileStream(path, fmRead)
  if s.isNil:
    raise newException(IOError, "loadNpy: cannot open '" & path & "'")
  defer: s.close()
  loadNpy(s)

proc saveNpy*(s: Stream; arr: NpyArray)
    {.raises: [NpyError, IOError, OSError].} =
  ## Writes `arr` as a v1.0 `.npy` array to `s`.
  let n = elementCount(arr.shape)
  if arr.data.len != n * arr.dtype.byteSize:
    raise newException(NpyError,
      "saveNpy: data length " & $arr.data.len & " does not match shape/dtype")
  var shapeStr = "("
  for i, d in arr.shape:
    if i > 0: shapeStr.add ", "
    shapeStr.add $d
  if arr.shape.len == 1: shapeStr.add ","
  shapeStr.add ")"
  var dict = "{'descr': '" & descrOf(arr.dtype) &
    "', 'fortran_order': False, 'shape': " & shapeStr & ", }"
  # Pad with spaces and a trailing newline so that magic + version + len
  # + header is a multiple of `HeaderAlignment` bytes.
  let prefix = NpyMagic.len + 2 + 2
  var totalLen = prefix + dict.len + 1
  let pad = (HeaderAlignment - (totalLen mod HeaderAlignment)) mod HeaderAlignment
  for _ in 0 ..< pad: dict.add ' '
  dict.add '\n'
  if dict.len > high(uint16).int:
    raise newException(NpyError, "saveNpy: header too large for v1.0")
  s.write(NpyMagic)
  s.write(uint8 1)
  s.write(uint8 0)
  s.write(uint16 dict.len)
  s.write(dict)
  if arr.data.len > 0:
    s.writeData(unsafeAddr arr.data[0], arr.data.len)

proc saveNpy*(path: string; arr: NpyArray)
    {.raises: [NpyError, IOError, OSError].} =
  ## Convenience wrapper that opens `path` for writing.
  let s = newFileStream(path, fmWrite)
  if s.isNil:
    raise newException(IOError, "saveNpy: cannot open '" & path & "'")
  defer: s.close()
  saveNpy(s, arr)

func initNpyArray*(dtype: DType; shape: openArray[int];
    data: sink seq[byte]): NpyArray =
  ## Builds an `NpyArray` from existing host bytes after validating the
  ## buffer size against `dtype.byteSize * prod(shape)`.
  let n = elementCount(shape)
  if data.len != n * dtype.byteSize:
    raise newException(NpyError,
      "initNpyArray: data length " & $data.len &
        " does not match dtype/shape (" & $(n * dtype.byteSize) & ")")
  NpyArray(dtype: dtype, shape: @shape, data: data)

# ---- Checkpoint helpers ---------------------------------------------------

type
  CheckpointError* = object of CatchableError

proc saveParams*[P](params: P; path: string; prefix: string = "param") =
  ## Saves `params` to directory `path`. Each pytree leaf tensor is
  ## written as a `.npy` file. A `manifest.json` records dtype, shape,
  ## and file mapping. Requires eager mode (host-side tensor access).
  let leaves = treeFlatten(params)
  createDir(path)
  var fileNames = newSeq[string](leaves.len)
  for i, leaf in leaves:
    let fname = prefix & "_" & $i & ".npy"
    let fullPath = path / fname
    let hostBuf = leaf.toHost()
    let npy = initNpyArray(leaf.dtype, leaf.shape, hostBuf)
    saveNpy(fullPath, npy)
    fileNames[i] = fname
  var manifest = newJObject()
  manifest["version"] = newJInt(1)
  var items = newJArray()
  for i, leaf in leaves:
    var item = newJObject()
    item["file"] = newJString(fileNames[i])
    item["dtype"] = newJString($leaf.dtype)
    item["shape"] = %leaf.shape
    items.add(item)
  manifest["parameters"] = items
  let manifestPath = path / "manifest.json"
  writeFile(manifestPath, $manifest)

proc loadParams*[P](path: string): P =
  ## Loads `params` from directory `path`. Reads `manifest.json`,
  ## loads each `.npy` file, and reconstructs the pytree structure.
  let manifestPath = path / "manifest.json"
  if not fileExists(manifestPath):
    raise newException(CheckpointError,
      "loadParams: manifest.json not found at " & manifestPath)
  let manifest = parseFile(manifestPath)
  let items = manifest["parameters"]
  let dev = defaultDevice()
  var leaves: seq[Tensor] = @[]
  for item in items:
    let fname = item["file"].getStr()
    let fullPath = path / fname
    if not fileExists(fullPath):
      raise newException(CheckpointError,
        "loadParams: file not found: " & fullPath)
    let arr = loadNpy(fullPath)
    var count = 1
    for d in arr.shape:
      count *= d
    if arr.dtype == dtFloat32:
      var data = newSeq[float32](count)
      copyMem(addr data[0], unsafeAddr arr.data[0], arr.data.len)
      let t = fromHostF32(dev, data, arr.shape)
      leaves.add(t)
    else:
      raise newException(CheckpointError,
        "loadParams: unsupported dtype " & $arr.dtype)
  treeUnflatten(result, leaves)
