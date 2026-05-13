## Tests for the NumPy `.npy` v1.0 reader/writer.

import std/[os, streams]
import rew

proc tmpPath(name: string): string =
  getTempDir() / ("rew_serialize_" & name)

block float32_2x3_round_trips_through_file:
  var src = newSeq[byte](6 * 4)
  let values = [1.0'f32, 2.0'f32, 3.0'f32, 4.0'f32, 5.0'f32, 6.0'f32]
  copyMem(addr src[0], unsafeAddr values[0], src.len)
  let arr = initNpyArray(dtFloat32, [2, 3], src)
  let path = tmpPath("f32.npy")
  saveNpy(path, arr)
  defer: removeFile(path)
  let loaded = loadNpy(path)
  doAssert loaded.dtype == dtFloat32
  doAssert loaded.shape == @[2, 3]
  doAssert loaded.data == arr.data

block uint8_1d_round_trip_preserves_trailing_comma_shape:
  var src = newSeq[byte](5)
  for i in 0 ..< src.len: src[i] = byte(10 + i)
  let arr = initNpyArray(dtUint8, [5], src)
  let path = tmpPath("u8.npy")
  saveNpy(path, arr)
  defer: removeFile(path)
  let loaded = loadNpy(path)
  doAssert loaded.dtype == dtUint8
  doAssert loaded.shape == @[5]
  doAssert loaded.data == arr.data

block scalar_0d_round_trip:
  var src = newSeq[byte](8)
  let v = 1234567890'i64
  copyMem(addr src[0], unsafeAddr v, src.len)
  let arr = initNpyArray(dtInt64, [], src)
  let path = tmpPath("scalar.npy")
  saveNpy(path, arr)
  defer: removeFile(path)
  let loaded = loadNpy(path)
  doAssert loaded.dtype == dtInt64
  doAssert loaded.shape == newSeq[int]()
  doAssert loaded.data == arr.data

block bad_magic_raises_npy_error:
  let path = tmpPath("bad.npy")
  writeFile(path, "NOTNUMPY")
  defer: removeFile(path)
  var raised = false
  try:
    discard loadNpy(path)
  except NpyError:
    raised = true
  doAssert raised

block fortran_order_true_is_rejected:
  # Hand-roll a header that would otherwise parse fine.
  let path = tmpPath("fortran.npy")
  let s = newFileStream(path, fmWrite)
  let dict = "{'descr': '<f4', 'fortran_order': True, 'shape': (2,), }   \n"
  s.write("\x93NUMPY")
  s.write(uint8 1)
  s.write(uint8 0)
  s.write(uint16 dict.len)
  s.write(dict)
  s.write(uint64 0)
  s.close()
  defer: removeFile(path)
  var raised = false
  try:
    discard loadNpy(path)
  except NpyError:
    raised = true
  doAssert raised

block size_mismatch_in_init_npy_array_is_rejected:
  var bytes = newSeq[byte](7)
  var raised = false
  try:
    discard initNpyArray(dtFloat32, [2, 3], bytes)
  except NpyError:
    raised = true
  doAssert raised

block negative_shape_in_init_npy_array_is_rejected:
  var bytes: seq[byte] = @[]
  doAssertRaises(NpyError):
    discard initNpyArray(dtFloat32, [-1], bytes)

block negative_shape_in_save_npy_is_rejected:
  let arr = NpyArray(dtype: dtFloat32, shape: @[-1], data: @[])
  let s = newStringStream()
  doAssertRaises(NpyError):
    saveNpy(s, arr)
