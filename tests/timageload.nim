## Image loading — tests for PPM/PGM parsing and image-folder sources.

import std/os
import rew

proc makePpmP6(w, h: int; data: seq[uint8]): string =
  ## Create a binary PPM (P6) file in memory.
  result = "P6\n" & $w & " " & $h & "\n255\n"
  for b in data:
    result.add(char(b))

proc makePgmP5(w, h: int; data: seq[uint8]): string =
  ## Create a binary PGM (P5) file in memory.
  result = "P5\n" & $w & " " & $h & "\n255\n"
  for b in data:
    result.add(char(b))

proc writeTestFile(name, content: string): string =
  let dir = getTempDir() / "rew_test_load"
  createDir(dir)
  let path = dir / name
  writeFile(path, content)
  path

# ---- PPM (P6) loading -------------------------------------------------------

block ppm_color_2x1:
  # A 2x1 RGB image: pixel0=(255,128,0), pixel1=(0,0,255)
  # Shape [C=3, H=1, W=2] → channel-first layout:
  # data[0]=R(pix0), data[1]=R(pix1), data[2]=G(pix0), data[3]=G(pix1),
  # data[4]=B(pix0), data[5]=B(pix1)
  let content = makePpmP6(2, 1, @[255'u8, 128, 0, 0, 0, 255])
  let path = writeTestFile("test_2x1.ppm", content)
  let img = loadImage(path)
  doAssert img.shape == @[3, 1, 2]
  doAssert img.data.len == 6
  # R of pixel0 = 255/255 = 1.0
  doAssert abs(img.data[0] - 1.0'f32) < 0.01
  # R of pixel1 = 0/255 = 0.0
  doAssert abs(img.data[1] - 0.0'f32) < 0.01
  # G of pixel0 = 128/255 ≈ 0.502
  doAssert abs(img.data[2] - (128.0'f32/255.0'f32)) < 0.02
  # G of pixel1 = 0/255 = 0.0
  doAssert abs(img.data[3] - 0.0'f32) < 0.01
  # B of pixel0 = 0/255 = 0.0
  doAssert abs(img.data[4] - 0.0'f32) < 0.01
  # B of pixel1 = 255/255 = 1.0
  doAssert abs(img.data[5] - 1.0'f32) < 0.01
  doAssert img.label == -1

block ppm_larger:
  # A 3x2 RGB image: all red (255,0,0)
  var data: seq[uint8] = @[]
  for i in 0 ..< 3*2*3:
    if i mod 3 == 0: data.add(255) else: data.add(0)
  let content = makePpmP6(3, 2, data)
  let path = writeTestFile("test_3x2.ppm", content)
  let img = loadImage(path)
  doAssert img.shape == @[3, 2, 3]
  doAssert img.data.len == 18
  # All Red channel pixels (first 6 elements) should be 1.0
  # data[0..5] = R channel (channel 0, 2x3=6 elements)
  for i in 0 ..< 6:
    doAssert abs(img.data[i] - 1.0'f32) < 0.01
  # Green and Blue channels should be 0.0
  for i in 6 ..< 17:
    doAssert abs(img.data[i] - 0.0'f32) < 0.01

# ---- PPM with PPM extension ------------------------------------------------

block ppm_extension:
  let content = makePpmP6(1, 1, @[255'u8, 255, 255])
  let path = writeTestFile("test_ext.PPM", content)
  let img = loadImage(path)
  doAssert img.shape == @[3, 1, 1]
  doAssert abs(img.data[0] - 1.0'f32) < 0.01

# ---- PGM (P5) loading -------------------------------------------------------

block pgm_grayscale:
  # A 2x2 grayscale image
  let content = makePgmP5(2, 2, @[0'u8, 64, 128, 255])
  let path = writeTestFile("test_2x2.pgm", content)
  let img = loadImage(path)
  doAssert img.shape == @[1, 2, 2]
  doAssert img.data.len == 4
  doAssert abs(img.data[0] - 0.0'f32) < 0.01
  doAssert abs(img.data[1] - 0.25'f32) < 0.02
  doAssert abs(img.data[2] - 0.5'f32) < 0.02
  doAssert abs(img.data[3] - 1.0'f32) < 0.01

block pgm_extension:
  let content = makePgmP5(1, 1, @[128'u8])
  let path = writeTestFile("test_ext.PGM", content)
  let img = loadImage(path)
  doAssert img.shape == @[1, 1, 1]

# ---- loadPpm / loadPgm directly --------------------------------------------

block loadPpm_direct:
  let content = makePpmP6(1, 1, @[100'u8, 150, 200])
  let path = writeTestFile("test_direct.ppm", content)
  let img = loadPpm(path)
  doAssert img.shape == @[3, 1, 1]

block loadPgm_direct:
  let content = makePgmP5(1, 1, @[128'u8])
  let path = writeTestFile("test_direct.pgm", content)
  let img = loadPgm(path)
  doAssert img.shape == @[1, 1, 1]

# ---- loadImage auto-detect --------------------------------------------------

block loadImage_ppm_auto:
  let content = makePpmP6(1, 1, @[255'u8, 255, 255])
  let path = writeTestFile("auto.ppm", content)
  let img = loadImage(path)
  doAssert img.shape == @[3, 1, 1]

block loadImage_unsupported:
  var caught = false
  try:
    let path = writeTestFile("test.txt", "not an image")
    discard loadImage(path)
  except ValueError:
    caught = true
  doAssert caught

# ---- fromImageFiles ---------------------------------------------------------

block fromImageFiles_basic:
  let content = makePpmP6(2, 2, @[
    255'u8, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255,
  ])
  let p1 = writeTestFile("ds1.ppm", content)
  let p2 = writeTestFile("ds2.ppm", content)
  let ds = fromImageFiles(@[p1, p2])
  var items: seq[Sample] = @[]
  for x in ds.items:
    items.add(x)
  doAssert items.len == 2
  doAssert items[0].dataShape == @[3, 2, 2]
  doAssert items[0].label == -1

# ---- fromImageFolder ---------------------------------------------------------

block fromImageFolder_basic:
  let baseDir = getTempDir() / "rew_test_folder"
  removeDir(baseDir)  # clean from any previous run
  createDir(baseDir / "class0")
  createDir(baseDir / "class1")
  let ppm1 = makePpmP6(1, 1, @[255'u8, 0, 0])
  let ppm2 = makePpmP6(1, 1, @[0'u8, 255, 0])
  let ppm3 = makePpmP6(1, 1, @[0'u8, 0, 255])
  writeFile(baseDir / "class0" / "img1.ppm", ppm1)
  writeFile(baseDir / "class0" / "img2.ppm", ppm2)
  writeFile(baseDir / "class0" / "img3.ppm", ppm3)
  writeFile(baseDir / "class1" / "img4.ppm", ppm1)
  writeFile(baseDir / "class1" / "img5.ppm", ppm2)

  let ds = fromImageFolder(baseDir, "*.ppm")
  var items: seq[Sample] = @[]
  for x in ds.items:
    items.add(x)
  doAssert items.len == 5
  doAssert items[0].label == 0  # first 3 from class0
  doAssert items[1].label == 0
  doAssert items[2].label == 0
  doAssert items[3].label == 1  # next 2 from class1
  doAssert items[4].label == 1
  for item in items:
    doAssert item.dataShape == @[3, 1, 1]

  # Cleanup
  removeDir(baseDir)

block fromImageFolder_different_pattern:
  let baseDir = getTempDir() / "rew_test_folder2"
  removeDir(baseDir)
  createDir(baseDir / "cats")
  writeFile(baseDir / "cats" / "img.ppm", makePpmP6(1, 1, @[128'u8, 128, 128]))
  writeFile(baseDir / "cats" / "not_an_image.txt", "hello")

  let ds = fromImageFolder(baseDir, "*.ppm")
  var items: seq[Sample] = @[]
  for x in ds.items:
    items.add(x)
  doAssert items.len == 1  # only the .ppm, not the .txt
  removeDir(baseDir)

# ---- Cleanup test temp dir --------------------------------------------------

removeDir(getTempDir() / "rew_test_load")

echo "timageload: all passed"
