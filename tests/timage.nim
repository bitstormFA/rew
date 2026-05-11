## Image transforms — tests for geometry, color, Compose, and dataset combinators.

import rew

proc makeImage(data: seq[float32]; shape: seq[int]; label: int = -1): Image =
  Image(data: data, shape: shape, label: label)

proc collect[T](ds: Dataset[T]): seq[T] =
  let it = ds.source()
  while true:
    let item = it()
    if finished(it): break
    result.add(item)

# ---- Image construction -----------------------------------------------------

block image_basic:
  let img = Image(data: @[0.5'f32, 0.3, 0.8, 0.1, 0.4, 0.9],
                   shape: @[3, 1, 2], label: 5)
  doAssert img.shape == @[3, 1, 2]
  doAssert img.label == 5
  doAssert img.data[0] == 0.5'f32
  doAssert img.data[5] == 0.9'f32

# ---- resize -----------------------------------------------------------------

block resize_identity_bilinear:
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0,
    4.0'f32, 5.0, 6.0, 7.0,
    8.0'f32, 9.0, 10.0, 11.0,
  ], @[3, 2, 2])
  let result = resize(img, (2, 2), ipBilinear)
  doAssert result.shape == @[3, 2, 2]
  doAssert result.data.len == 12

block resize_downsample_nearest:
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0,
    4.0'f32, 5.0, 6.0, 7.0,
  ], @[2, 2, 2])
  let result = resize(img, (1, 1), ipNearest)
  doAssert result.shape == @[2, 1, 1]

block resize_upsample_bilinear:
  let img = makeImage(@[
    0.0'f32, 0.5, 1.0, 0.5,
  ], @[2, 1, 2])
  let result = resize(img, (2, 4), ipBilinear)
  doAssert result.shape == @[2, 2, 4]

block resize_bicubic:
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0,
    4.0'f32, 5.0, 6.0, 7.0,
    8.0'f32, 9.0, 10.0, 11.0,
  ], @[3, 2, 2])
  let result = resize(img, (3, 3), ipBicubic)
  doAssert result.shape == @[3, 3, 3]
  doAssert result.data.len == 3 * 3 * 3

block resize_invalid_shape:
  var caught = false
  try:
    discard resize(Image(shape: @[0, 1, 2]), (2, 2))
  except ValueError:
    caught = true
  doAssert caught

# ---- centerCrop -------------------------------------------------------------

block centerCrop_basic:
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0,
    4.0'f32, 5.0, 6.0, 7.0,
    8.0'f32, 9.0, 10.0, 11.0,
  ], @[3, 2, 2])
  let result = centerCrop(img, (1, 1))
  doAssert result.shape == @[3, 1, 1]

block centerCrop_preserves_label:
  let img = makeImage(@[0.0'f32, 1.0, 2.0, 3.0], @[2, 1, 2], label = 7)
  let result = centerCrop(img, (1, 1))
  doAssert result.label == 7

# ---- crop -------------------------------------------------------------------

block crop_explicit:
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
    5.0'f32, 6.0, 7.0, 8.0,
  ], @[2, 2, 2])
  let result = crop(img, 0, 1, (2, 1))
  doAssert result.shape == @[2, 2, 1]

# ---- randomCrop (deterministic) ---------------------------------------------

block randomCrop_deterministic:
  let key = initKey(123'u64)
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  ], @[1, 3, 3])
  let r1 = randomCrop(img, key, (2, 2))
  let r2 = randomCrop(img, key, (2, 2))
  doAssert r1.data == r2.data
  doAssert r1.shape == r2.shape

# ---- randomResizedCrop ------------------------------------------------------

block randomResizedCrop_deterministic:
  let key = initKey(42'u64)
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
  ], @[1, 3, 3])
  let r1 = randomResizedCrop(img, key, (2, 2))
  let r2 = randomResizedCrop(img, key, (2, 2))
  doAssert r1.shape == r2.shape
  doAssert r1.data == r2.data

# ---- horizontalFlip / verticalFlip ------------------------------------------

block horizontalFlip_symmetry:
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
    5.0'f32, 6.0, 7.0, 8.0,
  ], @[2, 2, 2])
  let flipped = horizontalFlip(img)
  doAssert flipped.data[0] == 2.0'f32
  doAssert flipped.data[1] == 1.0'f32

block horizontalFlip_double_restore:
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
  ], @[2, 1, 2])
  let flipped = horizontalFlip(horizontalFlip(img))
  doAssert flipped.data == img.data

block verticalFlip_symmetry:
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
    5.0'f32, 6.0, 7.0, 8.0,
  ], @[2, 2, 2])
  let flipped = verticalFlip(img)
  # (c=0, y=0, x=0) was (c=0, y=1, x=0) = 3.0
  doAssert flipped.data[0] == 3.0'f32

block verticalFlip_double_restore:
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
    5.0'f32, 6.0, 7.0, 8.0,
  ], @[4, 1, 2])
  let flipped = verticalFlip(verticalFlip(img))
  doAssert flipped.data == img.data

# ---- randomHorizontalFlip / randomVerticalFlip ------------------------------

block randomHorizontalFlip_deterministic:
  let key = initKey(999'u64)
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
    5.0'f32, 6.0, 7.0, 8.0,
  ], @[2, 2, 2])
  let r1 = randomHorizontalFlip(img, key)
  let r2 = randomHorizontalFlip(img, key)
  doAssert r1.data == r2.data

block randomVerticalFlip_deterministic:
  let key = initKey(888'u64)
  let img = makeImage(@[
    1.0'f32, 2.0, 3.0, 4.0,
  ], @[2, 1, 2])
  let r1 = randomVerticalFlip(img, key)
  let r2 = randomVerticalFlip(img, key)
  doAssert r1.data == r2.data

# ---- pad --------------------------------------------------------------------

block pad_symmetric_int:
  let img = makeImage(@[
    0.1'f32, 0.2, 0.3, 0.4,
  ], @[1, 2, 2])
  let result = pad(img, 1, 0.0)
  doAssert result.shape == @[1, 4, 4]

block pad_asymmetric:
  let img = makeImage(@[
    0.5'f32, 1.0,
  ], @[1, 1, 2])
  let result = pad(img, @[1, 0, 2, 0], 0.0)
  doAssert result.shape == @[1, 1, 5]
  doAssert result.data[0] == 0.0'f32  # left pad
  doAssert result.data[1] == 0.5'f32  # original
  doAssert result.data[2] == 1.0'f32  # original
  doAssert result.data[3] == 0.0'f32  # right pad 1
  doAssert result.data[4] == 0.0'f32  # right pad 2

# ---- rotate -----------------------------------------------------------------

block rotate_90:
  let img = makeImage(@[
    1.0'f32, 0.0, 0.0, 0.0,
  ], @[1, 2, 2])
  let result = rotate(img, 90.0, ipNearest)
  doAssert result.shape == @[1, 2, 2]

block rotate_identity:
  let img = makeImage(@[
    0.5'f32, 1.0, 1.0, 0.5,
  ], @[1, 2, 2])
  let result = rotate(img, 0.0, ipBilinear)
  doAssert result.shape == @[1, 2, 2]

block randomRotation_deterministic:
  let key = initKey(77'u64)
  let img = makeImage(@[
    0.0'f32, 1.0, 2.0, 3.0,
    4.0'f32, 5.0, 6.0, 7.0,
    8.0'f32, 9.0, 10.0, 11.0,
  ], @[3, 2, 2])
  let r1 = randomRotation(img, key, 30.0)
  let r2 = randomRotation(img, key, 30.0)
  doAssert r1.data == r2.data

# ---- normalize --------------------------------------------------------------

block normalize_basic:
  let img = makeImage(@[
    0.5'f32, 1.0, 0.5, 1.0,
  ], @[2, 1, 2])
  let result = normalize(img, [0.0'f32, 0.0], [1.0'f32, 0.5])
  doAssert result.data[0] == 0.5'f32
  doAssert result.data[1] == 1.0'f32
  doAssert abs(result.data[2] - 1.0'f32) < 0.001
  doAssert abs(result.data[3] - 2.0'f32) < 0.001

# ---- grayscale --------------------------------------------------------------

block grayscale_3_to_1:
  let img = makeImage(@[
    1.0'f32, 0.0, 0.0, 0.0,
    0.0'f32, 1.0, 0.0, 0.0,
  ], @[3, 1, 2])
  let result = grayscale(img, 1)
  doAssert result.shape == @[1, 1, 2]

block grayscale_3_to_3:
  let img = makeImage(@[
    1.0'f32, 0.0, 0.0, 0.0,
    0.0'f32, 1.0, 0.0, 0.0,
  ], @[3, 1, 2])
  let result = grayscale(img, 3)
  doAssert result.shape == @[3, 1, 2]

block grayscale_luminance:
  let img = makeImage(@[
    1.0'f32, 0.0,  # R
    0.0'f32, 0.0,  # G
    0.0'f32, 0.0,  # B
  ], @[3, 1, 2])
  let result = grayscale(img, 1)
  doAssert abs(result.data[0] - 0.299'f32) < 0.001

# ---- randomInvert -----------------------------------------------------------

block randomInvert_deterministic:
  let key = initKey(555'u64)
  let img = makeImage(@[0.2'f32, 0.8], @[1, 1, 2])
  let r1 = randomInvert(img, key)
  let r2 = randomInvert(img, key)
  doAssert r1.data == r2.data

# ---- adjustBrightness -------------------------------------------------------

block adjustBrightness_identity:
  let img = makeImage(@[0.5'f32, 1.0, 0.25], @[3, 1, 1])
  let result = adjustBrightness(img, 1.0)
  doAssert result.data == img.data

block adjustBrightness_double:
  let img = makeImage(@[0.2'f32, 0.4], @[2, 1, 1])
  let result = adjustBrightness(img, 2.0)
  doAssert result.data[0] == 0.4'f32

block adjustBrightness_clamp:
  let img = makeImage(@[0.8'f32], @[1, 1, 1])
  let result = adjustBrightness(img, 3.0)
  doAssert result.data[0] == 1.0'f32

# ---- adjustContrast ---------------------------------------------------------

block adjustContrast_identity:
  let img = makeImage(@[0.5'f32, 0.75], @[2, 1, 1])
  let result = adjustContrast(img, 1.0)
  doAssert result.data == img.data

block adjustContrast_zero:
  let img = makeImage(@[0.2'f32, 0.8], @[2, 1, 1])
  let result = adjustContrast(img, 0.0)
  for v in result.data:
    doAssert abs(v - 0.5'f32) < 0.001

# ---- adjustSaturation -------------------------------------------------------

block adjustSaturation_identity:
  let img = makeImage(@[
    1.0'f32, 0.0, 0.0, 0.0, 0.0, 1.0,
  ], @[3, 1, 2])
  let result = adjustSaturation(img, 1.0)
  doAssert result.data == img.data

block adjustSaturation_zero:
  let img = makeImage(@[
    1.0'f32, 0.0, 0.0, 0.0, 0.0, 1.0,
  ], @[3, 1, 2])
  let result = adjustSaturation(img, 0.0)
  # Same pixel across channels should be equal (all become gray)
  doAssert abs(result.data[0] - result.data[2]) < 0.001
  doAssert abs(result.data[2] - result.data[4]) < 0.001

# ---- colorJitter ------------------------------------------------------------

block colorJitter_deterministic:
  let key = initKey(333'u64)
  let img = makeImage(@[
    0.5'f32, 0.2, 0.8, 0.3, 0.7, 0.1,
  ], @[3, 1, 2])
  let r1 = colorJitter(img, key, brightness = (0.5'f32, 1.5'f32), contrast = (0.5'f32, 1.5'f32))
  let r2 = colorJitter(img, key, brightness = (0.5'f32, 1.5'f32), contrast = (0.5'f32, 1.5'f32))
  doAssert r1.data == r2.data

# ---- gaussianBlur -----------------------------------------------------------

block gaussianBlur_identity:
  let img = makeImage(@[
    0.5'f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
  ], @[1, 3, 3])
  let result = gaussianBlur(img, 3, 1.0)
  doAssert result.shape == img.shape
  # Center pixel of a uniform image should stay near 0.5
  # (edge pixels may deviate due to zero-padding)
  # Center index: (c=0, y=1, x=1) = ((0*3)+1)*3+1 = 4
  doAssert abs(result.data[4] - 0.5'f32) < 0.01

block gaussianBlur_invalid_kernel:
  var caught = false
  try:
    discard gaussianBlur(Image(shape: @[1, 2, 2]), 2, 1.0)
  except ValueError:
    caught = true
  doAssert caught

# ---- gaussianNoise ----------------------------------------------------------

block gaussianNoise_deterministic:
  let key = initKey(111'u64)
  let img = makeImage(@[0.5'f32, 0.5, 0.5, 0.5], @[1, 2, 2])
  let r1 = gaussianNoise(img, key, 0.0, 0.01)
  let r2 = gaussianNoise(img, key, 0.0, 0.01)
  doAssert r1.data == r2.data

block gaussianNoise_noclip:
  let key = initKey(222'u64)
  let img = makeImage(@[0.0'f32], @[1, 1, 1])
  let result = gaussianNoise(img, key, 0.0, 0.5, clip = false)
  # With large sigma, values may leave [0,1] when not clipping
  discard  # just verify shape is preserved
  doAssert result.shape == img.shape

# ---- toDtype ----------------------------------------------------------------

block toDtype_float32_scale:
  let img = makeImage(@[255.0'f32, 128.0, 0.0], @[3, 1, 1])
  let result = toDtype(img, dtFloat32, scale = true)
  doAssert abs(result.data[0] - 1.0'f32) < 0.01
  doAssert abs(result.data[1] - 0.502'f32) < 0.01
  doAssert abs(result.data[2] - 0.0'f32) < 0.01

block toDtype_uint8_scale:
  let img = makeImage(@[1.0'f32, 0.5, 0.0], @[3, 1, 1])
  let result = toDtype(img, dtUint8, scale = true)
  doAssert abs(result.data[0] - 255.0'f32) < 0.01
  doAssert abs(result.data[1] - 128.0'f32) < 1.01  # rounding
  doAssert abs(result.data[2] - 0.0'f32) < 0.01

# ---- Compose ----------------------------------------------------------------

block compose_basic:
  let img = makeImage(@[
    0.0'f32, 0.5, 0.0, 0.5,
  ], @[2, 1, 2])
  let comp = initCompose(@[
    proc(img: Image): Image = adjustBrightness(img, 2.0'f32),
  ])
  let result = comp.apply(img)
  doAssert result.data[0] == 0.0'f32
  doAssert result.data[1] == 1.0'f32

block compose_pipeline:
  let img = makeImage(@[
    0.5'f32, 1.0, 0.5, 1.0,
  ], @[2, 1, 2])
  let comp = initCompose(@[
    proc(img: Image): Image = resize(img, (2, 2), ipNearest),
    proc(img: Image): Image = normalize(img, [0.0'f32, 0.5], [1.0'f32, 1.0]),
  ])
  let result = comp.apply(img)
  doAssert result.shape == @[2, 2, 2]

# ---- Dataset combinators ----------------------------------------------------

block ds_resize:
  let samples = @[
    Sample(data: @[0.0'f32, 1.0, 2.0, 3.0], dataShape: @[1, 2, 2], label: 0),
  ]
  let ds = fromSeq(samples).resize((1, 1))
  let results = collect(ds)
  doAssert results.len == 1
  doAssert results[0].dataShape == @[1, 1, 1]

block ds_centerCrop:
  let samples = @[
    Sample(data: @[0.0'f32, 1.0, 2.0, 3.0], dataShape: @[1, 2, 2], label: 0),
  ]
  let ds = fromSeq(samples).centerCrop((1, 1))
  let results = collect(ds)
  doAssert results.len == 1
  doAssert results[0].dataShape == @[1, 1, 1]

block ds_normalize:
  let samples = @[
    Sample(data: @[0.5'f32, 1.0], dataShape: @[2, 1, 1], label: 0),
  ]
  let ds = fromSeq(samples).normalize([0.0'f32, 0.5], [1.0'f32, 1.0])
  let results = collect(ds)
  doAssert abs(results[0].data[0] - 0.5'f32) < 0.001
  doAssert abs(results[0].data[1] - 0.5'f32) < 0.001

block ds_grayscale:
  let samples = @[
    Sample(data: @[1.0'f32, 0.0, 0.0, 0.0, 0.0, 1.0], dataShape: @[3, 1, 2],
      label: 0),
  ]
  let ds = fromSeq(samples).grayscale()
  let results = collect(ds)
  doAssert results.len == 1
  doAssert results[0].dataShape == @[1, 1, 2]

block ds_randomHorizontalFlip_deterministic:
  let key = initKey(100'u64)
  let samples = @[
    Sample(data: @[1.0'f32, 2.0, 3.0, 4.0], dataShape: @[1, 2, 2], label: 0),
  ]
  let ds = fromSeq(samples).randomHorizontalFlip(key)
  let r1 = collect(ds)
  let r2 = collect(ds)
  doAssert r1 == r2

echo "timage: all passed"
