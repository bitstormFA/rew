## Image transform primitives — host-side, pure-Nim geometry and color ops.
##
## All transforms operate on float32 pixel data in [0, 1] with shape [C, H, W].
## Random transforms accept a `Key` for deterministic reproducibility.
##
## Image loading: PPM/PGM (pure Nim), PNG/JPEG via bundled stb_image FFI.

import std/[math, os, strutils]
import ../rng
import ../dtype

type
  Image* = object
    ## A host-side image as a flat float32 buffer with shape [C, H, W].
    ## Pixel values are in [0, 1]. `label` is -1 when unlabeled.
    data*: seq[float32]
    shape*: seq[int]   ## [C, H, W]
    label*: int

  Interpolation* = enum
    ipNearest
    ipBilinear
    ipBicubic

  ImageTransform* = proc(img: Image): Image {.nimcall.}
    ## A pure function from Image to Image.

  Compose* = object
    ## A chain of Image→Image transform procs applied sequentially.
    transforms: seq[ImageTransform]

# ---- helpers ---------------------------------------------------------------

func shapeOk(shape: seq[int]): bool =
  shape.len == 3 and shape[0] >= 1 and shape[1] >= 1 and shape[2] >= 1

func flattenIdx(shape: openArray[int]; c, y, x: int): int =
  let (h, w) = (shape[1], shape[2])
  ((c * h) + y) * w + x

func clampf(v: float32; lo, hi: float32): float32 {.inline.} =
  if v < lo: lo elif v > hi: hi else: v

func lerpf(a, b, t: float32): float32 {.inline.} =
  a + (b - a) * t

func pixelBilinear(img: Image; fy, fx: float32; c: int): float32 =
  let (h, w) = (img.shape[1], img.shape[2])
  if fy < -0.5 or fy > float32(h) - 0.5 or fx < -0.5 or fx > float32(w) - 0.5:
    return 0.0'f32
  let
    yy = fy - 0.5'f32
    xx = fx - 0.5'f32
    y0 = max(0, int(floor(yy)))
    x0 = max(0, int(floor(xx)))
    y1 = min(h - 1, y0 + 1)
    x1 = min(w - 1, x0 + 1)
    ty = yy - floor(yy)
    tx = xx - floor(xx)
  let
    v00 = if y0 >= 0 and x0 >= 0 and y0 < h and x0 < w: img.data[flattenIdx(img.shape, c, y0, x0)] else: 0.0'f32
    v01 = if y0 >= 0 and x1 >= 0 and y0 < h and x1 < w: img.data[flattenIdx(img.shape, c, y0, x1)] else: 0.0'f32
    v10 = if y1 >= 0 and x0 >= 0 and y1 < h and x0 < w: img.data[flattenIdx(img.shape, c, y1, x0)] else: 0.0'f32
    v11 = if y1 >= 0 and x1 >= 0 and y1 < h and x1 < w: img.data[flattenIdx(img.shape, c, y1, x1)] else: 0.0'f32
  let
    v0 = lerpf(v00, v01, tx)
    v1 = lerpf(v10, v11, tx)
  lerpf(v0, v1, ty)

func cubicWeight(x: float32): float32 =
  ## Catmull-Rom bicubic kernel weight.
  let ax = abs(x)
  if ax <= 1.0'f32:
    (1.5'f32 * ax - 2.5'f32) * ax * ax + 1.0'f32
  elif ax < 2.0'f32:
    ((-0.5'f32 * ax + 2.5'f32) * ax - 4.0'f32) * ax + 2.0'f32
  else:
    0.0'f32

func pixelBicubic(img: Image; fy, fx: float32; c: int): float32 =
  let (h, w) = (img.shape[1], img.shape[2])
  let yy = fy - 0.5'f32
  let xx = fx - 0.5'f32
  let y0 = int(floor(yy))
  let x0 = int(floor(xx))
  var acc = 0.0'f32
  for dy in -1 .. 2:
    let py = y0 + dy
    let wy = cubicWeight(float32(dy) - (yy - float32(y0)))
    for dx in -1 .. 2:
      let px = x0 + dx
      let wx = cubicWeight(float32(dx) - (xx - float32(x0)))
      let v = if py >= 0 and py < h and px >= 0 and px < w:
        img.data[flattenIdx(img.shape, c, py, px)] else: 0.0'f32
      acc += v * wy * wx
  clampf(acc, 0.0'f32, 1.0'f32)

# ---- RNG helpers -----------------------------------------------------------

func rngUniform(k: Key): float32 {.inline.} =
  float32(k.a) / float32(high(uint32))

func rngNormal(k: Key; mean, std: float32): float32 =
  let u1 = max(1e-10'f32, rngUniform(k))
  let u2 = rngUniform(foldIn(k, k.a.uint64))
  let z = sqrt(-2.0'f32 * ln(u1)) * cos(2.0'f32 * 3.1415926535'f32 * u2)
  mean + z * std

# ---- Geometry transforms ---------------------------------------------------

proc resize*(img: Image; size: (int, int); interp: Interpolation = ipBilinear): Image =
  ## Resize the image to `(H, W)`. Supports nearest, bilinear, bicubic.
  if not img.shape.shapeOk:
    raise newException(ValueError, "resize: invalid image shape " & $img.shape)
  let (nc, srcH, srcW) = (img.shape[0], img.shape[1], img.shape[2])
  let (dstH, dstW) = size
  if dstH <= 0 or dstW <= 0:
    raise newException(ValueError, "resize: size must be positive, got " & $size)
  result.data = newSeq[float32](nc * dstH * dstW)
  result.shape = @[nc, dstH, dstW]
  result.label = img.label
  let scaleY = float32(srcH) / float32(dstH)
  let scaleX = float32(srcW) / float32(dstW)
  for c in 0 ..< nc:
    for y in 0 ..< dstH:
      for x in 0 ..< dstW:
        let srcY = (float32(y) + 0.5'f32) * scaleY
        let srcX = (float32(x) + 0.5'f32) * scaleX
        let idx = flattenIdx(result.shape, c, y, x)
        case interp
        of ipNearest:
          let sy = clamp(int(srcY), 0, srcH - 1)
          let sx = clamp(int(srcX), 0, srcW - 1)
          result.data[idx] = img.data[flattenIdx(img.shape, c, sy, sx)]
        of ipBilinear:
          result.data[idx] = pixelBilinear(img, srcY, srcX, c)
        of ipBicubic:
          result.data[idx] = pixelBicubic(img, srcY, srcX, c)

proc resizeImage*(img: Image; size: (int, int); interp: Interpolation = ipBilinear): Image {.inline.} =
  resize(img, size, interp)

proc centerCrop*(img: Image; size: (int, int)): Image =
  ## Crop the center `(H, W)` region. Pads with 0 if image is smaller.
  if not img.shape.shapeOk:
    raise newException(ValueError, "centerCrop: invalid image shape " & $img.shape)
  let (nc, srcH, srcW) = (img.shape[0], img.shape[1], img.shape[2])
  let (cropH, cropW) = size
  if cropH <= 0 or cropW <= 0:
    raise newException(ValueError, "centerCrop: size must be positive, got " & $size)
  result.data = newSeq[float32](nc * cropH * cropW)
  result.shape = @[nc, cropH, cropW]
  result.label = img.label

  let yOff = (srcH - cropH) div 2
  let xOff = (srcW - cropW) div 2

  for c in 0 ..< nc:
    for y in 0 ..< cropH:
      for x in 0 ..< cropW:
        let sy = yOff + y
        let sx = xOff + x
        let val = if sy >= 0 and sy < srcH and sx >= 0 and sx < srcW:
          img.data[flattenIdx(img.shape, c, sy, sx)]
        else:
          0.0'f32
        result.data[flattenIdx(result.shape, c, y, x)] = val

proc crop*(img: Image; top, left: int; size: (int, int)): Image =
  ## Crop from `(top, left)` with given `(H, W)`. Pads with 0 if out of bounds.
  if not img.shape.shapeOk:
    raise newException(ValueError, "crop: invalid image shape " & $img.shape)
  let (nc, srcH, srcW) = (img.shape[0], img.shape[1], img.shape[2])
  let (ch, cw) = size
  if ch <= 0 or cw <= 0:
    raise newException(ValueError, "crop: size must be positive, got " & $size)
  result.data = newSeq[float32](nc * ch * cw)
  result.shape = @[nc, ch, cw]
  result.label = img.label
  for c in 0 ..< nc:
    for y in 0 ..< ch:
      for x in 0 ..< cw:
        let sy = top + y
        let sx = left + x
        let val = if sy >= 0 and sy < srcH and sx >= 0 and sx < srcW:
          img.data[flattenIdx(img.shape, c, sy, sx)]
        else:
          0.0'f32
        result.data[flattenIdx(result.shape, c, y, x)] = val

proc randomCrop*(img: Image; key: Key; size: (int, int)): Image =
  ## Crop a random `(H, W)` region.
  if not img.shape.shapeOk:
    raise newException(ValueError, "randomCrop: invalid image shape " & $img.shape)
  let (srcH, srcW) = (img.shape[1], img.shape[2])
  let (ch, cw) = size
  if ch > srcH or cw > srcW:
    return centerCrop(img, size)
  let k1 = foldIn(key, 0'u64)
  let k2 = foldIn(key, 1'u64)
  let top = int(rngUniform(k1) * float32(srcH - ch))
  let left = int(rngUniform(k2) * float32(srcW - cw))
  crop(img, top, left, size)

proc randomResizedCrop*(img: Image; key: Key; size: (int, int);
    scale: (float32, float32) = (0.08'f32, 1.0'f32);
    ratio: (float32, float32) = (0.75'f32, 1.3333'f32);
    interp: Interpolation = ipBilinear): Image =
  ## Random crop with random scale/ratio, then resize to `size`.
  if not img.shape.shapeOk:
    raise newException(ValueError, "randomResizedCrop: invalid image shape " & $img.shape)
  let (srcH, srcW) = (img.shape[1], img.shape[2])
  var k = key
  # Sample area fraction
  k = foldIn(k, 0'u64)
  let s = lerpf(scale[0], scale[1], rngUniform(k))
  # Sample aspect ratio
  k = foldIn(k, 1'u64)
  let r = exp(lerpf(ln(ratio[0]), ln(ratio[1]), rngUniform(k)))
  # Compute crop size
  var cropH = int(sqrt(s * float32(srcH) * float32(srcW) * r))
  var cropW = int(sqrt(s * float32(srcH) * float32(srcW) / r))
  cropH = min(cropH, srcH)
  cropW = min(cropW, srcW)
  if cropH <= 0: cropH = 1
  if cropW <= 0: cropW = 1
  # Sample position
  k = foldIn(k, 2'u64)
  let top = int(rngUniform(k) * float32(srcH - cropH))
  k = foldIn(k, 3'u64)
  let left = int(rngUniform(k) * float32(srcW - cropW))
  let cropped = crop(img, max(0, top), max(0, left), (cropH, cropW))
  resize(cropped, size, interp)

proc horizontalFlip*(img: Image): Image =
  ## Deterministic horizontal flip.
  if not img.shape.shapeOk:
    raise newException(ValueError, "horizontalFlip: invalid image shape " & $img.shape)
  let (nc, h, w) = (img.shape[0], img.shape[1], img.shape[2])
  result.data = newSeq[float32](nc * h * w)
  result.shape = @[nc, h, w]
  result.label = img.label
  for c in 0 ..< nc:
    for y in 0 ..< h:
      for x in 0 ..< w:
        result.data[flattenIdx(result.shape, c, y, x)] =
          img.data[flattenIdx(img.shape, c, y, w - 1 - x)]

proc verticalFlip*(img: Image): Image =
  ## Deterministic vertical flip.
  if not img.shape.shapeOk:
    raise newException(ValueError, "verticalFlip: invalid image shape " & $img.shape)
  let (nc, h, w) = (img.shape[0], img.shape[1], img.shape[2])
  result.data = newSeq[float32](nc * h * w)
  result.shape = @[nc, h, w]
  result.label = img.label
  for c in 0 ..< nc:
    for y in 0 ..< h:
      for x in 0 ..< w:
        result.data[flattenIdx(result.shape, c, y, x)] =
          img.data[flattenIdx(img.shape, c, h - 1 - y, x)]

proc randomHorizontalFlip*(img: Image; key: Key; p: float32 = 0.5): Image =
  ## Horizontal flip with probability `p`.
  let k = foldIn(key, 0'u64)
  if rngUniform(k) < p:
    horizontalFlip(img)
  else:
    img

proc randomVerticalFlip*(img: Image; key: Key; p: float32 = 0.5): Image =
  ## Vertical flip with probability `p`.
  let k = foldIn(key, 0'u64)
  if rngUniform(k) < p:
    verticalFlip(img)
  else:
    img

proc pad*(img: Image; padding: seq[int]; fill: float32 = 0.0): Image =
  ## Pad all sides. `padding` is [left, top, right, bottom].
  if not img.shape.shapeOk:
    raise newException(ValueError, "pad: invalid image shape " & $img.shape)
  let (nc, h, w) = (img.shape[0], img.shape[1], img.shape[2])
  var padSeq: seq[int]
  if padding.len == 1:
    padSeq = @[padding[0], padding[0], padding[0], padding[0]]
  elif padding.len == 2:
    padSeq = @[padding[0], padding[1], padding[0], padding[1]]
  elif padding.len == 4:
    padSeq = padding
  else:
    raise newException(ValueError, "pad: padding must have 1, 2, or 4 elements, got " & $padding.len)
  let (left, top, right, bottom) = (padSeq[0], padSeq[1], padSeq[2], padSeq[3])
  let dstH = h + top + bottom
  let dstW = w + left + right
  result.data = newSeq[float32](nc * dstH * dstW)
  result.shape = @[nc, dstH, dstW]
  result.label = img.label
  for c in 0 ..< nc:
    for y in 0 ..< dstH:
      for x in 0 ..< dstW:
        let sidx = flattenIdx(result.shape, c, y, x)
        if y >= top and y < top + h and x >= left and x < left + w:
          result.data[sidx] = img.data[flattenIdx(img.shape, c, y - top, x - left)]
        else:
          result.data[sidx] = fill

proc pad*(img: Image; padding: int; fill: float32 = 0.0): Image =
  ## Pad all sides by the same amount.
  pad(img, @[padding], fill)

proc rotate*(img: Image; angleDeg: float32; interp: Interpolation = ipBilinear;
    expand: bool = false): Image =
  ## Rotate by `angleDeg` degrees counter-clockwise.
  if not img.shape.shapeOk:
    raise newException(ValueError, "rotate: invalid image shape " & $img.shape)
  let (nc, srcH, srcW) = (img.shape[0], img.shape[1], img.shape[2])
  let radians = angleDeg * 3.1415926535'f32 / 180.0'f32
  let cosA = cos(radians)
  let sinA = sin(radians)
  var dstH, dstW: int
  if expand:
    let aCos = abs(cosA.float64)
    let aSin = abs(sinA.float64)
    dstH = int(float64(srcH) * aCos + float64(srcW) * aSin)
    dstW = int(float64(srcH) * aSin + float64(srcW) * aCos)
  else:
    dstH = srcH
    dstW = srcW
  result.data = newSeq[float32](nc * dstH * dstW)
  result.shape = @[nc, dstH, dstW]
  result.label = img.label
  let cxSrc = float32(srcW) / 2.0'f32
  let cySrc = float32(srcH) / 2.0'f32
  let cxDst = float32(dstW) / 2.0'f32
  let cyDst = float32(dstH) / 2.0'f32
  for c in 0 ..< nc:
    for y in 0 ..< dstH:
      for x in 0 ..< dstW:
        let dx = float32(x) - cxDst
        let dy = float32(y) - cyDst
        let sx = cosA * dx + sinA * dy + cxSrc
        let sy = -sinA * dx + cosA * dy + cySrc
        let val = case interp
          of ipNearest:
            let siy = clamp(int(sy), 0, srcH - 1)
            let six = clamp(int(sx), 0, srcW - 1)
            img.data[flattenIdx(img.shape, c, siy, six)]
          of ipBilinear:
            pixelBilinear(img, sy, sx, c)
          of ipBicubic:
            pixelBicubic(img, sy, sx, c)
        result.data[flattenIdx(result.shape, c, y, x)] = val

proc randomRotation*(img: Image; key: Key; degrees: float32;
    interp: Interpolation = ipBilinear; expand: bool = false): Image =
  ## Rotate by a random angle in `[-degrees, degrees]`.
  let k = foldIn(key, 0'u64)
  let angle = (rngUniform(k) * 2.0'f32 - 1.0'f32) * degrees
  rotate(img, angle, interp, expand)

# ---- Color / pixel-level transforms ----------------------------------------

proc normalize*(img: Image; mean, std: openArray[float32]): Image =
  ## Channel-wise `(pixel - mean[c]) / std[c]`. Output may leave [0,1] range.
  if not img.shape.shapeOk:
    raise newException(ValueError, "normalize: invalid image shape " & $img.shape)
  let nc = img.shape[0]
  if mean.len < nc or std.len < nc:
    raise newException(ValueError, "normalize: need mean/std for " & $nc & " channels")
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  let n = img.data.len
  for i in 0 ..< n:
    let c = (i div (img.shape[1] * img.shape[2])) mod nc
    result.data[i] = (img.data[i] - mean[c]) / std[c]

proc grayscale*(img: Image; numOutputChannels: int = 1): Image =
  ## Convert RGB to grayscale using luminance weights (0.299, 0.587, 0.114).
  if not img.shape.shapeOk:
    raise newException(ValueError, "grayscale: invalid image shape " & $img.shape)
  let nc = img.shape[0]
  if nc != 3:
    raise newException(ValueError, "grayscale: expected 3 input channels, got " & $nc)
  if numOutputChannels != 1 and numOutputChannels != 3:
    raise newException(ValueError, "grayscale: numOutputChannels must be 1 or 3")
  let (h, w) = (img.shape[1], img.shape[2])
  let oc = numOutputChannels
  result.data = newSeq[float32](oc * h * w)
  result.shape = @[oc, h, w]
  result.label = img.label
  for y in 0 ..< h:
    for x in 0 ..< w:
      let g = 0.299'f32 * img.data[flattenIdx(img.shape, 0, y, x)] +
              0.587'f32 * img.data[flattenIdx(img.shape, 1, y, x)] +
              0.114'f32 * img.data[flattenIdx(img.shape, 2, y, x)]
      if oc == 1:
        result.data[y * w + x] = g
      else:
        result.data[flattenIdx(result.shape, 0, y, x)] = g
        result.data[flattenIdx(result.shape, 1, y, x)] = g
        result.data[flattenIdx(result.shape, 2, y, x)] = g

proc randomInvert*(img: Image; key: Key; p: float32 = 0.5): Image =
  ## Invert pixel values `1.0 - v` with probability `p`.
  if not img.shape.shapeOk:
    raise newException(ValueError, "randomInvert: invalid image shape " & $img.shape)
  let k = foldIn(key, 0'u64)
  if rngUniform(k) < p:
    result.data = newSeq[float32](img.data.len)
    result.shape = img.shape
    result.label = img.label
    for i in 0 ..< img.data.len:
      result.data[i] = 1.0'f32 - img.data[i]
  else:
    result = img

proc adjustBrightness*(img: Image; factor: float32): Image =
  ## Multiply all pixels by `factor`, clamp to [0, 1].
  if not img.shape.shapeOk:
    raise newException(ValueError, "adjustBrightness: invalid image shape " & $img.shape)
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  for i in 0 ..< img.data.len:
    result.data[i] = clampf(img.data[i] * factor, 0.0'f32, 1.0'f32)

proc adjustContrast*(img: Image; factor: float32): Image =
  ## Adjust contrast around mean gray: `(v - 0.5) * factor + 0.5`, clamp to [0, 1].
  if not img.shape.shapeOk:
    raise newException(ValueError, "adjustContrast: invalid image shape " & $img.shape)
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  for i in 0 ..< img.data.len:
    result.data[i] = clampf((img.data[i] - 0.5'f32) * factor + 0.5'f32, 0.0'f32, 1.0'f32)

proc adjustSaturation*(img: Image; factor: float32): Image =
  ## Convert RGB to grayscale, then blend `v = gray * (1 - factor) + original * factor`.
  if not img.shape.shapeOk:
    raise newException(ValueError, "adjustSaturation: invalid image shape " & $img.shape)
  let nc = img.shape[0]
  if nc != 3:
    raise newException(ValueError, "adjustSaturation: expected 3 input channels, got " & $nc)
  let (h, w) = (img.shape[1], img.shape[2])
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  for y in 0 ..< h:
    for x in 0 ..< w:
      let r = img.data[flattenIdx(img.shape, 0, y, x)]
      let g = img.data[flattenIdx(img.shape, 1, y, x)]
      let b = img.data[flattenIdx(img.shape, 2, y, x)]
      let gray = 0.299'f32 * r + 0.587'f32 * g + 0.114'f32 * b
      for c in 0 ..< 3:
        result.data[flattenIdx(result.shape, c, y, x)] =
          clampf(gray * (1.0'f32 - factor) + img.data[flattenIdx(img.shape, c, y, x)] * factor, 0.0'f32, 1.0'f32)

proc hsvToRgb(h, s, v: float32): tuple[r, g, b: float32] =
  if s == 0.0'f32:
    return (v, v, v)
  let hh = h * 6.0'f32
  let i = int(floor(hh))
  let f = hh - float32(i)
  let p = v * (1.0'f32 - s)
  let q = v * (1.0'f32 - s * f)
  let t = v * (1.0'f32 - s * (1.0'f32 - f))
  case i mod 6
  of 0: (v, t, p)
  of 1: (q, v, p)
  of 2: (p, v, t)
  of 3: (p, q, v)
  of 4: (t, p, v)
  else: (v, p, q)

proc rgbToHsv(r, g, b: float32): tuple[h, s, v: float32] =
  let mx = max(max(r, g), b)
  let mn = min(min(r, g), b)
  let chroma = mx - mn
  var h: float32 = 0.0'f32
  if chroma > 0.0'f32:
    if mx == r:
      h = (g - b) / chroma
      if h < 0.0'f32: h += 6.0'f32
    elif mx == g:
      h = (b - r) / chroma + 2.0'f32
    else:
      h = (r - g) / chroma + 4.0'f32
    h = h / 6.0'f32
  let s = if mx == 0.0'f32: 0.0'f32 else: chroma / mx
  (h, s, mx)

proc adjustHue*(img: Image; factor: float32): Image =
  ## Rotate hue in RGB→HSV→RGB cycle. `factor` in [-0.5, 0.5].
  if not img.shape.shapeOk:
    raise newException(ValueError, "adjustHue: invalid image shape " & $img.shape)
  let nc = img.shape[0]
  if nc != 3:
    raise newException(ValueError, "adjustHue: expected 3 input channels, got " & $nc)
  let (h, w) = (img.shape[1], img.shape[2])
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  for y in 0 ..< h:
    for x in 0 ..< w:
      let r = img.data[flattenIdx(img.shape, 0, y, x)]
      let g = img.data[flattenIdx(img.shape, 1, y, x)]
      let b = img.data[flattenIdx(img.shape, 2, y, x)]
      var (hv, sv, vv) = rgbToHsv(r, g, b)
      hv = hv + factor
      if hv < 0.0'f32: hv += 1.0'f32
      if hv >= 1.0'f32: hv -= 1.0'f32
      let (nr, ng, nb) = hsvToRgb(hv, sv, vv)
      result.data[flattenIdx(result.shape, 0, y, x)] = clampf(nr, 0.0'f32, 1.0'f32)
      result.data[flattenIdx(result.shape, 1, y, x)] = clampf(ng, 0.0'f32, 1.0'f32)
      result.data[flattenIdx(result.shape, 2, y, x)] = clampf(nb, 0.0'f32, 1.0'f32)

proc colorJitter*(img: Image; key: Key;
    brightness: (float32, float32) = (1.0'f32, 1.0'f32);
    contrast: (float32, float32) = (1.0'f32, 1.0'f32);
    saturation: (float32, float32) = (1.0'f32, 1.0'f32);
    hue: (float32, float32) = (0.0'f32, 0.0'f32)): Image =
  ## Randomly jitter brightness, contrast, saturation, and hue.
  var k = key
  var img = img
  if brightness[0] != brightness[1]:
    k = foldIn(k, 0'u64)
    let bf = lerpf(brightness[0], brightness[1], rngUniform(k))
    img = adjustBrightness(img, bf)
  if contrast[0] != contrast[1]:
    k = foldIn(k, 1'u64)
    let cf = lerpf(contrast[0], contrast[1], rngUniform(k))
    img = adjustContrast(img, cf)
  if saturation[0] != saturation[1]:
    k = foldIn(k, 2'u64)
    let sf = lerpf(saturation[0], saturation[1], rngUniform(k))
    img = adjustSaturation(img, sf)
  if hue[0] != hue[1]:
    k = foldIn(k, 3'u64)
    let hf = lerpf(hue[0], hue[1], rngUniform(k))
    img = adjustHue(img, hf)
  img

# ---- Filter transforms -----------------------------------------------------

func gaussianKernel1d(sigma: float32; size: int): seq[float32] =
  result = newSeq[float32](size)
  let center = float32(size - 1) / 2.0'f32
  var sum = 0.0'f32
  for i in 0 ..< size:
    let x = float32(i) - center
    result[i] = exp(-0.5'f32 * (x * x) / (sigma * sigma))
    sum += result[i]
  if sum > 0.0'f32:
    for i in 0 ..< size:
      result[i] = result[i] / sum

proc gaussianBlur*(img: Image; kernelSize: int; sigma: float32 = 1.0): Image =
  ## Separable 2D Gaussian blur of given kernel size.
  if not img.shape.shapeOk:
    raise newException(ValueError, "gaussianBlur: invalid image shape " & $img.shape)
  if kernelSize < 3 or kernelSize mod 2 == 0:
    raise newException(ValueError, "gaussianBlur: kernelSize must be odd and >= 3, got " & $kernelSize)
  let (nc, h, w) = (img.shape[0], img.shape[1], img.shape[2])
  let kernel = gaussianKernel1d(sigma, kernelSize)
  let pad = kernelSize div 2
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label

  # horizontal pass
  var tmp = newSeq[float32](img.data.len)
  for c in 0 ..< nc:
    for y in 0 ..< h:
      for x in 0 ..< w:
        var sum = 0.0'f32
        for kx in 0 ..< kernelSize:
          let px = x + kx - pad
          let v = if px >= 0 and px < w:
            img.data[flattenIdx(img.shape, c, y, px)] else: 0.0'f32
          sum += v * kernel[kx]
        tmp[flattenIdx(img.shape, c, y, x)] = sum

  # vertical pass
  for c in 0 ..< nc:
    for y in 0 ..< h:
      for x in 0 ..< w:
        var sum = 0.0'f32
        for ky in 0 ..< kernelSize:
          let py = y + ky - pad
          let v = if py >= 0 and py < h:
            tmp[flattenIdx(img.shape, c, py, x)] else: 0.0'f32
          sum += v * kernel[ky]
        result.data[flattenIdx(img.shape, c, y, x)] = sum

proc gaussianNoise*(img: Image; key: Key; mean: float32 = 0.0;
    sigma: float32 = 0.01; clip: bool = true): Image =
  ## Add Gaussian noise. When `clip=true`, clamp to [0, 1].
  if not img.shape.shapeOk:
    raise newException(ValueError, "gaussianNoise: invalid image shape " & $img.shape)
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  var k = key
  for i in 0 ..< img.data.len:
    k = foldIn(k, uint64(i))
    let noise = rngNormal(k, mean, sigma)
    let v = img.data[i] + noise
    result.data[i] = if clip: clampf(v, 0.0'f32, 1.0'f32) else: v

# ---- Type conversion -------------------------------------------------------

proc toDtype*(img: Image; dtype: DType; scale: bool = false): Image =
  ## Convert pixel value range. `dtype` indicates the target representation:
  ## `dtFloat32` → values in [0, 1]; `dtUint8` → values in [0, 255].
  ## When `scale=true`, multiply/divide by 255.
  if not img.shape.shapeOk:
    raise newException(ValueError, "toDtype: invalid image shape " & $img.shape)
  result.data = newSeq[float32](img.data.len)
  result.shape = img.shape
  result.label = img.label
  case dtype
  of dtFloat32:
    if scale:
      for i in 0 ..< img.data.len:
        result.data[i] = img.data[i] / 255.0'f32
    else:
      result.data = img.data
  of dtUint8:
    if scale:
      for i in 0 ..< img.data.len:
        result.data[i] = clampf(floor(img.data[i] * 255.0'f32 + 0.5'f32), 0.0'f32, 255.0'f32)
    else:
      for i in 0 ..< img.data.len:
        result.data[i] = clampf(floor(img.data[i] + 0.5'f32), 0.0'f32, 255.0'f32)
  else:
    raise newException(ValueError, "toDtype: unsupported dtype " & $dtype)

# ---- Compose ----------------------------------------------------------------

proc initCompose*(transforms: openArray[ImageTransform]): Compose =
  result.transforms = @transforms

proc apply*(c: Compose; img: Image): Image =
  result = img
  for t in c.transforms:
    result = t(result)

# ---- Image loading — PPM/PGM (pure Nim) ------------------------------------

proc parsePgmHeader(s: string): tuple[w, h, maxVal: int, dataStart: int] =
  var parts: seq[string] = @[]
  var cur = ""
  var inComment = false
  var i = 0
  while i < s.len:
    let ch = s[i]
    if inComment:
      if ch == '\n' or ch == '\r':
        inComment = false
      inc i
      continue
    if ch == '#':
      inComment = true
      inc i
      continue
    if ch in Whitespace:
      if cur.len > 0:
        parts.add(cur)
        cur = ""
      if parts.len >= 4:
        # Found all tokens; data starts after this whitespace
        while i < s.len and s[i] in Whitespace:
          inc i
        result.dataStart = i
        result.w = parseInt(parts[1])
        result.h = parseInt(parts[2])
        result.maxVal = parseInt(parts[3])
        return
      inc i
      continue
    cur.add(ch)
    inc i
  if cur.len > 0:
    parts.add(cur)
  if parts.len < 4:
    raise newException(ValueError, "PPM/PGM: invalid header, got " & $parts.len & " tokens")
  result.dataStart = s.len
  result.w = parseInt(parts[1])
  result.h = parseInt(parts[2])
  result.maxVal = parseInt(parts[3])

proc loadPpmOrPgm*(path: string): Image =
  ## Load a PPM (P6, color) or PGM (P5, grayscale) binary image.
  let raw = readFile(path)
  if raw.len < 3:
    raise newException(ValueError, "PPM/PGM: file too short")

  let isP6 = raw[0] == 'P' and raw[1] == '6'
  let isP5 = raw[0] == 'P' and raw[1] == '5'
  if not isP6 and not isP5:
    raise newException(ValueError, "PPM/PGM: unsupported format, expected P5 or P6")

  let (w, h, maxVal, pixelStart) = parsePgmHeader(raw)

  let nc = if isP6: 3 else: 1
  let expectedLen = pixelStart + w * h * nc * (if maxVal <= 255: 1 else: 2)
  if raw.len < expectedLen:
    raise newException(ValueError, "PPM/PGM: truncated pixel data, expected " &
      $expectedLen & " bytes, got " & $raw.len)

  result.data = newSeq[float32](nc * w * h)
  result.shape = @[nc, h, w]
  result.label = -1

  var pos = pixelStart
  if maxVal <= 255:
    for y in 0 ..< h:
      for x in 0 ..< w:
        for c in 0 ..< nc:
          if pos >= raw.len:
            raise newException(ValueError, "PPM/PGM: truncated pixel data")
          let v = float32(raw[pos].uint8) / 255.0'f32
          result.data[flattenIdx(result.shape, c, y, x)] = v
          inc pos
  else:
    for y in 0 ..< h:
      for x in 0 ..< w:
        for c in 0 ..< nc:
          if pos + 1 >= raw.len:
            raise newException(ValueError, "PPM/PGM: truncated pixel data")
          let hi = raw[pos].uint16
          let lo = raw[pos + 1].uint16
          let v = float32((hi shl 8) or lo) / float32(maxVal)
          result.data[flattenIdx(result.shape, c, y, x)] = v
          inc pos, 2

proc loadPpm*(path: string): Image =
  ## Load a binary PPM (P6) image.
  loadPpmOrPgm(path)

proc loadPgm*(path: string): Image =
  ## Load a binary PGM (P5) image.
  loadPpmOrPgm(path)

# ---- Image loading — stb_image FFI ------------------------------------------

when defined(rewStbImage):
  {.compile(currentSourcePath().parentDir() / "stb_image_impl.c").}

  proc stbiLoadFromMemory(data: ptr uint8; length: int;
      width, height, channels: var int32;
      desiredChannels: int32): ptr uint8 {.
      importc: "stbi_load_from_memory_wrapper", nodecl.}

  proc stbiFree(data: pointer) {.
      importc: "stbi_free_wrapper", nodecl.}

  proc loadStbImage*(path: string): Image =
    let raw = readFile(path)
    var w, h, c: int32
    let pixels = stbiLoadFromMemory(cast[ptr uint8](raw[0].unsafeAddr),
      raw.len.cint, w, h, c, 3'i32)
    if pixels.isNil:
      raise newException(ValueError, "stb_image: failed to decode " & path)
    let nc = 3
    result.data = newSeq[float32](nc * int(w) * int(h))
    result.shape = @[nc, int(h), int(w)]
    result.label = -1
    for i in 0 ..< nc * int(w) * int(h):
      result.data[i] = float32(pixels[i]) / 255.0'f32
    stbiFree(pixels)

proc loadImage*(path: string): Image =
  ## Auto-detects format from extension and loads PNG, JPEG, PPM, or PGM.
  let ext = path.splitFile().ext.toLowerAscii()
  case ext
  of ".ppm", ".pgm":
    loadPpmOrPgm(path)
  of ".png", ".jpg", ".jpeg":
    when defined(rewStbImage):
      loadStbImage(path)
    else:
      raise newException(ValueError,
        "loadImage: " & ext & " support requires stb_image. " &
        "Place stb_image.h in src/rew/data/ and build with -d:rewStbImage.")
  else:
    raise newException(ValueError, "loadImage: unsupported format " & ext)

proc loadImageFile*(path: string): Image {.inline.} =
  loadImage(path)
