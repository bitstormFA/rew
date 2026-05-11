## File-backed data sources — `.npy` image + label loading.
##
## Bridges rew's existing `serialize.nim` (`loadNpy`) with the data
## pipeline. Data is loaded eagerly into host memory on construction;
## the returned `Dataset` yields one `Sample` per call from the
## cached arrays.

import ../serialize
import ../dtype
import ./dataset
import ./sample

proc fromNpy*(imagesPath, labelsPath: string;
    normalise: bool = true): Dataset[Sample] =
  ## Loads images and labels from `.npy` files and returns a dataset of
  ## `Sample` values. Images are expected as uint8 or float32; labels as
  ## uint8 or int64.
  ##
  ## When `normalise` is true (default), uint8 pixel values are scaled
  ## to `[0, 1]`. Float32 images are passed through unchanged.
  ##
  ## The image array's first axis is the sample count. Remaining axes
  ## become the per-sample `dataShape` (e.g. `[28, 28]` for MNIST).
  let imgArr = loadNpy(imagesPath)
  let lblArr = loadNpy(labelsPath)
  if imgArr.shape.len < 2:
    raise newException(DataError,
      "fromNpy: image array must be at least rank-2, got shape " &
        $imgArr.shape)
  if lblArr.shape.len != 1:
    raise newException(DataError,
      "fromNpy: label array must be rank-1, got shape " & $lblArr.shape)
  let n = imgArr.shape[0]
  if lblArr.shape[0] != n:
    raise newException(DataError,
      "fromNpy: image count " & $n & " != label count " &
        $lblArr.shape[0])

  let sampleShape = imgArr.shape[1 .. ^1]
  var elemCount = 1
  for d in sampleShape: elemCount *= d

  # Pre-convert all images to float32.
  var pixels = newSeq[seq[float32]](n)
  case imgArr.dtype
  of dtUint8:
    for i in 0 ..< n:
      pixels[i] = newSeq[float32](elemCount)
      for j in 0 ..< elemCount:
        let v = float32(imgArr.data[i * elemCount + j])
        pixels[i][j] = if normalise: v / 255.0'f32 else: v
  of dtFloat32:
    for i in 0 ..< n:
      pixels[i] = newSeq[float32](elemCount)
      let offset = i * elemCount * sizeof(float32)
      for j in 0 ..< elemCount:
        var f: float32
        copyMem(addr f, unsafeAddr imgArr.data[offset + j * sizeof(float32)],
          sizeof(float32))
        pixels[i][j] = f
  else:
    raise newException(DataError,
      "fromNpy: unsupported image dtype " & $imgArr.dtype &
        " (expected uint8 or float32)")

  # Pre-convert all labels to int.
  var labels = newSeq[int](n)
  case lblArr.dtype
  of dtUint8:
    for i in 0 ..< n:
      labels[i] = int(lblArr.data[i])
  of dtInt64:
    for i in 0 ..< n:
      var v: int64
      copyMem(addr v, unsafeAddr lblArr.data[i * 8], 8)
      labels[i] = int(v)
  of dtInt32:
    for i in 0 ..< n:
      var v: int32
      copyMem(addr v, unsafeAddr lblArr.data[i * 4], 4)
      labels[i] = int(v)
  else:
    raise newException(DataError,
      "fromNpy: unsupported label dtype " & $lblArr.dtype &
        " (expected uint8, int32, or int64)")

  let shape = sampleShape
  result.source = proc(): iterator(): Sample =
    let px = pixels
    let lb = labels
    let sh = shape
    let count = n
    result = iterator(): Sample {.closure.} =
      for i in 0 ..< count:
        yield Sample(data: px[i], dataShape: sh, label: lb[i])
