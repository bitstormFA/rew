## Dataset-level image transform combinators and image-folder sources.
##
## Each combinator takes a `Dataset[Sample]` and returns a new `Dataset[Sample]`
## by applying an `image.nim` transform to the `Sample.data`/`Sample.dataShape`
## via the existing `map` combinator.

import std/[algorithm, os]
import ./dataset
import ./transform
import ./sample
import ./image
import ../rng

# ---- Internal helpers -------------------------------------------------------

proc sampleToImage(s: Sample): Image =
  Image(data: s.data, shape: s.dataShape, label: s.label)

proc imageToSample(img: Image): Sample =
  Sample(data: img.data, dataShape: img.shape, label: img.label)

proc transformSample(s: Sample; fn: proc(img: Image): Image): Sample =
  imageToSample(fn(sampleToImage(s)))

# ---- Dataset-level convenience combinators ----------------------------------

proc resize*(ds: Dataset[Sample]; size: (int, int);
    interp: Interpolation = ipBilinear): Dataset[Sample] =
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = resize(img, size, interp)))

proc centerCrop*(ds: Dataset[Sample]; size: (int, int)): Dataset[Sample] =
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = centerCrop(img, size)))

proc randomHorizontalFlip*(ds: Dataset[Sample]; key: Key;
    p: float32 = 0.5): Dataset[Sample] =
  let k = key
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = randomHorizontalFlip(img, k, p)))

proc randomVerticalFlip*(ds: Dataset[Sample]; key: Key;
    p: float32 = 0.5): Dataset[Sample] =
  let k = key
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = randomVerticalFlip(img, k, p)))

proc normalize*(ds: Dataset[Sample]; mean, std: openArray[float32]): Dataset[Sample] =
  let m = @mean
  let s = @std
  ds.map(proc(smpl: Sample): Sample =
    transformSample(smpl, proc(img: Image): Image = normalize(img, m, s)))

proc grayscale*(ds: Dataset[Sample]): Dataset[Sample] =
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = grayscale(img)))

proc randomCrop*(ds: Dataset[Sample]; key: Key;
    size: (int, int)): Dataset[Sample] =
  let k = key
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image = randomCrop(img, k, size)))

proc randomResizedCrop*(ds: Dataset[Sample]; key: Key; size: (int, int);
    scale: (float32, float32) = (0.08'f32, 1.0'f32);
    ratio: (float32, float32) = (0.75'f32, 1.3333'f32);
    interp: Interpolation = ipBilinear): Dataset[Sample] =
  let k = key
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image =
      randomResizedCrop(img, k, size, scale, ratio, interp)))

proc colorJitter*(ds: Dataset[Sample]; key: Key;
    brightness: (float32, float32) = (1.0'f32, 1.0'f32);
    contrast: (float32, float32) = (1.0'f32, 1.0'f32);
    saturation: (float32, float32) = (1.0'f32, 1.0'f32);
    hue: (float32, float32) = (0.0'f32, 0.0'f32)): Dataset[Sample] =
  let k = key
  ds.map(proc(s: Sample): Sample =
    transformSample(s, proc(img: Image): Image =
      colorJitter(img, k, brightness, contrast, saturation, hue)))

# ---- Image folder / file sources -------------------------------------------

proc fromImageFiles*(paths: seq[string]): Dataset[Sample] =
  ## Yields images from an explicit list of paths.
  let ps = paths
  result.source = proc(): iterator(): Sample =
    let p = ps
    result = iterator(): Sample {.closure.} =
      for path in p:
        let img = loadImage(path)
        yield Sample(data: img.data, dataShape: img.shape, label: img.label)

proc fromImageFolder*(root: string; pattern: string = "*.png"): Dataset[Sample] =
  ## Yields all images under `root` matching `pattern`. Labels are derived from
  ## subdirectory names (ImageFolder convention). Subdirectories are sorted so
  ## label assignment is deterministic.
  var allPaths: seq[string] = @[]
  var labels: seq[int] = @[]

  var classDirs: seq[string] = @[]
  for kind, path in walkDir(root):
    if kind == pcDir:
      classDirs.add(path)
  classDirs.sort(system.cmp)

  for ci, classDir in classDirs:
    var files: seq[string] = @[]
    for file in walkFiles(classDir / pattern):
      files.add(file)
    files.sort(system.cmp)
    for f in files:
      allPaths.add(f)
      labels.add(ci)

  let ps = allPaths
  let ls = labels
  result.source = proc(): iterator(): Sample =
    let p = ps
    let l = ls
    result = iterator(): Sample {.closure.} =
      for i in 0 ..< p.len:
        let img = loadImage(p[i])
        yield Sample(data: img.data, dataShape: img.shape, label: l[i])
