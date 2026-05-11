## Sample and Batch — host-side data containers and device transfer helpers.
##
## `Sample` holds a single data point as a flat `seq[float32]` with shape
## metadata plus an integer label. `Batch` is the collated form ready for
## device transfer. Both are plain value types (no `ref`).

import ./dataset
import ../dtype
import ../device
import ../tensor
import ../eager

type
  Sample* = object
    ## A single labeled data point on the host.
    data*: seq[float32]
    dataShape*: seq[int]
    label*: int

  Batch* = object
    ## A collated batch of samples, ready for device transfer.
    data*: seq[float32]
    dataShape*: seq[int]
    labels*: seq[int]
    batchSize*: int

proc shapeElementCount(shape: openArray[int]; opName: string): int =
  result = 1
  for d in shape:
    if d < 0:
      raise newException(DataError,
        opName & ": shape contains negative dimension " & $d)
    result *= d

proc collate*(samples: seq[Sample]): Batch =
  ## Stacks a sequence of samples into a single `Batch`. All samples
  ## must share the same `dataShape`. The batch data shape becomes
  ## `[batchSize] & sampleShape`.
  if samples.len == 0:
    return Batch(data: @[], dataShape: @[0], labels: @[], batchSize: 0)
  let shape = samples[0].dataShape
  let elemCount = shapeElementCount(shape, "collate")
  for i in 0 ..< samples.len:
    if samples[i].dataShape != shape:
      raise newException(DataError,
        "collate: sample " & $i & " shape " & $samples[i].dataShape &
          " differs from sample 0 shape " & $shape)
    if samples[i].data.len != elemCount:
      raise newException(DataError,
        "collate: sample " & $i & " data length " & $samples[i].data.len &
          " does not match shape " & $shape)
  let n = samples.len
  var batchData = newSeq[float32](n * elemCount)
  var labels = newSeq[int](n)
  for i in 0 ..< n:
    for j in 0 ..< elemCount:
      batchData[i * elemCount + j] = samples[i].data[j]
    labels[i] = samples[i].label
  Batch(
    data: batchData,
    dataShape: @[n] & shape,
    labels: labels,
    batchSize: n,
  )

proc oneHotF32*(labels: seq[int]; numClasses: int): seq[float32] =
  ## Converts integer labels to one-hot float32 vectors.
  if numClasses <= 0:
    raise newException(DataError,
      "oneHotF32: numClasses must be positive, got " & $numClasses)
  result = newSeq[float32](labels.len * numClasses)
  for i, c in labels:
    if c < 0 or c >= numClasses:
      raise newException(DataError,
        "oneHotF32: label " & $c & " out of range [0, " & $numClasses & ")")
    result[i * numClasses + c] = 1.0'f32

proc shapeDims(shape: openArray[int]): seq[int64] =
  result = newSeq[int64](shape.len)
  for i, v in shape: result[i] = int64(v)

proc f32ToDevice*(d: Device; data: var seq[float32];
    shape: openArray[int]): Tensor =
  ## Transfers a host float32 buffer to a device tensor.
  let n = shapeElementCount(shape, "f32ToDevice")
  if data.len != n:
    raise newException(DataError,
      "f32ToDevice: shape product " & $n & " != data length " & $data.len)
  let dims = shapeDims(shape)
  let bytes = data.len * sizeof(float32)
  let ptrIn = if data.len == 0: nil else: addr data[0]
  let h = transferToDevice(d, ptrIn, dtFloat32, dims, bytes)
  initEagerTensor(h, dtFloat32, shape, d)

proc toTensors*(d: Device; b: Batch; numClasses: int): tuple[x, y: Tensor] =
  ## Transfers a batch to device, producing a data tensor and a one-hot
  ## label tensor. `numClasses` sets the width of the one-hot encoding.
  var batchData = b.data
  let x = f32ToDevice(d, batchData, b.dataShape)
  var oh = oneHotF32(b.labels, numClasses)
  let y = f32ToDevice(d, oh, [b.batchSize, numClasses])
  (x: x, y: y)
