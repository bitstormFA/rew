## Phase 8 — DataPipe construction and iteration.

block init_datapipe_train_only:
  let trainFn = fromSeq(@[1, 2, 3])
  let pipe = initDataPipe(trainFn)
  doAssert isSome(pipe.val) == false
  doAssert isSome(pipe.test) == false
  doAssert isSome(pipe.predict) == false

block init_datapipe_with_val:
  let trainFn = fromSeq(@[1, 2, 3])
  let valFn = fromSeq(@[4, 5])
  let pipe = initDataPipe(trainFn, val = some(valFn))
  doAssert isSome(pipe.val)

block datapipe_iteration:
  let data = fromSeq(@[1, 2, 3, 4, 5])
  let pipe = initDataPipe(data)
  let iter = pipe.train.source()
  var items: seq[int] = @[]
  while true:
    let item = iter()
    if finished(iter): break
    items.add item
  doAssert items == @[1, 2, 3, 4, 5]

block datapipe_with_batch:
  var batches: seq[Batch] = @[]
  for i in 0 ..< 3:
    batches.add Batch(data: @[float32(i)], dataShape: @[1],
        labels: @[i], batchSize: 1)
  let pipe = initDataPipe(fromSeq(batches))
  let iter = pipe.train.source()
  var count = 0
  while true:
    let b = iter()
    if finished(iter): break
    doAssert b.batchSize == 1
    count += 1
  doAssert count == 3

block datapipe_val_iteration:
  let trainFn = fromSeq(@[1, 2])
  let valFn = fromSeq(@[10, 20, 30])
  let pipe = initDataPipe(trainFn, val = some(valFn))
  let valIter = pipe.val.get().source()
  var sum = 0
  while true:
    let item = valIter()
    if finished(valIter): break
    sum += item
  doAssert sum == 60
