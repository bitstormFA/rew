## Phase 8 — DataSplits construction and iteration.

block init_datasplits_train_only:
  let trainFn = fromSeq(@[1, 2, 3])
  let splits = initDataSplits(trainFn)
  doAssert isSome(splits.val) == false
  doAssert isSome(splits.test) == false
  doAssert isSome(splits.predict) == false

block init_datasplits_with_val:
  let trainFn = fromSeq(@[1, 2, 3])
  let valFn = fromSeq(@[4, 5])
  let splits = initDataSplits(trainFn, val = some(valFn))
  doAssert isSome(splits.val)

block datasplits_train_iteration:
  let data = fromSeq(@[1, 2, 3, 4, 5])
  let splits = initDataSplits(data)
  let iter = splits.train.source()
  var items: seq[int] = @[]
  while true:
    let item = iter()
    if finished(iter): break
    items.add item
  doAssert items == @[1, 2, 3, 4, 5]

block datasplits_with_batch:
  var batches: seq[Batch] = @[]
  for i in 0 ..< 3:
    batches.add Batch(data: @[float32(i)], dataShape: @[1],
        labels: @[i], batchSize: 1)
  let splits = initDataSplits(fromSeq(batches))
  let iter = splits.train.source()
  var count = 0
  while true:
    let b = iter()
    if finished(iter): break
    doAssert b.batchSize == 1
    count += 1
  doAssert count == 3

block datasplits_val_iteration:
  let trainFn = fromSeq(@[1, 2])
  let valFn = fromSeq(@[10, 20, 30])
  let splits = initDataSplits(trainFn, val = some(valFn))
  let valIter = splits.val.get().source()
  var sum = 0
  while true:
    let item = valIter()
    if finished(valIter): break
    sum += item
  doAssert sum == 60
