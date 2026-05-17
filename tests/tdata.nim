## Dataset Pipeline — tests for sources, transforms, sample/batch helpers.

import std/algorithm
import rew

proc collect[T](ds: Dataset[T]): seq[T] =
  let it = ds.source()
  while true:
    let item = it()
    if finished(it): break
    result.add(item)

# ---- fromSeq ---------------------------------------------------------------

block fromSeq_basic:
  let ds = fromSeq(@[10, 20, 30])
  doAssert collect(ds) == @[10, 20, 30]

block fromSeq_empty:
  let ds = fromSeq(newSeq[int]())
  doAssert collect(ds).len == 0

block fromSeq_reentrant:
  let ds = fromSeq(@[1, 2, 3])
  doAssert collect(ds) == collect(ds)

# ---- fromRange --------------------------------------------------------------

block fromRange_basic:
  let ds = fromRange(0, 5)
  doAssert collect(ds) == @[0, 1, 2, 3, 4]

block fromRange_step:
  let ds = fromRange(0, 10, 3)
  doAssert collect(ds) == @[0, 3, 6, 9]

block fromRange_negative_step:
  let ds = fromRange(5, 0, -1)
  doAssert collect(ds) == @[5, 4, 3, 2, 1]

block fromRange_empty:
  let ds = fromRange(5, 5)
  doAssert collect(ds).len == 0

block fromRange_zero_step_raises:
  var caught = false
  try:
    discard fromRange(0, 5, 0)
  except DataError:
    caught = true
  doAssert caught

# ---- map --------------------------------------------------------------------

block map_double:
  let ds = fromSeq(@[1, 2, 3]).map(proc(x: int): int = x * 2)
  doAssert collect(ds) == @[2, 4, 6]

block map_type_change:
  let ds = fromSeq(@[1, 2, 3]).map(proc(x: int): string = $x)
  doAssert collect(ds) == @["1", "2", "3"]

# ---- filter -----------------------------------------------------------------

block filter_evens:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).filter(proc(x: int): bool = x mod 2 == 0)
  doAssert collect(ds) == @[2, 4]

block filter_none:
  let ds = fromSeq(@[1, 3, 5]).filter(proc(x: int): bool = x mod 2 == 0)
  doAssert collect(ds).len == 0

block filter_all:
  let ds = fromSeq(@[2, 4, 6]).filter(proc(x: int): bool = x mod 2 == 0)
  doAssert collect(ds) == @[2, 4, 6]

# ---- batch ------------------------------------------------------------------

block batch_exact:
  let ds = fromSeq(@[1, 2, 3, 4]).batch(2)
  let batches = collect(ds)
  doAssert batches.len == 2
  doAssert batches[0] == @[1, 2]
  doAssert batches[1] == @[3, 4]

block batch_remainder:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).batch(2)
  let batches = collect(ds)
  doAssert batches.len == 3
  doAssert batches[2] == @[5]

block batch_drop_last:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).batch(2, dropLast = true)
  let batches = collect(ds)
  doAssert batches.len == 2

block batch_larger_than_data:
  let ds = fromSeq(@[1, 2]).batch(10)
  let batches = collect(ds)
  doAssert batches.len == 1
  doAssert batches[0] == @[1, 2]

block batch_drop_last_larger:
  let ds = fromSeq(@[1, 2]).batch(10, dropLast = true)
  doAssert collect(ds).len == 0

block batch_zero_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2]).batch(0)
  except DataError:
    caught = true
  doAssert caught

# ---- repeat -----------------------------------------------------------------

block repeat_finite:
  let ds = fromSeq(@[1, 2]).repeat(3)
  doAssert collect(ds) == @[1, 2, 1, 2, 1, 2]

block repeat_zero:
  let ds = fromSeq(@[1, 2]).repeat(0)
  doAssert collect(ds).len == 0

block repeat_infinite_with_take:
  let ds = fromSeq(@[10, 20]).repeat(-1).take(5)
  doAssert collect(ds) == @[10, 20, 10, 20, 10]

block repeat_infinite_empty:
  let ds = fromSeq(newSeq[int]()).repeat(-1).take(5)
  doAssert collect(ds).len == 0

block repeat_invalid_epochs_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2]).repeat(-2)
  except DataError:
    caught = true
  doAssert caught

# ---- take -------------------------------------------------------------------

block take_less_than_size:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).take(3)
  doAssert collect(ds) == @[1, 2, 3]

block take_more_than_size:
  let ds = fromSeq(@[1, 2]).take(10)
  doAssert collect(ds) == @[1, 2]

block take_zero:
  let ds = fromSeq(@[1, 2, 3]).take(0)
  doAssert collect(ds).len == 0

block take_negative_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2, 3]).take(-1)
  except DataError:
    caught = true
  doAssert caught

# ---- skip -------------------------------------------------------------------

block skip_some:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).skip(2)
  doAssert collect(ds) == @[3, 4, 5]

block skip_all:
  let ds = fromSeq(@[1, 2, 3]).skip(5)
  doAssert collect(ds).len == 0

block skip_zero:
  let ds = fromSeq(@[1, 2, 3]).skip(0)
  doAssert collect(ds) == @[1, 2, 3]

block skip_negative_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2, 3]).skip(-1)
  except DataError:
    caught = true
  doAssert caught

# ---- zip --------------------------------------------------------------------

block zip_equal:
  let a = fromSeq(@[1, 2, 3])
  let b = fromSeq(@["a", "b", "c"])
  let zipped = collect(zip(a, b))
  doAssert zipped.len == 3
  doAssert zipped[0] == (a: 1, b: "a")
  doAssert zipped[2] == (a: 3, b: "c")

block zip_unequal:
  let a = fromSeq(@[1, 2, 3, 4])
  let b = fromSeq(@[10, 20])
  let zipped = collect(zip(a, b))
  doAssert zipped.len == 2

# ---- enumerate --------------------------------------------------------------

block enumerate_basic:
  let ds = fromSeq(@["a", "b", "c"]).enumerate()
  let items = collect(ds)
  doAssert items.len == 3
  doAssert items[0] == (idx: 0, val: "a")
  doAssert items[2] == (idx: 2, val: "c")

block enumerate_start:
  let ds = fromSeq(@[10, 20]).enumerate(start = 5)
  let items = collect(ds)
  doAssert items[0] == (idx: 5, val: 10)
  doAssert items[1] == (idx: 6, val: 20)

# ---- shuffle ----------------------------------------------------------------

block shuffle_deterministic:
  let key = initKey(42'u64)
  let ds = fromSeq(@[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).shuffle(key, 5)
  let run1 = collect(ds)
  let run2 = collect(ds)
  doAssert run1 == run2, "shuffle must be deterministic with same key"
  doAssert run1.len == 10

block shuffle_preserves_elements:
  let key = initKey(123'u64)
  let original = @[1, 2, 3, 4, 5]
  let ds = fromSeq(original).shuffle(key, 3)
  var result = collect(ds)
  result.sort()
  var expected = original
  expected.sort()
  doAssert result == expected

block shuffle_different_keys:
  let ds1 = fromSeq(@[1, 2, 3, 4, 5, 6, 7, 8]).shuffle(initKey(1'u64), 4)
  let ds2 = fromSeq(@[1, 2, 3, 4, 5, 6, 7, 8]).shuffle(initKey(2'u64), 4)
  let r1 = collect(ds1)
  let r2 = collect(ds2)
  doAssert r1 != r2, "different keys should produce different orders"

block shuffle_zero_buffer_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2]).shuffle(initKey(1'u64), 0)
  except DataError:
    caught = true
  doAssert caught

# ---- concat -----------------------------------------------------------------

block concat_basic:
  let a = fromSeq(@[1, 2])
  let b = fromSeq(@[3, 4, 5])
  doAssert collect(concat(a, b)) == @[1, 2, 3, 4, 5]

block concat_empty:
  let a = fromSeq(newSeq[int]())
  let b = fromSeq(@[1, 2])
  doAssert collect(concat(a, b)) == @[1, 2]

# ---- pipeline composition --------------------------------------------------

block compose_map_filter_batch:
  let ds = fromRange(0, 20)
    .map(proc(x: int): int = x * x)
    .filter(proc(x: int): bool = x < 50)
    .batch(3)
  let batches = collect(ds)
  doAssert batches[0] == @[0, 1, 4]
  doAssert batches[1] == @[9, 16, 25]
  doAssert batches[2] == @[36, 49]

block compose_skip_take:
  let ds = fromRange(0, 100).skip(10).take(5)
  doAssert collect(ds) == @[10, 11, 12, 13, 14]

block compose_shuffle_batch:
  let key = initKey(99'u64)
  let ds = fromRange(0, 12).shuffle(key, 6).batch(4)
  let batches = collect(ds)
  doAssert batches.len == 3
  var all: seq[int] = @[]
  for b in batches:
    doAssert b.len == 4
    for x in b: all.add(x)
  all.sort()
  doAssert all == @[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# ---- Sample / Batch / collate -----------------------------------------------

block collate_basic:
  let samples = @[
    Sample(data: @[1.0'f32, 2.0, 3.0], dataShape: @[3], label: 0),
    Sample(data: @[4.0'f32, 5.0, 6.0], dataShape: @[3], label: 1),
  ]
  let b = collate(samples)
  doAssert b.batchSize == 2
  doAssert b.dataShape == @[2, 3]
  doAssert b.labels == @[0, 1]
  doAssert b.data.len == 6
  doAssert b.data[0] == 1.0'f32
  doAssert b.data[3] == 4.0'f32

block collate_empty:
  let b = collate(newSeq[Sample]())
  doAssert b.batchSize == 0

block collate_shape_mismatch_raises:
  let samples = @[
    Sample(data: @[1.0'f32], dataShape: @[1], label: 0),
    Sample(data: @[1.0'f32, 2.0], dataShape: @[2], label: 1),
  ]
  var caught = false
  try:
    discard collate(samples)
  except DataError:
    caught = true
  doAssert caught

block collate_first_data_length_mismatch_raises:
  let samples = @[
    Sample(data: @[1.0'f32], dataShape: @[2], label: 0),
  ]
  var caught = false
  try:
    discard collate(samples)
  except DataError:
    caught = true
  doAssert caught

# ---- oneHotF32 --------------------------------------------------------------

block onehot_basic:
  let oh = oneHotF32(@[0, 2, 1], 3)
  doAssert oh.len == 9
  doAssert oh[0] == 1.0'f32  # class 0
  doAssert oh[1] == 0.0'f32
  doAssert oh[2] == 0.0'f32
  doAssert oh[3] == 0.0'f32  # class 2
  doAssert oh[4] == 0.0'f32
  doAssert oh[5] == 1.0'f32
  doAssert oh[6] == 0.0'f32  # class 1
  doAssert oh[7] == 1.0'f32
  doAssert oh[8] == 0.0'f32

block onehot_out_of_range_raises:
  var caught = false
  try:
    discard oneHotF32(@[0, 5], 3)
  except DataError:
    caught = true
  doAssert caught

block onehot_bad_class_count_raises:
  var caught = false
  try:
    discard oneHotF32(@[0], 0)
  except DataError:
    caught = true
  doAssert caught

# ---- Dataset type / items ---------------------------------------------------

block dataset_items:
  let ds = fromSeq(@[10, 20, 30])
  var items: seq[int] = @[]
  for x in ds.items:
    items.add(x)
  doAssert items == @[10, 20, 30]

block dataset_to_dataset:
  let fn = proc(): iterator(): int =
    let captured = @[1, 2]
    result = iterator(): int {.closure.} =
      for x in captured: yield x
  let ds = toDataset(fn)
  doAssert collect(ds) == @[1, 2]

block dataset_reentrant:
  let ds = fromSeq(@[7, 8, 9])
  doAssert collect(ds) == @[7, 8, 9]
  doAssert collect(ds) == @[7, 8, 9]

# ---- prefetch ----------------------------------------------------------------

block prefetch_basic:
  let ds = fromSeq(@[1, 2, 3, 4, 5]).prefetch(3)
  doAssert collect(ds) == @[1, 2, 3, 4, 5]

block prefetch_empty:
  let ds = fromSeq(newSeq[int]()).prefetch(2)
  doAssert collect(ds).len == 0

block prefetch_single_element:
  let ds = fromSeq(@[99]).prefetch(5)
  doAssert collect(ds) == @[99]

block prefetch_buffer_one:
  let ds = fromSeq(@[10, 20, 30]).prefetch(1)
  doAssert collect(ds) == @[10, 20, 30]

block prefetch_zero_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2]).prefetch(0)
  except DataError:
    caught = true
  doAssert caught

block prefetch_reentrant:
  let ds = fromSeq(@[5, 6, 7, 8]).prefetch(3)
  doAssert collect(ds) == @[5, 6, 7, 8]
  doAssert collect(ds) == @[5, 6, 7, 8]

block prefetch_with_shuffle:
  let key = initKey(42'u64)
  let ds = fromSeq(@[1, 2, 3, 4, 5, 6]).shuffle(key, 3).prefetch(3)
  let run1 = collect(ds)
  let run2 = collect(ds)
  doAssert run1 == run2, "prefetch must not affect determinism"
  doAssert run1.len == 6

# ---- parMap ------------------------------------------------------------------

block parmap_identity:
  let ds = fromSeq(@[1, 2, 3, 4]).parMap(proc(x: int): int = x, numWorkers = 2)
  var result = collect(ds)
  result.sort()
  doAssert result == @[1, 2, 3, 4]

block parmap_double:
  let ds = fromSeq(@[1, 2, 3]).parMap(proc(x: int): int = x * 2, numWorkers = 3)
  var result = collect(ds)
  result.sort()
  doAssert result == @[2, 4, 6]

block parmap_empty:
  let ds = fromSeq(newSeq[int]()).parMap(proc(x: int): int = x * 2,
    numWorkers = 2)
  doAssert collect(ds).len == 0

block parmap_order_preserved:
  let ds = fromSeq(@[10, 20, 30]).parMap(proc(x: int): int = x, numWorkers = 2)
  doAssert collect(ds) == @[10, 20, 30]

block parmap_zero_workers_raises:
  var caught = false
  try:
    discard fromSeq(@[1, 2]).parMap(proc(x: int): int = x, numWorkers = 0)
  except DataError:
    caught = true
  doAssert caught

block parmap_reentrant:
  let ds = fromSeq(@[1, 2, 3]).parMap(proc(x: int): int = x + 1,
    numWorkers = 2)
  doAssert collect(ds) == @[2, 3, 4]
  doAssert collect(ds) == @[2, 3, 4]

# ---- pipeline composition with prefetch --------------------------------------

block compose_prefetch_batch:
  let ds = fromRange(0, 10).batch(3).prefetch(2)
  let batches = collect(ds)
  doAssert batches.len == 4
  doAssert batches[0] == @[0, 1, 2]
  doAssert batches[3] == @[9]

block compose_map_prefetch:
  let ds = fromSeq(@[1, 2, 3]).map(proc(x: int): int = x * 10).prefetch(2)
  doAssert collect(ds) == @[10, 20, 30]

echo "tdata: all passed"
