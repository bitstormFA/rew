## Pipeline combinators — lazy transforms over `Dataset[T]`.
##
## Each combinator takes a `Dataset[T]` and returns a new `Dataset[U]`,
## enabling chainable pipelines: `ds.map(fn).shuffle(key, 1000).batch(32)`.
## All transforms are lazy — elements are produced one at a time through
## closure iterators.

import ./dataset
import ../rng

proc map*[T, U](ds: Dataset[T]; fn: proc(x: T): U): Dataset[U] =
  ## Applies `fn` to each element.
  result.source = proc(): iterator(): U =
    let src = ds.source()
    result = iterator(): U {.closure.} =
      while true:
        let item = src()
        if finished(src): break
        yield fn(item)

proc filter*[T](ds: Dataset[T]; pred: proc(x: T): bool): Dataset[T] =
  ## Yields only elements for which `pred` returns true.
  result.source = proc(): iterator(): T =
    let src = ds.source()
    result = iterator(): T {.closure.} =
      while true:
        let item = src()
        if finished(src): break
        if pred(item):
          yield item

proc batch*[T](ds: Dataset[T]; batchSize: int;
    dropLast: bool = false): Dataset[seq[T]] =
  ## Groups consecutive elements into batches of `batchSize`. The final
  ## batch may be smaller unless `dropLast` is true.
  if batchSize <= 0:
    raise newException(DataError, "batch: batchSize must be positive")
  result.source = proc(): iterator(): seq[T] =
    let src = ds.source()
    let bs = batchSize
    let dl = dropLast
    result = iterator(): seq[T] {.closure.} =
      var buf = newSeq[T](0)
      while true:
        let item = src()
        if finished(src): break
        buf.add(item)
        if buf.len == bs:
          yield buf
          buf = newSeq[T](0)
      if buf.len > 0 and not dl:
        yield buf

proc repeat*[T](ds: Dataset[T]; epochs: int = -1): Dataset[T] =
  ## Repeats the dataset `epochs` times. Pass -1 for infinite repetition.
  if epochs < -1:
    raise newException(DataError,
      "repeat: epochs must be -1 or non-negative")
  result.source = proc(): iterator(): T =
    let ep = epochs
    let sourceFn = ds.source
    result = iterator(): T {.closure.} =
      if ep < 0:
        while true:
          let it = sourceFn()
          var yielded = false
          while true:
            let item = it()
            if finished(it): break
            yielded = true
            yield item
          if not yielded:
            break
      else:
        for _ in 0 ..< ep:
          let it = sourceFn()
          while true:
            let item = it()
            if finished(it): break
            yield item

proc take*[T](ds: Dataset[T]; n: int): Dataset[T] =
  ## Yields at most `n` elements from the dataset.
  if n < 0:
    raise newException(DataError, "take: n must be non-negative")
  result.source = proc(): iterator(): T =
    let src = ds.source()
    let limit = n
    result = iterator(): T {.closure.} =
      var count = 0
      while count < limit:
        let item = src()
        if finished(src): break
        yield item
        inc count

proc skip*[T](ds: Dataset[T]; n: int): Dataset[T] =
  ## Drops the first `n` elements, then yields the rest.
  if n < 0:
    raise newException(DataError, "skip: n must be non-negative")
  result.source = proc(): iterator(): T =
    let src = ds.source()
    let toSkip = n
    result = iterator(): T {.closure.} =
      var skipped = 0
      while skipped < toSkip:
        discard src()
        if finished(src): return
        inc skipped
      while true:
        let item = src()
        if finished(src): break
        yield item

proc zip*[A, B](a: Dataset[A]; b: Dataset[B]):
    Dataset[tuple[a: A, b: B]] =
  ## Pairs elements from two datasets. Stops when either is exhausted.
  result.source = proc(): iterator(): tuple[a: A, b: B] =
    let srcA = a.source()
    let srcB = b.source()
    result = iterator(): tuple[a: A, b: B] {.closure.} =
      while true:
        let va = srcA()
        if finished(srcA): break
        let vb = srcB()
        if finished(srcB): break
        yield (a: va, b: vb)

proc enumerate*[T](ds: Dataset[T]; start: int = 0):
    Dataset[tuple[idx: int, val: T]] =
  ## Pairs each element with a running index starting at `start`.
  result.source = proc(): iterator(): tuple[idx: int, val: T] =
    let src = ds.source()
    let s = start
    result = iterator(): tuple[idx: int, val: T] {.closure.} =
      var idx = s
      while true:
        let item = src()
        if finished(src): break
        yield (idx: idx, val: item)
        inc idx

proc shuffle*[T](ds: Dataset[T]; key: Key;
    bufferSize: int): Dataset[T] =
  ## Shuffles elements using a reservoir buffer of `bufferSize`. Uses
  ## rew's explicit `Key` PRNG — each epoch call gets a deterministically
  ## derived key so runs are reproducible.
  ##
  ## Algorithm: fill a buffer of `bufferSize` elements, then for each new
  ## element swap it with a random buffer slot and yield the displaced
  ## element. At exhaustion, Fisher-Yates shuffle the remaining buffer.
  if bufferSize <= 0:
    raise newException(DataError, "shuffle: bufferSize must be positive")
  result.source = proc(): iterator(): T =
    let src = ds.source()
    let bs = bufferSize
    let k = key
    result = iterator(): T {.closure.} =
      var buf = newSeq[T](0)
      var rngKey = k
      var idx = 0

      while buf.len < bs:
        let item = src()
        if finished(src): break
        buf.add(item)

      if buf.len > 0:
        while true:
          let item = src()
          if finished(src): break
          let slot = int(foldIn(rngKey, uint64(idx)).a) mod buf.len
          yield buf[slot]
          buf[slot] = item
          inc idx

        # Fisher-Yates shuffle the remaining buffer.
        for i in countdown(buf.len - 1, 1):
          let j = int(foldIn(rngKey, uint64(idx)).a) mod (i + 1)
          swap(buf[i], buf[j])
          inc idx
        for i in 0 ..< buf.len:
          yield buf[i]

proc concat*[T](a, b: Dataset[T]): Dataset[T] =
  ## Concatenates two datasets: yields all elements of `a` followed by
  ## all elements of `b`.
  result.source = proc(): iterator(): T =
    let srcA = a.source()
    let srcB = b.source()
    result = iterator(): T {.closure.} =
      while true:
        let item = srcA()
        if finished(srcA): break
        yield item
      while true:
        let item = srcB()
        if finished(srcB): break
        yield item

proc window*[T](ds: Dataset[T]; size, shift: int): Dataset[seq[T]] =
  ## Sliding window of `size` elements, advanced by `shift` each step.
  ## Partial windows at the end are dropped. Each window is yielded as a
  ## `seq[T]`.
  if size <= 0:
    raise newException(DataError, "window: size must be positive")
  if shift <= 0:
    raise newException(DataError, "window: shift must be positive")
  result.source = proc(): iterator(): seq[T] =
    let src = ds.source()
    let sz = size
    let sh = shift
    result = iterator(): seq[T] {.closure.} =
      var buf = newSeq[T](0)
      while buf.len < sz:
        let item = src()
        if finished(src): break
        buf.add(item)
      if buf.len == sz:
        yield buf
        while true:
          for _ in 0 ..< sh:
            buf.delete(0)
            let item = src()
            if finished(src): return
            buf.add(item)
          yield buf

proc unbatch*[T](ds: Dataset[seq[T]]): Dataset[T] =
  ## Flattens a dataset of batches into a dataset of individual elements.
  result.source = proc(): iterator(): T =
    let src = ds.source()
    result = iterator(): T {.closure.} =
      while true:
        let batch = src()
        if finished(src): break
        for item in batch:
          yield item

proc flatMap*[T, U](ds: Dataset[T]; fn: proc(x: T): Dataset[U]): Dataset[U] =
  ## Applies `fn` to each element and flattens the resulting datasets.
  result.source = proc(): iterator(): U =
    let src = ds.source()
    result = iterator(): U {.closure.} =
      while true:
        let item = src()
        if finished(src): break
        let inner = fn(item).source()
        while true:
          let innerItem = inner()
          if finished(inner): break
          yield innerItem
