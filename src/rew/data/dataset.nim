## Dataset — core type and in-memory sources.
##
## A `Dataset[T]` wraps a `DatasetFn[T]` factory proc that returns a fresh
## closure iterator each time it is called. One call = one epoch.
## This avoids `ref object` while giving tf.data-style re-entrant semantics.
##
## `DatasetFn[T]` is the low-level factory type; `Dataset[T]` is the
## user-facing chainable pipeline object.

type
  DatasetFn*[T] = proc(): iterator(): T
    ## A dataset factory that returns a fresh closure iterator per epoch.
    ## Each call produces an independent traversal of the data.

  Dataset*[T] = object
    ## A value-object dataset pipeline. Holds a source factory and supports
    ## chainable transforms (`.map`, `.batch`, `.shuffle`, `.prefetch`, …).
    source*: DatasetFn[T]

  DataError* = object of CatchableError
    ## Raised by data pipeline operations on validation failures.

proc toDataset*[T](fn: DatasetFn[T]): Dataset[T] =
  ## Wraps a bare `DatasetFn[T]` into a `Dataset[T]` pipeline object.
  Dataset[T](source: fn)

proc fromSeq*[T](data: seq[T]): Dataset[T] =
  ## Creates a dataset from an in-memory sequence. Each epoch iterates
  ## over `data` in order.
  result.source = proc(): iterator(): T =
    let captured = data
    result = iterator(): T {.closure.} =
      for i in 0 ..< captured.len:
        yield captured[i]

proc fromRange*(start, stop: int; step: int = 1): Dataset[int] =
  ## Creates a dataset that yields integers in `[start, stop)` with the
  ## given step. Useful for index-based pipelines and tests.
  if step == 0:
    raise newException(DataError, "fromRange: step must be non-zero")
  result.source = proc(): iterator(): int =
    let s = start
    let e = stop
    let st = step
    result = iterator(): int {.closure.} =
      var i = s
      if st > 0:
        while i < e:
          yield i
          i += st
      else:
        while i > e:
          yield i
          i += st

iterator items*[T](ds: Dataset[T]): T =
  ## Iterates over all elements of `ds`. One complete traversal.
  let it = ds.source()
  while true:
    let x = it()
    if finished(it): break
    yield x
