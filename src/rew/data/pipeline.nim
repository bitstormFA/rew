## Performance-oriented transforms — prefetch and parallel map.
##
## These transforms use background threads to decouple data preparation
## from model training. Requires `--threads:on` compilation flag.
##
## ## Design note
##
## Nim's closure iterators cannot be passed to `{.thread.}` procs directly.
## To work around this, `prefetch` and `parMap` materialise the source
## elements into a `seq[T]` before spawning worker threads. For datasets
## that are already backed by an in-memory `seq` (e.g. `fromSeq`, `fromNpy`,
## `fromRange`), this doubles memory usage for the source data. A future
## streaming version will use chunked materialisation to keep memory bounded.

import std/[typedthreads, options]
import ./dataset

type
  PrefetchState[T] = object
    ch: Channel[Option[T]]

  ParMapWorkerArgs[T, U] = object
    items: ptr seq[T]
    outputs: ptr seq[U]
    fn: proc(x: T): U {.gcsafe.}
    first, last: int

proc prefetchProducer[T](args: tuple[s: ptr PrefetchState[T], items: seq[T]]) {.thread.} =
  for x in args.items:
    args.s.ch.send(some(x))

proc parMapWorker[T, U](args: ParMapWorkerArgs[T, U]) {.thread.} =
  for i in args.first ..< args.last:
    args.outputs[][i] = args.fn(args.items[][i])

proc prefetch*[T](ds: Dataset[T]; bufferSize: int = 2): Dataset[T] =
  ## Prefetches elements from `ds` using a background thread and a bounded
  ## channel of `bufferSize` elements. The producer thread runs ahead of
  ## the consumer, decoupling data preparation from model training.
  ##
  ## Requires `--threads:on` compilation flag.
  ##
  ## The source elements are materialised into a `seq[T]` before the
  ## producer thread starts. For datasets already backed by in-memory
  ## sequences (`fromSeq`, `fromNpy`), this doubles the resident memory;
  ## for streaming sources, the entire stream is buffered.
  if bufferSize <= 0:
    raise newException(DataError, "prefetch: bufferSize must be positive")
  result.source = proc(): iterator(): T =
    let inner = ds.source
    let innerIter = inner()

    var items = newSeq[T]()
    while true:
      let x = innerIter()
      if finished(innerIter): break
      items.add(x)

    var state: PrefetchState[T]
    state.ch.open(bufferSize)

    var thr: Thread[tuple[s: ptr PrefetchState[T], items: seq[T]]]
    createThread(thr, prefetchProducer[T], (s: addr state, items: items))

    let count = items.len
    result = iterator(): T {.closure.} =
      var received = 0
      while received < count:
        let (has, val) = state.ch.tryRecv()
        if has:
          yield val.get()
          inc received
      joinThread(thr)
      state.ch.close()

proc parMap*[T, U](ds: Dataset[T]; fn: proc(x: T): U {.gcsafe.};
    numWorkers: int = 4): Dataset[U] =
  ## Applies `fn` to each element using up to `numWorkers` worker threads.
  ## Results preserve source order.
  ##
  ## Unlike `map`, `parMap` materialises the entire source before yielding
  ## so workers can write into fixed output slots safely.
  if numWorkers <= 0:
    raise newException(DataError, "parMap: numWorkers must be positive")
  result.source = proc(): iterator(): U =
    let inner = ds.source
    let innerIter = inner()

    var items = newSeq[T]()
    while true:
      let x = innerIter()
      if finished(innerIter): break
      items.add(x)

    result = iterator(): U {.closure.} =
      var outputs = newSeq[U](items.len)
      if items.len == 0:
        discard
      elif numWorkers == 1 or items.len == 1:
        for i, x in items:
          outputs[i] = fn(x)
      else:
        let workerCount = min(numWorkers, items.len)
        let chunkSize = (items.len + workerCount - 1) div workerCount
        var threads = newSeq[Thread[ParMapWorkerArgs[T, U]]](workerCount)
        var args = newSeq[ParMapWorkerArgs[T, U]](workerCount)
        for worker in 0 ..< workerCount:
          let first = worker * chunkSize
          let last = min(first + chunkSize, items.len)
          args[worker] = ParMapWorkerArgs[T, U](
            items: addr items,
            outputs: addr outputs,
            fn: fn,
            first: first,
            last: last,
          )
          createThread(threads[worker], parMapWorker[T, U], args[worker])
        for worker in 0 ..< workerCount:
          joinThread(threads[worker])
      for y in outputs:
        yield y
