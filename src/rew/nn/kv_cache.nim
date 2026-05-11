## KV Cache for efficient autoregressive inference.
##
## Caches previous key and value tensors so each generation step
## only computes attention for the new token, reusing past K/V.
##
## Pure value type following rew's functional nn invariant.

import ../tensor
import ../ops/concat

type
  KvCache* = object
    ## Cached key and value tensors for one attention layer.
    ##
    ## `kCache` and `vCache` have shape `[batch, numHeads, cachedLen, headDim]`.
    ## On each step, new K/V are concatenated along the seq dim (dim 2).
    kCache*: Tensor
    vCache*: Tensor
    batchSize*: int
    numHeads*: int
    headDim*: int
    cachedLen*: int
    maxLen*: int

  SlidingWindowKvCache* = object
    ## KV cache with a sliding window: only the last `windowSize` tokens
    ## are retained.
    kv*: KvCache
    windowSize*: int

proc initKvCache*(batchSize, numHeads, maxLen, headDim: int): KvCache =
  ## Constructs an empty KV cache.
  ##
  ## `batchSize` — number of sequences in the batch.
  ## `numHeads` — number of attention heads.
  ## `maxLen` — maximum cache length.
  ## `headDim` — per-head dimension.
  KvCache(
    batchSize: batchSize,
    numHeads: numHeads,
    headDim: headDim,
    cachedLen: 0,
    maxLen: maxLen,
  )

proc append*(cache: KvCache; kNew, vNew: Tensor): KvCache =
  ## Returns a new `KvCache` with `kNew` and `vNew` concatenated.
  ##
  ## `kNew` / `vNew` have shape `[batch, numHeads, 1, headDim]`
  ## (single new token) or `[batch, numHeads, newLen, headDim]`.
  if cache.cachedLen == 0:
    result = KvCache(
      kCache: kNew,
      vCache: vNew,
      batchSize: cache.batchSize,
      numHeads: cache.numHeads,
      headDim: cache.headDim,
      cachedLen: kNew.shape[2],
      maxLen: cache.maxLen,
    )
  else:
    let newLen = kNew.shape[2]
    let totalLen = cache.cachedLen + newLen
    if totalLen > cache.maxLen:
      raise newException(TensorError,
        "KvCache.append: total length " & $totalLen &
          " exceeds maxLen " & $cache.maxLen)
    result = KvCache(
      kCache: concat([cache.kCache, kNew], 2),
      vCache: concat([cache.vCache, vNew], 2),
      batchSize: cache.batchSize,
      numHeads: cache.numHeads,
      headDim: cache.headDim,
      cachedLen: totalLen,
      maxLen: cache.maxLen,
    )

proc window*(cache: KvCache; startPos, windowLen: int): (Tensor, Tensor) =
  ## Retrieves K/V tensors from `startPos` with length `windowLen`.
  ## Returns `(kSlice, vSlice)`.
  if startPos < 0 or windowLen <= 0:
    raise newException(TensorError,
      "KvCache.window: invalid startPos=" & $startPos &
        " windowLen=" & $windowLen)
  if startPos + windowLen > cache.cachedLen:
    raise newException(TensorError,
      "KvCache.window: range [" & $startPos & ":" & $(startPos + windowLen) &
        ") exceeds cachedLen " & $cache.cachedLen)
  let rank = cache.kCache.shape.len
  var starts = newSeq[int](rank)
  var limits = newSeq[int](rank)
  var strides = newSeq[int](rank)
  for i in 0 ..< rank:
    starts[i] = 0
    limits[i] = cache.kCache.shape[i]
    strides[i] = 1
  starts[2] = startPos
  limits[2] = startPos + windowLen
  let kSlice = slice(cache.kCache, starts, limits, strides)
  let vSlice = slice(cache.vCache, starts, limits, strides)
  (kSlice, vSlice)

# ---- Sliding window variant ------------------------------------------------

proc initSlidingWindowKvCache*(batchSize, numHeads, maxLen, headDim: int;
    windowSize: int): SlidingWindowKvCache =
  ## Constructs a sliding-window KV cache. Only the last `windowSize`
  ## tokens are retained.
  if windowSize > maxLen:
    raise newException(TensorError,
      "initSlidingWindowKvCache: windowSize " & $windowSize &
        " must not exceed maxLen " & $maxLen)
  SlidingWindowKvCache(
    kv: initKvCache(batchSize, numHeads, maxLen, headDim),
    windowSize: windowSize,
  )

proc append*(cache: SlidingWindowKvCache; kNew, vNew: Tensor):
    SlidingWindowKvCache =
  ## Appends new K/V and truncates to the sliding window.
  var full = cache.kv.append(kNew, vNew)
  let totalLen = full.cachedLen
  if totalLen > cache.windowSize:
    let dropLen = totalLen - cache.windowSize
    let rank = kNew.shape.len
    var starts = newSeq[int](rank)
    var limits = newSeq[int](rank)
    var strides = newSeq[int](rank)
    for i in 0 ..< rank:
      starts[i] = 0
      limits[i] = full.kCache.shape[i]
      strides[i] = 1
    starts[2] = dropLen
    limits[2] = totalLen
    full = KvCache(
      kCache: slice(full.kCache, starts, limits, strides),
      vCache: slice(full.vCache, starts, limits, strides),
      batchSize: full.batchSize,
      numHeads: full.numHeads,
      headDim: full.headDim,
      cachedLen: cache.windowSize,
      maxLen: full.maxLen,
    )
  SlidingWindowKvCache(kv: full, windowSize: cache.windowSize)
