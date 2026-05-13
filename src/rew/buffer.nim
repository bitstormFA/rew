## BufferHandle — owning handle for a PJRT device buffer.
##
## This is the **only** approved `ref object` in the codebase outside of
## `src/rew/pjrt/`. The underlying object owns the PJRT buffer; Nim's `ref`
## machinery provides the refcount, and the `=destroy` hook on the inner
## object releases the device buffer exactly once when the last reference
## drops.
##
## **Donation.** When a buffer is consumed by a `jit`'d call as a donated
## argument, its state is flipped to `bsDonated`. Any further use raises
## `BufferDonatedError` with a message naming the consuming `jit` call. The
## buffer object is still released by `=destroy`; donation invalidates user
## access to the value, not the handle's responsibility to destroy the PJRT
## buffer wrapper.
##
## **Releaser.** The destructor delegates to `pjrt/registry.releaseBuffer`
## using the `target` field to look up the correct API table. This keeps
## the layering rules clean (this module does not import the PJRT API
## table directly) and supports multi-plugin usage.

import ./pjrt/capi
import ./binaries/target

export target.Target

type
  BufferState* = enum
    ## Lifecycle state of a `BufferHandle`.
    bsLive      ## Owns a live PJRT buffer; `=destroy` will release it.
    bsDonated   ## Consumed by a donating `jit` call; unusable but still owned.
    bsReleased  ## Already released (post-`=destroy` or post-explicit free).

  BufferReleaser* = proc (t: Target; raw: PjrtBufferRaw) {.nimcall, raises: [].}
    ## Function that frees a raw PJRT buffer for a given target. Provided
    ## at handle creation by the eager layer. `nimcall` keeps the proc
    ## non-capturing; `raises: []` matches the destructor contract.

  BufferDonatedError* = object of CatchableError
    ## Raised on any operation against a `BufferHandle` whose state is
    ## `bsDonated` (or, defensively, `bsReleased`).

  BufferHandleObj* = object
    ## The owning object. Constructed only via `newBufferHandle`. The `ref`
    ## alias `BufferHandle` is what the rest of the codebase passes around.
    state*: BufferState
    target*: Target
    raw*: PjrtBufferRaw
    raws*: seq[PjrtBufferRaw]
      ## Optional per-addressable-device storage for a global sharded tensor.
      ## `raw` remains the primary/first buffer for existing single-device
      ## code paths. When `raws` is non-empty the destructor owns every entry
      ## in `raws` and does not release `raw` separately.
    rawShardIndices*: seq[int]
      ## Global mesh-linear shard index for each entry in `raws`. Empty means
      ## legacy order (`raws[i]` stores shard `i`).
    releaser*: BufferReleaser
    donatedBy*: string
      ## Free-form context (typically the consuming `jit` callsite) attached
      ## when the handle is donated; surfaces in the `BufferDonatedError`
      ## message.
    sizeBytes*: int
      ## On-device size in bytes if known. Used by debug/HLO-dump helpers;
      ## a value of 0 means "unknown".

  BufferHandle* = ref BufferHandleObj
    ## Refcounted alias. Cheap to copy (just bumps Nim's ARC counter); the
    ## inner buffer is released exactly once when the last reference drops.

proc `=destroy`*(h: BufferHandleObj) {.raises: [].} =
  ## Owning destructor. Releases the PJRT buffer iff the handle still owns
  ## it (`bsLive`/`bsDonated`, non-nil `raw`, non-nil `releaser`). Donated
  ## handles are logically unusable but still own their PJRT buffer object.
  if (h.state == bsLive or h.state == bsDonated) and h.releaser != nil:
    if h.raws.len > 0:
      for raw in h.raws:
        if not raw.isNil:
          h.releaser(h.target, raw)
    elif not h.raw.isNil:
      h.releaser(h.target, h.raw)
  `=destroy`(h.raws)
  `=destroy`(h.rawShardIndices)
  `=destroy`(h.donatedBy)

proc newBufferHandle*(target: Target; raw: PjrtBufferRaw;
    releaser: BufferReleaser; sizeBytes: int = 0): BufferHandle =
  ## Constructs a fresh `BufferHandle` in the `bsLive` state.
  ##
  ## `target` identifies which plugin's API table to use for release.
  ## `raw` is the device-buffer pointer returned by PJRT. `releaser` is
  ## the non-capturing thunk that frees it. `sizeBytes` is optional
  ## metadata used by debug helpers.
  ##
  ## Caller-side rule: each raw buffer must be wrapped exactly once.
  BufferHandle(state: bsLive, target: target, raw: raw, releaser: releaser,
    sizeBytes: sizeBytes)

proc newBufferSetHandle*(target: Target; raws: openArray[PjrtBufferRaw];
    releaser: BufferReleaser; sizeBytes: int = 0;
    shardIndices: openArray[int] = []): BufferHandle =
  ## Constructs an owning handle for one logical global tensor backed by one
  ## PJRT buffer per addressable device. `raws[0]` is also exposed through
  ## `raw` so legacy single-buffer diagnostics can still identify storage.
  ## `shardIndices`, when provided, maps each raw buffer to its global
  ## mesh-linear shard index.
  ##
  ## Caller-side rule: each raw buffer in `raws` must be wrapped exactly once.
  if shardIndices.len > 0 and shardIndices.len != raws.len:
    raise newException(ValueError,
      "newBufferSetHandle: shardIndices length " & $shardIndices.len &
        " does not match raws length " & $raws.len)
  let primary =
    if raws.len == 0: nil.PjrtBufferRaw
    else: raws[0]
  var indices: seq[int] = @[]
  if shardIndices.len > 0:
    indices = @shardIndices
  BufferHandle(state: bsLive, target: target, raw: primary, raws: @raws,
    rawShardIndices: indices, releaser: releaser, sizeBytes: sizeBytes)

proc isBufferSet*(h: BufferHandle): bool =
  ## True when this handle owns one buffer per addressable device.
  not h.isNil and h.raws.len > 0

proc bufferCount*(h: BufferHandle): int =
  ## Number of owned raw PJRT buffers.
  if h.isNil:
    0
  elif h.raws.len > 0:
    h.raws.len
  elif not h.raw.isNil:
    1
  else:
    0

proc shardIndices*(h: BufferHandle): seq[int] =
  ## Returns the global shard index for each raw buffer in this handle.
  if h.isNil or h.raws.len == 0:
    return @[]
  if h.rawShardIndices.len > 0:
    result = h.rawShardIndices
  else:
    result = newSeq[int](h.raws.len)
    for i in 0 ..< result.len:
      result[i] = i

proc markDonated*(h: BufferHandle; consumer: string) =
  ## Flips the handle to the donated state. Idempotent on already-donated
  ## handles (later donations replace the `consumer` string).
  ##
  ## After donation, reads and transfers are rejected. ARC still calls the
  ## releaser when the last reference drops, because the wrapper object
  ## remains ours even if PJRT may have reused the backing allocation.
  if h.isNil:
    raise newException(ValueError, "markDonated: nil buffer handle")
  h.state = bsDonated
  h.donatedBy = consumer

proc raiseDonated(h: BufferHandle; opName: string) {.noreturn, noinline.} =
  ## Centralised helper that builds the `BufferDonatedError` message.
  let suffix =
    if h.donatedBy.len > 0: " (donated to: " & h.donatedBy & ")"
    else: ""
  raise newException(BufferDonatedError,
    opName & ": buffer has been donated and may no longer be used" & suffix)

proc requireLive*(h: BufferHandle; opName: string) =
  ## Guard called by every op that reads from a `BufferHandle`. Raises
  ## `BufferDonatedError` for `bsDonated` and `bsReleased` states.
  if h.isNil:
    raise newException(ValueError, opName & ": nil buffer handle")
  case h.state
  of bsLive: discard
  of bsDonated, bsReleased: raiseDonated(h, opName)

proc isLive*(h: BufferHandle): bool =
  ## Convenience predicate; true iff `requireLive` would not raise.
  not h.isNil and h.state == bsLive

proc isDonated*(h: BufferHandle): bool =
  ## True iff the handle is in the `bsDonated` state.
  not h.isNil and h.state == bsDonated
