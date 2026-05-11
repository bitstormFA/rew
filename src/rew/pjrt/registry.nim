## Registry — multi-plugin PJRT API and client cache.
##
## Replaces the single `eagerApi` global from the old `eager.nim` with a
## per-`Target` API table and per-`(Target, ordinal)` client cache. This
## enables true multi-plugin usage (e.g. CPU + CUDA tensors in one process).
##
## The registry is process-global (not thread-local) because PJRT plugins
## are loaded via `dlopen` and their handles are process-scoped. Thread
## safety for concurrent first-use materialization is deferred to v1.1;
## v1 assumes single-threaded init (all plugins loaded from the main
## thread before any parallel work).

import std/tables
import ./capi
import ./client
import ../binaries/target

export target.Target

type
  RegistryError* = object of CatchableError
    ## Raised on registry lookup failures.

  DeviceKey* = tuple[target: Target, ordinal: int]
    ## Composite key for the per-device client cache.

var
  apiTable: Table[Target, PjrtApiHandle]
    ## Loaded plugin API tables, one per target.
  clientTable: Table[DeviceKey, PjrtClient]
    ## Per-(target, ordinal) PJRT clients.

proc registerApi*(t: Target; api: PjrtApiHandle) =
  ## Stores a loaded API handle for `t`. Idempotent if the same handle
  ## is registered twice; raises if a *different* handle is registered
  ## for the same target (indicates a double-load bug).
  if t in apiTable:
    if cast[pointer](apiTable[t]) != cast[pointer](api):
      raise newException(RegistryError,
        "attempted to register a second PJRT API handle for target " & $t)
    return
  apiTable[t] = api

proc lookupApi*(t: Target): PjrtApiHandle =
  ## Returns the API handle for `t`. Raises `RegistryError` if the target
  ## has not been loaded yet.
  if t notin apiTable:
    raise newException(RegistryError,
      "no PJRT plugin loaded for target " & $t &
      ". Call `ensureClient` or `resolvePluginPath` first.")
  apiTable[t]

proc hasApi*(t: Target): bool =
  ## Returns true if a plugin has been loaded for `t`.
  t in apiTable

proc registerClient*(key: DeviceKey; c: PjrtClient) =
  ## Stores a client for the given (target, ordinal) pair.
  clientTable[key] = c

proc lookupClient*(key: DeviceKey): PjrtClient =
  ## Returns the cached client for `key`, or raises `RegistryError`.
  if key notin clientTable:
    raise newException(RegistryError,
      "no PJRT client for " & $key.target & ":" & $key.ordinal)
  clientTable[key]

proc hasClient*(key: DeviceKey): bool =
  ## Returns true if a client exists for `key`.
  key in clientTable

proc releaseBuffer*(t: Target; raw: PjrtBufferRaw) {.raises: [].} =
  ## Frees a raw PJRT buffer against the API table for `t`. Designed to
  ## be called from `BufferHandle`'s destructor. Errors are swallowed so
  ## the destructor contract (`raises: []`) holds.
  if raw.isNil: return
  try:
    if not apiTable.hasKey(t): return
    let api = apiTable[t]
    bufferDestroy(api, raw)
  except CatchableError:
    discard
  except Exception:
    discard

proc loadedTargets*(): seq[Target] =
  ## Returns all targets that have a loaded API handle.
  result = @[]
  for t in apiTable.keys:
    result.add t

proc clearRegistry*() =
  ## Drops all cached API handles and clients. Used by tests.
  apiTable.clear()
  clientTable.clear()
