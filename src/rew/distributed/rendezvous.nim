## Small stdlib rendezvous KV store used by tests and worker bootstrap.

import std/[os, tables, times]

type
  RendezvousTimeoutError* = object of CatchableError

  RendezvousStore* = object
    values: Table[string, string]

proc initRendezvousStore*(): RendezvousStore =
  RendezvousStore(values: initTable[string, string]())

proc put*(store: var RendezvousStore; key, value: string) =
  store.values[key] = value

proc tryGet*(store: RendezvousStore; key: string): string =
  if store.values.hasKey(key):
    store.values[key]
  else:
    ""

proc contains*(store: RendezvousStore; key: string): bool =
  store.values.hasKey(key)

proc get*(store: RendezvousStore; key: string;
    timeoutMs: int = 30000): string =
  let start = epochTime()
  while true:
    if store.values.hasKey(key):
      return store.values[key]
    if int((epochTime() - start) * 1000.0) >= timeoutMs:
      raise newException(RendezvousTimeoutError,
        "rendezvous get timed out for key '" & key & "'")
    sleep(1)
