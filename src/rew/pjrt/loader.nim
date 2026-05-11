## PJRT plugin discovery and load.
##
## Resolves the plugin shared library for a `Target` via the binaries
## resolver (manifest-driven download, env-var overrides, or build path),
## then `dlopen`s it and calls `GetPjrtApi` to obtain the function-pointer
## table.
##
## This is the only module that bridges `binaries/resolver` and the raw
## PJRT C API. Everything outside `src/rew/pjrt/` uses the typed wrappers
## in `client.nim` and the registry.

import std/dynlib
import ./capi
import ../binaries/target
import ../binaries/resolver

export capi
export target.Target

proc loadPlugin*(t: Target): PjrtApiHandle =
  ## Loads the PJRT plugin for `t` and returns its API table. The plugin
  ## path is resolved via the binaries resolver (cached downloads, env-var
  ## overrides, or build artifacts). Raises `PjrtError` on failure.
  var path: string
  try:
    path = resolvePluginPath(t)
  except ResolveError as e:
    raisePjrt("Could not resolve PJRT plugin for " & $t & ": " & e.msg)

  let lib = loadLib(path, globalSymbols = false)
  if lib == nil:
    raisePjrt("Could not dlopen PJRT plugin at '" & path &
      "' for target " & $t & ". Check that the shared library exists " &
      "and all runtime dependencies (CUDA, cuDNN, etc.) are available.")

  let sym = symAddr(lib, GetPjrtApiSymbol)
  if sym == nil:
    unloadLib(lib)
    raisePjrt("PJRT plugin '" & path & "' is missing the '" &
      GetPjrtApiSymbol & "' symbol")

  let getter = cast[GetPjrtApiProc](sym)
  let api = getter()
  if api.isNil:
    unloadLib(lib)
    raisePjrt("PJRT plugin '" & path & "' returned a null API table")
  api

proc loadPluginByPath*(path: string): PjrtApiHandle =
  ## Loads a PJRT plugin from an explicit path. Used by tests and tooling
  ## that bypass the manifest-driven resolver.
  let lib = loadLib(path, globalSymbols = false)
  if lib == nil:
    raisePjrt("Could not dlopen PJRT plugin at '" & path & "'")

  let sym = symAddr(lib, GetPjrtApiSymbol)
  if sym == nil:
    unloadLib(lib)
    raisePjrt("PJRT plugin '" & path & "' is missing the '" &
      GetPjrtApiSymbol & "' symbol")

  let getter = cast[GetPjrtApiProc](sym)
  let api = getter()
  if api.isNil:
    unloadLib(lib)
    raisePjrt("PJRT plugin '" & path & "' returned a null API table")
  api

proc pluginFileName*(t: Target): string =
  ## Returns the conventional file name for a plugin of the given target.
  target.pluginFileName(t)
