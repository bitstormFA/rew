## Metadata for StableHLO custom calls and Tokamax-backed kernels.
##
## The actual `stablehlo.custom_call` op lands with the full StableHLO op
## expansion. This module establishes the public metadata shape first so
## custom kernels can be described without pulling Tokamax into rew core.

import ./tools

type
  CustomCallApiVersion* = enum
    ccApiUnspecified
    ccApiTypedFfi

  CustomCallTarget* = object
    ## Backend-specific custom-call target.
    name*: string
    platform*: string

  CustomCallSpec* = object
    ## Description of a custom call payload.
    target*: CustomCallTarget
    hasSideEffect*: bool
    apiVersion*: CustomCallApiVersion
    backendConfig*: string

  CustomCallRegistry* = object
    ## Explicit registry of custom-call targets known to a program.
    specs*: seq[CustomCallSpec]

  TokamaxKernelSpec* = object
    ## Build metadata for a Tokamax-backed custom-call kernel.
    call*: CustomCallSpec
    sources*: seq[string]
    outputLibrary*: string
    buildArgs*: seq[string]

proc initCustomCallTarget*(name: string; platform = ""): CustomCallTarget =
  ## Creates a custom-call target descriptor.
  CustomCallTarget(name: name, platform: platform)

proc initCustomCallSpec*(target: CustomCallTarget; backendConfig = "";
    hasSideEffect = false;
    apiVersion = ccApiUnspecified): CustomCallSpec =
  ## Creates a custom-call spec used by future StableHLO lowering.
  CustomCallSpec(
    target: target,
    hasSideEffect: hasSideEffect,
    apiVersion: apiVersion,
    backendConfig: backendConfig)

proc initTokamaxCall*(kernelName: string; backendConfig = "";
    platform = ""): CustomCallSpec =
  ## Convenience constructor for Tokamax-style custom kernels.
  initCustomCallSpec(
    initCustomCallTarget(kernelName, platform),
    backendConfig = backendConfig,
    apiVersion = ccApiTypedFfi)

func initCustomCallRegistry*(): CustomCallRegistry =
  ## Creates an empty explicit custom-call registry.
  CustomCallRegistry(specs: @[])

proc register*(r: var CustomCallRegistry; spec: CustomCallSpec): int =
  ## Registers `spec`, replacing any existing target/platform entry.
  for i, existing in r.specs:
    if existing.target.name == spec.target.name and
        existing.target.platform == spec.target.platform:
      r.specs[i] = spec
      return i
  r.specs.add spec
  r.specs.len - 1

func hasCustomCall*(r: CustomCallRegistry; name: string;
    platform = ""): bool =
  ## Returns whether `name`/`platform` is registered.
  for spec in r.specs:
    if spec.target.name == name and spec.target.platform == platform:
      return true
  false

proc lookupCustomCall*(r: CustomCallRegistry; name: string;
    platform = ""): CustomCallSpec =
  ## Returns a registered custom call or raises `KeyError`.
  for spec in r.specs:
    if spec.target.name == name and spec.target.platform == platform:
      return spec
  raise newException(KeyError,
    "custom call not registered: " & name & " (" & platform & ")")

proc initTokamaxKernel*(kernelName: string; sources: openArray[string];
    outputLibrary = ""; buildArgs: openArray[string] = [];
    backendConfig = ""; platform = ""): TokamaxKernelSpec =
  ## Creates build metadata for a Tokamax-backed custom-call kernel.
  TokamaxKernelSpec(
    call: initTokamaxCall(kernelName, backendConfig, platform),
    sources: @sources,
    outputLibrary: outputLibrary,
    buildArgs: @buildArgs)

func tokamaxBuildArgs*(spec: TokamaxKernelSpec): seq[string] =
  ## Converts Tokamax kernel metadata into command arguments.
  result = @["build"]
  for src in spec.sources:
    result.add src
  if spec.outputLibrary.len > 0:
    result.add "--output"
    result.add spec.outputLibrary
  for arg in spec.buildArgs:
    result.add arg

proc buildTokamaxKernel*(spec: TokamaxKernelSpec): ToolResult =
  ## Invokes the optional Tokamax tool for `spec`.
  tokamax(tokamaxBuildArgs(spec))
