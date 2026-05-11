## Lazy eager batching.
##
## `lazy(fn)` batches a user-supplied eager op sequence into one traced,
## cached PJRT program. It is intentionally runtime-only, like `jit`: no
## macro syntax, no hidden host transfers, and the same explicit donation
## behavior as `JitFunction`.

import ../tensor
import ../stablehlo/ir
import ./jit

type
  LazyFunction* = object
    ## Cached lazy eager batch.
    compiled*: JitFunction

proc lazy*(fn: JitFn; funcName = "lazy_fn";
    donateArgs: openArray[int] = []): LazyFunction =
  ## Wraps `fn` as a lazy eager batch.
  ##
  ## The first call for a shape/dtype/device signature traces and compiles
  ## the whole op sequence; later calls reuse the cached executable.
  LazyFunction(compiled: jit(fn, funcName, donateArgs))

proc call*(lazyFn: LazyFunction; args: openArray[Tensor]): seq[Tensor] =
  ## Executes the lazy batch for `args`.
  lazyFn.compiled.call(args)

proc lower*(lazyFn: LazyFunction; args: openArray[Tensor]): ShModule =
  ## Returns the lowered StableHLO module for this input signature.
  lazyFn.compiled.lower(args)

proc text*(lazyFn: LazyFunction; args: openArray[Tensor]): string =
  ## Returns StableHLO text for this input signature.
  lazyFn.compiled.text(args)

proc executableText*(lazyFn: LazyFunction; args: openArray[Tensor]): string =
  ## Returns PJRT-ready StableHLO text for this input signature.
  lazyFn.compiled.executableText(args)

proc cacheSize*(lazyFn: LazyFunction): int =
  ## Number of cached input signatures.
  lazyFn.compiled.cacheSize()

proc clearCache*(lazyFn: LazyFunction) =
  ## Clears cached lowerings and executables.
  lazyFn.compiled.clearCache()

proc lazyCall*(fn: JitFn; args: openArray[Tensor]; funcName = "lazy_fn";
    donateArgs: openArray[int] = []): seq[Tensor] =
  ## One-shot lazy eager batch construction and execution.
  lazy(fn, funcName, donateArgs).call(args)
