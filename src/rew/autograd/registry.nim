## Autograd VJP registry — single source of truth for reverse-mode rules.
##
## ## Invariant #5
## One vjp registry feeds both tape-based eager `backward()` and the
## functional `grad`/`vjp` transform. Reverse-mode only.
##
## Differentiable ops are declared with `registerVjp("opName")` and wired
## to a concrete closure with `registerVjpRule("opName", rule)`. Ops that
## intentionally do not participate in reverse mode use `registerNoGrad`.
## The coverage lint checks that every `{.rewOp.}` has exactly one policy,
## and the test suite checks that every differentiable policy has a real rule.

import std/sets
import std/tables
import ../tensor
import ./tape

type
  GradientPolicy* = enum
    ## Autograd policy for a public op.
    gpDifferentiable
    gpNoGradient

  VjpRuleStatus* = enum
    ## Runtime status for an op in the reverse-mode registry.
    vrsUnregistered
    vrsNoGradient
    vrsDeclared
    vrsInstalled

  VjpRegistryError* = object of CatchableError
    ## Raised on duplicate registration. Names must be unique.

  VjpRule* = proc (primals: openArray[Tensor]; output: Tensor;
      cotangent: Tensor; intAttrs: IntAttrs): seq[Tensor] {.closure.}
    ## Reverse-mode rule. Returns the cotangent of every input in
    ## `primals` order. Implementations call the public op API to build
    ## the gradient subgraph; the tape is automatically paused while a
    ## rule runs.

var registered {.threadvar.}: HashSet[string]
var noGradient {.threadvar.}: HashSet[string]
var rules {.threadvar.}: TableRef[string, VjpRule]

proc registerVjp*(opName: string) =
  ## Declares `opName` as differentiable.
  ##
  ## A concrete closure must be installed with `registerVjpRule` before
  ## transforms can differentiate through this op.
  if opName in registered or opName in noGradient:
    raise newException(VjpRegistryError,
      "autograd policy for '" & opName & "' is already registered")
  registered.incl opName

proc registerNoGrad*(opName: string) =
  ## Records that `opName` is intentionally non-differentiable.
  if opName in registered or opName in noGradient:
    raise newException(VjpRegistryError,
      "autograd policy for '" & opName & "' is already registered")
  noGradient.incl opName

proc registerVjpRule*(opName: string; rule: VjpRule) =
  ## Attaches the actual reverse-mode closure for `opName`. The name
  ## must already be registered via `registerVjp` (the lint checks for
  ## that). Re-registration overwrites; the autograd module installs all
  ## rules at startup.
  if opName notin registered:
    raise newException(VjpRegistryError,
      "registerVjpRule: '" & opName & "' has no registerVjp(...) entry")
  if rules.isNil: rules = newTable[string, VjpRule]()
  rules[opName] = rule

proc hasVjp*(opName: string): bool =
  ## True iff `opName` is declared differentiable.
  opName in registered

proc hasNoGradient*(opName: string): bool =
  ## True iff `opName` is explicitly registered as non-differentiable.
  opName in noGradient

proc gradientPolicy*(opName: string): GradientPolicy =
  ## Returns the autograd policy for `opName`.
  if opName in registered:
    return gpDifferentiable
  if opName in noGradient:
    return gpNoGradient
  raise newException(VjpRegistryError,
    "no autograd policy registered for '" & opName & "'")

proc hasVjpRule*(opName: string): bool =
  ## True iff a real reverse-mode closure has been installed for
  ## `opName`. Distinct from `hasVjp`, which only checks the declared
  ## differentiable policy.
  not rules.isNil and opName in rules

proc vjpRuleStatus*(opName: string): VjpRuleStatus =
  ## Returns the full reverse-mode registration status for `opName`.
  if opName in noGradient:
    return vrsNoGradient
  if opName in registered:
    if not rules.isNil and opName in rules:
      return vrsInstalled
    return vrsDeclared
  vrsUnregistered

proc getVjpRule*(opName: string): VjpRule =
  ## Returns the installed rule. Raises `VjpRegistryError` when no rule
  ## is wired \u2014 callers should guard with `hasVjpRule` if a missing
  ## rule is recoverable.
  if rules.isNil or opName notin rules:
    raise newException(VjpRegistryError,
      "no vjp rule installed for '" & opName & "'")
  rules[opName]

proc registeredVjps*(): seq[string] =
  ## Snapshot of all differentiable op names. Stable order is not
  ## guaranteed.
  for n in registered:
    result.add n

proc registeredNoGradOps*(): seq[string] =
  ## Snapshot of all explicit no-gradient op names.
  for n in noGradient:
    result.add n

proc registeredAutogradOps*(): seq[string] =
  ## Snapshot of all public ops with any autograd policy.
  for n in registered:
    result.add n
  for n in noGradient:
    result.add n

proc missingVjpRules*(): seq[string] =
  ## Snapshot of differentiable ops whose concrete VJP rule is not installed.
  for n in registered:
    if rules.isNil or n notin rules:
      result.add n

# --- Autograd policy registrations ------------------------------------------
#
# Names listed here must match every `{.rewOp.}`-marked proc exported by
# `src/rew/tensor.nim` and `src/rew/ops/*.nim`. The architectural lint
# `tools/check_vjp_coverage.nim` enforces the bidirectional match.
# Differentiable entries below must have concrete closures installed from
# `src/rew/autograd/rules.nim`; intentional token/discrete/stateful ops are
# registered with `registerNoGrad`.

registerVjp("add")
registerVjp("sub")
registerVjp("mul")
registerVjp("neg")
registerVjp("divide")
registerVjp("maximum")
registerVjp("minimum")
registerVjp("atan2")
registerVjp("power")
registerNoGrad("remainder")
registerNoGrad("bitwiseAnd")
registerNoGrad("bitwiseOr")
registerNoGrad("bitwiseXor")
registerNoGrad("shiftLeft")
registerNoGrad("shiftRightArithmetic")
registerNoGrad("shiftRightLogical")
registerVjp("exp")
registerVjp("log")
registerVjp("sqrt")
registerVjp("abs")
registerVjp("tanh")
registerVjp("cbrt")
registerNoGrad("ceil")
registerVjp("expm1")
registerNoGrad("floor")
registerVjp("log1p")
registerVjp("logistic")
registerVjp("tan")
registerNoGrad("sign")
registerNoGrad("roundNearestAfz")
registerNoGrad("roundNearestEven")
registerNoGrad("bitwiseNot")
registerNoGrad("countLeadingZeros")
registerNoGrad("popcnt")
registerVjp("optimizationBarrier")
registerVjp("stopGradient")
registerNoGrad("astype")
registerNoGrad("bitcastConvert")
registerNoGrad("isFinite")
registerNoGrad("reducePrecision")
registerNoGrad("complex")
registerNoGrad("real")
registerNoGrad("imag")
registerVjp("reshape")
registerVjp("transpose")
registerVjp("reverse")
registerVjp("reduceSum")
registerVjp("reduceMax")
registerVjp("reduceMin")
registerVjp("reduceProd")
registerNoGrad("all")
registerNoGrad("any")
registerVjp("dot")
registerVjp("dotGeneral")
registerVjp("matmul")
registerVjp("broadcastTo")
registerVjp("concat")
registerVjp("slice")
registerVjp("sine")
registerVjp("cosine")
registerVjp("rsqrt")
registerVjp("clamp")
registerVjp("conv2d")
registerVjp("maxPool2d")
registerNoGrad("batchNormInference")
registerNoGrad("batchNormTraining")
registerNoGrad("batchNormGrad")
registerNoGrad("cholesky")
registerNoGrad("getDimensionSize")
registerNoGrad("pad")
registerNoGrad("broadcast")
registerNoGrad("dynamicSlice")
registerNoGrad("dynamicUpdateSlice")
registerNoGrad("iota")
registerNoGrad("replicaId")
registerNoGrad("partitionId")
registerNoGrad("allGather")
registerNoGrad("allReduce")
registerNoGrad("reduceScatter")
registerNoGrad("allToAll")
registerNoGrad("collectiveBroadcast")
registerNoGrad("collectivePermute")
registerNoGrad("setDimensionSize")
registerNoGrad("dynamicReshape")
registerNoGrad("dynamicPad")
registerNoGrad("dynamicIota")
registerNoGrad("realDynamicSlice")
registerNoGrad("dynamicBroadcastInDim")
registerNoGrad("fft")
registerNoGrad("triangularSolve")
registerNoGrad("einsum")
registerNoGrad("unaryEinsum")
registerNoGrad("torchIndexSelect")
registerNoGrad("select")
registerNoGrad("compare")
registerNoGrad("gather")
registerNoGrad("dynamicGather")
registerNoGrad("indexSelect")
registerNoGrad("scatter")
registerNoGrad("segmentSum")
registerNoGrad("segmentMax")
registerNoGrad("selectAndScatter")
registerNoGrad("sort")
registerNoGrad("mapOp")
registerNoGrad("rng")
registerNoGrad("rngBitGenerator")
registerNoGrad("uniformQuantize")
registerNoGrad("uniformDequantize")
registerNoGrad("dynamicConv")
registerNoGrad("reduceWindow")
