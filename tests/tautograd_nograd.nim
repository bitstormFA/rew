## Explicit no-gradient registry entries.

import rew
import rew/dev

block register_no_grad_policy:
  registerNoGrad("test.token_op")
  doAssert hasNoGradient("test.token_op")
  doAssert gradientPolicy("test.token_op") == gpNoGradient
  doAssert vjpRuleStatus("test.token_op") == vrsNoGradient
  doAssert "test.token_op" in registeredNoGradOps()
  doAssert "test.token_op" in registeredAutogradOps()

block openxla_unary_no_grad_ops_are_registered:
  for op in ["ceil", "floor", "remainder", "sign",
             "roundNearestAfz", "roundNearestEven",
             "bitwiseAnd", "bitwiseOr", "bitwiseXor", "bitwiseNot",
             "shiftLeft", "shiftRightArithmetic", "shiftRightLogical",
             "countLeadingZeros", "popcnt",
             "astype", "bitcastConvert", "isFinite", "reducePrecision",
             "complex", "real", "imag",
             "batchNormInference", "batchNormTraining", "batchNormGrad",
             "cholesky", "getDimensionSize", "pad",
             "broadcast", "dynamicSlice", "dynamicUpdateSlice", "iota",
             "replicaId", "partitionId",
             "setDimensionSize", "dynamicReshape", "dynamicPad",
             "dynamicIota", "realDynamicSlice",
             "dynamicBroadcastInDim", "fft", "triangularSolve",
             "einsum", "unaryEinsum", "torchIndexSelect"]:
    doAssert hasNoGradient(op)
    doAssert gradientPolicy(op) == gpNoGradient
    doAssert vjpRuleStatus(op) == vrsNoGradient
    doAssert op in registeredNoGradOps()

block unknown_policy_status:
  doAssert vjpRuleStatus("test.unknown") == vrsUnregistered

block duplicate_policy_rejected:
  var raised = false
  try:
    registerVjp("test.token_op")
  except VjpRegistryError:
    raised = true
  doAssert raised

echo "tautograd_nograd: OK"
