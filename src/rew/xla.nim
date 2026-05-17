## Raw OpenXLA/StableHLO and transform surface.
##
## Most user code should import `rew`. Import `rew/xla` when you need raw
## `jit`, StableHLO inspection, lowering, or explicit trace construction.

import ./[dtype, sharding, device, tensor]
export dtype, sharding, device, tensor

import ./ops/[arith, unary, shape, reduce, linalg, literal, factory,
  concat, conv, pool, normalization, ternary, compare, gather,
  scatter, segment, sort, map, random, interpolate, fold, gridsample]
import ./ops/distributed as distributed_ops
export arith, unary, shape, reduce, linalg, literal, factory, concat, conv,
  pool, normalization, distributed_ops, ternary, compare, gather, scatter,
  segment, sort, map, random, interpolate, fold, gridsample

import ./openxla
export openxla

import ./stablehlo/[ir, ops, verify, text]
export ir, ops, verify, text

import ./value
export value

import ./dispatch
export dispatch

# Install built-in VJP closures without exposing the registration API on the
# compiler surface.
from ./autograd/rules import installAllVjpRules
installAllVjpRules()

from ./autograd/transform import GradError, VjpResult, ValueAndGradResult,
  gradMode, vjp, valueAndGrad, grad
export GradError, VjpResult, ValueAndGradResult, gradMode, vjp, valueAndGrad,
  grad

import ./transform
export transform
