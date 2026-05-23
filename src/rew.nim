## Rechenwerk (rew) — high-level user API for tensor and training code.
##
## This module is the user-level public surface. Raw StableHLO/OpenXLA,
## explicit tracing, lowering, JIT handles, dispatch internals, and extension
## hooks live in `rew/xla` or `rew/dev`.
##
## See `docs/architecture.md` for the layered design and the ten locked
## architectural decisions the implementation upholds.

import ./rew/[dtype, sharding, device]
export dtype, sharding, device

from ./rew/buffer import BufferDonatedError
export BufferDonatedError

from ./rew/tensor import Tensor, TensorError, TensorModeError, numElements,
  isEager, isTrace, withSharding, shard, manualShard, replicate
export Tensor, TensorError, TensorModeError, numElements, isEager, isTrace,
  withSharding, shard, manualShard, replicate

import ./rew/ops/[arith, unary, shape, reduce, linalg, literal, factory,
  concat, conv, pool, normalization, ternary, compare, gather,
  scatter, segment, sort, map, random, interpolate, fold, gridsample]
import ./rew/ops/distributed as distributed_ops
export arith, unary, shape, reduce, linalg, literal, factory, concat, conv,
  pool, normalization, distributed_ops, ternary, compare, gather, scatter,
  segment, sort, map, random, interpolate, fold, gridsample

import ./rew/[pytree, rng]
export pytree, rng

# Install built-in VJP closures without exposing the registration API on the
# user-level surface.
from ./rew/autograd/rules import installAllVjpRules
installAllVjpRules()

from ./rew/autograd/transform import GradError, VjpResult,
  ValueAndGradResult, gradMode, vjp, valueAndGrad, grad
export GradError, VjpResult, ValueAndGradResult, gradMode, vjp, valueAndGrad,
  grad

from ./rew/transform/control import CondError, ScanBody, condN, cond,
  whileLoop, fori, caseOp, scan
from ./rew/transform/vmap import vmap
export CondError, ScanBody, condN, cond, whileLoop, fori, caseOp, scan, vmap

import ./rew/[nn, optim]
export nn, optim

from ./rew/eager import EagerError, HostShardLoader, HostByteShardLoader,
  HostTensorShard, transferToDevice, transferToHost, fromHostF32, fromHost,
  fromHostByteShards, fromHostShards, fromHostSharded, zerosSharded,
  zerosLikeEager, scalarF32, scalar, toHost, toHostShards, toHostBytes, item,
  `to`, shardToMesh, installEagerBackend
export EagerError, HostShardLoader, HostByteShardLoader, HostTensorShard,
  transferToDevice, transferToHost, fromHostF32, fromHost, fromHostByteShards,
  fromHostShards, fromHostSharded, zerosSharded, zerosLikeEager, scalarF32,
  scalar, toHost, toHostShards, toHostBytes, item, `to`, shardToMesh,
  installEagerBackend

import ./rew/serialize
export serialize

import ./rew/data
export data

import ./rew/dataframe
export dataframe

import ./rew/optim/scheduler
export scheduler

import ./rew/train
export train

import ./rew/safetensors
export safetensors

import ./rew/models
export models

import ./rew/hf
export hf

import ./rew/infer
export infer

import ./rew/quantize
export quantize

import ./rew/checkpoint
export checkpoint

import ./rew/gguf
export gguf

import ./rew/multimodal
export multimodal

const
  RewVersion* = "0.2.0"
    ## Library version string.
