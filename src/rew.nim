## Rechenwerk (rew) — eager, debuggable Nim deep learning framework on OpenXLA/PJRT.
##
## This module is the public API surface. All user-facing types and procs are
## re-exported from here; everything under `rew/<layer>/...` is internal and
## may change without notice.
##
## See `docs/architecture.md` for the layered design and the ten locked
## architectural decisions the implementation upholds.

import ./rew/binaries/[target, manifest]
export target, manifest

import ./rew/[dtype, sharding, device, buffer]
export dtype, sharding, device, buffer

import ./rew/value
export value

import ./rew/openxla
export openxla

import ./rew/stablehlo/[ir, ops, verify, text]
export ir, ops, verify, text

import ./rew/[tensor, dispatch]
export tensor, dispatch

import ./rew/autograd/registry
export registry

import ./rew/ops/[arith, unary, shape, reduce, linalg, literal, factory,
  concat, conv, pool, normalization, ternary, compare, gather,
  scatter, segment, sort, map, random, interpolate, fold, gridsample]
import ./rew/ops/distributed as distributed_ops
export arith, unary, shape, reduce, linalg, literal, factory, concat, conv,
  pool, normalization, distributed_ops, ternary, compare, gather, scatter,
  segment, sort, map, random, interpolate, fold, gridsample

import ./rew/[pytree, rng]
export pytree, rng

import ./rew/autograd
export autograd

import ./rew/transform
export transform

import ./rew/[nn, optim]
export nn, optim

import ./rew/eager
export eager

import ./rew/serialize
export serialize

import ./rew/onnx
export onnx

import ./rew/tflite
export tflite

import ./rew/data
export data

import ./rew/pjrt/registry
export registry

import ./rew/optim/scheduler
export scheduler

import ./rew/train
export train

import ./rew/distributed as distributed_api
export distributed_api

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
