## Phase 8 — Runtime: init, setup, PRNG keys, distributed collectives.

import std/strutils

block init_runtime_defaults:
  let runtime = initRuntime()
  doAssert runtime.devices == 1
  doAssert runtime.precision == prFloat32
  doAssert runtime.globalRank == 0
  doAssert runtime.worldSize == 1
  doAssert runtime.isGlobalZero()

block init_runtime_cpu:
  let runtime = initRuntime(akCpu)
  doAssert runtime.device.target == tCpu

block setup_model_is_identity:
  var runtime = initRuntime(akCpu)
  let model = @[1, 2, 3]
  let result = runtime.setup(model)
  doAssert result == model

block setup_data_is_identity:
  var runtime = initRuntime(akCpu)
  let data = fromSeq(@[1, 2, 3])
  let result = runtime.setup(data)
  doAssert result.source != nil  # Dataset.source is a closure, verify non-nil

block distributed_single_process_fast_paths:
  let runtime = initRuntime(akCpu)
  doAssert runtime.isGlobalZero()
  runtime.barrier()  # no-op, must not crash

block distributed_collectives_trace:
  var runtime = initRuntime(akCpu, devices = 2)
  withTrace ctx, "runtime_collectives", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    let gathered = runtime.allGather(inputs[0])
    let reduced = runtime.allReduce(inputs[0])
    let broadcasted = runtime.broadcast(inputs[0])
    doAssert gathered.shape == @[4]
    doAssert reduced.shape == @[2]
    doAssert broadcasted.shape == @[2]
    ctx.traceReturn([gathered, reduced, broadcasted])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.all_gather" in text
  doAssert "stablehlo.all_reduce" in text
  doAssert "stablehlo.collective_broadcast" in text

block distributed_collectives_require_trace_for_multi:
  let runtime = initRuntime(akCpu, devices = 2)
  let t = initTraceTensor(ShValueId(1), dtFloat32, @[2], cpu(0))
  doAssertRaises(TensorError):
    discard runtime.allReduce(t)

block prng_next_key:
  var runtime = initRuntime(akCpu)
  let k1 = runtime.nextKey()
  let k2 = runtime.nextKey()
  doAssert k1 != k2

block seed_everything:
  seedEverything(42)
  let k = initKeyFromGlobalSeed()
  doAssert k != initKey(0)

block accelerator_enum_values:
  doAssert akCpu != akCuda
  doAssert akAuto != akTpu

block precision_enum_values:
  doAssert prFloat32 != prFloat16
  doAssert prBFloat16 != prMixedF16
