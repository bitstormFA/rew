## Phase 8 — Workbench: init, setup, PRNG keys, distributed collectives.

import std/strutils

block init_workbench_defaults:
  let wb = initWorkbench()
  doAssert wb.devices == 1
  doAssert wb.precision == prFloat32
  doAssert wb.globalRank == 0
  doAssert wb.worldSize == 1
  doAssert wb.isGlobalZero()

block init_workbench_cpu:
  let wb = initWorkbench(akCpu)
  doAssert wb.device.target == tCpu

block setup_model_is_identity:
  var wb = initWorkbench(akCpu)
  let model = @[1, 2, 3]
  let result = wb.setup(model)
  doAssert result == model

block setup_data_is_identity:
  var wb = initWorkbench(akCpu)
  let data = fromSeq(@[1, 2, 3])
  let result = wb.setup(data)
  doAssert result.source != nil  # Dataset.source is a closure, verify non-nil

block distributed_single_process_fast_paths:
  let wb = initWorkbench(akCpu)
  doAssert wb.isGlobalZero()
  wb.barrier()  # no-op, must not crash

block distributed_collectives_trace:
  var wb = initWorkbench(akCpu, devices = 2)
  withTrace ctx, "workbench_collectives", cpu(0):
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2]])
    let gathered = wb.allGather(inputs[0])
    let reduced = wb.allReduce(inputs[0])
    let broadcasted = wb.broadcast(inputs[0])
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
  let wb = initWorkbench(akCpu, devices = 2)
  let t = initTraceTensor(ShValueId(1), dtFloat32, @[2], cpu(0))
  doAssertRaises(TensorError):
    discard wb.allReduce(t)

block prng_next_key:
  var wb = initWorkbench(akCpu)
  let k1 = wb.nextKey()
  let k2 = wb.nextKey()
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
