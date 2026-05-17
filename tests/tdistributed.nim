## Distributed planning and OpenXLA compile-option smoke tests.

import std/[os, strutils]
import rew
import rew/xla
import rew/distributed
import rew/eager
import rew/binaries/target
import rew/openxla/compile_options
import rew/pjrt/[capi, loader]
import rew/distributed/rendezvous

type EnvSnapshot = tuple[exists: bool, value: string]

proc saveEnv(key: string): EnvSnapshot =
  (exists: existsEnv(key), value: getEnv(key))

proc restoreEnv(key: string; snapshot: EnvSnapshot) =
  if snapshot.exists:
    putEnv(key, snapshot.value)
  else:
    delEnv(key)

proc makeInput(dtype: DType; shape: openArray[int];
    dev = cpu(0); sharding: Sharding = initReplicated()): Tensor =
  initTraceTensor(ShValueId(1), dtype, @shape, dev, sharding)

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "tdistributed: eager shard round-trip skipped - ", e.msg
    false

block compile_options_default_matches_pjrt_stub:
  let opts = initCompileOptionsSpec()
  doAssert hexBytes(encodeCompileOptions(opts)) == "1a0420012801"

block compile_options_data_parallel_assignment:
  let opts = initCompileOptionsSpec(numReplicas = 2, numPartitions = 1,
    deviceAssignment = [0'i64, 1'i64])
  doAssert hexBytes(encodeCompileOptions(opts)) ==
    "1a10200228014a0a080210011a040a020001"

block compile_options_tensor_parallel_shardy:
  let opts = initCompileOptionsSpec(numReplicas = 1, numPartitions = 2,
    deviceAssignment = [0'i64, 1'i64],
    useSpmdPartitioning = true,
    useShardyPartitioner = true)
  doAssert hexBytes(encodeCompileOptions(opts)) ==
    "1a192001280230014a0e080110021a030a01001a030a0101980101"

block rendezvous_kv:
  var store = initRendezvousStore()
  doAssert not store.contains("rank/0")
  doAssert store.tryGet("rank/0") == ""
  store.put("rank/0", "ready")
  doAssert store.contains("rank/0")
  doAssert store.tryGet("rank/0") == "ready"
  doAssert store.get("rank/0", timeoutMs = 1) == "ready"
  doAssertRaises(RendezvousTimeoutError):
    discard store.get("missing", timeoutMs = 1)

block init_distributed_reads_env:
  let keys = ["REW_DIST_RANK", "REW_DIST_WORLD_SIZE",
              "REW_DIST_LOCAL_RANK", "REW_DIST_LOCAL_SIZE",
              "REW_DIST_PROCESS_INDEX", "REW_DIST_PROCESS_COUNT",
              "REW_DIST_RUN_ID", "REW_DIST_HOST", "REW_DIST_PORT"]
  var snapshots: seq[EnvSnapshot] = @[]
  for key in keys:
    snapshots.add saveEnv(key)
  try:
    putEnv("REW_DIST_RANK", "3")
    putEnv("REW_DIST_WORLD_SIZE", "8")
    putEnv("REW_DIST_LOCAL_RANK", "1")
    putEnv("REW_DIST_LOCAL_SIZE", "2")
    putEnv("REW_DIST_PROCESS_INDEX", "3")
    putEnv("REW_DIST_PROCESS_COUNT", "8")
    putEnv("REW_DIST_RUN_ID", "unit-test")
    putEnv("REW_DIST_HOST", "127.0.0.9")
    putEnv("REW_DIST_PORT", "23456")
    let ctx = initDistributed([cpu(0), cpu(1)])
    doAssert ctx.process.rank == 3
    doAssert ctx.process.worldSize == 8
    doAssert ctx.process.localRank == 1
    doAssert ctx.process.localSize == 2
    doAssert ctx.rendezvous.runId == "unit-test"
    doAssert ctx.rendezvous.host == "127.0.0.9"
    doAssert ctx.rendezvous.port == 23456
  finally:
    for i, key in keys:
      restoreEnv(key, snapshots[i])

block stablehlo_emits_sdy_sharding:
  let mesh = initMesh("mesh", ["x"], [2], [cpu(0), cpu(1)])
  let sharding = initPartitioned(mesh, initPartitionSpec(["x"]))
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[add(args[0], args[0]).withSharding(args[0].sharding)]
  let j = jit(f, "sharded_add")
  let x = makeInput(dtFloat32, [8], sharding = sharding)
  let txt = j.text([x])
  doAssert "sdy.mesh @mesh" in txt
  doAssert "sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>" in txt

block shard_layouts_split_global_shape:
  let mesh = initMesh("mesh", ["x"], [2], [cpu(0), cpu(1)])
  let layouts = shardLayouts([8], mesh, initPartitionSpec(["x"]))
  doAssert layouts.len == 2
  doAssert layouts[0].index == 0
  doAssert layouts[0].localShape == @[4]
  doAssert layouts[0].offsets == @[0]
  doAssert layouts[1].index == 1
  doAssert layouts[1].localShape == @[4]
  doAssert layouts[1].offsets == @[4]

block sharded_host_round_trip_or_skip:
  if canLoadCpu():
    let mesh = initMesh("mesh", ["x"], [1], [cpu(0)])
    let sharding = initPartitioned(mesh, initPartitionSpec(["x"]))
    let x = fromHostSharded([1'f32, 2'f32, 3'f32, 4'f32], [4], sharding)
    doAssert x.buffer.isBufferSet
    doAssert x.buffer.shardIndices == @[0]
    doAssert x.toHost(float32) == @[1'f32, 2'f32, 3'f32, 4'f32]
    let shards = x.toHostShards()
    doAssert shards.len == 1
    doAssert shards[0].layout.localShape == @[4]
    let z = zerosSharded([4], dtFloat32, sharding)
    doAssert z.toHost(float32) == @[0'f32, 0'f32, 0'f32, 0'f32]

    let count = addressableDeviceCountFor(cpu(0))
    if count >= 2:
      let mesh2 = initMesh("mesh2", ["x"], [2], [cpu(0), cpu(1)])
      let sharding2 = initPartitioned(mesh2, initPartitionSpec(["x"]))
      let y = fromHostSharded(
        [0'f32, 1'f32, 2'f32, 3'f32, 4'f32, 5'f32, 6'f32, 7'f32],
        [8], sharding2)
      doAssert y.buffer.isBufferSet
      doAssert y.buffer.shardIndices == @[0, 1]
      doAssert y.toHost(float32) ==
        @[0'f32, 1'f32, 2'f32, 3'f32, 4'f32, 5'f32, 6'f32, 7'f32]
    else:
      echo "tdistributed: multi-device shard round-trip skipped - only ",
        count, " CPU device(s)"

block planner_and_dist_jit_text:
  let ctx = initDistributed([cpu(0), cpu(1)])
  let mesh = meshFromTopology(ctx, ["data"], [2])
  let sharding = initPartitioned(mesh, initPartitionSpec(["data"]))
  let f = proc(args: openArray[Tensor]): seq[Tensor] =
    @[mul(args[0], args[0]).withSharding(args[0].sharding)]
  let x = makeInput(dtFloat32, [8], sharding = sharding)
  let plan = planParallelism(f, [x], autoParallel(), ctx)
  doAssert plan.numReplicas == 2
  doAssert plan.numPartitions == 1
  doAssert plan.compileOptions.len > 0
  doAssert plan.cacheKey.len > 0
  let df = distJit(f, plan, funcName = "distributed_square")
  let txt = df.text([x])
  doAssert "mhlo.num_replicas = 2 : i32" in txt
  doAssert "mhlo.num_partitions = 1 : i32" in txt
  doAssert "sdy.mesh @mesh" in txt

echo "tdistributed: OK"
