## Distributed training API built on OpenXLA/PJRT.
##
## This module is deliberately high-level: it owns meshes, policies, plans,
## distributed `jit` wrappers, and environment-driven process metadata. Raw
## PJRT access stays behind `eager.nim`, preserving the layer invariant.

import std/[os, strutils, tables]
import ./device
import ./dtype
import ./tensor
import ./sharding
import ./eager
import ./transform/jit
import ./stablehlo/ir
import ./stablehlo/text
import ./stablehlo/verify
import ./openxla/compile_options
import ./distributed/rendezvous

export sharding.Mesh, sharding.PartitionSpec, sharding.Sharding
export sharding.initMesh, sharding.initPartitionSpec
export sharding.initPartitionSpecGroups, sharding.initPartitioned
export sharding.initManualSharding
export compile_options.CompileOptionsSpec, compile_options.encodeCompileOptions
export rendezvous

type
  RendezvousKind* = enum
    rvEnv
    rvTcp
    rvManual

  RendezvousConfig* = object
    ## How worker processes discover each other. `rvEnv` reads `REW_DIST_*`.
    kind*: RendezvousKind
    host*: string
    port*: int
    runId*: string
    timeoutMs*: int

  HostSpec* = object
    ## Launcher-visible host description.
    host*: string
    port*: int
    slots*: int
    sshUser*: string

  ProcessSpec* = object
    ## Process/rank identity for the current worker.
    rank*: int
    worldSize*: int
    localRank*: int
    localSize*: int
    processIndex*: int
    processCount*: int

  DistributedContext* = object
    ## Process-local distributed runtime metadata.
    rendezvous*: RendezvousConfig
    process*: ProcessSpec
    hosts*: seq[HostSpec]
    devices*: seq[Device]
    coordinator*: string
    nextLaunchId*: int

  CostObjective* = enum
    coThroughput
    coMemory
    coBalanced

  ParallelPolicyKind* = enum
    ppkAuto
    ppkData
    ppkTensor
    ppkZero
    ppkPipeline
    ppkHybrid
    ppkManual

  ParallelPolicy* = object
    ## User intent for the planner. Zero counts mean "infer from topology".
    kind*: ParallelPolicyKind
    objective*: CostObjective
    requestedReplicas*: int
    requestedPartitions*: int
    pipelineStages*: int
    zeroStage*: int
    beamWidth*: int

  PlanScore* = object
    ## Deterministic planner metadata. Runtime cost-analysis fields are filled
    ## when the active PJRT plugin can compile the candidate.
    objective*: CostObjective
    estimatedMemoryBytes*: int64
    throughputScore*: float
    memoryScore*: float
    collectiveScore*: float

  ParallelPlan* = object
    ## Concrete distributed execution plan for one traced function.
    policy*: ParallelPolicy
    mesh*: Mesh
    devices*: seq[Device]
    process*: ProcessSpec
    numReplicas*: int
    numPartitions*: int
    useSpmdPartitioning*: bool
    useAutoSpmdPartitioning*: bool
    useShardyPartitioner*: bool
    deviceAssignment*: seq[int64]
    compileOptions*: string
    topologyFingerprint*: string
    stableHloHash*: string
    cacheKey*: string
    score*: PlanScore

  DistributedJitFunction* = object
    ## A `jit` handle plus a distributed plan.
    jit*: JitFunction
    plan*: ParallelPlan

var launchCounters {.threadvar.}: Table[string, int]

func stableHash(s: string): string =
  ## FNV-1a, sufficient for deterministic cache-key fragments.
  var h = 1469598103934665603'u64
  for ch in s:
    h = h xor uint64(ord(ch))
    h = h * 1099511628211'u64
  const Hex = "0123456789abcdef"
  for shift in countdown(60, 0, 4):
    result.add Hex[int((h shr shift) and 0xf)]

proc envInt(name: string; defaultValue: int): int =
  let value = getEnv(name).strip()
  if value.len == 0:
    return defaultValue
  try:
    parseInt(value)
  except ValueError:
    defaultValue

proc envStr(name, defaultValue: string): string =
  let value = getEnv(name)
  if value.len == 0: defaultValue else: value

func initRendezvousConfig*(kind: RendezvousKind = rvEnv;
    host: string = "127.0.0.1"; port: int = 0; runId: string = "";
    timeoutMs: int = 30000): RendezvousConfig =
  RendezvousConfig(kind: kind, host: host, port: port, runId: runId,
    timeoutMs: timeoutMs)

func initHostSpec*(host: string; port: int = 0; slots: int = 1;
    sshUser: string = ""): HostSpec =
  HostSpec(host: host, port: port, slots: slots, sshUser: sshUser)

func initProcessSpec*(rank: int = 0; worldSize: int = 1;
    localRank: int = 0; localSize: int = 1;
    processIndex: int = 0; processCount: int = 1): ProcessSpec =
  ProcessSpec(rank: rank, worldSize: worldSize, localRank: localRank,
    localSize: localSize, processIndex: processIndex,
    processCount: processCount)

func autoParallel*(objective: CostObjective = coThroughput;
    beamWidth: int = 8): ParallelPolicy =
  ParallelPolicy(kind: ppkAuto, objective: objective, beamWidth: beamWidth)

func dataParallel*(replicas: int = 0;
    objective: CostObjective = coThroughput): ParallelPolicy =
  ParallelPolicy(kind: ppkData, objective: objective,
    requestedReplicas: replicas)

func tensorParallel*(partitions: int = 0;
    objective: CostObjective = coThroughput): ParallelPolicy =
  ParallelPolicy(kind: ppkTensor, objective: objective,
    requestedPartitions: partitions)

func zeroParallel*(stage: int = 2; replicas: int = 0;
    objective: CostObjective = coMemory): ParallelPolicy =
  ParallelPolicy(kind: ppkZero, objective: objective,
    requestedReplicas: replicas, zeroStage: stage)

func pipelineParallel*(stages: int = 0;
    objective: CostObjective = coThroughput): ParallelPolicy =
  ParallelPolicy(kind: ppkPipeline, objective: objective,
    requestedPartitions: stages, pipelineStages: stages)

func hybridParallel*(replicas: int = 0; partitions: int = 0;
    objective: CostObjective = coBalanced): ParallelPolicy =
  ParallelPolicy(kind: ppkHybrid, objective: objective,
    requestedReplicas: replicas, requestedPartitions: partitions)

proc defaultContextDevices(process: ProcessSpec): seq[Device] =
  let base = defaultDevice()
  let localSize = max(1, process.localSize)
  result = newSeq[Device](localSize)
  for i in 0 ..< localSize:
    result[i] = initDevice(base.target, i)

proc initDistributed*(config: RendezvousConfig;
    devices: openArray[Device] = []): DistributedContext =
  ## Creates or joins a distributed context from explicit config plus
  ## `REW_DIST_*` environment overrides.
  let process = initProcessSpec(
    rank = envInt("REW_DIST_RANK", envInt("RANK", 0)),
    worldSize = envInt("REW_DIST_WORLD_SIZE", envInt("WORLD_SIZE", 1)),
    localRank = envInt("REW_DIST_LOCAL_RANK", envInt("LOCAL_RANK", 0)),
    localSize = envInt("REW_DIST_LOCAL_SIZE", envInt("LOCAL_WORLD_SIZE", 1)),
    processIndex = envInt("REW_DIST_PROCESS_INDEX",
      envInt("REW_DIST_RANK", envInt("RANK", 0))),
    processCount = envInt("REW_DIST_PROCESS_COUNT",
      envInt("REW_DIST_WORLD_SIZE", envInt("WORLD_SIZE", 1))),
  )
  var cfg = config
  cfg.host = envStr("REW_DIST_HOST", cfg.host)
  cfg.port = envInt("REW_DIST_PORT", cfg.port)
  cfg.runId = envStr("REW_DIST_RUN_ID", cfg.runId)
  result.rendezvous = cfg
  result.process = process
  result.coordinator = envStr("REW_DIST_COORDINATOR",
    cfg.host & ":" & $cfg.port)
  result.devices =
    if devices.len > 0: @devices
    else: defaultContextDevices(process)
  result.nextLaunchId = envInt("REW_DIST_LAUNCH_ID_START", 0)

proc initDistributed*(devices: openArray[Device] = []): DistributedContext =
  ## Creates a context using `rvEnv` and `REW_DIST_*` metadata.
  initDistributed(initRendezvousConfig(), devices)

func product(xs: openArray[int]): int =
  result = 1
  for x in xs:
    result *= x

proc meshFromTopology*(ctx: DistributedContext; axes: openArray[string];
    sizes: openArray[int]; name: string = "mesh"): Mesh =
  ## Builds a logical mesh using the context's local/global process metadata.
  let total = product(sizes)
  if total <= 0:
    raise newException(ValueError, "meshFromTopology: mesh size must be > 0")
  let target =
    if ctx.devices.len > 0: ctx.devices[0].target
    else: defaultDevice().target
  var devices = newSeq[Device](total)
  if ctx.devices.len == total:
    devices = ctx.devices
  else:
    let localSize = max(1, ctx.devices.len)
    for i in 0 ..< total:
      let ordinal =
        if ctx.devices.len > 0: ctx.devices[i mod localSize].ordinal
        else: i
      devices[i] = initDevice(target, ordinal)
  var processes = newSeq[int](total)
  let processCount = max(1, ctx.process.processCount)
  let perProcess = max(1, (total + processCount - 1) div processCount)
  for i in 0 ..< total:
    processes[i] = min(processCount - 1, i div perProcess)
  initMesh(name, axes, sizes, devices, processes)

func boundedCount(requested, total: int): int =
  if requested <= 0: total
  else: max(1, min(requested, total))

func memoryBytes(args: openArray[Tensor]; entry: JitCacheEntry): int64 =
  for t in args:
    result += int64(t.numElements * t.dtype.byteSize)
  for i, dt in entry.outDtypes:
    var n = 1
    for dim in entry.outShapes[i]:
      n *= dim
    result += int64(n * dt.byteSize)

proc fallbackAssignment(devices: openArray[Device]; count: int): seq[int64] =
  result = newSeq[int64](count)
  for i in 0 ..< count:
    result[i] =
      if devices.len == 0: int64 i
      else: int64 devices[i mod devices.len].ordinal

proc addressableDevices(ctx: DistributedContext; mesh: Mesh): seq[Device] =
  if mesh.devices.len > 0 and mesh.processes.len == mesh.devices.len:
    for i, device in mesh.devices:
      if mesh.processes[i] == ctx.process.processIndex:
        result.add device
  if result.len == 0:
    result = ctx.devices

proc makeCompileOptions(plan: var ParallelPlan) =
  let meshShape = newSeq[int64](plan.mesh.sizes.len)
  var shape = meshShape
  for i, size in plan.mesh.sizes:
    shape[i] = int64 size
  let spec = initCompileOptionsSpec(
    numReplicas = plan.numReplicas,
    numPartitions = plan.numPartitions,
    deviceAssignment = plan.deviceAssignment,
    useSpmdPartitioning = plan.useSpmdPartitioning,
    useAutoSpmdPartitioning = plan.useAutoSpmdPartitioning,
    useShardyPartitioner = plan.useShardyPartitioner,
    processIndex = plan.process.processIndex,
    processCount = plan.process.processCount,
    autoSpmdMeshShape = shape,
    autoSpmdMeshIds = plan.deviceAssignment)
  plan.compileOptions = encodeCompileOptions(spec)

proc chooseCounts(policy: ParallelPolicy; mesh: Mesh;
    sampleArgs: openArray[Tensor]): tuple[replicas, partitions: int] =
  let total = max(1, mesh.meshSize)
  case policy.kind
  of ppkData, ppkZero:
    result.replicas = boundedCount(policy.requestedReplicas, total)
    result.partitions = max(1, total div result.replicas)
    if policy.kind == ppkData:
      result.partitions = 1
      result.replicas = total
  of ppkTensor, ppkPipeline:
    result.partitions = boundedCount(policy.requestedPartitions, total)
    result.replicas = max(1, total div result.partitions)
    if policy.kind == ppkTensor:
      result.replicas = 1
      result.partitions = total
  of ppkHybrid, ppkManual:
    result.replicas =
      if policy.requestedReplicas > 0: boundedCount(policy.requestedReplicas,
        total)
      else: max(1, total div max(1, policy.requestedPartitions))
    result.partitions =
      if policy.requestedPartitions > 0: boundedCount(
        policy.requestedPartitions, total)
      else: max(1, total div result.replicas)
  of ppkAuto:
    if total == 1:
      result.replicas = 1
      result.partitions = 1
    elif sampleArgs.len > 0 and sampleArgs[0].shape.len > 0 and
        sampleArgs[0].shape[0] mod total == 0:
      result.replicas = total
      result.partitions = 1
    else:
      result.replicas = 1
      result.partitions = total
  if result.replicas * result.partitions > total:
    result.partitions = max(1, total div result.replicas)
  if result.replicas <= 0: result.replicas = 1
  if result.partitions <= 0: result.partitions = 1

proc planParallelism*(jit: JitFunction; sampleArgs: openArray[Tensor];
    policy: ParallelPolicy; ctx: DistributedContext): ParallelPlan =
  ## Traces once, chooses a deterministic OpenXLA-compatible execution plan,
  ## and encodes the PJRT compile options for that plan.
  let entry = jit.compileFor(sampleArgs)
  let mesh =
    if sampleArgs.len > 0 and not sampleArgs[0].sharding.isReplicated:
      sampleArgs[0].sharding.activeMesh()
    else:
      meshFromTopology(ctx, ["data"], [max(1, ctx.devices.len)])
  let counts = chooseCounts(policy, mesh, sampleArgs)
  result.policy = policy
  result.mesh = mesh
  result.devices = addressableDevices(ctx, mesh)
  result.process = ctx.process
  result.numReplicas = counts.replicas
  result.numPartitions = counts.partitions
  result.useSpmdPartitioning = result.numPartitions > 1
  result.useAutoSpmdPartitioning = policy.kind == ppkAuto and
    result.numPartitions > 1
  result.useShardyPartitioner = result.numPartitions > 1 or
    policy.kind in {ppkTensor, ppkZero, ppkPipeline, ppkHybrid}
  let assignmentCount = result.numReplicas * result.numPartitions
  try:
    result.deviceAssignment = defaultDeviceAssignmentFor(result.devices[0],
      result.numReplicas, result.numPartitions)
  except CatchableError:
    result.deviceAssignment = fallbackAssignment(result.devices,
      assignmentCount)
  except Exception:
    result.deviceAssignment = fallbackAssignment(result.devices,
      assignmentCount)
  try:
    result.topologyFingerprint = topologyFingerprintFor(result.devices[0])
  except CatchableError:
    result.topologyFingerprint = stableHash($result.devices)
  except Exception:
    result.topologyFingerprint = stableHash($result.devices)
  result.stableHloHash = stableHash(entry.text)
  result.score = PlanScore(
    objective: policy.objective,
    estimatedMemoryBytes: memoryBytes(sampleArgs, entry),
    throughputScore: float(result.numReplicas * result.numPartitions),
    memoryScore: 1.0 / float(max(1, result.numPartitions)),
    collectiveScore:
      if result.numReplicas > 1 or result.numPartitions > 1: 1.0 else: 0.0)
  makeCompileOptions(result)
  result.cacheKey = stableHash(result.topologyFingerprint & "\n" &
    result.stableHloHash & "\n" & $policy & "\n" &
    hexBytes(result.compileOptions))

proc planParallelism*(fn: JitFn; sampleArgs: openArray[Tensor];
    policy: ParallelPolicy; ctx: DistributedContext): ParallelPlan =
  ## Convenience overload for planning an unwrapped function.
  let j = jit(fn, "plan_probe")
  planParallelism(j, sampleArgs, policy, ctx)

proc executableText(entry: JitCacheEntry; plan: ParallelPlan): string =
  var m: ShModule = entry.module
  if m.funcs.len > 0:
    m.funcs[0].name = "main"
  m.numReplicas = plan.numReplicas
  m.numPartitions = plan.numPartitions
  let meshOp = shardyMeshOp(plan.mesh)
  var seen = false
  for existing in m.shardyMeshOps:
    if existing == meshOp:
      seen = true
  if not seen:
    m.shardyMeshOps.add meshOp
  verify(m)
  emitText(m)

proc distJit*(fn: JitFn; plan: ParallelPlan;
    donateArgs: openArray[int] = [];
    funcName: string = "dist_jit"): DistributedJitFunction =
  ## Wraps a Nim training step in distributed `jit`.
  DistributedJitFunction(jit: jit(fn, funcName, donateArgs), plan: plan)

proc text*(f: DistributedJitFunction; args: openArray[Tensor]): string =
  ## Returns the PJRT-ready StableHLO text with mesh/count metadata.
  executableText(f.jit.compileFor(args), f.plan)

proc nextLaunchId(key: string): int =
  if not launchCounters.hasKey(key):
    launchCounters[key] = 0
  inc launchCounters[key]
  launchCounters[key]

proc call*(f: DistributedJitFunction; args: openArray[Tensor]): seq[Tensor] =
  ## Executes the distributed `jit` via OpenXLA/PJRT.
  let entry = f.jit.compileFor(args)
  let moduleText = executableText(entry, f.plan)
  let key = signatureOf(args) & "::" & f.plan.cacheKey
  executeDistributedJit(f.jit.funcName, key, moduleText, args,
    entry.outDtypes, entry.outShapes, f.jit.donateArgs, entry.outShardings,
    f.plan.devices, f.plan.compileOptions, nextLaunchId(f.plan.cacheKey))
