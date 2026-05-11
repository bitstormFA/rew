## Sharding and SPMD annotations.
##
## `Replicated` remains the default and keeps the current single-device eager
## path intact. `Mesh` and `PartitionSpec` provide the public model needed for
## Shardy/SPMD lowering and multi-host execution.

import ./device

type
  Mesh* = object
    ## Logical device mesh. Axis names and sizes have the same length.
    name*: string
    axes*: seq[string]
    sizes*: seq[int]
    devices*: seq[Device]
      ## Optional row-major physical device assignment. Empty means the mesh
      ## is symbolic and may be mapped by a later compile/load step.
    processes*: seq[int]
      ## Optional process index per physical device, also row-major.

  PartitionSpec* = object
    ## Per-tensor-dimension sharding axes. Empty entries mean replicated
    ## along that tensor dimension.
    axes*: seq[string]
    axisGroups*: seq[seq[string]]
      ## General form: each tensor dimension may be split across zero, one,
      ## or more mesh axes. `axes` is kept as the common one-axis view.

  ShardLayout* = object
    ## Row-major layout for one shard of a global tensor.
    ##
    ## `index` is the global mesh-linear shard index. `localShape` is the
    ## shard shape stored on `device`; `offsets` gives the start coordinate
    ## of that shard in the global tensor.
    index*: int
    process*: int
    device*: Device
    globalShape*: seq[int]
    localShape*: seq[int]
    offsets*: seq[int]

  ShardingKind* = enum
    ## Tag for the variant `Sharding` object.
    skReplicated
    skPartitioned
    skManual

  Sharding* = object
    ## Sharding annotation carried on every `Tensor`.
    case kind*: ShardingKind
    of skReplicated:
      discard
    of skPartitioned:
      mesh*: Mesh
      spec*: PartitionSpec
    of skManual:
      manualMesh*: Mesh
      manualSpec*: PartitionSpec

func initReplicated*(): Sharding =
  ## Constructs the canonical replicated sharding.
  Sharding(kind: skReplicated)

func axisIndex(mesh: Mesh; axis: string): int =
  for i, candidate in mesh.axes:
    if candidate == axis:
      return i
  -1

func meshSize*(mesh: Mesh): int =
  ## Returns the product of all mesh-axis sizes.
  result = 1
  for size in mesh.sizes:
    result *= size

func elementCount*(layout: ShardLayout): int =
  ## Number of elements in this shard.
  result = 1
  for dim in layout.localShape:
    result *= dim

func containsAxis*(mesh: Mesh; axis: string): bool =
  ## True when `axis` is a named axis of `mesh`.
  mesh.axisIndex(axis) >= 0

func initMesh*(name: string; axes: openArray[string];
    sizes: openArray[int]; devices: openArray[Device] = [];
    processes: openArray[int] = []): Mesh =
  ## Constructs a logical mesh. Raises `ValueError` on malformed inputs.
  if axes.len != sizes.len:
    raise newException(ValueError,
      "initMesh: axes and sizes must have the same length")
  for i, axis in axes:
    if axis.len == 0:
      raise newException(ValueError,
        "initMesh: mesh axis names must not be empty")
    for j in 0 ..< i:
      if axes[j] == axis:
        raise newException(ValueError,
          "initMesh: duplicate mesh axis '" & axis & "'")
  for size in sizes:
    if size <= 0:
      raise newException(ValueError,
        "initMesh: mesh axis sizes must be positive")
  var total = 1
  for size in sizes:
    total *= size
  if devices.len > 0 and devices.len != total:
    raise newException(ValueError,
      "initMesh: device assignment length " & $devices.len &
        " does not match mesh size " & $total)
  if processes.len > 0:
    if devices.len == 0:
      raise newException(ValueError,
        "initMesh: process indices require an explicit device assignment")
    if processes.len != devices.len:
      raise newException(ValueError,
        "initMesh: processes length must match devices length")
    for p in processes:
      if p < 0:
        raise newException(ValueError,
          "initMesh: process indices must be non-negative")
  Mesh(name: name, axes: @axes, sizes: @sizes, devices: @devices,
    processes: @processes)

func initPartitionSpec*(axes: openArray[string]): PartitionSpec =
  ## Constructs a partition spec. Use `""` for a replicated dimension.
  result.axes = @axes
  result.axisGroups = newSeq[seq[string]](axes.len)
  for i, axis in axes:
    if axis.len > 0:
      result.axisGroups[i] = @[axis]

func initPartitionSpecGroups*(axisGroups: openArray[seq[string]]):
    PartitionSpec =
  ## Constructs a general partition spec where one tensor dimension can be
  ## partitioned by multiple mesh axes. Use an empty group for replication.
  result.axisGroups = newSeq[seq[string]](axisGroups.len)
  result.axes = newSeq[string](axisGroups.len)
  for i, group in axisGroups:
    result.axisGroups[i] = @group
    result.axes[i] =
      if group.len == 1: group[0]
      elif group.len == 0: ""
      else:
        var joined = group[0]
        for j in 1 ..< group.len:
          joined.add "+"
          joined.add group[j]
        joined

func initPartitioned*(mesh: Mesh; spec: PartitionSpec): Sharding =
  ## Constructs a mesh-partitioned sharding annotation.
  Sharding(kind: skPartitioned, mesh: mesh, spec: spec)

func initManualSharding*(mesh: Mesh; spec: PartitionSpec): Sharding =
  ## Constructs a manual sharding annotation.
  Sharding(kind: skManual, manualMesh: mesh, manualSpec: spec)

func isReplicated*(s: Sharding): bool =
  ## Returns true when `s` is replicated.
  s.kind == skReplicated

func isPartitioned*(s: Sharding): bool =
  ## Returns true when `s` is partitioned by a mesh/spec pair.
  s.kind == skPartitioned

func isManual*(s: Sharding): bool =
  ## Returns true when `s` uses manual sharding.
  s.kind == skManual

func activeMesh*(s: Sharding): Mesh =
  ## Returns the mesh carried by a partitioned/manual sharding.
  case s.kind
  of skReplicated:
    raise newException(ValueError,
      "activeMesh: replicated sharding has no mesh")
  of skPartitioned:
    s.mesh
  of skManual:
    s.manualMesh

func activeSpec*(s: Sharding): PartitionSpec =
  ## Returns the partition spec carried by a partitioned/manual sharding.
  case s.kind
  of skReplicated:
    raise newException(ValueError,
      "activeSpec: replicated sharding has no partition spec")
  of skPartitioned:
    s.spec
  of skManual:
    s.manualSpec

func validatePartitionSpec*(mesh: Mesh; spec: PartitionSpec;
    tensorRank: int) =
  ## Validates that `spec` is rank-compatible and only references axes in
  ## `mesh`, with no mesh axis used twice.
  if spec.axisGroups.len != tensorRank:
    raise newException(ValueError,
      "validatePartitionSpec: spec rank " & $spec.axisGroups.len &
        " does not match tensor rank " & $tensorRank)
  var used: seq[string] = @[]
  for group in spec.axisGroups:
    for axis in group:
      if axis.len == 0:
        continue
      if not mesh.containsAxis(axis):
        raise newException(ValueError,
          "validatePartitionSpec: unknown mesh axis '" & axis & "'")
      for seen in used:
        if seen == axis:
          raise newException(ValueError,
            "validatePartitionSpec: mesh axis '" & axis &
              "' is used more than once")
      used.add axis

func validateSharding*(s: Sharding; tensorRank: int) =
  ## Validates a sharding annotation against a tensor rank.
  case s.kind
  of skReplicated:
    discard
  of skPartitioned:
    validatePartitionSpec(s.mesh, s.spec, tensorRank)
  of skManual:
    validatePartitionSpec(s.manualMesh, s.manualSpec, tensorRank)

func meshAxisCoord(mesh: Mesh; linearIndex, axisIndex: int): int =
  var stride = 1
  for i in countdown(mesh.sizes.high, axisIndex + 1):
    stride *= mesh.sizes[i]
  (linearIndex div stride) mod mesh.sizes[axisIndex]

func processFor(mesh: Mesh; index: int): int =
  if mesh.processes.len > 0: mesh.processes[index] else: 0

func deviceFor(mesh: Mesh; index: int): Device =
  if mesh.devices.len > 0: mesh.devices[index] else: cpu(index)

func shardLayout*(shape: openArray[int]; mesh: Mesh; spec: PartitionSpec;
    index: int): ShardLayout =
  ## Computes the local shape and global offsets for one mesh shard.
  validatePartitionSpec(mesh, spec, shape.len)
  let total = mesh.meshSize
  if index < 0 or index >= total:
    raise newException(ValueError,
      "shardLayout: shard index " & $index &
        " is outside mesh size " & $total)
  result = ShardLayout(
    index: index,
    process: processFor(mesh, index),
    device: deviceFor(mesh, index),
    globalShape: @shape,
    localShape: @shape,
    offsets: newSeq[int](shape.len))
  for dim, group in spec.axisGroups:
    if group.len == 0:
      continue
    var factor = 1
    var groupCoord = 0
    for axis in group:
      let axisIdx = mesh.axisIndex(axis)
      factor *= mesh.sizes[axisIdx]
      groupCoord = groupCoord * mesh.sizes[axisIdx] +
        meshAxisCoord(mesh, index, axisIdx)
    if shape[dim] mod factor != 0:
      raise newException(ValueError,
        "shardLayout: dimension " & $dim & " of shape " & $(@shape) &
          " is not divisible by sharding factor " & $factor)
    result.localShape[dim] = shape[dim] div factor
    result.offsets[dim] = groupCoord * result.localShape[dim]

func shardLayouts*(shape: openArray[int]; mesh: Mesh; spec: PartitionSpec;
    processIndex: int = -1): seq[ShardLayout] =
  ## Computes shard layouts for a mesh/spec pair.
  ##
  ## When `processIndex >= 0`, only shards owned by that process are returned.
  let total = mesh.meshSize
  for index in 0 ..< total:
    let p = processFor(mesh, index)
    if processIndex >= 0 and p != processIndex:
      continue
    result.add shardLayout(shape, mesh, spec, index)

func shardLayouts*(shape: openArray[int]; sharding: Sharding;
    processIndex: int = -1): seq[ShardLayout] =
  ## Computes shard layouts for a partitioned/manual sharding annotation.
  case sharding.kind
  of skReplicated:
    result.add ShardLayout(index: 0, process: 0, device: cpu(0),
      globalShape: @shape, localShape: @shape,
      offsets: newSeq[int](shape.len))
  of skPartitioned:
    result = shardLayouts(shape, sharding.mesh, sharding.spec, processIndex)
  of skManual:
    result = shardLayouts(shape, sharding.manualMesh, sharding.manualSpec,
      processIndex)

func shardingKey*(s: Sharding): string =
  ## Stable cache-key fragment for trace/eager executable specialization.
  case s.kind
  of skReplicated:
    result = "replicated"
  of skPartitioned, skManual:
    let mesh =
      if s.kind == skPartitioned: s.mesh else: s.manualMesh
    let spec =
      if s.kind == skPartitioned: s.spec else: s.manualSpec
    result =
      (if s.kind == skPartitioned: "partitioned:" else: "manual:") &
      mesh.name & "["
    for i, axis in mesh.axes:
      if i > 0: result.add ","
      result.add axis & "=" & $mesh.sizes[i]
    if mesh.devices.len > 0:
      result.add "@devices="
      for i, d in mesh.devices:
        if i > 0: result.add ","
        result.add $d
    if mesh.processes.len > 0:
      result.add "@processes="
      for i, p in mesh.processes:
        if i > 0: result.add ","
        result.add $p
    result.add "]("
    for i, group in spec.axisGroups:
      if i > 0: result.add ";"
      if group.len == 0:
        result.add "*"
      else:
        for j, axis in group:
          if j > 0: result.add "+"
          result.add axis
    result.add ")"

func `$`*(mesh: Mesh): string =
  ## Human-readable mesh summary.
  result = mesh.name & "["
  for i, axis in mesh.axes:
    if i > 0: result.add ", "
    result.add axis & "=" & $mesh.sizes[i]
  result.add "]"

func `$`*(spec: PartitionSpec): string =
  ## Human-readable partition spec.
  result = "("
  for i, group in spec.axisGroups:
    if i > 0: result.add ", "
    if group.len == 0:
      result.add "*"
    else:
      for j, axis in group:
        if j > 0: result.add "+"
        result.add axis
  result.add ")"

func `$`*(s: Sharding): string =
  ## Human-readable sharding annotation.
  case s.kind
  of skReplicated:
    "replicated"
  of skPartitioned:
    "partitioned(" & $s.mesh & ", " & $s.spec & ")"
  of skManual:
    "manual(" & $s.manualMesh & ", " & $s.manualSpec & ")"

func renderAxisGroups(groups: seq[seq[string]]): string =
  result = "["
  for i, group in groups:
    if i > 0: result.add ", "
    result.add "{"
    for j, axis in group:
      if j > 0: result.add ", "
      result.add "\"" & axis & "\""
    result.add "}"
  result.add "]"

func shardyMeshAttr*(mesh: Mesh): string =
  ## Renders `mesh` as an SDY mesh attribute body.
  result = "<["
  for i, axis in mesh.axes:
    if i > 0: result.add ", "
    result.add "\"" & axis & "\"=" & $mesh.sizes[i]
  result.add "]"
  if mesh.devices.len > 0:
    result.add ", device_ids=["
    for i, device in mesh.devices:
      if i > 0: result.add ", "
      result.add $device.ordinal
    result.add "]"
  result.add ">"

func shardyMeshOp*(mesh: Mesh): string =
  ## Renders an `sdy.mesh` symbol definition for optional Shardy tooling.
  "sdy.mesh @" & mesh.name & " = " & shardyMeshAttr(mesh)

func shardyTensorSharding*(s: Sharding; tensorRank: int): string =
  ## Renders the stripped SDY tensor sharding form used by Shardy collectives
  ## and `sdy.sharding_per_value`. Replicated sharding has no SDY attribute
  ## because absence of an attribute is fully open/replicated.
  if s.kind == skReplicated:
    return ""
  validateSharding(s, tensorRank)
  let mesh = s.activeMesh()
  let spec = s.activeSpec()
  "<@" & mesh.name & ", " & renderAxisGroups(spec.axisGroups) & ">"

func shardyPerValueAttr*(shardings: openArray[Sharding];
    ranks: openArray[int]): string =
  ## Renders `#sdy.sharding_per_value<[...]>` for values with explicit
  ## sharding. Replicated entries are omitted as open shardings (`<>`).
  if shardings.len != ranks.len:
    raise newException(ValueError,
      "shardyPerValueAttr: sharding/rank length mismatch")
  result = "#sdy.sharding_per_value<["
  for i, sharding in shardings:
    if i > 0: result.add ", "
    let item = sharding.shardyTensorSharding(ranks[i])
    result.add(if item.len == 0: "<>" else: item)
  result.add "]>"
