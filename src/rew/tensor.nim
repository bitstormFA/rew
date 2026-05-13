## Tensor — the public, non-generic, value-typed array.
##
## ## Invariant #4
## A `Tensor` is a value `object` carrying dynamic dtype + shape, the device
## it lives on, a sharding annotation, and a `ref BufferHandle` to the
## on-device storage. ARC hooks live on `BufferHandle` only; `Tensor` itself
## is trivially copyable and just bumps the buffer's refcount.
##
## ## Trace and eager
## A `Tensor` is in one of two modes, distinguished by the `buffer` and
## `traceId` fields:
##
## | mode  | `buffer`  | `traceId`            |
## |-------|-----------|----------------------|
## | eager | non-nil   | `InvalidShValueId`   |
## | trace | nil       | a valid SSA value id |
##
## The dispatcher (`src/rew/dispatch.nim`) decides which mode an op runs in
## based on the current trace context. **Mixing modes inside a single op
## raises `TensorModeError`** so traced functions cannot accidentally close
## over eager tensors and vice versa.
##
## Predicates `isEager` / `isTrace` and the helpers `requireEager` /
## `requireTrace` are the only sanctioned ways to branch on mode.

import ./dtype
import ./sharding
import ./device
import ./buffer
import ./stablehlo/ir
import ./value

type
  TensorError* = object of CatchableError
    ## Base type for tensor-construction and shape/dtype validation errors.

  TensorModeError* = object of TensorError
    ## Raised when a trace-mode and an eager-mode tensor meet inside an op.
    ## Indicates a programming bug — closing over an eager tensor inside a
    ## traced function, or vice versa.

  Tensor* = object
    ## Public tensor value. See module doc for field semantics.
    dtype*: DType
    shape*: seq[int]
    device*: Device
    sharding*: Sharding
    buffer*: BufferHandle
      ## Non-nil in eager mode, nil in trace mode.
    traceId*: ShValueId
      ## `InvalidShValueId` (sentinel 0) in eager mode; a valid SSA id in
      ## trace mode.

func checkedElementCount(shape: openArray[int]; opName: string): int =
  result = 1
  for i, d in shape:
    if d < 0:
      raise newException(TensorError,
        opName & ": shape dimension #" & $i &
          " must be non-negative, got " & $d)
    if d != 0 and result > high(int) div d:
      raise newException(TensorError,
        opName & ": shape element count overflows int")
    result *= d

func numElements*(t: Tensor): int =
  ## Product of the shape dimensions; 1 for a 0-d (scalar) tensor.
  checkedElementCount(t.shape, "numElements")

func isEager*(t: Tensor): bool =
  ## True iff `t` is an eager-mode tensor (carries a live `BufferHandle`).
  not t.buffer.isNil

func isTrace*(t: Tensor): bool =
  ## True iff `t` is a trace-mode tensor (carries a valid SSA id, no
  ## buffer).
  t.buffer.isNil and t.traceId.int != 0

func tensorTypeOf*(t: Tensor): ShTensorType =
  ## Returns the `ShTensorType` matching this tensor's dtype + shape. Used
  ## by the dispatcher when threading trace tensors through the StableHLO
  ## builder.
  initTensorType(t.dtype, t.shape)

func shValueTypeOf*(t: Tensor): ShValueType =
  ## Returns the general StableHLO value descriptor for this tensor.
  initValueType(t.tensorTypeOf)

func valueTypeOf*(t: Tensor): ValueType =
  ## Returns the public general value type matching this tensor.
  initTensorValueType(t.dtype, t.shape)

proc requireEager*(t: Tensor; opName: string) =
  ## Guard called by eager-only code paths. Raises `TensorModeError` for
  ## trace tensors, and surfaces buffer-donation problems via
  ## `requireLive`.
  if t.buffer.isNil:
    raise newException(TensorModeError,
      opName & ": expected an eager tensor, got a trace tensor")
  t.buffer.requireLive(opName)

proc requireTrace*(t: Tensor; opName: string) =
  ## Guard called by trace-only code paths. Raises `TensorModeError` for
  ## eager tensors.
  if not t.buffer.isNil:
    raise newException(TensorModeError,
      opName & ": expected a trace tensor, got an eager tensor")
  if t.traceId.int == 0:
    raise newException(TensorModeError,
      opName & ": trace tensor has no SSA id (uninitialised)")

proc initTraceTensor*(id: ShValueId; dtype: DType; shape: openArray[int];
    device: Device; sharding: Sharding = initReplicated()): Tensor =
  ## Constructs a trace-mode tensor from a fresh SSA id. Called by the
  ## dispatcher after every op emission.
  if id.int == 0:
    raise newException(TensorModeError,
      "initTraceTensor: trace tensor requires a non-zero SSA id")
  discard checkedElementCount(shape, "initTraceTensor")
  validateSharding(sharding, shape.len)
  Tensor(
    dtype: dtype,
    shape: @shape,
    device: device,
    sharding: sharding,
    buffer: nil,
    traceId: id,
  )

proc initEagerTensor*(buffer: BufferHandle; dtype: DType;
    shape: openArray[int]; device: Device;
    sharding: Sharding = initReplicated()): Tensor =
  ## Constructs an eager-mode tensor wrapping an existing `BufferHandle`.
  ## The handle's refcount is bumped by the assignment.
  if buffer.isNil:
    raise newException(TensorError,
      "initEagerTensor: buffer must not be nil")
  discard checkedElementCount(shape, "initEagerTensor")
  validateSharding(sharding, shape.len)
  Tensor(
    dtype: dtype,
    shape: @shape,
    device: device,
    sharding: sharding,
    buffer: buffer,
    traceId: ShValueId(0),
  )

proc withSharding*(t: Tensor; sharding: Sharding): Tensor =
  ## Returns a copy of `t` annotated with `sharding`. This is metadata-only:
  ## it never moves buffers or performs implicit cross-device transfer.
  validateSharding(sharding, t.shape.len)
  result = t
  result.sharding = sharding

proc shard*(t: Tensor; mesh: Mesh; spec: PartitionSpec): Tensor =
  ## Returns `t` annotated as mesh-partitioned.
  t.withSharding(initPartitioned(mesh, spec))

proc manualShard*(t: Tensor; mesh: Mesh; spec: PartitionSpec): Tensor =
  ## Returns `t` annotated for manual sharding.
  t.withSharding(initManualSharding(mesh, spec))

proc replicate*(t: Tensor): Tensor =
  ## Returns `t` annotated as replicated.
  t.withSharding(initReplicated())

proc requireSameDevice*(a, b: Tensor; opName: string) =
  ## Raises `DeviceError` when `a` and `b` live on different devices. Use
  ## at the entry of every binary op — the framework never moves data
  ## implicitly.
  if a.device != b.device:
    raise newException(DeviceError,
      opName & ": cross-device operation between " & $a.device &
        " and " & $b.device & " (use .to(device) to move explicitly)")

proc requireSameMode*(a, b: Tensor; opName: string) =
  ## Raises `TensorModeError` when `a` and `b` are in different modes (one
  ## trace, one eager).
  if a.isEager xor b.isEager:
    raise newException(TensorModeError,
      opName & ": cannot mix eager and trace tensors in the same op")
