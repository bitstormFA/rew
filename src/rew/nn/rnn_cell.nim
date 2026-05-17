## Recurrent cells — LSTMCell, GRUCell.
##
## Pure value types following rew's functional nn invariant.
## Each cell operates on a single timestep: `forward(cell, x, state)` returns
## `(output, newState)`. Multi-layer unrolling is handled by `LSTM` / `GRU`
## in `rnn.nim`.

import std/math
import ../tensor
import ../pytree
import ../rng
import ../ops/literal
import ../ops/arith
import ../ops/unary
import ../ops/linalg
import ../ops/shape
import ../ops/concat
import ./init
import ./activation

type
  LSTMCell* = object
    ## Single LSTM cell. All weight matrices are fused into `wIh` and `wHh`
    ## for efficiency: `wIh` is `[4*hiddenDim, inputDim]` (i, f, g, o stacked),
    ## `wHh` is `[4*hiddenDim, hiddenDim]`.
    wIh*: Param[Tensor]
    wHh*: Param[Tensor]
    bIh*: Param[Tensor]
    bHh*: Param[Tensor]
    inputDim*: int
    hiddenDim*: int

  LSTMState* = object
    ## LSTM hidden state and cell state, each `[batch, hiddenDim]`.
    h*: Buffer[Tensor]
    c*: Buffer[Tensor]

  GRUCell* = object
    ## Single GRU cell. Weights are fused: `wIh` is `[3*hiddenDim, inputDim]`
    ## (r, z, n stacked), `wHh` is `[3*hiddenDim, hiddenDim]`.
    wIh*: Param[Tensor]
    wHh*: Param[Tensor]
    bIh*: Param[Tensor]
    bHh*: Param[Tensor]
    inputDim*: int
    hiddenDim*: int

  GRUState* = object
    ## GRU hidden state `[batch, hiddenDim]`.
    h*: Buffer[Tensor]

# ---- LSTMCell ----------------------------------------------------------------

proc initLSTMCell*(key: Key; inputDim, hiddenDim: int): LSTMCell =
  ## Constructs an LSTMCell with Kaiming uniform initialization.
  if inputDim <= 0 or hiddenDim <= 0:
    raise newException(TensorError,
      "initLSTMCell: dims must be positive")
  let keys = split(key, 2)
  let bound = (1.0f32 / float32(hiddenDim)).sqrt
  # wIh: [4*hiddenDim, inputDim]
  let wihData = uniformF32(keys[0], 4 * hiddenDim * inputDim, -bound, bound)
  # wHh: [4*hiddenDim, hiddenDim]
  let whhData = uniformF32(keys[1], 4 * hiddenDim * hiddenDim, -bound, bound)
  # biases: zeros
  var bihData = newSeq[float32](4 * hiddenDim)
  let bhhData = newSeq[float32](4 * hiddenDim)
  # forget gate bias init to 1.0 for better convergence
  for i in hiddenDim ..< 2 * hiddenDim:
    bihData[i] = 1'f32
  LSTMCell(
    wIh: param(constantF32([4 * hiddenDim, inputDim], wihData)),
    wHh: param(constantF32([4 * hiddenDim, hiddenDim], whhData)),
    bIh: param(constantF32([4 * hiddenDim], bihData)),
    bHh: param(constantF32([4 * hiddenDim], bhhData)),
    inputDim: inputDim,
    hiddenDim: hiddenDim,
  )

proc initLSTMState*(batchSize, hiddenDim: int): LSTMState =
  ## Initializes zero LSTM state for a given batch size.
  let hData = newSeq[float32](batchSize * hiddenDim)
  let cData = newSeq[float32](batchSize * hiddenDim)
  LSTMState(
    h: buffer(constantF32([batchSize, hiddenDim], hData)),
    c: buffer(constantF32([batchSize, hiddenDim], cData)),
  )

proc forward*(cell: LSTMCell; x: Tensor; state: LSTMState): (Tensor, LSTMState) =
  ## Single LSTM timestep. `x` is `[batch, inputDim]`, state holds `h` and
  ## `c` of shape `[batch, hiddenDim]`. Returns `(output, newState)`.
  if x.shape.len != 2 or x.shape[1] != cell.inputDim:
    raise newException(TensorError,
      "LSTMCell.forward: expected [batch, " & $cell.inputDim & "], got " & $x.shape)
  let batchSize = x.shape[0]
  # Gates: [batch, 4*hiddenDim]
  let gates = add(
    matmul(x, transpose(cell.wIh, [1, 0])),
    broadcastTo(cell.bIh, [batchSize, 4 * cell.hiddenDim], @[0]))
  let hGates = add(
    matmul(state.h, transpose(cell.wHh, [1, 0])),
    broadcastTo(cell.bHh, [batchSize, 4 * cell.hiddenDim], @[0]))
  let total = add(gates, hGates)
  # Split gates: i, f, g, o
  let hd = cell.hiddenDim
  let i = sigmoid(slice(total, [0, 0], [batchSize, hd], [1, 1]))
  let f = sigmoid(slice(total, [0, hd], [batchSize, 2 * hd], [1, 1]))
  let g = tanh(slice(total, [0, 2 * hd], [batchSize, 3 * hd], [1, 1]))
  let o = sigmoid(slice(total, [0, 3 * hd], [batchSize, 4 * hd], [1, 1]))
  # c_new = f * c + i * g
  let cNew = add(mul(f, state.c), mul(i, g))
  # h_new = o * tanh(c_new)
  let hNew = mul(o, tanh(cNew))
  (hNew, LSTMState(h: buffer(hNew), c: buffer(cNew)))

# ---- GRUCell -----------------------------------------------------------------

proc initGRUCell*(key: Key; inputDim, hiddenDim: int): GRUCell =
  ## Constructs a GRUCell with Kaiming uniform initialization.
  if inputDim <= 0 or hiddenDim <= 0:
    raise newException(TensorError,
      "initGRUCell: dims must be positive")
  let keys = split(key, 2)
  let bound = (1.0f32 / float32(hiddenDim)).sqrt
  # wIh: [3*hiddenDim, inputDim]
  let wihData = uniformF32(keys[0], 3 * hiddenDim * inputDim, -bound, bound)
  # wHh: [3*hiddenDim, hiddenDim]
  let whhData = uniformF32(keys[1], 3 * hiddenDim * hiddenDim, -bound, bound)
  let bihData = newSeq[float32](3 * hiddenDim)
  let bhhData = newSeq[float32](3 * hiddenDim)
  GRUCell(
    wIh: param(constantF32([3 * hiddenDim, inputDim], wihData)),
    wHh: param(constantF32([3 * hiddenDim, hiddenDim], whhData)),
    bIh: param(constantF32([3 * hiddenDim], bihData)),
    bHh: param(constantF32([3 * hiddenDim], bhhData)),
    inputDim: inputDim,
    hiddenDim: hiddenDim,
  )

proc initGRUState*(batchSize, hiddenDim: int): GRUState =
  ## Initializes zero GRU state for a given batch size.
  let hData = newSeq[float32](batchSize * hiddenDim)
  GRUState(h: buffer(constantF32([batchSize, hiddenDim], hData)))

proc forward*(cell: GRUCell; x: Tensor; state: GRUState): (Tensor, GRUState) =
  ## Single GRU timestep. `x` is `[batch, inputDim]`, state holds `h` of
  ## shape `[batch, hiddenDim]`. Returns `(output, newState)`.
  if x.shape.len != 2 or x.shape[1] != cell.inputDim:
    raise newException(TensorError,
      "GRUCell.forward: expected [batch, " & $cell.inputDim & "], got " & $x.shape)
  let batchSize = x.shape[0]
  let hd = cell.hiddenDim
  # Input projections: [batch, 3*hiddenDim]
  let xGates = add(
    matmul(x, transpose(cell.wIh, [1, 0])),
    broadcastTo(cell.bIh, [batchSize, 3 * hd], @[0]))
  # Hidden projections: [batch, 3*hiddenDim]
  let hGates = add(
    matmul(state.h, transpose(cell.wHh, [1, 0])),
    broadcastTo(cell.bHh, [batchSize, 3 * hd], @[0]))
  # Split: r, z, n
  let xR = slice(xGates, [0, 0], [batchSize, hd], [1, 1])
  let xZ = slice(xGates, [0, hd], [batchSize, 2 * hd], [1, 1])
  let xN = slice(xGates, [0, 2 * hd], [batchSize, 3 * hd], [1, 1])
  let hR = slice(hGates, [0, 0], [batchSize, hd], [1, 1])
  let hZ = slice(hGates, [0, hd], [batchSize, 2 * hd], [1, 1])
  let hN = slice(hGates, [0, 2 * hd], [batchSize, 3 * hd], [1, 1])
  # r = sigmoid(x_r + h_r)
  let r = sigmoid(add(xR, hR))
  # z = sigmoid(x_z + h_z)
  let z = sigmoid(add(xZ, hZ))
  # n = tanh(x_n + r * h_n)
  let n = tanh(add(xN, mul(r, hN)))
  # h_new = (1 - z) * n + z * h
  let one = scalarF32(1'f32)
  var dims: seq[int] = @[]
  let oneB = broadcastTo(one, z.shape, dims)
  let hNew = add(mul(sub(oneB, z), n), mul(z, state.h))
  (hNew, GRUState(h: buffer(hNew)))
