## Multi-layer recurrent modules — LSTM, GRU, Bidirectional.
##
## These unroll cells over a sequence. Each layer is a value type carrying
## its cells. Bidirectional runs a module forward and backward over the
## time axis (axis=1 in NTC layout).

import ../tensor
import ../pytree
import ../rng
import ../ops/shape
import ../ops/concat
import ./rnn_cell

type
  LSTM* = object
    ## Multi-layer LSTM. `cells` is indexed `[layer][direction]`.
    ## Input shape: `[batch, seqLen, inputDim]` (NTC layout).
    ## Output shape: `[batch, seqLen, numDirs * hiddenDim]`.
    cells*: seq[seq[LSTMCell]]
    numLayers*: int
    hiddenDim*: int
    inputDim*: int
    bidirectional*: bool
    dropout*: float32

  LSTMStates* = object
    ## Stack of LSTM states indexed by `[layer][direction]`.
    states*: seq[seq[LSTMState]]

  GRU* = object
    ## Multi-layer GRU.
    cells*: seq[seq[GRUCell]]
    numLayers*: int
    hiddenDim*: int
    inputDim*: int
    bidirectional*: bool
    dropout*: float32

  GRUStates* = object
    ## Stack of GRU states indexed by `[layer][direction]`.
    states*: seq[seq[GRUState]]

# ---- LSTM --------------------------------------------------------------------

proc initLSTM*(key: Key; inputDim, hiddenDim: int; numLayers = 1;
    bidirectional = false; dropout: float32 = 0'f32): LSTM =
  ## Constructs a multi-layer LSTM.
  if inputDim <= 0 or hiddenDim <= 0 or numLayers <= 0:
    raise newException(TensorError,
      "initLSTM: dims and numLayers must be positive")
  if dropout < 0'f32 or dropout >= 1'f32:
    raise newException(TensorError,
      "initLSTM: dropout must be in [0, 1)")
  let numDirs = if bidirectional: 2 else: 1
  var cells: seq[seq[LSTMCell]] = @[]
  for layer in 0 ..< numLayers:
    var layerCells: seq[LSTMCell] = @[]
    let layerInputDim = if layer == 0: inputDim else: hiddenDim * numDirs
    for dir in 0 ..< numDirs:
      let k = foldIn(key, uint64(layer * numDirs + dir))
      layerCells.add initLSTMCell(k, layerInputDim, hiddenDim)
    cells.add layerCells
  LSTM(cells: cells, numLayers: numLayers, hiddenDim: hiddenDim,
    inputDim: inputDim, bidirectional: bidirectional, dropout: dropout)

proc initLSTMStates*(batchSize: int; layer: LSTM): LSTMStates =
  ## Initializes zero states for each layer and direction.
  let numDirs = if layer.bidirectional: 2 else: 1
  var states: seq[seq[LSTMState]] = @[]
  for _ in 0 ..< layer.numLayers:
    var layerStates: seq[LSTMState] = @[]
    for _ in 0 ..< numDirs:
      layerStates.add initLSTMState(batchSize, layer.hiddenDim)
    states.add layerStates
  LSTMStates(states: states)

proc forward*(layer: LSTM; x: Tensor; states: LSTMStates): (Tensor, LSTMStates) =
  ## Unrolls the LSTM over the sequence. `x` is `[batch, seqLen, inputDim]`.
  ## `states` holds initial states per layer/direction. Returns
  ## `(output, newStates)` where output is `[batch, seqLen, numDirs*hiddenDim]`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "LSTM.forward: expected [batch, seqLen, inputDim], got " & $x.shape)
  let numDirs = if layer.bidirectional: 2 else: 1
  let batchSize = x.shape[0]
  let seqLen = x.shape[1]
  var layerOutput = x
  var newStatesSeq: seq[seq[LSTMState]] = @[]
  for l in 0 ..< layer.numLayers:
    var dirOutputs: seq[Tensor] = @[]
    var layerNewStates: seq[LSTMState] = @[]
    for d in 0 ..< numDirs:
      var h: Tensor = states.states[l][d].h
      var c: Tensor = states.states[l][d].c
      var stepOutputs: seq[Tensor] = @[]
      if d == 0:
        # Forward direction: iterate 0..seqLen-1
        for s in 0 ..< seqLen:
          let xt = slice(layerOutput, [0, s, 0],
            [batchSize, s + 1, layerOutput.shape[2]], [1, 1, 1])
          let xt2d = reshape(xt, [batchSize, layerOutput.shape[2]])
          let (hNew, newState) = layer.cells[l][d].forward(xt2d,
            LSTMState(h: buffer(h), c: buffer(c)))
          h = hNew
          c = newState.c
          stepOutputs.add hNew
        dirOutputs.add stack(stepOutputs, 1)
      else:
        # Backward direction: iterate seqLen-1..0
        for si in countdown(seqLen - 1, 0):
          let xt = slice(layerOutput, [0, si, 0],
            [batchSize, si + 1, layerOutput.shape[2]], [1, 1, 1])
          let xt2d = reshape(xt, [batchSize, layerOutput.shape[2]])
          let (hNew, newState) = layer.cells[l][d].forward(xt2d,
            LSTMState(h: buffer(h), c: buffer(c)))
          h = hNew
          c = newState.c
          stepOutputs.add hNew
        # Reverse backward outputs back to correct order
        var reversed = newSeq[Tensor](stepOutputs.len)
        for i, t in stepOutputs:
          reversed[stepOutputs.len - 1 - i] = t
        dirOutputs.add stack(reversed, 1)
      layerNewStates.add LSTMState(h: buffer(h), c: buffer(c))
    # Concatenate direction outputs
    if numDirs == 2:
      layerOutput = concat(dirOutputs, 2)
    else:
      layerOutput = dirOutputs[0]
    # Apply dropout between layers (not on last layer)
    if l < layer.numLayers - 1 and layer.dropout > 0'f32:
      # dropout in inference is no-op, training uses Dropout layer
      discard
    newStatesSeq.add layerNewStates
  (layerOutput, LSTMStates(states: newStatesSeq))

# ---- GRU ---------------------------------------------------------------------

proc initGRU*(key: Key; inputDim, hiddenDim: int; numLayers = 1;
    bidirectional = false; dropout: float32 = 0'f32): GRU =
  ## Constructs a multi-layer GRU.
  if inputDim <= 0 or hiddenDim <= 0 or numLayers <= 0:
    raise newException(TensorError,
      "initGRU: dims and numLayers must be positive")
  if dropout < 0'f32 or dropout >= 1'f32:
    raise newException(TensorError,
      "initGRU: dropout must be in [0, 1)")
  let numDirs = if bidirectional: 2 else: 1
  var cells: seq[seq[GRUCell]] = @[]
  for layer in 0 ..< numLayers:
    var layerCells: seq[GRUCell] = @[]
    let layerInputDim = if layer == 0: inputDim else: hiddenDim * numDirs
    for dir in 0 ..< numDirs:
      let k = foldIn(key, uint64(layer * numDirs + dir))
      layerCells.add initGRUCell(k, layerInputDim, hiddenDim)
    cells.add layerCells
  GRU(cells: cells, numLayers: numLayers, hiddenDim: hiddenDim,
    inputDim: inputDim, bidirectional: bidirectional, dropout: dropout)

proc initGRUStates*(batchSize: int; layer: GRU): GRUStates =
  ## Initializes zero states for each layer and direction.
  let numDirs = if layer.bidirectional: 2 else: 1
  var states: seq[seq[GRUState]] = @[]
  for _ in 0 ..< layer.numLayers:
    var layerStates: seq[GRUState] = @[]
    for _ in 0 ..< numDirs:
      layerStates.add initGRUState(batchSize, layer.hiddenDim)
    states.add layerStates
  GRUStates(states: states)

proc forward*(layer: GRU; x: Tensor; states: GRUStates): (Tensor, GRUStates) =
  ## Unrolls the GRU over the sequence. `x` is `[batch, seqLen, inputDim]`.
  if x.shape.len != 3:
    raise newException(TensorError,
      "GRU.forward: expected [batch, seqLen, inputDim], got " & $x.shape)
  let numDirs = if layer.bidirectional: 2 else: 1
  let batchSize = x.shape[0]
  let seqLen = x.shape[1]
  var layerOutput = x
  var newStatesSeq: seq[seq[GRUState]] = @[]
  for l in 0 ..< layer.numLayers:
    var dirOutputs: seq[Tensor] = @[]
    var layerNewStates: seq[GRUState] = @[]
    for d in 0 ..< numDirs:
      var h: Tensor = states.states[l][d].h
      var stepOutputs: seq[Tensor] = @[]
      if d == 0:
        for s in 0 ..< seqLen:
          let xt = slice(layerOutput, [0, s, 0],
            [batchSize, s + 1, layerOutput.shape[2]], [1, 1, 1])
          let xt2d = reshape(xt, [batchSize, layerOutput.shape[2]])
          let (hNew, _) = layer.cells[l][d].forward(xt2d,
            GRUState(h: buffer(h)))
          h = hNew
          stepOutputs.add h
        dirOutputs.add stack(stepOutputs, 1)
      else:
        for si in countdown(seqLen - 1, 0):
          let xt = slice(layerOutput, [0, si, 0],
            [batchSize, si + 1, layerOutput.shape[2]], [1, 1, 1])
          let xt2d = reshape(xt, [batchSize, layerOutput.shape[2]])
          let (hNew, _) = layer.cells[l][d].forward(xt2d,
            GRUState(h: buffer(h)))
          h = hNew
          stepOutputs.add h
        var reversed = newSeq[Tensor](stepOutputs.len)
        for i, t in stepOutputs:
          reversed[stepOutputs.len - 1 - i] = t
        dirOutputs.add stack(reversed, 1)
      layerNewStates.add GRUState(h: buffer(h))
    if numDirs == 2:
      layerOutput = concat(dirOutputs, 2)
    else:
      layerOutput = dirOutputs[0]
    newStatesSeq.add layerNewStates
  (layerOutput, GRUStates(states: newStatesSeq))
