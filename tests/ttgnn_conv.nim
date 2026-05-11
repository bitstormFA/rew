## GNN convolution layers — trace-mode shape and IR checks.

import rew
import std/strutils

let TestDevice = cpu(0)

# ---- GCNConv ---------------------------------------------------------------

block gcn_conv_forward_shape:
  let key = initKey(0x42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initGCNConv(key, 8, 16)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    doAssert y.dtype == dtFloat32
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.dot_general" in text
  doAssert "stablehlo.scatter" in text

block gcn_conv_no_bias:
  let key = initKey(0x42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initGCNConv(key, 8, 16, bias = false)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  verify(ctx.builder.build())

# ---- GATConv ---------------------------------------------------------------

block gat_conv_forward_shape:
  let key = initKey(0x43u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initGATConv(key, 8, 4, 2)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 8]  # outCh * heads = 4 * 2 = 8
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.scatter" in text

block gat_conv_average_heads:
  let key = initKey(0x43u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initGATConv(key, 8, 4, 2, concat = false)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 4]  # outCh = 4
    ctx.traceReturn([y])
  verify(ctx.builder.build())

# ---- SAGEConv --------------------------------------------------------------

block sage_conv_forward_shape:
  let key = initKey(0x44u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initSAGEConv(key, 8, 16)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block sage_conv_with_normalize:
  let key = initKey(0x44u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initSAGEConv(key, 8, 16, normalize = true)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  verify(ctx.builder.build())

# ---- GINConv ---------------------------------------------------------------

block gin_conv_forward_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    # Build a simple MLP: 8 -> 16 -> 16
    var mlp = initSequential()
    let key = initKey(0x45u64)
    let keys = split(key, 2)
    mlp.add initLinear(keys[0], 8, 16)
    mlp.add relu
    mlp.add initLinear(keys[1], 16, 16)
    var layer = initGINConv(mlp)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.scatter" in text

# ---- EdgeConv --------------------------------------------------------------

block edge_conv_forward_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var mlp = initSequential()
    let key = initKey(0x46u64)
    let keys = split(key, 2)
    mlp.add initLinear(keys[0], 16, 32)
    mlp.add relu
    mlp.add initLinear(keys[1], 32, 16)
    var layer = initEdgeConv(mlp)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# ---- GraphConv -------------------------------------------------------------

block graph_conv_forward_shape:
  let key = initKey(0x47u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtInt32],
      @[@[5, 8], @[2, 10]])
    var layer = initGraphConv(key, 8, 16)
    let y = layer.forward(inputs[0], inputs[1])
    doAssert y.shape == @[5, 16]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
