import rew
let TestDevice = cpu(0)

block test_rope_debug:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[1, 3, 4]])
    let x = inputs[0]
    let half = 2
    let xPairs = reshape(x, x.shape[0..1] & @[half, 2])
    doAssert xPairs.shape == @[1, 3, 2, 2]
    var starts0 = @[0, 0, 0, 0]
    var limits0 = @[1, 3, 2, 1]
    var strides = @[1, 1, 1, 1]
    let x0 = slice(xPairs, starts0, limits0, strides)
    let x0Flat = reshape(x0, x0.shape[0 ..< x0.shape.len - 1])
    doAssert x0Flat.shape == @[1, 3, 2]
    ctx.traceReturn([x0Flat])
    discard ctx.builder.build()
