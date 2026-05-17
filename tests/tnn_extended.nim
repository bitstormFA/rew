## Tests for new nn layers, activations, losses, and optimizers.

import rew
import rew/xla
import std/strutils

let TestDevice = cpu(0)

# --- Activation functions ---

block sigmoid_shape_and_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 3]])
    let y = sigmoid(inputs[0])
    doAssert y.shape == @[4, 3]
    doAssert y.dtype == dtFloat32
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.exponential" in text
  doAssert "stablehlo.negate" in text
  doAssert "stablehlo.divide" in text

block gelu_shape_and_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 5]])
    let y = gelu(inputs[0])
    doAssert y.shape == @[2, 5]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.tanh" in text

block silu_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let y = silu(inputs[0])
    doAssert y.shape == @[3]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block leaky_relu_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 4]])
    let y = leakyRelu(inputs[0], 0.1'f32)
    doAssert y.shape == @[4, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.maximum" in text

block softmax_shape_and_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 10]])
    let y = softmax(inputs[0], 1)
    doAssert y.shape == @[4, 10]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.exponential" in text
  doAssert "stablehlo.reduce" in text

block log_softmax_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 10]])
    let y = logSoftmax(inputs[0], 1)
    doAssert y.shape == @[4, 10]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- New unary ops ---

block sine_cosine_rsqrt_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 2]])
    let s = sine(inputs[0])
    let c = cosine(inputs[0])
    let r = rsqrt(inputs[0])
    doAssert s.shape == @[3, 2]
    doAssert c.shape == @[3, 2]
    doAssert r.shape == @[3, 2]
    ctx.traceReturn([s, c, r])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.sine" in text
  doAssert "stablehlo.cosine" in text
  doAssert "stablehlo.rsqrt" in text

# --- Concatenation and slice ---

block concat_shape_and_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[3, 4], @[5, 4]])
    let y = concat(inputs, 0)
    doAssert y.shape == @[8, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.concatenate" in text

block slice_shape_and_verify:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[10, 8]])
    let y = slice(inputs[0], [2, 0], [7, 4])
    doAssert y.shape == @[5, 4]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.slice" in text
  doAssert "stablehlo.slice %v1 [2:7, 0:4]" in text

block concat_invalid_dim_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 4]])
    doAssertRaises(TensorError):
      discard concat(inputs, 5)
    ctx.traceReturn(inputs)

block slice_out_of_bounds_raises:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[5, 3]])
    doAssertRaises(TensorError):
      discard slice(inputs[0], [0, 0], [10, 3])
    ctx.traceReturn(inputs)

# --- LayerNorm ---

block layernorm_forward_shape:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 8]])
    let ln = initLayerNorm([8])
    let y = ln.forward(inputs[0])
    doAssert y.shape == @[4, 8]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  doAssert "stablehlo.rsqrt" in text

block layernorm_3d:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[2, 4, 8]])
    let ln = initLayerNorm([4, 8])
    let y = ln.forward(inputs[0])
    doAssert y.shape == @[2, 4, 8]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- Dropout ---

block dropout_training_shape:
  let key = initKey(42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 8]])
    let dp = initDropout(0.5'f32)
    let y = dp.forward(inputs[0], key, training = true)
    doAssert y.shape == @[4, 8]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

block dropout_inference_passthrough:
  let key = initKey(42u64)
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 8]])
    let dp = initDropout(0.5'f32)
    let y = dp.forward(inputs[0], key, training = false)
    doAssert y.traceId == inputs[0].traceId
    ctx.traceReturn([y])

# --- Loss functions ---

block bce_loss_is_scalar:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 1], @[4, 1]])
    let l = binaryCrossEntropy(inputs[0], inputs[1])
    doAssert l.shape.len == 0
    ctx.traceReturn([l])
  let m = ctx.builder.build()
  verify(m)

block huber_loss_is_scalar:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[4, 3]])
    let l = huberLoss(inputs[0], inputs[1], delta = 1.0'f32)
    doAssert l.shape.len == 0
    ctx.traceReturn([l])
  let m = ctx.builder.build()
  verify(m)

block stabilized_softmax_ce_is_scalar:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 10], @[4, 10]])
    let l = softmaxCrossEntropy(inputs[0], inputs[1])
    doAssert l.shape.len == 0
    ctx.traceReturn([l])
  let m = ctx.builder.build()
  verify(m)
  let text = emitText(m)
  # Stabilized version should use reduceMax.
  doAssert "reduce" in text

# --- Embedding ---

block embedding_forward_shape:
  let key = initKey(99u64)
  withTrace ctx, "main", TestDevice:
    # Input is one-hot encoded indices [seqLen, vocabSize]
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 100]])
    let emb = initEmbedding(key, 100, 32)
    doAssert emb.weight.shape == @[100, 32]
    let y = emb.forward(inputs[0])
    doAssert y.shape == @[3, 32]
    ctx.traceReturn([y])
  let m = ctx.builder.build()
  verify(m)

# --- VJP rules for new ops ---

block abs_vjp_gradient:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(abs(args[0]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[4]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block reduce_max_vjp:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 4]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(reduceMax(args[0], [1]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3, 4]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block concat_vjp:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[3, 4], @[2, 4]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(concat(args, 0), [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 2
    doAssert grads[0].shape == @[3, 4]
    doAssert grads[1].shape == @[2, 4]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block slice_vjp:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[8, 4]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(slice(args[0], [2, 0], [6, 4]), [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[8, 4]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block sine_cosine_vjp:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(add(sine(args[0]), cosine(args[0])), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block rsqrt_vjp:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(rsqrt(args[0]), [0])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[4]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block sigmoid_differentiable:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[3, 2]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(sigmoid(args[0]), [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[3, 2]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

block softmax_differentiable:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 10]])
    let f = proc(args: openArray[Tensor]): Tensor =
      reduceSum(softmax(args[0], 1), [0, 1])
    let grads = grad(f, inputs)
    doAssert grads.len == 1
    doAssert grads[0].shape == @[4, 10]
    ctx.traceReturn(grads)
  let m = ctx.builder.build()
  verify(m)

# --- Optimizer tests ---

block adam_step_basic:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[4, 3]])
    let lr = scalarF32(0.001'f32)
    let opt = initAdam(lr)
    type Params = object
      w: Tensor
    let params = Params(w: inputs[0])
    let grads = Params(w: inputs[1])
    let state = initAdamState(params)
    let (newParams, newState) = opt.step(params, grads, state)
    doAssert newParams.w.shape == @[4, 3]
    doAssert newState.t == 1
    doAssert newState.beta1Power == opt.beta1
    doAssert newState.beta2Power == opt.beta2
    ctx.traceReturn([newParams.w])
  let m = ctx.builder.build()
  verify(m)

block adamw_step_basic:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[4, 3]])
    let lr = scalarF32(0.001'f32)
    let opt = initAdamW(lr, weightDecay = 0.01'f32)
    type Params = object
      w: Tensor
    let params = Params(w: inputs[0])
    let grads = Params(w: inputs[1])
    let state = initAdamState(params)
    let (newParams, newState) = opt.step(params, grads, state)
    doAssert newParams.w.shape == @[4, 3]
    doAssert newState.t == 1
    doAssert newState.beta1Power == opt.beta1
    doAssert newState.beta2Power == opt.beta2
    ctx.traceReturn([newParams.w])
  let m = ctx.builder.build()
  verify(m)

block momentum_sgd_step:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32, dtFloat32],
      @[@[4, 3], @[4, 3]])
    let lr = scalarF32(0.01'f32)
    let opt = initMomentumSgd(lr, momentum = 0.9'f32)
    type Params = object
      w: Tensor
    let params = Params(w: inputs[0])
    let grads = Params(w: inputs[1])
    let state = initMomentumState(params)
    let (newParams, newState) = opt.step(params, grads, state)
    doAssert newParams.w.shape == @[4, 3]
    doAssert newState.velocity.len == 1
    ctx.traceReturn([newParams.w])
  let m = ctx.builder.build()
  verify(m)

block clip_grad_norm_basic:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 3]])
    type Params = object
      w: Tensor
    let grads = Params(w: inputs[0])
    let clipped = clipGradNorm(grads, 1.0'f32)
    doAssert clipped.w.shape == @[4, 3]
    ctx.traceReturn([clipped.w])
  let m = ctx.builder.build()
  verify(m)

block clip_grad_value_basic:
  withTrace ctx, "main", TestDevice:
    let inputs = ctx.traceInputs(@[dtFloat32], @[@[4, 3]])
    type Params = object
      w: Tensor
    let grads = Params(w: inputs[0])
    let clipped = clipGradValue(grads, 0.5'f32)
    doAssert clipped.w.shape == @[4, 3]
    ctx.traceReturn([clipped.w])
  let m = ctx.builder.build()
  verify(m)

# --- Init helpers ---

block normal_f32_sanity:
  let key = initKey(123u64)
  let samples = normalF32(key, 1000, 0'f32, 1'f32)
  doAssert samples.len == 1000
  var sum: float32 = 0
  for s in samples: sum += s
  let mean = sum / 1000'f32
  doAssert mean > -0.5 and mean < 0.5  # loose check

block xavier_uniform_sanity:
  let key = initKey(77u64)
  let data = xavierUniformF32(key, 128, 64)
  doAssert data.len == 128 * 64

block kaiming_normal_sanity:
  let key = initKey(88u64)
  let data = kaimingNormalF32(key, 128, 128 * 64)
  doAssert data.len == 128 * 64
