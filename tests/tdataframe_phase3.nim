## DataFrame phase-3 typed batch and Dataset bridge tests.

import rew
import rew/dev
import rew/dataframe

type
  FeatureBatch = object
    x: Tensor
    y: Tensor

  TinyModel = object
    w: Param[Tensor]

proc initTinyModel(d: Device): TinyModel =
  TinyModel(w: param(fromHostF32(d, [0'f32], [1])))

proc tinyLoss(model: TinyModel; batch: FeatureBatch; ctx: CallCtx): Tensor =
  discard ctx
  let pred = mul(model.w, batch.x)
  let diff = sub(pred, batch.y)
  reduceMean(mul(diff, diff), [0])

proc setupCpu(d: var Device): bool =
  d = cpu()
  setDefaultDevice(d)
  installEagerBackend()
  try:
    discard scalarF32(d, 0'f32)
    true
  except EagerError as e:
    echo "  (skip) no CPU plugin - skipping DataFrame tensor bridge tests: ", e.msg
    false

var d: Device
if setupCpu(d):

  block collect_as_typed_batch:
    let df = sql("""
        select * from (values
          (1.0::double, 10.0::double),
          (2.0::double, 20.0::double),
          (3.0::double, 30.0::double)
        ) as t(x, y)
    """)
    let batch = collectAs[FeatureBatch](df, d)

    doAssert batch.x.shape == @[3]
    doAssert batch.y.shape == @[3]
    doAssert batch.x.dtype == dtFloat32
    doAssert batch.y.dtype == dtFloat32
    doAssert batch.x.toHost(float32) == @[1'f32, 2'f32, 3'f32]
    doAssert batch.y.toHost(float32) == @[10'f32, 20'f32, 30'f32]

  block dataframe_to_dataset_batches:
    let df = sql("""
        select * from (values
          (1.0::double, 10.0::double),
          (2.0::double, 20.0::double),
          (3.0::double, 30.0::double)
        ) as t(x, y)
    """)
    let ds = toDataset[FeatureBatch](df, batchSize = 2, device = d)

    var seen: seq[int]
    for batch in ds:
      seen.add batch.x.shape[0]
    doAssert seen == @[2, 1]

  block dataframe_dataset_feeds_trainer:
    let df = sql("""
        select * from (values
          (1.0::double, 2.0::double),
          (2.0::double, 4.0::double),
          (3.0::double, 6.0::double)
        ) as t(x, y)
    """)
    let ds = toDataset[FeatureBatch](df, batchSize = 1, device = d)
    let data = initDataSplits(ds)
    var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.1'f32)))
    var trainer = initTrainer(maxEpochs = 1, accelerator = akCpu)

    trainer.fit(state, data, tinyLoss)

    doAssert state.step == 3

  block dataframe_batch_feeds_compute_grads:
    let batch = collectAs[FeatureBatch](sql("""
        select * from (values
          (1.0::double, 2.0::double)
        ) as t(x, y)
    """), d)
    let runtime = initRuntime(akCpu)
    let gradStep = jit(proc(args: openArray[Tensor]): seq[Tensor] =
      let batchX = args[1]
      let batchY = args[2]
      let fn = proc(params: openArray[Tensor]): Tensor =
        let pred = mul(params[0], batchX)
        let diff = sub(pred, batchY)
        reduceMean(mul(diff, diff), [0])
      @[runtime.computeGrads(fn, args[0])])
    let w = fromHostF32(d, [1'f32], [1])
    let grads = gradStep.call([w, batch.x, batch.y])

    doAssert grads.len == 1
    doAssert grads[0].shape == @[1]
    let host = grads[0].toHost(float32)
    doAssert host[0] > -2.1'f32 and host[0] < -1.9'f32

  block explicit_multi_column_mapping_builds_feature_matrix:
    type MatrixBatch = object
      features: Tensor
      target: Tensor

    let mapping = batchMapping[MatrixBatch](
      tensorField("features", ["a", "b"], dtFloat32),
      tensorField("target", "target", dtFloat32),
    )
    let df = sql("""
        select * from (values
          (1.0::double, 2.0::double, 0.0::double),
          (3.0::double, 4.0::double, 1.0::double)
        ) as t(a, b, target)
    """)
    let batch = collectAs[MatrixBatch](df, d, mapping = mapping)

    doAssert batch.features.shape == @[2, 2]
    doAssert batch.target.shape == @[2]
    doAssert batch.features.toHost(float32) == @[1'f32, 2'f32, 3'f32, 4'f32]
