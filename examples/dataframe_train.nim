## Train from a DuckDB-backed DataFrame using typed batches.

import rew
import rew/dataframe

type
  TinyModel = object
    w: Param[Tensor]

  TinyBatch = object
    x: Tensor
    y: Tensor

proc initTinyModel(d: Device): TinyModel =
  TinyModel(w: param(fromHostF32(d, [0'f32, 0'f32], [2])))

proc loss(model: TinyModel; batch: TinyBatch; ctx: CallCtx): Tensor =
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
    echo "CPU plugin unavailable: ", e.msg
    false

proc main() =
  var d: Device
  if not setupCpu(d):
    return

  let df = sql("""
      select * from (values
        (1.0::double, 2.0::double),
        (2.0::double, 4.0::double),
        (3.0::double, 6.0::double),
        (4.0::double, 8.0::double)
      ) as t(x, y)
    """)
  let train = toDataset[TinyBatch](df, batchSize = 2, device = d)
  let data = initDataSplits(train)

  var state = initTrainState(initTinyModel(d), sgd(scalarF32(d, 0.05'f32)))
  var trainer = initTrainer(maxEpochs = 2, accelerator = akCpu)
  trainer.fit(state, data, loss)
  echo "trained steps: ", state.step

when isMainModule:
  main()
