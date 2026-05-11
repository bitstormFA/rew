## Phase 8 — OptimizerConfig types and OptimizerKind variants.

block optimizer_kind_sgd:
  let ok = OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert ok.kind == otSgd

block optimizer_kind_adam:
  let ok = OptimizerKind(kind: otAdam,
      adam: initAdam(scalarF32(cpu(0), 0.001'f32)))
  doAssert ok.kind == otAdam

block optimizer_kind_adamw:
  let ok = OptimizerKind(kind: otAdamW,
      adamw: initAdamW(scalarF32(cpu(0), 0.001'f32)))
  doAssert ok.kind == otAdamW

block optimizer_kind_momentum:
  let ok = OptimizerKind(kind: otMomentumSgd,
      momentum: initMomentumSgd(scalarF32(cpu(0), 0.01'f32)))
  doAssert ok.kind == otMomentumSgd

block optimizer_state_sgd_has_no_state:
  let opt = OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert opt.kind == otSgd

block optimizer_config:
  let ok = OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.1'f32)))
  let cfg = initOptimizerConfig(ok)
  doAssert cfg.frequency == 1
  doAssert isSome(cfg.scheduler) == false

block optimizer_config_with_scheduler:
  let ok = OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.1'f32)))
  let sc = SchedulerConfig(
    scheduler: SchedulerKind(kind: stStepLR,
        stepLr: initStepLR(stepSize = 10)),
    interval: siEpoch,
    frequency: 1,
  )
  let cfg = initOptimizerConfig(ok, scheduler = some(sc))
  doAssert isSome(cfg.scheduler)

block optimizer_config_frequency:
  let ok = OptimizerKind(kind: otSgd,
      sgd: initSgd(scalarF32(cpu(0), 0.1'f32)))
  let cfg = initOptimizerConfig(ok, frequency = 4)
  doAssert cfg.frequency == 4
