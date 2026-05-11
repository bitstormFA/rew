## Phase 8 — LR schedulers: StepLR, CosineAnnealing, ReduceOnPlateau.

block step_lr_init:
  let s = initStepLR(stepSize = 5, gamma = 0.5'f32)
  doAssert s.stepSize == 5
  doAssert s.gamma == 0.5'f32

block step_lr_decay:
  let s = initStepLR(stepSize = 3, gamma = 0.5'f32)
  var lr = 0.1'f32
  # No decay at epoch 0 (epoch+1 mod 3 != 0)
  doAssert step(s, lr, 0) == lr
  # Decay at epoch 2 (epoch+1 mod 3 == 0)
  doAssert abs(step(s, lr, 2) - 0.05'f32) < 1e-6'f32
  # No decay at epoch 3
  lr = step(s, lr, 2)
  doAssert step(s, lr, 3) == lr

block step_lr_no_decay_at_zero:
  let s = initStepLR(stepSize = 2, gamma = 0.5'f32)
  doAssert step(s, 0.1'f32, 0) == 0.1'f32

block cosine_annealing_init:
  let s = initCosineAnnealingLR(tMax = 10, etaMin = 0.0'f32)
  doAssert s.tMax == 10
  doAssert s.etaMin == 0.0'f32

block cosine_annealing_values:
  let s = initCosineAnnealingLR(tMax = 10, etaMin = 0.0'f32)
  let baseLr = 0.1'f32
  # At epoch 0: max (cos(0) = 1, value = baseLr)
  doAssert abs(step(s, baseLr, 0) - 0.1'f32) < 1e-4'f32
  # At epoch 5: halfway (cos(pi/2) = 0, value = 0.5 * baseLr)
  doAssert abs(step(s, baseLr, 5) - 0.05'f32) < 1e-4'f32
  # At epoch 10: wraps back to 0 (same as epoch 0)
  doAssert abs(step(s, baseLr, 10) - 0.1'f32) < 1e-4'f32

block reduce_on_plateau_init:
  let s = initReduceOnPlateau(factor = 0.5'f32, patience = 3, minLr = 1e-6'f32)
  doAssert s.patience == 3
  doAssert s.factor == 0.5'f32

block reduce_on_plateau_improves:
  var s = initReduceOnPlateau(factor = 0.5'f32, patience = 3)
  var state = initPlateauState(0.1'f32, modeIsMin = true)
  var lr = 0.1'f32
  # Metric improves (goes down), no decay
  let newLr = step(s, lr, 0.4'f32, state, modeIsMin = true)
  doAssert newLr == lr

block reduce_on_plateau_patience_exhausted:
  var s = initReduceOnPlateau(factor = 0.5'f32, patience = 2)
  var state = initPlateauState(0.1'f32, modeIsMin = true)
  var lr = 0.1'f32
  # First call sets baseline
  discard step(s, lr, 0.4'f32, state, modeIsMin = true)
  # No improvement
  discard step(s, lr, 0.5'f32, state, modeIsMin = true)
  # Still no improvement, patience exhausted
  let newLr = step(s, lr, 0.5'f32, state, modeIsMin = true)
  doAssert newLr < lr  # LR should be reduced

block reduce_on_plateau_min_lr:
  var s = initReduceOnPlateau(factor = 0.5'f32, patience = 1, minLr = 0.01'f32)
  var state = initPlateauState(0.02'f32, modeIsMin = true)
  var lr = 0.02'f32
  discard step(s, lr, 0.1'f32, state, modeIsMin = true)  # set baseline
  discard step(s, lr, 0.2'f32, state, modeIsMin = true)  # worse
  let newLr = step(s, lr, 0.2'f32, state, modeIsMin = true)  # patience=1
  doAssert newLr >= 0.01'f32  # should not go below minLr

block scheduler_kind_dispatch:
  let sk = SchedulerKind(kind: stStepLR,
      stepLr: initStepLR(stepSize = 5, gamma = 0.5'f32))
  doAssert sk.kind == stStepLR
  let newLr = step(sk, 0.1'f32, 0, baseLr = 0.1'f32)
  doAssert newLr == 0.1'f32  # no decay at epoch 0

block scheduler_config:
  let sc = SchedulerConfig(
    scheduler: SchedulerKind(kind: stStepLR,
        stepLr: initStepLR(stepSize = 10)),
    interval: siEpoch,
    frequency: 1,
  )
  doAssert sc.interval == siEpoch
  doAssert sc.frequency == 1
