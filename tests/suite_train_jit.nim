## Phase 8 — Trainer jit/donateParams fields (reserved for HLO integration).

block trainer_jit_field:
  var trainer = initTrainer()
  doAssert not trainer.jit  # off by default
  trainer.jit = true
  doAssert trainer.jit

block trainer_donate_params_field:
  var trainer = initTrainer()
  doAssert not trainer.donateParams  # off by default
  trainer.donateParams = true
  doAssert trainer.donateParams

block trainer_jit_and_donate_together:
  var trainer = initTrainer()
  trainer.jit = true
  trainer.donateParams = true
  doAssert trainer.jit
  doAssert trainer.donateParams
