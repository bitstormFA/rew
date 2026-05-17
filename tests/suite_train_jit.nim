## Phase 8 — Trainer donation controls.

block trainer_donate_params_field:
  var trainer = initTrainer()
  doAssert not trainer.donateParams
  trainer.donateParams = true
  doAssert trainer.donateParams
