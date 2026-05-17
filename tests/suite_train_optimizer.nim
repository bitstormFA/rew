## Phase 8 — GradientTransform optimizer language.

block gradient_transform_sgd:
  let tx = sgd(scalarF32(cpu(0), 0.1'f32))
  doAssert tx.kind == gtkSgd

block gradient_transform_adamw:
  let tx = adamw(scalarF32(cpu(0), 0.001'f32))
  doAssert tx.kind == gtkAdamW

block gradient_transform_chain:
  let tx = chain(clipByGlobalNorm(1'f32),
    sgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert tx.kind == gtkChain
  doAssert tx.transforms.len == 2

block gradient_transform_schedule:
  let schedule = proc(step: int): Tensor {.closure.} =
    scalarF32(cpu(0), float32(step + 1))
  let tx = chain(scaleBySchedule(schedule),
    sgd(scalarF32(cpu(0), 0.1'f32)))
  let state = initState(tx, scalarF32(cpu(0), 1'f32))
  doAssert state.kind == gtkChain
  doAssert state.states.len == 2

block gradient_transform_freeze:
  let tx = chain(freeze(["weight"]), sgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert tx.kind == gtkChain
  doAssert tx.transforms[0].kind == gtkFreeze
