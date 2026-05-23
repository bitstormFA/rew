## Phase 8 - open optimizer protocol.

block optimizer_protocol_sgd:
  let tx = sgd(scalarF32(cpu(0), 0.1'f32))
  let state: EmptyOptState = initState(tx, scalarF32(cpu(0), 1'f32))
  discard state

block optimizer_protocol_adamw:
  let tx = adamw(scalarF32(cpu(0), 0.001'f32))
  let state: AdamState = initState(tx, scalarF32(cpu(0), 1'f32))
  discard state

block optimizer_protocol_chain:
  let tx = chain(clipByGlobalNorm(1'f32),
    sgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert tx.transforms.len == 2
  doAssert tx.transforms[0].kind == gtkClipByGlobalNorm
  doAssert tx.transforms[1].kind == gtkSgd

block optimizer_protocol_schedule:
  let schedule = proc(step: int): Tensor {.closure.} =
    scalarF32(cpu(0), float32(step + 1))
  let tx = chain(scaleBySchedule(schedule),
    sgd(scalarF32(cpu(0), 0.1'f32)))
  let state = initState(tx, scalarF32(cpu(0), 1'f32))
  doAssert state.states.len == 2

block optimizer_protocol_freeze:
  let tx = chain(freeze(["weight"]), sgd(scalarF32(cpu(0), 0.1'f32)))
  doAssert tx.transforms[0].kind == gtkFreeze
