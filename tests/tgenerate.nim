import rew
import rew/pjrt/loader
import rew/binaries/target

proc canLoadCpu(): bool =
  try:
    discard loadPlugin(tCpu)
    true
  except PjrtError as e:
    echo "tgenerate: skipped - ", e.msg
    false

block generation_config_validation:
  var raised = false
  try:
    discard initGenerationConfig(topP = 0.0'f32)
  except ValueError:
    raised = true
  doAssert raised

block generate_greedy_uses_prompt_and_offsets:
  if not canLoadCpu(): break generate_greedy_uses_prompt_and_offsets
  let d = cpu(0)
  setDefaultDevice(d)
  installEagerBackend()

  var calls = 0
  var offsets: seq[int] = @[]
  var inputs: seq[seq[int32]] = @[]

  proc forward(inputIds: Tensor; offset: int;
      kvCache: Tensor = Tensor()): Tensor =
    offsets.add offset
    inputs.add inputIds.toHost(int32)
    let nextToken = if calls == 0: 4 else: 2
    calls += 1
    var logits = @[-10.0'f32, -10.0'f32, -10.0'f32, -10.0'f32, -10.0'f32]
    logits[nextToken] = 10.0'f32
    fromHost(logits, [1, 1, logits.len])

  let config = initGenerationConfig(
    maxNewTokens = 3,
    temperature = 0.0'f32,
    topK = 0,
    doSample = false,
  )
  let result = generate(forward, @[7, 8, 9], config, eosTokenId = 2)

  doAssert result.tokenIds == @[7, 8, 9, 4, 2]
  doAssert result.logProbs.len == 2
  doAssert offsets == @[0, 3]
  doAssert inputs == @[@[7'i32, 8'i32, 9'i32], @[4'i32]]
