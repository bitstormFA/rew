## Target enum parsing, host triplet detection, and default target probe.

import std/os
import rew/binaries/target

block parse_target:
  doAssert parseTarget("cpu") == tCpu
  doAssert parseTarget("cuda12") == tCuda12
  doAssert parseTarget("cuda13") == tCuda13
  doAssert parseTarget("rocm") == tRocm
  doAssert parseTarget("metal") == tMetal
  doAssert parseTarget("tpu") == tTpu
  doAssert parseTarget("CPU") == tCpu
  doAssert parseTarget(" Cuda12 ") == tCuda12

block parse_target_unknown_raises:
  var raised = false
  try:
    discard parseTarget("nonexistent")
  except TargetError:
    raised = true
  doAssert raised

block target_name_round_trip:
  for t in [tCpu, tCuda12, tCuda13, tRocm, tMetal, tTpu]:
    doAssert parseTarget(targetName(t)) == t

block host_triplet_detection:
  let t = detectHostTriplet()
  doAssert t.arch in {aX86_64, aAarch64}
  when defined(linux):
    doAssert t.os == osLinux
  when defined(macosx):
    doAssert t.os == osDarwin

block host_triplet_env_override:
  putEnv("REW_TARGET_PLATFORM", "aarch64-linux-gnu")
  let t = detectHostTriplet()
  doAssert t.arch == aAarch64
  doAssert t.os == osLinux
  doAssert t.abi == abiGnu
  delEnv("REW_TARGET_PLATFORM")

block default_target_env:
  putEnv("REW_TARGET", "cpu")
  resetDefaultTarget()
  doAssert defaultTarget() == tCpu
  delEnv("REW_TARGET")

block default_target_cached:
  putEnv("REW_TARGET", "cuda12")
  resetDefaultTarget()
  let t1 = defaultTarget()
  doAssert t1 == tCuda12
  let t2 = defaultTarget()
  doAssert t2 == t1
  delEnv("REW_TARGET")
  resetDefaultTarget()

block set_default_target:
  setDefaultTarget(tTpu)
  doAssert defaultTarget() == tTpu
  resetDefaultTarget()

echo "ttarget: OK"
