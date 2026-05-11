## Optional OpenXLA tool wrapper behavior.

import std/[os, strutils]
import rew

block tool_names_and_env:
  doAssert toolName(otRunHloModule) == "run_hlo_module"
  doAssert toolEnvVar(otRunHloModule) == "REW_RUN_HLO_MODULE"
  doAssert toolName(otXprof) == "xprof"
  doAssert toolName(otSdyOpt) == "sdy_opt"
  doAssert toolEnvVar(otSdyTranslate) == "REW_SDY_TRANSLATE"
  doAssert toolName(otMpmdOpt) == "mpmd_opt"

block cli_tool_parsing:
  doAssert parseOpenXlaToolName("hlo-opt") == otHloOpt
  doAssert parseOpenXlaToolName("ptx_opt") == otPtxOpt
  doAssert parseOpenXlaToolName("sdy-translate") == otSdyTranslate
  doAssert "run_hlo_module" in openXlaCliUsage()
  doAssert openXlaToolChoices().len == 10
  doAssertRaises(OpenXlaCliError):
    discard parseOpenXlaToolName("missing")

block missing_env_path_raises:
  let old = getEnv("REW_RUN_HLO_MODULE")
  putEnv("REW_RUN_HLO_MODULE", "/definitely/missing/rew/run_hlo_module")
  var raised = false
  try:
    discard requireTool(otRunHloModule)
  except OpenXlaToolError:
    raised = true
  if old.len == 0:
    delEnv("REW_RUN_HLO_MODULE")
  else:
    putEnv("REW_RUN_HLO_MODULE", old)
  doAssert raised

block shardy_missing_env_path_raises:
  let old = getEnv("REW_SDY_OPT")
  putEnv("REW_SDY_OPT", "/definitely/missing/rew/sdy_opt")
  var raised = false
  try:
    discard requireTool(otSdyOpt)
  except OpenXlaToolError:
    raised = true
  if old.len == 0:
    delEnv("REW_SDY_OPT")
  else:
    putEnv("REW_SDY_OPT", old)
  doAssert raised

block xprof_profile_metadata:
  let spec = initXprofProfile("view",
    ["--logdir=/tmp/rew-trace"], metadata = [("run", "smoke")])
  doAssert xprofArgs(spec) == @["view", "--logdir=/tmp/rew-trace"]
  doAssert spec.metadata[0] == ("run", "smoke")

block custom_call_metadata:
  let spec = initTokamaxCall("tokamax.layer_norm", platform = "cuda")
  doAssert spec.target.name == "tokamax.layer_norm"
  doAssert spec.target.platform == "cuda"
  doAssert spec.apiVersion == ccApiTypedFfi

block custom_call_registry_and_tokamax_build_args:
  var registry = initCustomCallRegistry()
  let spec = initTokamaxCall("tokamax.rms_norm", backendConfig = "{}", platform = "cuda")
  doAssert registry.register(spec) == 0
  doAssert registry.hasCustomCall("tokamax.rms_norm", "cuda")
  doAssert registry.lookupCustomCall("tokamax.rms_norm", "cuda").backendConfig == "{}"

  let replacement = initTokamaxCall("tokamax.rms_norm", backendConfig = "{\"eps\":1e-5}", platform = "cuda")
  doAssert registry.register(replacement) == 0
  doAssert registry.specs.len == 1
  doAssert registry.lookupCustomCall("tokamax.rms_norm", "cuda").backendConfig == "{\"eps\":1e-5}"

  let kernel = initTokamaxKernel("tokamax.rms_norm",
    ["kernels/rms_norm.cc"], outputLibrary = "librms_norm.so",
    buildArgs = ["--target=cuda"], platform = "cuda")
  let args = tokamaxBuildArgs(kernel)
  doAssert args == @["build", "kernels/rms_norm.cc", "--output",
                     "librms_norm.so", "--target=cuda"]

echo "topenxla_tools: OK"
