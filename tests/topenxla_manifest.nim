## OpenXLA manifest and coverage model.

import std/tables
import rew
import rew/xla

block manifest_pins_components:
  let m = openXlaManifest()
  doAssert m.schemaVersion == 1
  for name in ["xla", "stablehlo", "shardy", "xprof", "tokamax"]:
    doAssert name in m.components
    doAssert m.components[name].repo.len > 0
    doAssert m.components[name].rev.len == 40

block stablehlo_inventory_counts:
  let m = openXlaManifest()
  doAssert m.stablehlo.opCount == 117
  doAssert m.stablehlo.ops.len == 117
  let cov = m.stableHloCoverage()
  doAssert cov.supported == 117
  doAssert cov.total == 117
  doAssert m.missingStableHloOps().len == 0

block optional_tools_are_recorded:
  let m = openXlaManifest()
  for name in ["run_hlo_module", "hlo_opt", "xprof", "tokamax",
               "sdy_opt", "sdy_translate", "mpmd_opt"]:
    doAssert name in m.tools
    doAssert m.tools[name].binary.len > 0
    doAssert not m.tools[name].required

echo "topenxla_manifest: OK"
