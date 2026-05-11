## Manifest parsing and slot lookup.

import std/[strutils, tables]
import rew/binaries/manifest
import rew/binaries/target

func isSha256(s: string): bool =
  if s.len != 64:
    return false
  for ch in s:
    if ch notin {'0'..'9', 'a'..'f'}:
      return false
  true

block manifest_parses:
  let m = manifest()
  doAssert m.rewVersion.len > 0
  doAssert m.openxlaXlaRev.len > 0
  doAssert m.expectedApiVersion.major >= 0
  doAssert m.expectedApiVersion.minor >= 0

block manifest_has_slots:
  let m = manifest()
  doAssert len(m.slots) >= 12,
    "expected at least 12 slots, got " & $len(m.slots)

block manifest_slot_keys_round_trip:
  let m = manifest()
  for t in [tCpu, tCuda12, tCuda13, tRocm, tTpu]:
    let key = slotKeyFor(HostTriplet(arch: aX86_64, os: osLinux, abi: abiGnu), t)
    doAssert key in m.slots, "missing slot: " & key

block manifest_cpu_slots:
  let m = manifest()
  doAssert "x86_64-linux-gnu-cpu" in m.slots
  doAssert "aarch64-linux-gnu-cpu" in m.slots
  doAssert "aarch64-darwin-cpu" in m.slots
  doAssert "x86_64-windows-cpu" in m.slots

block manifest_gpu_slots:
  let m = manifest()
  doAssert "x86_64-linux-gnu-cuda12" in m.slots
  doAssert "x86_64-linux-gnu-cuda13" in m.slots
  doAssert "aarch64-linux-gnu-cuda12" in m.slots
  doAssert "aarch64-linux-gnu-cuda13" in m.slots
  doAssert "x86_64-linux-gnu-rocm" in m.slots

block manifest_metal_slots:
  let m = manifest()
  doAssert "x86_64-darwin-metal" in m.slots
  doAssert "aarch64-darwin-metal" in m.slots

block manifest_tpu_slot:
  let m = manifest()
  doAssert "x86_64-linux-gnu-tpu" in m.slots

block slot_source_types:
  let m = manifest()
  let cpuSlot = m.slots["x86_64-linux-gnu-cpu"]
  doAssert cpuSlot.source == ssVendor
  let cudaSlot = m.slots["x86_64-linux-gnu-cuda12"]
  doAssert cudaSlot.source == ssVendorWheel

block manifest_checksums_are_pinned:
  let m = manifest()
  for key, slot in m.slots:
    doAssert isSha256(slot.sha256), "invalid sha256 for " & key
    doAssert not slot.sha256.allCharsInSet({'0'}),
      "placeholder sha256 for " & key

block lookup_missing_raises:
  let m = manifest()
  var raised = false
  try:
    discard m.lookupSlot("nonexistent-slot-xyz")
  except KeyError:
    raised = true
  doAssert raised

echo "tmanifest: OK"
