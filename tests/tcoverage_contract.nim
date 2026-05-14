## Fast source-level guard for the primitive-op testing contract.
##
## This intentionally avoids importing `rew`, so it stays cheap while catching
## missing autograd policies and unmentioned primitive ops early in `bau test`.

import std/[algorithm, os, sets, strutils]

const
  OpsDir = "src/rew/ops"
  RegistryFile = "src/rew/autograd/registry.nim"
  RulesFile = "src/rew/autograd/rules.nim"
  TestsDir = "tests"

proc declName(line: string): string =
  let s = line.strip()
  let offset =
    if s.startsWith("proc "): 5
    elif s.startsWith("func "): 5
    else: return ""

  var i = offset
  while i < s.len and s[i].isSpaceAscii:
    inc i
  let start = i
  while i < s.len and (s[i].isAlphaNumeric or s[i] == '_'):
    inc i
  if i == start: return ""
  s[start ..< i]

proc collectRewOps(): HashSet[string] =
  for path in walkDirRec(OpsDir):
    if not path.endsWith(".nim"):
      continue
    var current = ""
    for raw in readFile(path).splitLines:
      let line = raw.strip()
      if line.len == 0 or line.startsWith("#"):
        continue
      let name = declName(line)
      if name.len > 0:
        current = name
      if current.len > 0 and "{.rewOp.}" in line:
        result.incl current
      if current.len > 0 and line.endsWith("="):
        current = ""

proc collectCalls(path: string; callName: string): HashSet[string] =
  for raw in readFile(path).splitLines:
    let line = raw.strip()
    if line.len == 0 or line.startsWith("#"):
      continue
    let needle = callName & "(\""
    var start = 0
    while true:
      let idx = line.find(needle, start)
      if idx < 0:
        break
      let nameStart = idx + needle.len
      let nameEnd = line.find('"', nameStart)
      doAssert nameEnd > nameStart, "unterminated call in " & path & ": " & line
      result.incl line[nameStart ..< nameEnd]
      start = nameEnd + 1

proc sortedItems(values: HashSet[string]): seq[string] =
  for value in values:
    result.add value
  result.sort()

proc merged(a, b: HashSet[string]): HashSet[string] =
  result = a
  for item in b:
    result.incl item

proc message(label: string; values: HashSet[string]): string =
  label & ": " & sortedItems(values).join(", ")

proc isIdentChar(c: char): bool =
  c.isAlphaNumeric or c == '_'

proc containsIdentifier(source, name: string): bool =
  var start = 0
  while true:
    let idx = source.find(name, start)
    if idx < 0:
      return false
    let beforeOk = idx == 0 or not isIdentChar(source[idx - 1])
    let after = idx + name.len
    let afterOk = after >= source.len or not isIdentChar(source[after])
    if beforeOk and afterOk:
      return true
    start = idx + 1

proc behaviorTestSource(): string =
  for path in walkDirRec(TestsDir):
    if not path.endsWith(".nim"):
      continue
    let name = path.extractFilename()
    if name in ["all.nim", "tcoverage_contract.nim"]:
      continue
    for raw in readFile(path).splitLines:
      let line = raw.strip()
      if line.len == 0 or line.startsWith("#"):
        continue
      result.add line
      result.add '\n'

let
  ops = collectRewOps()
  vjps = collectCalls(RegistryFile, "registerVjp")
  noGrad = collectCalls(RegistryFile, "registerNoGrad")
  policies = merged(vjps, noGrad)
  rules = collectCalls(RulesFile, "registerVjpRule")

block all_rew_ops_have_exactly_one_autograd_policy:
  doAssert ops.len > 0, "expected to find primitive ops"
  let missing = ops - policies
  let extra = policies - ops
  doAssert missing.len == 0, message("missing autograd policy", missing)
  doAssert extra.len == 0, message("policy without rewOp", extra)
  let ambiguous = vjps * noGrad
  doAssert ambiguous.len == 0, message("both vjp and no-grad", ambiguous)

block differentiable_ops_have_real_vjp_rules:
  let missingRules = vjps - rules
  doAssert missingRules.len == 0, message("missing VJP rule", missingRules)
  let ruleWithoutPolicy = rules - vjps
  doAssert ruleWithoutPolicy.len == 0,
    message("VJP rule without differentiable policy", ruleWithoutPolicy)

block every_primitive_op_is_mentioned_by_behavior_tests:
  let source = behaviorTestSource()
  var unmentioned: HashSet[string]
  for op in ops:
    if not source.containsIdentifier(op):
      unmentioned.incl op
  doAssert unmentioned.len == 0, message("op missing behavior-test mention",
    unmentioned)
