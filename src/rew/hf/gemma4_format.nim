## Gemma 4 chat/tool-call formatting helpers.
##
## Implements the subset of `chat_template.jinja` needed for Hermes
## function-calling SFT rows.

import std/[json, strutils]

const
  BosToken = "<bos>"
  TurnStart = "<|turn>"
  TurnEnd = "<turn|>"
  ToolStart = "<|tool>"
  ToolEnd = "<tool|>"
  ToolCallStart = "<|tool_call>"
  ToolCallEnd = "<tool_call|>"
  QuoteTok = "<|\"|>"

proc quoteGemmaString(value: string): string =
  QuoteTok & value.replace("\\", "\\\\").replace("\n", "\\n") & QuoteTok

proc formatGemmaJsonValue*(node: JsonNode): string =
  ## Formats JSON in the compact key/value style used inside Gemma tool tags.
  case node.kind
  of JObject:
    result.add "{"
    var first = true
    for key, value in node.pairs:
      if not first:
        result.add ","
      first = false
      result.add key
      result.add ":"
      result.add formatGemmaJsonValue(value)
    result.add "}"
  of JArray:
    result.add "["
    var index = 0
    for value in node:
      if index > 0:
        result.add ","
      result.add formatGemmaJsonValue(value)
      inc index
    result.add "]"
  of JString:
    let value = node.getStr()
    if value in ["object", "array", "string", "number", "integer",
        "boolean", "null"]:
      result = quoteGemmaString(value.toUpperAscii())
    else:
      result = quoteGemmaString(value)
  of JInt:
    result = $node.getBiggestInt()
  of JFloat:
    result = $node.getFloat()
  of JBool:
    result = if node.getBool(): "true" else: "false"
  of JNull:
    result = "null"

proc toolFunctionNode(tool: JsonNode): JsonNode =
  if tool.kind == JObject and tool.hasKey("function") and
      tool["function"].kind == JObject:
    tool["function"]
  else:
    tool

proc formatGemmaToolDeclaration*(tool: JsonNode): string =
  ## Formats one tool declaration as a Gemma 4 system tool block.
  let fn = toolFunctionNode(tool)
  let name = if fn.hasKey("name"): fn["name"].getStr() else: "tool"
  let desc = if fn.hasKey("description"): fn["description"].getStr() else: ""
  let params = if fn.hasKey("parameters"): fn["parameters"] else: newJObject()
  ToolStart & "declaration:" & name & "{" &
    "description:" & quoteGemmaString(desc) & "," &
    "parameters:" & formatGemmaJsonValue(params) & "}" & ToolEnd

proc parseHermesTools*(row: JsonNode): seq[JsonNode] =
  ## Parses the Hermes `tools` string column.
  if row.hasKey("tools"):
    let toolsNode =
      if row["tools"].kind == JString: parseJson(row["tools"].getStr())
      else: row["tools"]
    if toolsNode.kind == JArray:
      for tool in toolsNode:
        result.add tool

proc stripHermesToolBlock*(content: string): string =
  ## Removes the embedded `<tools>...</tools>` block from Hermes system text.
  let start = content.find("<tools>")
  let stop = content.find("</tools>")
  if start >= 0 and stop > start:
    let tailStart = stop + "</tools>".len
    let suffix = if tailStart < content.len: content[tailStart .. ^1] else: ""
    result = (content[0 ..< start] & suffix).strip()
  else:
    result = content.strip()

proc extractTagBlocks(content, openTag, closeTag: string): seq[string] =
  var pos = 0
  while pos < content.len:
    let start = content.find(openTag, pos)
    if start < 0:
      pos = content.len
    else:
      let bodyStart = start + openTag.len
      let stop = content.find(closeTag, bodyStart)
      if stop < 0:
        pos = content.len
      else:
        result.add content[bodyStart ..< stop].strip()
        pos = stop + closeTag.len

proc formatGemmaToolCall(callNode: JsonNode): string =
  let name =
    if callNode.hasKey("name"): callNode["name"].getStr()
    elif callNode.hasKey("function") and callNode["function"].hasKey("name"):
      callNode["function"]["name"].getStr()
    else:
      "tool"
  var args = newJObject()
  if callNode.hasKey("arguments"):
    if callNode["arguments"].kind == JString:
      args = parseJson(callNode["arguments"].getStr())
    else:
      args = callNode["arguments"]
  elif callNode.hasKey("function") and callNode["function"].hasKey("arguments"):
    let fnArgs = callNode["function"]["arguments"]
    args = if fnArgs.kind == JString: parseJson(fnArgs.getStr()) else: fnArgs
  ToolCallStart & "call:" & name & formatGemmaJsonValue(args) & ToolCallEnd

proc formatAssistantContent(content: string): string =
  let blocks = extractTagBlocks(content, "<tool_call>", "</tool_call>")
  if blocks.len == 0:
    return content.strip()
  for item in blocks:
    result.add formatGemmaToolCall(parseJson(item))

proc roleName(fromValue: string): string =
  case fromValue
  of "system": "system"
  of "human", "user": "user"
  of "gpt", "assistant", "model": "model"
  else: fromValue

proc formatGemma4ToolSft*(row: JsonNode; addBos = true): string =
  ## Converts a Hermes function-calling row to Gemma 4 SFT text.
  if addBos:
    result.add BosToken
  if not row.hasKey("conversations") or row["conversations"].kind != JArray:
    return result

  let tools = parseHermesTools(row)
  for turn in row["conversations"]:
    let role = roleName(turn["from"].getStr())
    let rawContent = turn["value"].getStr()
    result.add TurnStart
    result.add role
    result.add "\n"
    if role == "system":
      let systemText = stripHermesToolBlock(rawContent)
      if systemText.len > 0:
        result.add systemText
        result.add "\n"
      for tool in tools:
        result.add formatGemmaToolDeclaration(tool)
        result.add "\n"
    elif role == "model":
      result.add formatAssistantContent(rawContent)
      result.add "\n"
    else:
      result.add rawContent.strip()
      result.add "\n"
    result.add TurnEnd
    result.add "\n"
