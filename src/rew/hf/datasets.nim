## Hugging Face dataset sources.
##
## Provides lazy `Dataset` adapters for the datasets-server `/rows` API and
## cached JSON/JSONL files downloaded from the Hub.

import std/[httpclient, json, os, strutils, uri]
import ../data/dataset
import ./hub

type
  HfDatasetRow* = object
    ## One row returned by the Hugging Face datasets-server `/rows` API.
    rowIdx*: int
    row*: JsonNode
    truncatedCells*: seq[string]

const
  DatasetsServerEndpoint = "https://datasets-server.huggingface.co"

proc parseHfRowsPayload*(payload: JsonNode): tuple[
    rows: seq[HfDatasetRow]; total: int] =
  ## Parses a datasets-server `/rows` response.
  result.total = -1
  if payload.hasKey("num_rows_total"):
    result.total = payload["num_rows_total"].getInt()
  if payload.hasKey("rows") and payload["rows"].kind == JArray:
    for item in payload["rows"]:
      var row = HfDatasetRow(rowIdx: -1, row: newJObject())
      if item.hasKey("row_idx"):
        row.rowIdx = item["row_idx"].getInt()
      if item.hasKey("row"):
        row.row = item["row"]
      if item.hasKey("truncated_cells") and
          item["truncated_cells"].kind == JArray:
        for cell in item["truncated_cells"]:
          row.truncatedCells.add cell.getStr()
      result.rows.add row

proc rowsUrl(endpoint, repoId, config, split: string; offset, length: int):
    string =
  endpoint & "/rows?dataset=" & encodeUrl(repoId) &
    "&config=" & encodeUrl(config) &
    "&split=" & encodeUrl(split) &
    "&offset=" & $offset &
    "&length=" & $length

proc fromHfRows*(repoId, config: string; split = "train"; token = "";
    endpoint = DatasetsServerEndpoint; pageSize = 100): Dataset[HfDatasetRow] =
  ## Streams rows from the Hugging Face datasets-server `/rows` API.
  ##
  ## Each dataset traversal creates a fresh HTTP client and paginates with
  ## 100-row pages by default.
  result.source = proc(): iterator(): HfDatasetRow =
    let capturedRepo = repoId
    let capturedConfig = config
    let capturedSplit = split
    let capturedToken = token
    let capturedEndpoint = endpoint
    let capturedPageSize = pageSize
    result = iterator(): HfDatasetRow {.closure.} =
      var headers = newHttpHeaders({"User-Agent": "rew-hf-datasets/0.1"})
      let tok = if capturedToken.len > 0: capturedToken else: getEnv("HF_TOKEN", "")
      if tok.len > 0:
        headers["Authorization"] = "Bearer " & tok
      var client = newHttpClient(timeout = 120_000, headers = headers)
      defer: client.close()
      var offset = 0
      var done = false
      while not done:
        let url = rowsUrl(capturedEndpoint, capturedRepo, capturedConfig,
          capturedSplit, offset, capturedPageSize)
        let parsed = parseHfRowsPayload(parseJson(client.getContent(url)))
        if parsed.rows.len == 0:
          done = true
        else:
          for row in parsed.rows:
            yield row
          offset += parsed.rows.len
          if parsed.total >= 0 and offset >= parsed.total:
            done = true

iterator jsonArrayItems(node: JsonNode): JsonNode =
  if node.kind == JArray:
    for item in node:
      yield item
  elif node.kind == JObject and node.hasKey("data") and
      node["data"].kind == JArray:
    for item in node["data"]:
      yield item
  else:
    raise newException(DataError,
      "fromHfJsonFile: expected top-level JSON array or object with data[]")

proc fromHfJsonFile*(path: string): Dataset[JsonNode] =
  ## Creates a dataset from a cached `.json` or `.jsonl` file.
  result.source = proc(): iterator(): JsonNode =
    let capturedPath = path
    result = iterator(): JsonNode {.closure.} =
      if capturedPath.endsWith(".jsonl"):
        for line in lines(capturedPath):
          let trimmed = line.strip()
          if trimmed.len > 0:
            yield parseJson(trimmed)
      else:
        let node = parseFile(capturedPath)
        for item in jsonArrayItems(node):
          yield item

proc firstExistingDataFile(snapshotDir: string): string =
  for file in walkDirRec(snapshotDir):
    if file.endsWith(".json") or file.endsWith(".jsonl"):
      return file
  raise newException(DataError,
    "fromHfJson: no cached JSON/JSONL files in " & snapshotDir)

proc fromHfJson*(repoId: string; config = ""; revision = "main";
    cacheDir = ""; token = ""): Dataset[JsonNode] =
  ## Downloads a Hugging Face dataset config and reads cached JSON/JSONL rows.
  let snapshotDir = hfDownloadDataset(repoId, config, revision, cacheDir, token)
  fromHfJsonFile(firstExistingDataFile(snapshotDir))

proc downloadHermesFunctionCalling*(
    cacheDir = ""; token = ""): string =
  ## Downloads the default Hermes single-turn function-calling JSON data.
  hfDownloadDataset(
    "NousResearch/hermes-function-calling-v1",
    config = "func_calling_singleturn",
    cacheDir = cacheDir,
    token = token,
  )
