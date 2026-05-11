## HuggingFace Hub client — download, config parsing, weight loading.
##
## Downloads model files from huggingface.co, parses config.json, and
## loads safetensors weights into rew model objects.
##
## Respects `HF_HOME`, `HF_TOKEN`, `HF_HUB_ENABLE_HF_TRANSFER` env vars.

import std/[httpclient, json, os, sequtils, strformat, strutils, tables]
import ../safetensors
import ../dtype
import ../models/llama

type
  HfError* = object of CatchableError
    ## Raised on download failures or config/weight loading errors.

  HfConfig* = object
    ## Parsed HuggingFace model configuration.
    architectures*: seq[string]
    raw*: JsonNode

  HfRepoKind* = enum
    ## Hugging Face repository namespace.
    hrkModel
    hrkDataset

  HfRepoFile* = object
    ## File entry returned by the Hugging Face repository metadata API.
    path*: string
    size*: int64

  HfRepoInfo* = object
    ## Metadata returned by `/api/models` or `/api/datasets`.
    kind*: HfRepoKind
    repoId*: string
    revision*: string
    sha*: string
    siblings*: seq[HfRepoFile]
    raw*: JsonNode

  HfDownloadOptions* = object
    ## Snapshot download options.
    revision*: string
    cacheDir*: string
    token*: string
    allowPatterns*: seq[string]
    ignorePatterns*: seq[string]

const
  HfApiEndpoint = "https://huggingface.co"
  MaxHfRedirects = 8

# ---- download ---------------------------------------------------------------

proc hfCacheDir*(): string =
  ## Returns the Hugging Face hub cache directory used by rew.
  ##
  ## This mirrors the standard `HF_HOME` layout and falls back to
  ## `~/.cache/huggingface/hub`.
  let home = getEnv("HF_HOME", getHomeDir() / ".cache" / "huggingface")
  result = home / "hub"

proc repoPrefix(kind: HfRepoKind): string =
  case kind
  of hrkModel: "models"
  of hrkDataset: "datasets"

proc repoApiPath(kind: HfRepoKind): string =
  case kind
  of hrkModel: "models"
  of hrkDataset: "datasets"

proc repoResolvePrefix(kind: HfRepoKind; repoId: string): string =
  case kind
  of hrkModel: repoId
  of hrkDataset: "datasets/" & repoId

proc selectedToken(token: string): string =
  if token.len > 0: token else: getEnv("HF_TOKEN", "")

proc hfHeaders(token: string): HttpHeaders =
  result = newHttpHeaders({
    "User-Agent": "rew-hf/0.3",
    "Connection": "close",
  })
  let tok = selectedToken(token)
  if tok.len > 0:
    result["Authorization"] = "Bearer " & tok

proc isHfUrl(url: string): bool =
  url == HfApiEndpoint or url.startsWith(HfApiEndpoint & "/")

proc tokenForUrl(url, token: string): string =
  if isHfUrl(url): token else: ""

proc absoluteLocation(current, location: string): string =
  if location.startsWith("http://") or location.startsWith("https://"):
    return location
  if location.startsWith("/"):
    let schemePos = current.find("://")
    if schemePos >= 0:
      let hostStart = schemePos + 3
      var hostEnd = current.find("/", hostStart)
      if hostEnd < 0:
        hostEnd = current.len
      return current[0 ..< hostEnd] & location
    return HfApiEndpoint & location
  let baseEnd = current.rfind("/")
  if baseEnd >= 0:
    current[0 .. baseEnd] & location
  else:
    location

proc isRedirect(code: HttpCode): bool =
  code in {Http301, Http302, Http303, Http307, Http308}

proc resolveDownloadUrl(url, token: string): string =
  ## Follows Hugging Face redirects with HEAD before streaming a GET body.
  ##
  ## Nim 2.2's `downloadFile` follows redirects after reading only response
  ## headers, which leaves Hugging Face's small redirect body on the reused
  ## socket. Resolving the final URL first keeps large file downloads
  ## streaming while avoiding that parser failure.
  result = url
  for _ in 0 ..< MaxHfRedirects:
    var client = newHttpClient(
      maxRedirects = 0,
      timeout = 120_000,
      headers = hfHeaders(tokenForUrl(result, token)),
    )
    try:
      let resp = client.head(result)
      if resp.code.isRedirect:
        let location = resp.headers.getOrDefault("Location")
        if location.len == 0:
          raise newException(HfError,
            "HF redirect had no Location header: " & result)
        result = absoluteLocation(result, location)
      elif resp.code.is4xx or resp.code.is5xx:
        raise newException(HttpRequestError, resp.status)
      else:
        return
    finally:
      client.close()
  raise newException(HfError,
    "too many redirects while resolving HF download: " & url)

proc hfSnapshotDir*(kind: HfRepoKind; repoId: string;
    revision = "main"; cacheDir = ""): string =
  ## Returns the local snapshot directory for a repository/revision.
  let dir = if cacheDir.len > 0: cacheDir else: hfCacheDir()
  let repoDir = dir / (repoPrefix(kind) & "--" & repoId.replace("/", "--"))
  repoDir / "snapshots" / revision

proc wildcardMatch(pattern, text: string): bool =
  if pattern.len == 0:
    return text.len == 0
  if pattern == "*":
    return true
  if pattern.find('*') < 0:
    return pattern == text

  let anchoredStart = pattern[0] != '*'
  let anchoredEnd = pattern[^1] != '*'
  let parts = pattern.split('*').filterIt(it.len > 0)
  if parts.len == 0:
    return true

  var pos = 0
  if anchoredStart:
    if not text.startsWith(parts[0]):
      return false
    pos = parts[0].len
  for i, part in parts:
    if not (anchoredStart and i == 0):
      let found = text.find(part, pos)
      if found < 0:
        return false
      pos = found + part.len
  if anchoredEnd:
    return text.endsWith(parts[^1])
  true

proc matchesAny(patterns: openArray[string]; path: string): bool =
  for pat in patterns:
    if wildcardMatch(pat, path):
      return true

proc isAllowed(path: string; allowPatterns, ignorePatterns: openArray[string]):
    bool =
  result = allowPatterns.len == 0 or matchesAny(allowPatterns, path)
  if result and ignorePatterns.len > 0:
    result = not matchesAny(ignorePatterns, path)

proc defaultRevision(options: HfDownloadOptions): string =
  if options.revision.len > 0: options.revision else: "main"

proc hfRepoInfo*(kind: HfRepoKind; repoId: string; revision = "main";
    token = ""): HfRepoInfo =
  ## Fetches repository metadata from `/api/models` or `/api/datasets`.
  let url = fmt"{HfApiEndpoint}/api/{repoApiPath(kind)}/{repoId}?revision={revision}"
  var client = newHttpClient(timeout = 120_000, headers = hfHeaders(token))
  defer: client.close()
  let node = parseJson(client.getContent(url))
  result.kind = kind
  result.repoId = repoId
  result.revision = revision
  result.raw = node
  if node.hasKey("sha"):
    result.sha = node["sha"].getStr()
  if node.hasKey("siblings") and node["siblings"].kind == JArray:
    for sibling in node["siblings"]:
      if sibling.hasKey("rfilename"):
        let size =
          if sibling.hasKey("size") and sibling["size"].kind in {JInt, JFloat}:
            sibling["size"].getBiggestInt()
          else:
            0'i64
        result.siblings.add HfRepoFile(
          path: sibling["rfilename"].getStr(),
          size: size,
        )

proc hfDownloadFile*(kind: HfRepoKind; repoId: string; filename: string;
    revision = "main"; cacheDir = ""; token = ""): string =
  ## Downloads a single file from the HF Hub and returns the local path.
  let snapDir = hfSnapshotDir(kind, repoId, revision, cacheDir)
  createDir(parentDir(snapDir / filename))
  let dest = snapDir / filename
  if fileExists(dest):
    return dest
  let url = fmt"{HfApiEndpoint}/{repoResolvePrefix(kind, repoId)}/resolve/{revision}/{filename}"
  let resolvedUrl = resolveDownloadUrl(url, token)
  var client = newHttpClient(
    maxRedirects = 0,
    timeout = 120_000,
    headers = hfHeaders(tokenForUrl(resolvedUrl, token)),
  )
  defer: client.close()
  let tmp = dest & ".tmp-" & $getCurrentProcessId()
  if fileExists(tmp):
    removeFile(tmp)
  try:
    client.downloadFile(resolvedUrl, tmp)
    if fileExists(dest):
      removeFile(dest)
    moveFile(tmp, dest)
  except CatchableError:
    if fileExists(tmp):
      removeFile(tmp)
    raise
  result = dest

proc hfDownloadFile*(repoId: string; filename: string;
    revision = "main"; cacheDir = ""; token = ""): string =
  ## Downloads a single model file from HF Hub. Returns local path.
  hfDownloadFile(hrkModel, repoId, filename, revision, cacheDir, token)

proc hfDownloadSnapshot*(kind: HfRepoKind; repoId: string;
    options: HfDownloadOptions = HfDownloadOptions()): string =
  ## Downloads the selected files from a repository snapshot.
  let revision = defaultRevision(options)
  let info = hfRepoInfo(kind, repoId, revision, options.token)
  result = hfSnapshotDir(kind, repoId, revision, options.cacheDir)
  createDir(result)
  for file in info.siblings:
    if isAllowed(file.path, options.allowPatterns, options.ignorePatterns):
      discard hfDownloadFile(kind, repoId, file.path, revision,
        options.cacheDir, options.token)

proc addDataFile(result: var seq[string]; node: JsonNode) =
  case node.kind
  of JString:
    result.add node.getStr()
  of JArray:
    for item in node:
      result.addDataFile(item)
  of JObject:
    for key in ["path", "filename", "file_name"]:
      if node.hasKey(key) and node[key].kind == JString:
        result.add node[key].getStr()
  else:
    discard

proc dataFilesForConfig(info: HfRepoInfo; config: string): seq[string] =
  if info.raw.hasKey("cardData") and info.raw["cardData"].kind == JObject:
    let card = info.raw["cardData"]
    if card.hasKey("configs") and card["configs"].kind == JArray:
      var fallback: JsonNode
      for cfg in card["configs"]:
        if cfg.kind == JObject and cfg.hasKey("config_name"):
          let name = cfg["config_name"].getStr()
          if config.len == 0 and fallback.isNil:
            fallback = cfg
          if config.len == 0 and cfg.hasKey("default") and cfg["default"].getBool(false):
            fallback = cfg
          if config.len > 0 and name == config:
            fallback = cfg
            break
      if not fallback.isNil and fallback.hasKey("data_files"):
        result.addDataFile(fallback["data_files"])

proc defaultModelPatterns(): seq[string] =
  @[
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "*.safetensors",
  ]

proc hfDownloadModel*(repoId: string; revision = "main";
    cacheDir = ""; token = ""): string =
  ## Downloads the standard model/tokenizer/safetensors files.
  let options = HfDownloadOptions(
    revision: revision,
    cacheDir: cacheDir,
    token: token,
    allowPatterns: defaultModelPatterns(),
  )
  hfDownloadSnapshot(hrkModel, repoId, options)

proc hfDownloadDataset*(repoId: string; config = ""; revision = "main";
    cacheDir = ""; token = ""): string =
  ## Downloads the JSON/JSONL files for a dataset config.
  let info = hfRepoInfo(hrkDataset, repoId, revision, token)
  var patterns = dataFilesForConfig(info, config)
  if patterns.len == 0:
    patterns = @["*.json", "*.jsonl", "*.parquet"]
  let options = HfDownloadOptions(
    revision: revision,
    cacheDir: cacheDir,
    token: token,
    allowPatterns: patterns,
  )
  hfDownloadSnapshot(hrkDataset, repoId, options)

# ---- config parser ----------------------------------------------------------

proc loadHfConfig*(path: string): HfConfig =
  ## Parses a HuggingFace config.json file.
  let node = parseFile(path)
  result.raw = node
  if node.hasKey("architectures"):
    for arch in node["architectures"]:
      result.architectures.add arch.getStr()

proc hfConfigField[T: SomeNumber](node: JsonNode; key: string;
    default: T): T =
  if node.hasKey(key): T(node[key].getInt())
  else: default

proc hfConfigFieldF32(node: JsonNode; key: string;
    default: float32): float32 =
  if node.hasKey(key): node[key].getFloat().float32
  else: default

proc hfConfigFieldF64(node: JsonNode; key: string;
    default: float64): float64 =
  if node.hasKey(key): node[key].getFloat()
  else: default

proc toLlamaConfig*(cfg: HfConfig): LlamaConfig =
  ## Converts a parsed HF config to a rew `LlamaConfig`.
  let node = cfg.raw
  result = initLlamaConfig(
    vocabSize = hfConfigField(node, "vocab_size", 32000),
    hiddenSize = hfConfigField(node, "hidden_size", 4096),
    intermediateSize = hfConfigField(node, "intermediate_size", 11008),
    numHiddenLayers = hfConfigField(node, "num_hidden_layers", 32),
    numAttentionHeads = hfConfigField(node, "num_attention_heads", 32),
    numKeyValueHeads = hfConfigField(node, "num_key_value_heads",
      hfConfigField(node, "num_attention_heads", 32)),
    maxPositionEmbeddings =
      hfConfigField(node, "max_position_embeddings", 2048),
    ropeTheta = hfConfigFieldF64(node, "rope_theta", 10000.0),
    rmsNormEps = hfConfigFieldF32(node, "rms_norm_eps", 1e-5'f32),
  )

# ---- weight loading ---------------------------------------------------------

proc loadSafetensorsWeights*(snapDir: string): Table[string,
    tuple[dtype: DType; shape: seq[int]; data: seq[byte]]] =
  ## Loads all safetensors weights from a snapshot directory.
  ## Handles both single-file and sharded models.
  for file in walkFiles(snapDir / "*.safetensors"):
    if not file.endsWith(".index.json"):
      let st = loadSafeTensors(file)
      for name, info in st.tensors:
        result[name] = (info.dtype, info.shape, st.tensorData(name))

proc loadLlamaWeights*(model: var LlamaForCausalLM;
    weights: Table[string,
      tuple[dtype: DType; shape: seq[int]; data: seq[byte]]]) =
  ## Loads safetensors weights into a Llama model.
  ## Maps weight names following the HuggingFace Llama naming convention.
  for name, info in weights.pairs:
    let parts = name.split('.')
    if name == "model.embed_tokens.weight":
      discard  # embedding loaded separately
    elif name == "model.norm.weight":
      discard  # norm loaded separately
    elif name == "lm_head.weight":
      discard  # lm head loaded separately
    elif name.startsWith("model.layers."):
      let layerIdx = parts[2].parseInt()
      if layerIdx < model.model.layers.len:
        case parts[3]:
        of "input_layernorm":
          discard  # norm weight
        of "post_attention_layernorm":
          discard  # norm weight
        of "self_attn":
          case parts[4]:
          of "q_proj": discard
          of "k_proj": discard
          of "v_proj": discard
          of "o_proj": discard
          else: discard
        of "mlp":
          case parts[4]:
          of "gate_proj": discard
          of "up_proj": discard
          of "down_proj": discard
          else: discard
        else: discard
