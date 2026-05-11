## Minimal Hugging Face `tokenizer.json` BPE loader.
##
## This covers the Gemma 4 text tokenizer shape used by the QLoRA example:
## vocab, merge ranks, added special tokens, space-to-sentinel normalization,
## and decode sentinel replacement.

import std/[algorithm, json, os, strutils, tables]

type
  HfTokenizer* = object
    ## Minimal BPE tokenizer loaded from a Hugging Face tokenizer.json file.
    vocab*: Table[string, int]
    idToToken*: Table[int, string]
    mergeRanks*: Table[string, int]
    specialTokens*: Table[string, int]
    specialTokenOrder*: seq[string]
    unkToken*: string
    bosToken*: string
    eosToken*: string
    padToken*: string

const
  SpaceMarker = "\226\150\129"
  PairSep = "\0"

proc pairKey(a, b: string): string =
  a & PairSep & b

proc startsAt(text, needle: string; pos: int): bool =
  if pos + needle.len > text.len:
    return false
  for i in 0 ..< needle.len:
    if text[pos + i] != needle[i]:
      return false
  true

proc addVocab(tok: var HfTokenizer; token: string; id: int) =
  tok.vocab[token] = id
  tok.idToToken[id] = token

proc addSpecial(tok: var HfTokenizer; token: string; id: int) =
  tok.specialTokens[token] = id
  if token notin tok.specialTokenOrder:
    tok.specialTokenOrder.add token
  addVocab(tok, token, id)

proc loadHfTokenizer*(path: string): HfTokenizer =
  ## Loads a minimal BPE tokenizer from Hugging Face `tokenizer.json`.
  let node = parseFile(path)
  result.unkToken = "<unk>"
  result.bosToken = "<bos>"
  result.eosToken = "<eos>"
  result.padToken = "<pad>"

  if node.hasKey("model") and node["model"].kind == JObject:
    let model = node["model"]
    if model.hasKey("unk_token"):
      result.unkToken = model["unk_token"].getStr()
    if model.hasKey("vocab") and model["vocab"].kind == JObject:
      for token, idNode in model["vocab"].pairs:
        result.addVocab(token, idNode.getInt())
    if model.hasKey("merges") and model["merges"].kind == JArray:
      var rank = 0
      for merge in model["merges"]:
        if merge.kind == JString:
          let parts = merge.getStr().splitWhitespace()
          if parts.len == 2:
            result.mergeRanks[pairKey(parts[0], parts[1])] = rank
            inc rank
        elif merge.kind == JArray and merge.len == 2:
          result.mergeRanks[pairKey(merge[0].getStr(),
            merge[1].getStr())] = rank
          inc rank

  if node.hasKey("added_tokens") and node["added_tokens"].kind == JArray:
    for added in node["added_tokens"]:
      if added.hasKey("content") and added.hasKey("id"):
        result.addSpecial(added["content"].getStr(), added["id"].getInt())

  for token in [result.unkToken, result.bosToken, result.eosToken,
      result.padToken]:
    if result.vocab.hasKey(token):
      result.addSpecial(token, result.vocab[token])

  result.specialTokenOrder.sort(proc(a, b: string): int = cmp(b.len, a.len))

proc normalizePiece(text: string): seq[string] =
  var i = 0
  while i < text.len:
    if text[i] == ' ':
      result.add SpaceMarker
      inc i
    else:
      result.add $text[i]
      inc i

proc applyBpe(tok: HfTokenizer; pieces: seq[string]): seq[string] =
  result = pieces
  var changed = true
  while changed and result.len > 1:
    changed = false
    var bestRank = high(int)
    var bestIndex = -1
    for i in 0 ..< result.len - 1:
      let key = pairKey(result[i], result[i + 1])
      if tok.mergeRanks.hasKey(key):
        let rank = tok.mergeRanks[key]
        if rank < bestRank:
          bestRank = rank
          bestIndex = i
    if bestIndex >= 0:
      result[bestIndex] = result[bestIndex] & result[bestIndex + 1]
      result.delete(bestIndex + 1)
      changed = true

proc encodeNormal(tok: HfTokenizer; text: string): seq[int] =
  let pieces = tok.applyBpe(normalizePiece(text))
  let unkId = tok.vocab.getOrDefault(tok.unkToken, 0)
  for piece in pieces:
    result.add tok.vocab.getOrDefault(piece, unkId)

proc encode*(tok: HfTokenizer; text: string; addBos = false;
    addEos = false): seq[int] =
  ## Encodes text to token ids using the loaded BPE vocab/merges.
  if addBos and tok.vocab.hasKey(tok.bosToken):
    result.add tok.vocab[tok.bosToken]
  var pos = 0
  var chunkStart = 0
  while pos < text.len:
    var matched = ""
    for special in tok.specialTokenOrder:
      if text.startsAt(special, pos):
        matched = special
        break
    if matched.len > 0:
      if chunkStart < pos:
        result.add tok.encodeNormal(text[chunkStart ..< pos])
      result.add tok.specialTokens[matched]
      pos += matched.len
      chunkStart = pos
    else:
      inc pos
  if chunkStart < text.len:
    result.add tok.encodeNormal(text[chunkStart ..< text.len])
  if addEos and tok.vocab.hasKey(tok.eosToken):
    result.add tok.vocab[tok.eosToken]

proc decode*(tok: HfTokenizer; ids: openArray[int]): string =
  ## Decodes ids to text with Gemma-style space sentinel replacement.
  for id in ids:
    result.add tok.idToToken.getOrDefault(id, tok.unkToken)
  result = result.replace(SpaceMarker, " ")

proc loadGemma4Tokenizer*(snapshotDir: string): HfTokenizer =
  ## Loads `tokenizer.json` from a downloaded Gemma 4 snapshot.
  loadHfTokenizer(snapshotDir / "tokenizer.json")
