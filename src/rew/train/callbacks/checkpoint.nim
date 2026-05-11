## Checkpoint — saves training state when a monitored metric improves.

import std/[options, os, strutils]
import ../callback
import ../context
import ./earlystop  # for CheckpointMode

type
  TrainerAccess* = object
    ## Interface that the Trainer provides to callbacks via the `trainer`
    ## pointer. The Trainer (Phase 5) allocates a `TrainerAccess`, fills in
    ## the proc fields, and passes `addr(access)` as the `trainer: pointer`
    ## argument to callback hooks.
    saveCheckpoint*: proc(path: string; ctx: TrainContext) {.closure.}

  Checkpoint* = object
    monitor*: string
    mode*: CheckpointMode
    saveLast*: bool
    saveTopK*: int
    dirPath*: string
    filename*: string

  CheckpointState = ref object
    bestScore*: float32
    saved*: seq[string]         # Paths of saved checkpoints, best first

func initCheckpoint*(monitor: string = "val/loss";
    dirPath: string = "checkpoints"; saveLast: bool = true;
    saveTopK: int = 1; mode: CheckpointMode = cmMin;
    filename: string = "ckpt-epoch={epoch}-step={step}"): Checkpoint =
  ## Creates a Checkpoint callback.
  ##
  ## - `monitor` — metric name to watch (e.g., "val/loss").
  ## - `mode` — `cmMin` (lower is better) or `cmMax` (higher is better).
  ## - `saveLast` — always save the most recent epoch.
  ## - `saveTopK` — keep at most this many best checkpoints (by monitor).
  ## - `dirPath` — directory for checkpoint files.
  ## - `filename` — template supporting `{epoch}`, `{step}`, `{metric}`.
  if saveTopK < 0:
    raise newException(ValueError,
      "initCheckpoint: saveTopK must be >= 0")
  Checkpoint(monitor: monitor, mode: mode, saveLast: saveLast,
             saveTopK: saveTopK, dirPath: dirPath, filename: filename)

func formatFilename(tmpl: string; epoch, step: int; metric: float32): string =
  result = tmpl
  result = result.replace("{epoch}", $epoch)
  result = result.replace("{step}", $step)
  result = result.replace("{metric}", formatFloat(float(metric), ffDecimal, 6))

proc pruneCheckpoints(state: CheckpointState; keep: int; dirPath: string) =
  ## Removes the oldest excess checkpoints beyond `keep` from disk and state.
  if keep <= 0:
    for path in state.saved:
      try: removeDir(path)
      except OSError: discard
    state.saved.setLen(0)
    return
  while state.saved.len > keep:
    let oldest = state.saved[^1]
    try: removeDir(oldest)
    except OSError: discard
    state.saved.delete(state.saved.high)

func toCallback*(ckpt: Checkpoint): Callback =
  ## Converts the Checkpoint config into a `Callback`.
  ##
  ## The callback casts `trainer` to `ptr TrainerAccess` to call
  ## `saveCheckpoint`. The Trainer (Phase 5) wires this up.
  let state = CheckpointState(
    bestScore: case ckpt.mode
      of cmMin: float32.high
      of cmMax: float32.low,
    saved: @[],
  )
  result = initCallback("Checkpoint")

  result.onTrainEpochEnd = some(proc(trainer, task: pointer;
      ctx: var TrainContext) {.closure.} =
    # Find the monitored metric
    var currentScore: float32
    var found = false
    for m in ctx.epochMetrics:
      if m.name == ckpt.monitor:
        currentScore = m.value
        found = true
        break
    if not found:
      if ckpt.saveLast:
        # Still save last even if monitor not found
        let path = ckpt.dirPath / formatFilename(ckpt.filename,
            ctx.epoch, ctx.globalStep, 0.0'f32)
        createDir(path)
        let access = cast[ptr TrainerAccess](trainer)
        access.saveCheckpoint(path, ctx)
      return

    let improved = case ckpt.mode
      of cmMin: currentScore < state.bestScore
      of cmMax: currentScore > state.bestScore

    let path = ckpt.dirPath / formatFilename(ckpt.filename,
        ctx.epoch, ctx.globalStep, currentScore)

    if improved:
      state.bestScore = currentScore
      createDir(path)
      let access = cast[ptr TrainerAccess](trainer)
      access.saveCheckpoint(path, ctx)
      # Insert at front (best first)
      state.saved.insert(path, 0)
      # Prune excess
      pruneCheckpoints(state, ckpt.saveTopK, ckpt.dirPath)
    elif ckpt.saveLast:
      createDir(path)
      let access = cast[ptr TrainerAccess](trainer)
      access.saveCheckpoint(path, ctx)
      # Append as non-best checkpoint
      state.saved.add(path)
      # Keep saved list manageable
      if state.saved.len > ckpt.saveTopK + 1:
        let oldest = state.saved[^1]
        try: removeDir(oldest)
        except OSError: discard
        state.saved.delete(state.saved.high)
  )
