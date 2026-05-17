## Train umbrella — re-exports the two-tier training API.
##
## ## Tier 1: Runtime
## Opt-in scaling. You own the loop.
##
## ## Tier 2: Trainer
## Typed automation over `TrainState`, `DataSplits`, and loss/custom-step procs.
##
## Usage:
##   import rew/train
##   # or, for individual modules:
##   import rew/train/[runtime, context, trainer]

import ./train/runtime
import ./train/context
import ./train/datasplits
import ./train/state
import ./train/callback
import ./train/callbacks/[checkpoint, earlystop, progress, logmonitor]
import ./train/trainer

export runtime
export context
export datasplits
export state
export callback
export checkpoint
export earlystop
export progress
export logmonitor
export trainer
