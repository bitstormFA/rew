## Train umbrella — re-exports the two-tier training API.
##
## ## Tier 1: Workbench
## Opt-in scaling. You own the loop.
##
## ## Tier 2: Trainer
## Full automation. The framework owns the loop.
##
## Usage:
##   import rew/train
##   # or, for individual modules:
##   import rew/train/[workbench, context, trainer]

import ./train/workbench
import ./train/context
import ./train/datapipe
import ./train/hooks
import ./train/optimizer
import ./train/callback
import ./train/callbacks/[checkpoint, earlystop, progress, logmonitor]
import ./train/trainer

export workbench
export context
export datapipe
export hooks
export optimizer
export callback
export checkpoint
export earlystop
export progress
export logmonitor
export trainer
