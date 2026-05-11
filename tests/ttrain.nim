## Consolidated training API suite.
##
## The included suite fragments are not standalone test binaries; this file is
## the single `t*.nim` entry point that keeps training coverage fast to compile.

import std/[math, options, os]
import rew
import rew/data/[dataset, sample]
import rew/optim/scheduler
import rew/pjrt/loader
import rew/train/[workbench, datapipe, context, optimizer, hooks, callback,
                  trainer]
import rew/train/callbacks/[checkpoint, earlystop, progress]

include suite_train_workbench
include suite_train_context
include suite_train_datapipe
include suite_train_optimizer
include suite_train_scheduler
include suite_train_callback
include suite_train_checkpoint
include suite_train_earlystop
include suite_train_jit
include suite_train_manual
include suite_train_trainer
