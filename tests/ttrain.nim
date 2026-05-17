## Consolidated training API suite.
##
## The included suite fragments are not standalone test binaries; this file is
## the single `t*.nim` entry point that keeps training coverage fast to compile.

import std/[math, options, os]
import rew
import rew/xla
import rew/data/[dataset, sample]
import rew/optim/scheduler
import rew/pjrt/loader
import rew/train/[runtime, datasplits, context, callback, trainer]
import rew/train/callbacks/[checkpoint, earlystop]

include suite_train_runtime
include suite_train_context
include suite_train_datasplits
include suite_train_optimizer
include suite_train_scheduler
include suite_train_callback
include suite_train_checkpoint
include suite_train_earlystop
include suite_train_jit
include suite_train_trainer
