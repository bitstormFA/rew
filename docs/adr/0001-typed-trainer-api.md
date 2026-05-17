# Typed Trainer API

REW's high-level Trainer uses `TrainState`, `DataSplits`, typed losses, and typed custom steps rather than Task objects with manual/automatic optimization hooks. Raw `JitFunction` plumbing remains available through the compiler tier, but the user-facing training API keeps model state, optimizer state, metrics, callbacks, and checkpoints in one typed value-state language; `CallCtx` is policy-only so dynamic step and PRNG state stay on `TrainState` instead of being frozen into compiled losses.
