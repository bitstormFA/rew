# Rechenwerk

Rechenwerk is a deep learning framework context. Its language centers on explicit
value state, typed training steps, device placement, and compiler-backed tensor
execution.

## Language

**Runtime**:
The explicit execution context for a training run.
_Avoid_: Workbench

**Train State**:
The value that carries model state, optimizer state, step count, and randomness through training.
_Avoid_: Hidden trainer state, module state

**Typed Step**:
A training computation expressed over structured model and batch values.
_Avoid_: Flat tensor step, raw JIT step

**Step Result**:
The value returned by a typed step, containing the next state and observable metrics.
_Avoid_: Training output tuple

**Param**:
A trainable model leaf.
_Avoid_: Weight when trainability is the point

**Buffer**:
A non-trainable model leaf that still belongs to model or training state.
_Avoid_: State tensor, running tensor

**Bare Tensor Leaf**:
A structured tensor value that is carried through pytrees without being trainable model state.
_Avoid_: Implicit parameter

**Gradient Transform**:
A composable optimizer update rule.
_Avoid_: Optimizer kind, optimizer enum

**Dataset**:
A typed stream of user batch values.
_Avoid_: Loader when structure matters

**Dataset Pipeline**:
A composable data manipulation flow over datasets or samples.
_Avoid_: Data Pipe

**Data Splits**:
A grouping of train, validation, test, and prediction datasets.
_Avoid_: Dataset Pipeline

**Trainer**:
An orchestration layer over runtime, train state, data splits, typed steps, metrics, and callbacks.
_Avoid_: Training loop owner when discussing the framework concept

**User Surface**:
The everyday API for tensor, model, data, optimizer, checkpoint, and training code.
_Avoid_: Compiler Surface, Extension Surface

**Compiler Surface**:
The API for tracing, lowering, StableHLO/OpenXLA inspection, raw JIT handles, and generated-program tooling.
_Avoid_: User Surface

**Extension Surface**:
The API for adding primitive operations, VJP policies, VJP rules, and dispatch/eager implementation hooks.
_Avoid_: User Surface

## Relationships

- A **Runtime** provides execution context for **Typed Steps**.
- A **Train State** contains **Param** leaves and **Buffer** leaves.
- A **Bare Tensor Leaf** is not a **Param** unless it is explicitly wrapped.
- A **Typed Step** consumes a **Train State** and a **Dataset** batch.
- A **Typed Step** produces a **Step Result**.
- A **Gradient Transform** updates **Param** leaves while preserving **Buffer** leaves.
- A **Dataset Pipeline** transforms or prepares **Dataset** values.
- **Data Splits** group one or more **Dataset** values for a **Trainer**.
- A **Trainer** repeatedly applies **Typed Steps** to **Data Splits**.
- The **User Surface** may use the **Compiler Surface** internally, but does not expose raw compiler handles as part of the high-level vocabulary.
- The **Extension Surface** supports framework authors extending primitive behavior; application code should normally stay on the **User Surface**.

## Example dialogue

> **Dev:** "Should this example build a raw JIT function over a flat list of tensors?"
> **Domain expert:** "No. Use a **Typed Step** over a **Train State** and batch value; the **Runtime** owns execution policy."

## Flagged ambiguities

- "Workbench" previously named the public execution context; resolved: use **Runtime**.
- "DataPipe" was previously used for grouped datasets; resolved language distinguishes **Dataset Pipeline** for data manipulation from **Data Splits** for train/validation/test/predict subsets.
- "OptimizerKind" previously described optimizer choice; resolved: use **Gradient Transform**.
- "Public API" previously meant a single umbrella surface; resolved language distinguishes **User Surface**, **Compiler Surface**, and **Extension Surface**.
