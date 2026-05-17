# Rechenwerk

Rechenwerk is a deep learning framework context. Its language centers on
explicit value state, typed training steps, device placement, and
compiler-backed tensor execution.

## Language

### Public Surfaces

**User Surface**:
The everyday API for tensor, model, data, optimizer, checkpoint, and training code.
_Avoid_: Compiler Surface, Extension Surface

**Compiler Surface**:
The API for tracing, lowering, StableHLO/OpenXLA inspection, raw JIT handles, and generated-program tooling.
_Avoid_: User Surface

**Extension Surface**:
The API for adding primitive behavior, autodiff policy, dispatch behavior, and runtime integration.
_Avoid_: User Surface

**Specialist Import**:
An explicit low-level import used by framework or integration code rather than ordinary model code.
_Avoid_: User Surface

### Tensor Execution

**Tensor**:
A typed, shaped value located on a specific device.
_Avoid_: ndarray, array, generic tensor

**DType**:
The element type carried by a tensor.
_Avoid_: Nim type when discussing tensor elements

**Shape**:
The runtime dimensions of a tensor.
_Avoid_: Static shape

**Device**:
A concrete execution location for tensor buffers.
_Avoid_: Target, backend

**Target**:
A closed accelerator family used to select a runtime plugin.
_Avoid_: Backend string, device string

**PJRT Plugin**:
The runtime backend that executes compiled tensor programs for a target.
_Avoid_: Driver, backend binary

**Plugin Manifest**:
The pinned catalog used to resolve and verify PJRT plugins.
_Avoid_: Download list, package index

**Device Transfer**:
An explicit move of tensor data to another device.
_Avoid_: Implicit copy, automatic placement

**Host Observation**:
An explicit read of device tensor data by host code.
_Avoid_: Implicit host transfer

**Buffer Donation**:
Permission for a compiled step to consume an input buffer and return replacement state.
_Avoid_: In-place mutation

**Sharding**:
The placement intent for partitioning a tensor across a device mesh.
_Avoid_: Device transfer

**Mesh**:
A named logical arrangement of devices for distributed placement.
_Avoid_: Device list

### Compiler And Autodiff

**Primitive Operation**:
A tensor operation with a lowering and an autodiff policy.
_Avoid_: Kernel when discussing framework semantics

**Composite Operation**:
A higher-level tensor operation expressed in terms of primitive operations.
_Avoid_: Primitive Operation

**Eager Execution**:
Operation-by-operation tensor execution through the same compiler path used by tracing.
_Avoid_: Interpreter, uncompiled mode

**Trace**:
A symbolic recording of tensor operations for compilation or transformation.
_Avoid_: Log, profile trace

**JIT**:
A compiled traced computation specialized to input tensor signatures.
_Avoid_: Macro JIT, source rewrite

**StableHLO Program**:
The compiler-level representation of a traced tensor computation.
_Avoid_: Runtime graph, Python graph

**Lowering**:
The translation from user-level tensor operations to compiler-level program form.
_Avoid_: Execution

**Control-Flow Combinator**:
An explicit in-graph control-flow construct.
_Avoid_: Host if/while inside JIT

**VJP Rule**:
The reverse-mode autodiff policy for a primitive operation.
_Avoid_: Gradient formula when registry behavior matters

**No-Grad Operation**:
A primitive operation that deliberately has no gradient.
_Avoid_: Missing gradient

**Gradient Tape**:
The eager-mode record used to replay gradients outside a compiled transform.
_Avoid_: JIT trace

### Value State And Models

**Value State**:
Plain values that carry framework state explicitly through calls.
_Avoid_: Hidden mutable state

**Model State**:
The structured value containing a model's trainable and non-trainable leaves.
_Avoid_: Module internals

**Functional Layer**:
A value object whose behavior is expressed by a forward call.
_Avoid_: Module class, stateful layer object

**Stateful Layer**:
A layer whose non-trainable state is visible in the model tree.
_Avoid_: Hidden layer state

**Pytree**:
A structured value that can be flattened into tensor leaves and rebuilt with the same shape.
_Avoid_: Flat tensor list when structure matters

**Tree Leaf**:
A tensor-like value carried by a pytree.
_Avoid_: Field when trainability or serialization matters

**Tree Path**:
The stable name of a leaf inside a pytree.
_Avoid_: Positional index

**Param**:
A trainable model leaf.
_Avoid_: Weight when trainability is the point

**Buffer**:
A non-trainable model leaf that still belongs to model or training state.
_Avoid_: State tensor, running tensor

**Bare Tensor Leaf**:
A structured tensor value that is carried through pytrees without being trainable model state.
_Avoid_: Implicit parameter

**Random Key**:
A value used to derive deterministic randomness explicitly.
_Avoid_: Global RNG, hidden seed

### Neural Network Domain

**Deep Model**:
A structured composition of layers that maps input batches to predictions or features.
_Avoid_: Network when state semantics matter

**Forward Pass**:
The application of a model or layer to input values to produce output values.
_Avoid_: Call when model semantics matter

**Representation**:
A learned feature value carried between layers or compared across modalities.
_Avoid_: Raw tensor

**Activation**:
A nonlinear transformation applied to features.
_Avoid_: Layer when no state is carried

**Logits**:
Unnormalized prediction scores produced before probability normalization.
_Avoid_: Probabilities

**Prediction Head**:
The task-facing projection that turns model features into logits or values.
_Avoid_: Attention Head

**Loss**:
A scalar training objective derived from predictions and targets.
_Avoid_: Metric

**Initialization**:
The deterministic construction of model state from random keys and static shape choices.
_Avoid_: Hidden random construction

**Linear Layer**:
A trainable affine projection over feature dimensions.
_Avoid_: Dense when project terminology matters

**Projection**:
A linear map that moves representations into another feature space.
_Avoid_: Device placement

**Bilinear Layer**:
A trainable interaction between two input feature values.
_Avoid_: Concatenation

**Convolutional Layer**:
A trainable local feature extractor over spatial dimensions.
_Avoid_: Filter when discussing model state

**Feature Map**:
A spatial tensor whose channels carry learned features at each location.
_Avoid_: Image

**Channel Axis**:
The tensor axis that indexes learned per-location features.
_Avoid_: Attention Head

**Convolution Kernel**:
The learned local window applied by a convolutional layer.
_Avoid_: PJRT kernel

**Stride**:
The step size used when a local operation moves across positions.
_Avoid_: Training step

**Dilation**:
The spacing between sampled positions inside a convolution kernel.
_Avoid_: Sharding

**Padding**:
Boundary values added before a spatial operation.
_Avoid_: Batch padding when sequence length is the topic

**Depthwise Convolution**:
A convolution that applies separate kernels to each input channel.
_Avoid_: Grouped Query Attention

**Pointwise Convolution**:
A one-by-one convolution that mixes channel features at each spatial location.
_Avoid_: Linear Layer when spatial layout is still present

**Separable Convolution**:
A convolutional block that combines depthwise and pointwise convolution.
_Avoid_: Plain Convolutional Layer

**Embedding Layer**:
A trainable lookup from discrete identifiers to feature vectors.
_Avoid_: Tokenizer

**Normalization Layer**:
A layer that rescales activations using trainable leaves, non-trainable leaves, or both.
_Avoid_: Activation

**Batch Normalization**:
A normalization layer that uses batch statistics and running-statistic buffers.
_Avoid_: Layer Normalization

**Layer Normalization**:
A normalization layer over each example's feature axes.
_Avoid_: Batch Normalization

**RMS Normalization**:
A normalization layer that rescales features by their root mean square.
_Avoid_: Layer Normalization when mean-centering matters

**Group Normalization**:
A normalization layer over channel groups within each example.
_Avoid_: Graph Pooling

**Instance Normalization**:
A normalization layer over each sample's spatial axes.
_Avoid_: Batch Normalization

**Spectral Normalization**:
A constraint that rescales a layer weight by its estimated spectral norm.
_Avoid_: Activation normalization

**Dropout Layer**:
A stochastic regularization layer controlled by an explicit random key.
_Avoid_: Implicit noise

**Residual Path**:
An additive path that carries features across a block.
_Avoid_: Hidden state

**Pooling Operation**:
A reduction that summarizes features across spatial, sequence, or graph axes.
_Avoid_: Collation

**Upsampling Operation**:
A transformation that increases spatial resolution.
_Avoid_: Device transfer

**Pixel Shuffle**:
A rearrangement that converts channel features into higher spatial resolution.
_Avoid_: Pooling Operation

**Sequence Model**:
A model whose primary axis represents ordered positions.
_Avoid_: Batch model

**Token**:
A discrete sequence item represented by an integer identifier.
_Avoid_: Embedding

**Vocabulary**:
The finite set of token identifiers a language model can embed or predict.
_Avoid_: Dataset

**Token Embedding**:
An embedding layer used specifically for vocabulary tokens.
_Avoid_: Positional Encoding

**Recurrent State**:
The hidden or cell value carried across sequence positions.
_Avoid_: Buffer when temporal semantics matter

**Recurrent Cell**:
A sequence unit that consumes one position and returns output plus next recurrent state.
_Avoid_: Feed Forward Block

**Gated Recurrent Cell**:
A recurrent cell that controls state flow with learned gates.
_Avoid_: Gated Feed Forward Block

**Cell State**:
The long-lived state value carried by an LSTM-style recurrent cell.
_Avoid_: Optimizer State

**Bidirectional Sequence Model**:
A sequence model that combines forward and reverse recurrent passes.
_Avoid_: Causal Language Model

**Attention**:
Content-based mixing of values using query and key similarity.
_Avoid_: Pooling

**Attention Projection**:
A projection that creates attention queries, keys, values, or output features.
_Avoid_: Prediction Head

**Attention Head**:
An independent attention subspace inside an attention layer.
_Avoid_: Device head

**Attention Mask**:
A rule or tensor that limits which positions may attend to which positions.
_Avoid_: Dropout mask

**Causal Mask**:
An attention mask that prevents a position from reading future positions.
_Avoid_: No-Grad Operation

**Full Attention**:
Attention where every permitted position can attend across the available context.
_Avoid_: Sliding Window Attention

**Sliding Window Attention**:
Attention restricted to a bounded neighborhood of recent positions.
_Avoid_: Full Attention

**Multi-Query Attention**:
Grouped query attention with one shared key/value head.
_Avoid_: Grouped Query Attention when more than one key/value head exists

**Grouped Query Attention**:
Attention where many query heads share fewer key/value heads.
_Avoid_: Multi-head attention when key/value head count differs

**Feed Forward Block**:
A position-wise MLP block used inside sequence architectures.
_Avoid_: Optimizer step

**Gated Feed Forward Block**:
A feed-forward block whose candidate features are modulated by a learned gate.
_Avoid_: Gated Recurrent Cell

**Transformer Block**:
A sequence block combining attention, feed-forward work, normalization, and residual paths.
_Avoid_: Generic layer stack

**Positional Encoding**:
A signal that lets a model distinguish positions.
_Avoid_: Token embedding

**Rotary Position Encoding**:
A positional encoding applied by rotating query and key feature pairs.
_Avoid_: Learned position embedding

**KV Cache**:
Decoder state that stores previously computed attention keys and values.
_Avoid_: Compile cache

**Causal Language Model**:
A sequence model that predicts next-token logits from earlier tokens.
_Avoid_: Classifier

**Vision Patch**:
A fixed-size image region represented as one sequence position.
_Avoid_: Pixel

**Class Token**:
A learned sequence position used to collect image-level information.
_Avoid_: Label

**Vision Transformer**:
A transformer model over image patches and a class token.
_Avoid_: Convolutional Layer

**Vision Tower**:
The image encoder portion of a multimodal model.
_Avoid_: Prediction Head

**Text Tower**:
The text encoder portion of a multimodal model.
_Avoid_: Tokenizer

**Projection Space**:
The shared representation space where modality embeddings are compared.
_Avoid_: Mesh

**Contrastive Pair**:
Two examples whose embeddings are compared as matched or unmatched views.
_Avoid_: Input batch

**Contrastive Logits**:
Similarity scores between embeddings in a shared projection space.
_Avoid_: Classification logits

**Adapter**:
A small trainable component attached to a larger model.
_Avoid_: Full fine-tuning

**Frozen Base**:
The non-trainable portion of a model that an adapter augments.
_Avoid_: Buffer when role is architectural

**LoRA Adapter**:
A low-rank trainable update attached to a linear projection.
_Avoid_: Weight replacement

**Adapter Rank**:
The low-rank dimension that controls LoRA adapter capacity.
_Avoid_: Tensor rank

**Adapter Scale**:
The scalar factor applied to a LoRA adapter update.
_Avoid_: Learning rate

**QLoRA Layer**:
A LoRA-adapted layer whose base projection is stored in quantized form.
_Avoid_: Quantized adapter

**Quantized Weight**:
A lower-precision representation of a weight tensor.
_Avoid_: Param when it is frozen model state

**NF4 Weight**:
A quantized weight stored in normal-float-four form.
_Avoid_: LoRA Adapter

**Quantization Scale**:
A per-group value used to reconstruct quantized weights.
_Avoid_: Loss scale

**Graph Data**:
Feature values together with connectivity between entities.
_Avoid_: Dataset graph

**Node Feature**:
A feature value attached to a graph node.
_Avoid_: Sequence position

**Edge Feature**:
A feature value attached to a graph edge.
_Avoid_: Edge Index

**Edge Index**:
A tensor representation of directed graph connectivity.
_Avoid_: Dense adjacency when sparse connectivity matters

**Self Loop**:
An edge from a node to itself used in graph message passing.
_Avoid_: Residual Path

**Message Passing**:
Feature updates produced by aggregating information along graph edges.
_Avoid_: Tensor transfer

**Graph Convolution**:
A message-passing layer that mixes neighbor features through learned projections.
_Avoid_: Spatial convolution

**Graph Attention**:
Message passing weighted by learned attention over neighboring nodes.
_Avoid_: Attention over token positions

**Graph Pooling**:
Aggregation from node-level features to graph-level features.
_Avoid_: Spatial pooling

**Graph Batch Index**:
A node-to-graph assignment used when multiple graphs share tensors.
_Avoid_: Batch

**Point Cloud**:
A set of points with coordinates and optional features.
_Avoid_: Image grid

**Neighborhood Graph**:
A graph connecting nearby points for local point-cloud computation.
_Avoid_: Data split

**Local Frame**:
A coordinate basis attached to a point or neighborhood.
_Avoid_: Device mesh

**Pair Distance**:
A geometric distance between points used for neighborhoods or equivariant features.
_Avoid_: Loss

**Equivariant Feature**:
A feature whose transformation law matters under spatial transformations.
_Avoid_: Plain feature vector

**Spherical Harmonic Feature**:
An angular basis feature used by equivariant point-cloud layers.
_Avoid_: Positional Encoding

**Tensor Product Feature Mix**:
An equivariant combination of features through representation-aware products.
_Avoid_: Matrix multiplication

**Equivariant Convolution**:
A point-cloud convolution that preserves spatial transformation behavior.
_Avoid_: Convolutional Layer when grid layout is assumed

**Diffusion Model**:
A generative model trained around progressive noising and denoising.
_Avoid_: Diffusion scheduler

**Forward Diffusion**:
The process that corrupts clean data with scheduled noise.
_Avoid_: Forward Pass

**Reverse Denoising**:
The process that iteratively removes noise to produce samples.
_Avoid_: Training step

**Noise Schedule**:
The sequence of noise levels used by a diffusion process.
_Avoid_: Learning rate schedule

**Timestep Embedding**:
A feature representation of a diffusion step.
_Avoid_: Positional encoding when denoising semantics matter

**Noise Prediction Target**:
A diffusion training target where the model predicts added noise.
_Avoid_: Logits

**Clean Sample Target**:
A diffusion training target where the model predicts the original clean sample.
_Avoid_: Reconstruction metric

**Diffusion Sampler**:
An inference loop that applies reverse denoising steps.
_Avoid_: Dataset sampler

**DDPM Sampler**:
A stochastic diffusion sampler following the DDPM reverse process.
_Avoid_: DDIM Sampler

**DDIM Sampler**:
A deterministic or reduced-step diffusion sampler derived from DDIM.
_Avoid_: DDPM Sampler

**U-Net**:
An encoder-decoder model with skip paths for dense prediction.
_Avoid_: Transformer

### Training

**Runtime**:
The explicit execution context for a training run.
_Avoid_: Workbench

**Train State**:
The value that carries model state, optimizer state, step count, and randomness through training.
_Avoid_: Hidden trainer state, module state

**Call Context**:
The policy context passed to typed losses and typed custom steps.
_Avoid_: Step state, randomness source

**Typed Loss**:
A loss function expressed over a structured model value and structured batch value.
_Avoid_: Flat tensor loss

**Typed Step**:
A training computation expressed over structured model, state, and batch values.
_Avoid_: Flat tensor step, raw JIT step

**Compiled Train Step**:
A cached compiled form of a typed training step.
_Avoid_: Exposed JIT handle

**Step Result**:
The value returned by a typed step, containing the next state and observable metrics.
_Avoid_: Training output tuple

**Gradient Transform**:
A composable optimizer update rule.
_Avoid_: Optimizer kind, optimizer enum

**Optimizer State**:
The value carried by a gradient transform between training steps.
_Avoid_: Optimizer object state

**Frozen Leaf**:
A model leaf intentionally excluded from gradient updates.
_Avoid_: Mutated requires-grad flag

**Metric**:
A named scalar observation emitted from training or validation.
_Avoid_: Log line

**Callback**:
A context-centric observer of training lifecycle and metrics.
_Avoid_: Task hook

**Checkpoint**:
A saved value state tree that can be restored by named paths.
_Avoid_: Model file

**Trainer**:
An orchestration layer over runtime, train state, data splits, typed steps, metrics, checkpoints, and callbacks.
_Avoid_: Training loop owner when discussing the framework concept

### Data

**Dataset**:
A typed stream of user batch values.
_Avoid_: Loader when structure matters

**Dataset Pipeline**:
A composable data manipulation flow over datasets or samples.
_Avoid_: Data Pipe

**Data Splits**:
A grouping of train, validation, test, and prediction datasets.
_Avoid_: Dataset Pipeline

**Sample**:
One host-side data item before batching.
_Avoid_: Batch

**Batch**:
A structured user value consumed by a typed loss or typed step.
_Avoid_: Flat tensor arguments

**Collation**:
The transformation from samples into a batch value.
_Avoid_: Device transfer

## Relationships

- A **User Surface** may call into the **Compiler Surface**, but it should not expose raw compiler handles.
- An **Extension Surface** changes primitive behavior; ordinary model code stays on the **User Surface**.
- A **Target** selects a **PJRT Plugin**; a **Device** is a concrete execution location within that target.
- A **Plugin Manifest** resolves PJRT plugins for one or more **Targets**.
- A **Tensor** has one **DType**, one **Shape**, one **Device**, and optional **Sharding**.
- A **Device Transfer** changes a tensor's **Device**; a **Host Observation** reads tensor data out of device execution.
- **Buffer Donation** consumes old tensor buffers so a **Compiled Train Step** can return replacement **Value State**.
- **Eager Execution** and **JIT** both rely on **Lowering** into a **StableHLO Program**.
- A **Trace** records **Primitive Operations**, **Composite Operations**, and **Control-Flow Combinators**.
- Every **Primitive Operation** has either a **VJP Rule** or is a **No-Grad Operation**.
- A **Gradient Tape** records eager tensor work; a **Trace** records compiler-oriented tensor work.
- A **Pytree** contains **Tree Leaves** named by **Tree Paths**.
- A **Model State** contains **Param** leaves, **Buffer** leaves, and possibly **Bare Tensor Leaf** values.
- A **Bare Tensor Leaf** is not a **Param** unless it is explicitly wrapped.
- A **Functional Layer** contributes **Model State** without hiding mutable state.
- A **Stateful Layer** stores its non-trainable state as **Buffer** leaves.
- A **Deep Model** is composed from **Functional Layers**, **Stateful Layers**, and plain tensor operations.
- A **Forward Pass** produces **Representation** values, **Logits**, or other predictions, often by applying **Activation** functions.
- A **Prediction Head** consumes a **Representation** and produces task-facing **Logits** or values.
- A **Loss** is optimized during training; a **Metric** is observed during or after training.
- **Initialization** uses **Random Keys** rather than hidden global state.
- A **Projection** appears inside a **Linear Layer**, **Attention Projection**, **Prediction Head**, or **LoRA Adapter**.
- A **Linear Layer**, **Convolutional Layer**, **Embedding Layer**, or **Normalization Layer** may contribute **Param** leaves, **Buffer** leaves, or both.
- A **Bilinear Layer** models feature interactions that a simple concatenation would leave to later layers.
- A **Convolutional Layer** applies a **Convolution Kernel** over **Feature Map** values along a **Channel Axis**.
- **Stride**, **Dilation**, and **Padding** define how a **Convolution Kernel** visits spatial positions.
- A **Separable Convolution** combines **Depthwise Convolution** and **Pointwise Convolution**.
- **Pixel Shuffle** is an **Upsampling Operation** that trades channel capacity for spatial resolution.
- **Batch Normalization** carries running-statistic **Buffer** leaves; **Layer Normalization**, **RMS Normalization**, **Group Normalization**, and **Instance Normalization** normalize within each example.
- **Spectral Normalization** constrains a layer's weight rather than normalizing its activations.
- A **Dropout Layer** consumes an explicit **Random Key** during training.
- A **Residual Path** preserves feature flow across a **Transformer Block**, **U-Net**, or other deep block.
- A **Token** belongs to a **Vocabulary** and is represented by a **Token Embedding**.
- A **Recurrent Cell** consumes one sequence position and returns a **Recurrent State**.
- A **Gated Recurrent Cell** updates **Recurrent State** through learned gates; an LSTM-style cell additionally carries **Cell State**.
- A **Bidirectional Sequence Model** is not a **Causal Language Model** because it reads both sequence directions.
- An **Attention Projection** creates the query, key, value, or output representation used by **Attention**.
- **Attention** uses **Attention Head** subspaces; **Grouped Query Attention** shares key/value heads across query heads.
- **Multi-Query Attention** is the one-key/value-head special case of **Grouped Query Attention**.
- **Attention Masks** restrict attention visibility; a **Causal Mask** is the next-token visibility rule for a **Causal Language Model**.
- **Full Attention** and **Sliding Window Attention** differ in how much context each position can see.
- A **Gated Feed Forward Block** is feed-forward gating, not recurrent state gating.
- A **KV Cache** stores attention keys and values for a **Causal Language Model**.
- **Positional Encoding** and **Rotary Position Encoding** give **Sequence Models** position information.
- A **Vision Patch** becomes a sequence position in a vision transformer; a **Class Token** represents image-level information.
- A **Vision Transformer** applies **Transformer Block** layers to **Vision Patch** positions.
- A **Vision Tower** and **Text Tower** produce representations in a shared **Projection Space**.
- A **Contrastive Pair** is scored as **Contrastive Logits** inside the **Projection Space**.
- An **Adapter** updates a **Frozen Base** without full fine-tuning.
- A **LoRA Adapter** augments a linear projection; **Adapter Rank** and **Adapter Scale** define its capacity and contribution.
- A **QLoRA Layer** combines a **LoRA Adapter** with a frozen **Quantized Weight** base.
- An **NF4 Weight** is a **Quantized Weight** reconstructed with one or more **Quantization Scale** values.
- **Graph Data** is processed by **Message Passing** over an **Edge Index**.
- **Graph Data** contains **Node Feature** values, may contain **Edge Feature** values, and may use **Self Loop** edges.
- A **Graph Convolution** is learned **Message Passing**; **Graph Attention** is attention-weighted **Message Passing**.
- **Graph Pooling** summarizes node features into graph-level features.
- A **Graph Batch Index** connects node-level features to graph-level pooling.
- A **Point Cloud** may be turned into a **Neighborhood Graph** with **Local Frames**.
- A **Neighborhood Graph** can be formed from **Pair Distance** values between **Point Cloud** points.
- **Equivariant Feature** values may use **Spherical Harmonic Feature** values and are combined through **Tensor Product Feature Mix** when spatial transformation behavior matters.
- An **Equivariant Convolution** applies that equivariant feature logic to point-cloud neighborhoods.
- A **Diffusion Model** uses a **Noise Schedule** and **Timestep Embedding** during denoising.
- **Forward Diffusion** produces noisy inputs; **Reverse Denoising** is performed by a **Diffusion Sampler**.
- **Noise Prediction Target** and **Clean Sample Target** are diffusion **Loss** targets, not **Logits**.
- **DDPM Sampler** and **DDIM Sampler** are forms of **Diffusion Sampler**.
- A **Train State** contains **Model State**, **Optimizer State**, a **Gradient Transform**, a step count, and a **Random Key**.
- A **Typed Loss** consumes **Model State**, a **Batch**, and a **Call Context**.
- A **Typed Step** consumes a **Train State** and a **Batch**.
- A **Typed Step** produces a **Step Result**.
- A **Compiled Train Step** is the compiled form of a **Typed Step**.
- A **Gradient Transform** updates **Param** leaves while preserving **Buffer** leaves.
- A **Frozen Leaf** remains in **Model State** while being excluded from updates.
- **Metrics** are observations of **Step Results** and validation work.
- A **Checkpoint** saves **Train State** or related **Value State** by **Tree Path**.
- A **Callback** observes **Metrics** and lifecycle context without owning **Model State**.
- A **Dataset Pipeline** transforms **Samples** or **Batches** into **Dataset** values.
- **Collation** turns **Samples** into a **Batch**.
- **Data Splits** group one or more **Dataset** values for a **Trainer**.
- A **Trainer** repeatedly applies **Typed Steps** to **Data Splits**.
- **Random Keys** are carried by **Train State** or caller-owned values, not by hidden global state.

## Example dialogue

> **Dev:** "Should this example pass a flat list of tensors into a raw JIT function and mutate the layer fields afterward?"
> **Domain expert:** "No. Put the model in **Train State**, mark trainable leaves as **Param**, keep non-trainable leaves as **Buffer**, and run a **Typed Step** through a **Compiled Train Step** or **Trainer**."

> **Dev:** "Can this dataset object also mean the train/validation split?"
> **Domain expert:** "No. A **Dataset Pipeline** manipulates data; **Data Splits** group datasets by training role."

> **Dev:** "Can the loss read the step number and RNG from **Call Context**?"
> **Domain expert:** "No. **Call Context** is policy context; dynamic step and randomness belong to **Train State** or explicit **Random Key** values."

> **Dev:** "Is this attention head the same kind of head as a device head?"
> **Domain expert:** "No. An **Attention Head** is a feature subspace inside **Attention**; a **Device** belongs to tensor execution."

> **Dev:** "Should this LoRA field be a normal trainable weight on the base model?"
> **Domain expert:** "No. The **Frozen Base** is non-trainable model state, while the **LoRA Adapter** carries the trainable update."

> **Dev:** "Is this graph batch just the same thing as the training batch?"
> **Domain expert:** "No. A **Graph Batch Index** assigns nodes to graphs inside **Graph Data**; the training **Batch** is the user value consumed by the step."

> **Dev:** "Can the diffusion sampler be described as the dataset sampler?"
> **Domain expert:** "No. A **Diffusion Sampler** performs **Reverse Denoising**; a **Dataset Pipeline** provides data."

> **Dev:** "Can I call the classifier output an attention head?"
> **Domain expert:** "No. A **Prediction Head** produces task-facing outputs, while an **Attention Head** is a subspace inside **Attention**."

## Flagged Ambiguities

- "Workbench" previously named the public execution context; resolved: use **Runtime**.
- "DataPipe" was previously used for grouped datasets; resolved language distinguishes **Dataset Pipeline** for data manipulation from **Data Splits** for train/validation/test/predict subsets.
- "OptimizerKind" previously described optimizer choice; resolved: use **Gradient Transform**.
- "Public API" previously meant a single umbrella surface; resolved language distinguishes **User Surface**, **Compiler Surface**, and **Extension Surface**.
- "Weight" can mean any learned tensor or specifically a trainable leaf; resolved: use **Param** when trainability is the point.
- "Buffer" can mean runtime storage or model state; resolved: use **Buffer** for non-trainable model leaves and **Buffer Donation** for compiled-step storage reuse.
- "Backend" can mean target family, plugin, or device; resolved: use **Target**, **PJRT Plugin**, or **Device**.
- "Trace" can mean compiler recording or eager gradient recording; resolved: use **Trace** for compiler recording and **Gradient Tape** for eager autodiff recording.
- "Head" can mean attention subspace, task projection, or device ordinal; resolved: use **Attention Head**, **Prediction Head**, or **Device**.
- "Projection" can mean learned feature mapping or device placement; resolved: use **Projection** for representation mapping and **Device Transfer** for placement.
- "Kernel" can mean convolutional window, compiler kernel, or runtime backend work; resolved: use **Convolution Kernel**, **Primitive Operation**, or **PJRT Plugin**.
- "Channel" can mean spatial feature axis or communication path; resolved: use **Channel Axis** for neural network tensors.
- "Normalization" can mean activation normalization, weight normalization, or graph aggregation; resolved: use **Batch Normalization**, **Layer Normalization**, **RMS Normalization**, **Group Normalization**, **Instance Normalization**, **Spectral Normalization**, or **Graph Pooling**.
- "Embedding" can mean token lookup, positional signal, timestep signal, or shared multimodal representation; resolved: use **Embedding Layer**, **Token Embedding**, **Positional Encoding**, **Timestep Embedding**, or **Projection Space**.
- "Mask" can mean attention visibility, dropout randomness, or trainability; resolved: use **Attention Mask**, **Dropout Layer**, or **Frozen Leaf**.
- "Cache" can mean compiled executable cache or attention state; resolved: use **Compiled Train Step**/**JIT** cache language for compilation and **KV Cache** for decoder attention state.
- "Schedule" can mean optimizer learning-rate policy or diffusion noise policy; resolved: use **Gradient Transform** schedule language for optimization and **Noise Schedule** for diffusion.
- "Sampler" can mean data iteration or diffusion generation; resolved: use **Dataset Pipeline** for data and **Diffusion Sampler** for generation.
- "Adapter" can mean any small attachment or specifically LoRA; resolved: use **Adapter** broadly and **LoRA Adapter** when the low-rank update is meant.
- "Rank" can mean tensor dimensionality or LoRA capacity; resolved: use tensor rank in tensor-shape discussions and **Adapter Rank** in LoRA discussions.
- "Scale" can mean adapter contribution, quantization reconstruction, loss scaling, or learning-rate policy; resolved: use **Adapter Scale**, **Quantization Scale**, or **Gradient Transform** schedule language.
- "Batch" can mean training batch, batch-normalization statistics, or graph membership; resolved: use **Batch**, **Batch Normalization**, or **Graph Batch Index**.
- "Quantized" can describe executable lowering, tensors, or model weights; resolved: use **Quantized Weight** for lower-precision model state.
