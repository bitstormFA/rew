# ExtensionPlan1: DuckDB DataFrames + Statistics for rew

## Vision

Expand rew from tensor-first ML and training into structured analytics, classical statistics, and Bayesian modeling. Implement a DuckDB-backed DataFrame layer with a composable expression DSL, lazy relational transforms, and seamless interoperability with rew's existing `Dataset` and pytree training pipelines.

## Guiding principles

- Keep the primary API expression-based, not raw SQL.
- Use `col(...)`, `lit(...)`, `when(...).then(...).else_(...)`, `groupBy`, `summarise`, `select`, `mutate`, `filter`, `arrange`, `limit`, and `collect`.
- Preserve `DataFrame -> Dataset[Batch] -> Tensor -> model -> loss` as the canonical training bridge.
- Support large, out-of-memory sources through DuckDB pushdown and CPU fallback when device compilation is impossible.
- Expose SQL as an escape hatch for advanced use cases.
- Build classical and Bayesian statistics on top of tensor computation and rew's value-state model.

## Phase 0: Alignment and scaffolding

1. Document the target API shape in `docs/high-level-api.md` or a new design note.
2. Add a top-level `rew/dataframe` or `rew/df` module umbrella.
3. Define the public DataFrame types and expression DSL in Nim.
4. Keep core rew invariants intact: no raw SQL as primary API, no implicit device moves, no global mutable state in public API.

## Phase 1: Core DataFrame expression engine

1. Create `DataFrame` as a lazy relational pipeline value type.
2. Define `Expr`/`ColumnExpr` and `LiteralExpr` AST nodes.
3. Implement `col`, `lit`, `sum`, `mean`, `count`, `min`, `max`, `alias`, `asc`, `desc`, `when`, string predicates, and date-part helpers.
4. Add chainable verbs: `select`, `filter`, `mutate`, `groupBy`, `summarise`, `arrange`, `limit`, `join`, `collect`.
5. Implement backend-neutral logical IR that can emit DuckDB SQL.
6. Add a SQL escape hatch like `df.sql(...)` or `df.toSql()` for debugging and custom queries.

## Phase 2: DuckDB backend integration

1. Add DuckDB as a new dependency and Nim cimport layer.
2. Implement a DuckDB table/source adapter for Parquet and CSV files.
3. Compile expression DAGs to DuckDB SQL with proper column aliasing, type coercion, and predicate pushdown.
4. Enable `collect(device = someDevice)` to materialize results as tensors and support optional runtime coercion.
5. Implement CPU fallback: if an expression cannot compile to device, run DuckDB on CPU and then convert results to tensors.
6. Support automatic column-to-field mapping for explicit Nim batch types.

## Phase 3: Training bridge and pytree support

1. Add `DataFrame.collectAs[Batch]` and `DataFrame.toDataset[Batch]` helpers.
2. Use automatic schema mapping from DataFrame columns to batch object fields.
3. Validate field names and dtypes at runtime, with optional coercion settings.
4. Ensure compatibility with `rew/train/Trainer`, `TrainState`, and `treeMap`.
5. Add examples that train a model directly from a DuckDB-backed DataFrame source.

## Phase 4: Classical statistics and ML primitives

1. Build a lightweight stats module for `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`, `StandardScaler`, `PCA`, and `train_test_split`.
2. Implement estimator fit/predict APIs that operate on `DataFrame` or `Dataset[Batch]`.
3. Provide evaluation helpers such as `accuracy`, `rocAuc`, `meanSquaredError`, and `crossValScore`.
4. Expose model parameter state as explicit Nim objects compatible with rew's `TrainState`/optimizer semantics where useful.

## Phase 5: Bayesian statistics and probabilistic modeling

1. Add a probabilistic modeling API inspired by PyMC, with explicit model specification and priors.
2. Implement a `MCMC` runner and NUTS sampler using rew tensor operations for log-probability, gradients, and transitions.
3. Provide host-side or device-supported sampling workflows, with DuckDB-based data ingestion for observed data.
4. Support posterior summarization, trace diagnostics, and predictive sampling.

## Phase 6: polish, docs, and testing

1. Add test coverage for expression translation, DuckDB SQL generation, and `collect` semantics.
2. Add train integration tests that show DataFrame-based datasets feeding `Trainer` and `computeGrads`.
3. Add numerical equivalence tests for eager vs. compiled transform paths.
4. Document the API in examples and `docs/high-level-api.md`.
5. Add lint rules or checks if needed for the new layer.

## Risk mitigation

- Keep the first release small: focus on query expressions, DuckDB SQL emission, and `collectAs[Batch]` before adding full estimator or NUTS support.
- Use DuckDB as the core relational engine; do not attempt to rewrite SQL internals in Nim.
- Default to CPU/DuckDB fallback for unsupported device-side operations.
- Treat SQL as an advanced escape hatch, not the user-facing DSL.

## Success criteria

- A user can write a chainable, expression-based DataFrame pipeline and collect it into a typed batch.
- A DataFrame source can feed a `Trainer` pipeline through `Dataset[Batch]` without ad hoc runtime plumbing.
- Classical model APIs exist for core regression/classification workflows.
- Bayesian sampling via NUTS is supported as a distinct probabilistic modeling path.
- The new layer preserves rew’s existing public API contract and does not leak raw JIT or PJRT internals.
