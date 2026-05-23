## Core DataFrame value types and expression DSL.
##
## This module defines the public, backend-neutral shape of rew DataFrames.
## It intentionally stops at lazy relational pipeline values: no DuckDB handle,
## no global connection, and no implicit tensor/device materialisation.

type
  DataFrameError* = object of CatchableError
    ## Raised when a DataFrame DSL constructor receives invalid input.

  DataFrameSourceKind* = enum
    dfSourceTable,
    dfSourceCsv,
    dfSourceParquet

  DataFrameSource* = object
    ## Describes a lazy host-side relational source.
    kind*: DataFrameSourceKind
    name*: string
    path*: string

  LiteralKind* = enum
    litNull,
    litBool,
    litInt,
    litFloat,
    litString

  LiteralValue* = object
    ## Scalar literal payload used by expression nodes.
    kind*: LiteralKind
    boolVal*: bool
    intVal*: int64
    floatVal*: float64
    stringVal*: string

  SortDirection* = enum
    sortAscending,
    sortDescending

  ExprKind* = enum
    exprColumn,
    exprLiteral,
    exprUnary,
    exprBinary,
    exprCall,
    exprAlias,
    exprSort,
    exprCase

  Expr* = object
    ## Backend-neutral DataFrame expression node.
    kind*: ExprKind
    name*: string
    literal*: LiteralValue
    args*: seq[Expr]
    direction*: SortDirection
    branches*: seq[CaseBranch]

  CaseBranch* = object
    ## One `when(condition).then(value)` branch in a case expression.
    condition*: Expr
    value*: Expr

  NamedExpr* = object
    ## A named expression used by `mutate` and `summarise`.
    name*: string
    expr*: Expr

  DataFrameStepKind* = enum
    dfStepSelect,
    dfStepFilter,
    dfStepMutate,
    dfStepGroupBy,
    dfStepSummarise,
    dfStepArrange,
    dfStepLimit

  DataFrameStep* = object
    ## One lazy relational transform in a DataFrame pipeline.
    kind*: DataFrameStepKind
    exprs*: seq[Expr]
    namedExprs*: seq[NamedExpr]
    limitCount*: int

  DataFrame* = object
    ## Lazy relational pipeline value.
    source*: DataFrameSource
    steps*: seq[DataFrameStep]

  GroupedDataFrame* = object
    ## Intermediate value returned by `groupBy`.
    parent*: DataFrame
    keys*: seq[Expr]

  WhenBuilder* = object
    condition: Expr

  WhenThenBuilder* = object
    condition: Expr
    value: Expr

proc requireNonEmpty(value, label: string) =
  if value.len == 0:
    raise newException(DataFrameError, label & " must not be empty")

proc requireAny[T](values: openArray[T]; label: string) =
  if values.len == 0:
    raise newException(DataFrameError, label & " must not be empty")

func exprCall(name: string; args: openArray[Expr]): Expr =
  Expr(kind: exprCall, name: name, args: @args)

func unaryExpr(name: string; arg: Expr): Expr =
  Expr(kind: exprUnary, name: name, args: @[arg])

func binaryExpr(name: string; left, right: Expr): Expr =
  Expr(kind: exprBinary, name: name, args: @[left, right])

func appendStep(df: DataFrame; step: DataFrameStep): DataFrame =
  result = df
  result.steps.add step

proc fromTable*(name: string): DataFrame =
  ## Creates a lazy DataFrame from a registered relational table.
  requireNonEmpty(name, "fromTable name")
  DataFrame(source: DataFrameSource(kind: dfSourceTable, name: name))

proc readCsv*(path: string): DataFrame =
  ## Creates a lazy DataFrame source for a CSV file path.
  requireNonEmpty(path, "readCsv path")
  DataFrame(source: DataFrameSource(kind: dfSourceCsv, path: path))

proc readParquet*(path: string): DataFrame =
  ## Creates a lazy DataFrame source for a Parquet file path.
  requireNonEmpty(path, "readParquet path")
  DataFrame(source: DataFrameSource(kind: dfSourceParquet, path: path))

proc col*(name: string): Expr =
  ## References a column by name.
  requireNonEmpty(name, "column name")
  Expr(kind: exprColumn, name: name)

func nullLit*(): Expr =
  ## Creates a null literal expression.
  Expr(kind: exprLiteral, literal: LiteralValue(kind: litNull))

func lit*(value: bool): Expr =
  ## Creates a boolean literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litBool, boolVal: value))

func lit*(value: int): Expr =
  ## Creates an integer literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litInt, intVal: int64(value)))

func lit*(value: int64): Expr =
  ## Creates an int64 literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litInt, intVal: value))

func lit*(value: float): Expr =
  ## Creates a float64 literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litFloat, floatVal: value))

func lit*(value: float32): Expr =
  ## Creates a float32 literal expression.
  lit(float(value))

func lit*(value: string): Expr =
  ## Creates a string literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litString, stringVal: value))

proc alias*(expr: Expr; name: string): Expr =
  ## Attaches an output alias to an expression.
  requireNonEmpty(name, "alias name")
  Expr(kind: exprAlias, name: name, args: @[expr])

proc named*(name: string; expr: Expr): NamedExpr =
  ## Names an expression for verbs that introduce or aggregate columns.
  requireNonEmpty(name, "named expression name")
  NamedExpr(name: name, expr: expr)

func asc*(expr: Expr): Expr =
  ## Marks an expression as ascending for `arrange`.
  Expr(kind: exprSort, args: @[expr], direction: sortAscending)

func desc*(expr: Expr): Expr =
  ## Marks an expression as descending for `arrange`.
  Expr(kind: exprSort, args: @[expr], direction: sortDescending)

func sum*(expr: Expr): Expr =
  ## Aggregate sum.
  exprCall("sum", [expr])

func mean*(expr: Expr): Expr =
  ## Aggregate mean.
  exprCall("mean", [expr])

func count*(): Expr =
  ## Aggregate row count.
  exprCall("count", [])

func count*(expr: Expr): Expr =
  ## Aggregate count over an expression.
  exprCall("count", [expr])

func min*(expr: Expr): Expr =
  ## Aggregate minimum.
  exprCall("min", [expr])

func max*(expr: Expr): Expr =
  ## Aggregate maximum.
  exprCall("max", [expr])

func contains*(expr, needle: Expr): Expr =
  ## String containment predicate.
  exprCall("contains", [expr, needle])

func startsWith*(expr, prefix: Expr): Expr =
  ## String prefix predicate.
  exprCall("startsWith", [expr, prefix])

func endsWith*(expr, suffix: Expr): Expr =
  ## String suffix predicate.
  exprCall("endsWith", [expr, suffix])

func year*(expr: Expr): Expr =
  ## Extracts a year part from a date-like expression.
  exprCall("year", [expr])

func month*(expr: Expr): Expr =
  ## Extracts a month part from a date-like expression.
  exprCall("month", [expr])

func day*(expr: Expr): Expr =
  ## Extracts a day part from a date-like expression.
  exprCall("day", [expr])

func hour*(expr: Expr): Expr =
  ## Extracts an hour part from a date-like expression.
  exprCall("hour", [expr])

func `+`*(left, right: Expr): Expr =
  binaryExpr("+", left, right)

func `-`*(left, right: Expr): Expr =
  binaryExpr("-", left, right)

func `*`*(left, right: Expr): Expr =
  binaryExpr("*", left, right)

func `/`*(left, right: Expr): Expr =
  binaryExpr("/", left, right)

func `==`*(left, right: Expr): Expr =
  binaryExpr("==", left, right)

func `!=`*(left, right: Expr): Expr =
  binaryExpr("!=", left, right)

func `<`*(left, right: Expr): Expr =
  binaryExpr("<", left, right)

func `<=`*(left, right: Expr): Expr =
  binaryExpr("<=", left, right)

func `>`*(left, right: Expr): Expr =
  binaryExpr(">", left, right)

func `>=`*(left, right: Expr): Expr =
  binaryExpr(">=", left, right)

func allOf*(left, right: Expr): Expr =
  ## Boolean conjunction for expression predicates.
  binaryExpr("and", left, right)

func anyOf*(left, right: Expr): Expr =
  ## Boolean disjunction for expression predicates.
  binaryExpr("or", left, right)

func notExpr*(expr: Expr): Expr =
  ## Boolean negation for expression predicates.
  unaryExpr("not", expr)

func `when`*(condition: Expr): WhenBuilder =
  ## Starts a case expression. Because `when` is a Nim keyword, callers may use
  ## either `` `when`(condition) `` or `caseWhen(condition)`.
  WhenBuilder(condition: condition)

func caseWhen*(condition: Expr): WhenBuilder =
  ## Starts a case expression without requiring keyword escaping.
  `when`(condition)

func then*(builder: WhenBuilder; value: Expr): WhenThenBuilder =
  ## Sets the value for a case expression condition.
  WhenThenBuilder(condition: builder.condition, value: value)

func thenValue*(builder: WhenBuilder; value: Expr): WhenThenBuilder =
  ## Alias for `then` for callers that prefer non-keyword-like names.
  then(builder, value)

func otherwise*(builder: WhenThenBuilder; fallback: Expr): Expr =
  ## Completes a case expression with a fallback value.
  Expr(kind: exprCase,
    branches: @[CaseBranch(condition: builder.condition, value: builder.value)],
    args: @[fallback])

proc select*(df: DataFrame; exprs: varargs[Expr]): DataFrame =
  ## Adds a projection step.
  requireAny(exprs, "select expressions")
  df.appendStep(DataFrameStep(kind: dfStepSelect, exprs: @exprs))

proc filter*(df: DataFrame; predicate: Expr): DataFrame =
  ## Adds a predicate step.
  df.appendStep(DataFrameStep(kind: dfStepFilter, exprs: @[predicate]))

proc mutate*(df: DataFrame; assignments: varargs[NamedExpr]): DataFrame =
  ## Adds derived columns to the lazy pipeline.
  requireAny(assignments, "mutate assignments")
  for assignment in assignments:
    requireNonEmpty(assignment.name, "mutate assignment name")
  df.appendStep(DataFrameStep(kind: dfStepMutate, namedExprs: @assignments))

proc groupBy*(df: DataFrame; keys: varargs[Expr]): GroupedDataFrame =
  ## Groups a lazy DataFrame by one or more key expressions.
  requireAny(keys, "groupBy keys")
  GroupedDataFrame(parent: df, keys: @keys)

proc summarise*(df: DataFrame; aggregations: varargs[NamedExpr]): DataFrame =
  ## Adds ungrouped aggregations to the lazy pipeline.
  requireAny(aggregations, "summarise aggregations")
  for aggregation in aggregations:
    requireNonEmpty(aggregation.name, "summarise aggregation name")
  df.appendStep(DataFrameStep(kind: dfStepSummarise,
    namedExprs: @aggregations))

proc summarise*(df: GroupedDataFrame;
    aggregations: varargs[NamedExpr]): DataFrame =
  ## Adds grouped aggregations to the lazy pipeline.
  requireAny(aggregations, "summarise aggregations")
  for aggregation in aggregations:
    requireNonEmpty(aggregation.name, "summarise aggregation name")
  result = df.parent.appendStep(DataFrameStep(kind: dfStepGroupBy,
    exprs: df.keys))
  result = result.appendStep(DataFrameStep(kind: dfStepSummarise,
    namedExprs: @aggregations))

proc arrange*(df: DataFrame; orderings: varargs[Expr]): DataFrame =
  ## Adds an ordering step. Bare expressions are treated as backend-default
  ## ascending order; use `asc` or `desc` to be explicit.
  requireAny(orderings, "arrange orderings")
  df.appendStep(DataFrameStep(kind: dfStepArrange, exprs: @orderings))

proc limit*(df: DataFrame; count: int): DataFrame =
  ## Adds a row limit step.
  if count < 0:
    raise newException(DataFrameError, "limit count must be non-negative")
  df.appendStep(DataFrameStep(kind: dfStepLimit, limitCount: count))
