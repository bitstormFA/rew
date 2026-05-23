## Core DataFrame value types and expression DSL.
##
## This module defines the public, backend-neutral shape of rew DataFrames.
## It intentionally stops at lazy relational pipeline values: no DuckDB handle,
## no global connection, and no implicit tensor/device materialisation.

from std/strutils import replace, split

type
  DataFrameError* = object of CatchableError
    ## Raised when a DataFrame DSL constructor receives invalid input.

  DataFrameSourceKind* = enum
    dfSourceTable,
    dfSourceCsv,
    dfSourceParquet,
    dfSourceSql

  DataFrameSource* = object
    ## Describes a lazy host-side relational source.
    kind*: DataFrameSourceKind
    name*: string
    path*: string
    query*: string

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

  JoinKind* = enum
    joinInner,
    joinLeft,
    joinRight,
    joinFull,
    joinCross

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

  ColumnExpr* = Expr
    ## Expression node that references a column.

  LiteralExpr* = Expr
    ## Expression node that holds a scalar literal.

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
    dfStepLimit,
    dfStepJoin

  DataFrameStep* = object
    ## One lazy relational transform in a DataFrame pipeline.
    kind*: DataFrameStepKind
    exprs*: seq[Expr]
    namedExprs*: seq[NamedExpr]
    limitCount*: int
    joinKind*: JoinKind
    joinSource*: DataFrameSource
    joinSteps*: seq[DataFrameStep]
    joinPredicate*: Expr
    hasJoinPredicate*: bool

  LogicalPlan* = object
    ## Backend-neutral relational IR for a lazy DataFrame.
    source*: DataFrameSource
    steps*: seq[DataFrameStep]

  DataFrame* = object
    ## Lazy relational pipeline value.
    source*: DataFrameSource
    steps*: seq[DataFrameStep]

  DataFrameCollection* = object
    ## Phase-1 collection boundary. It records the compiled query until a
    ## concrete backend is attached in a later phase.
    plan*: DataFrame
    sql*: string

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

func toLogicalPlan*(df: DataFrame): LogicalPlan =
  ## Returns the backend-neutral logical plan represented by `df`.
  LogicalPlan(source: df.source, steps: df.steps)

func initDataFrame*(plan: LogicalPlan): DataFrame =
  ## Reconstructs a lazy DataFrame value from a logical plan.
  DataFrame(source: plan.source, steps: plan.steps)

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

proc fromSql*(query: string): DataFrame =
  ## Creates a lazy DataFrame from a raw SQL query escape hatch.
  requireNonEmpty(query, "SQL query")
  DataFrame(source: DataFrameSource(kind: dfSourceSql, query: query))

proc sql*(query: string): DataFrame =
  ## Alias for `fromSql` for advanced callers that need a raw SQL source.
  fromSql(query)

proc col*(name: string): ColumnExpr =
  ## References a column by name.
  requireNonEmpty(name, "column name")
  Expr(kind: exprColumn, name: name)

func nullLit*(): LiteralExpr =
  ## Creates a null literal expression.
  Expr(kind: exprLiteral, literal: LiteralValue(kind: litNull))

func lit*(value: bool): LiteralExpr =
  ## Creates a boolean literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litBool, boolVal: value))

func lit*(value: int): LiteralExpr =
  ## Creates an integer literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litInt, intVal: int64(value)))

func lit*(value: int64): LiteralExpr =
  ## Creates an int64 literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litInt, intVal: value))

func lit*(value: float): LiteralExpr =
  ## Creates a float64 literal expression.
  Expr(kind: exprLiteral,
    literal: LiteralValue(kind: litFloat, floatVal: value))

func lit*(value: float32): LiteralExpr =
  ## Creates a float32 literal expression.
  lit(float(value))

func lit*(value: string): LiteralExpr =
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

func contains*(expr: Expr; needle: string): Expr =
  ## String containment predicate with a string literal needle.
  contains(expr, lit(needle))

func startsWith*(expr, prefix: Expr): Expr =
  ## String prefix predicate.
  exprCall("startsWith", [expr, prefix])

func startsWith*(expr: Expr; prefix: string): Expr =
  ## String prefix predicate with a string literal prefix.
  startsWith(expr, lit(prefix))

func endsWith*(expr, suffix: Expr): Expr =
  ## String suffix predicate.
  exprCall("endsWith", [expr, suffix])

func endsWith*(expr: Expr; suffix: string): Expr =
  ## String suffix predicate with a string literal suffix.
  endsWith(expr, lit(suffix))

func like*(expr, pattern: Expr): Expr =
  ## SQL LIKE string predicate.
  exprCall("like", [expr, pattern])

func like*(expr: Expr; pattern: string): Expr =
  ## SQL LIKE string predicate with a string literal pattern.
  like(expr, lit(pattern))

func matches*(expr, pattern: Expr): Expr =
  ## Regular-expression string predicate.
  exprCall("matches", [expr, pattern])

func matches*(expr: Expr; pattern: string): Expr =
  ## Regular-expression string predicate with a string literal pattern.
  matches(expr, lit(pattern))

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

func minute*(expr: Expr): Expr =
  ## Extracts a minute part from a date-like expression.
  exprCall("minute", [expr])

func second*(expr: Expr): Expr =
  ## Extracts a second part from a date-like expression.
  exprCall("second", [expr])

func dayOfWeek*(expr: Expr): Expr =
  ## Extracts a weekday number from a date-like expression.
  exprCall("dayOfWeek", [expr])

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

func elseValue*(builder: WhenThenBuilder; fallback: Expr): Expr =
  ## Completes a case expression with a fallback value.
  otherwise(builder, fallback)

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

proc join*(left, right: DataFrame; on: Expr;
    kind: JoinKind = joinInner): DataFrame =
  ## Adds a lazy relational join against another DataFrame.
  if kind == joinCross:
    raise newException(DataFrameError,
      "cross joins must use crossJoin without a predicate")
  left.appendStep(DataFrameStep(
    kind: dfStepJoin,
    joinKind: kind,
    joinSource: right.source,
    joinSteps: right.steps,
    joinPredicate: on,
    hasJoinPredicate: true))

proc crossJoin*(left, right: DataFrame): DataFrame =
  ## Adds a lazy cross join against another DataFrame.
  left.appendStep(DataFrameStep(
    kind: dfStepJoin,
    joinKind: joinCross,
    joinSource: right.source,
    joinSteps: right.steps))

proc joinStrings(parts: openArray[string]; separator: string): string =
  for i, part in parts:
    if i > 0:
      result.add separator
    result.add part

proc quoteSqlString(value: string): string =
  result = "'" & value.replace("'", "''") & "'"

proc quoteIdentifierPart(value: string): string =
  requireNonEmpty(value, "identifier part")
  result = "\"" & value.replace("\"", "\"\"") & "\""

proc quoteIdentifier(value: string): string =
  let parts = value.split('.')
  var quoted: seq[string]
  for part in parts:
    quoted.add quoteIdentifierPart(part)
  result = quoted.joinStrings(".")

proc literalSql(value: LiteralValue): string =
  case value.kind
  of litNull:
    result = "NULL"
  of litBool:
    if value.boolVal:
      result = "TRUE"
    else:
      result = "FALSE"
  of litInt:
    result = $value.intVal
  of litFloat:
    result = $value.floatVal
  of litString:
    result = quoteSqlString(value.stringVal)

proc binarySqlName(name: string): string =
  case name
  of "==":
    result = "="
  of "!=":
    result = "<>"
  of "and":
    result = "AND"
  of "or":
    result = "OR"
  else:
    result = name

proc datePartName(name: string): string =
  case name
  of "dayOfWeek":
    result = "dow"
  else:
    result = name

proc callSql(name: string; args: openArray[Expr]): string
proc exprSql(expr: Expr; allowAlias = true): string

proc callSql(name: string; args: openArray[Expr]): string =
  case name
  of "mean":
    requireAny(args, "mean arguments")
    result = "avg(" & exprSql(args[0], false) & ")"
  of "count":
    if args.len == 0:
      result = "count(*)"
    else:
      result = "count(" & exprSql(args[0], false) & ")"
  of "sum", "min", "max":
    requireAny(args, name & " arguments")
    result = name & "(" & exprSql(args[0], false) & ")"
  of "contains":
    requireAny(args, "contains arguments")
    if args.len != 2:
      raise newException(DataFrameError, "contains requires two arguments")
    result = "contains(" & exprSql(args[0], false) & ", " &
      exprSql(args[1], false) & ")"
  of "startsWith":
    requireAny(args, "startsWith arguments")
    if args.len != 2:
      raise newException(DataFrameError, "startsWith requires two arguments")
    result = "starts_with(" & exprSql(args[0], false) & ", " &
      exprSql(args[1], false) & ")"
  of "endsWith":
    requireAny(args, "endsWith arguments")
    if args.len != 2:
      raise newException(DataFrameError, "endsWith requires two arguments")
    result = "ends_with(" & exprSql(args[0], false) & ", " &
      exprSql(args[1], false) & ")"
  of "like":
    requireAny(args, "like arguments")
    if args.len != 2:
      raise newException(DataFrameError, "like requires two arguments")
    result = "(" & exprSql(args[0], false) & " LIKE " &
      exprSql(args[1], false) & ")"
  of "matches":
    requireAny(args, "matches arguments")
    if args.len != 2:
      raise newException(DataFrameError, "matches requires two arguments")
    result = "regexp_matches(" & exprSql(args[0], false) & ", " &
      exprSql(args[1], false) & ")"
  of "year", "month", "day", "hour", "minute", "second", "dayOfWeek":
    requireAny(args, name & " arguments")
    result = "date_part(" & quoteSqlString(datePartName(name)) & ", " &
      exprSql(args[0], false) & ")"
  else:
    var rendered: seq[string]
    for arg in args:
      rendered.add exprSql(arg, false)
    result = name & "(" & rendered.joinStrings(", ") & ")"

proc exprSql(expr: Expr; allowAlias = true): string =
  case expr.kind
  of exprColumn:
    result = quoteIdentifier(expr.name)
  of exprLiteral:
    result = literalSql(expr.literal)
  of exprUnary:
    if expr.name == "not":
      requireAny(expr.args, "not arguments")
      result = "(NOT " & exprSql(expr.args[0], false) & ")"
    else:
      requireAny(expr.args, expr.name & " arguments")
      result = expr.name & "(" & exprSql(expr.args[0], false) & ")"
  of exprBinary:
    if expr.args.len != 2:
      raise newException(DataFrameError,
        "binary expression requires two arguments")
    result = "(" & exprSql(expr.args[0], false) & " " &
      binarySqlName(expr.name) &
      " " & exprSql(expr.args[1], false) & ")"
  of exprCall:
    result = callSql(expr.name, expr.args)
  of exprAlias:
    requireAny(expr.args, "alias arguments")
    let rendered = exprSql(expr.args[0], false)
    if allowAlias:
      result = rendered & " AS " & quoteIdentifier(expr.name)
    else:
      result = rendered
  of exprSort:
    requireAny(expr.args, "sort arguments")
    let direction =
      case expr.direction
      of sortAscending: "ASC"
      of sortDescending: "DESC"
    result = exprSql(expr.args[0], false) & " " & direction
  of exprCase:
    if expr.args.len != 1:
      raise newException(DataFrameError,
        "case expression requires one fallback expression")
    requireAny(expr.branches, "case branches")
    result = "CASE"
    for branch in expr.branches:
      result.add " WHEN "
      result.add exprSql(branch.condition, false)
      result.add " THEN "
      result.add exprSql(branch.value, false)
    result.add " ELSE "
    result.add exprSql(expr.args[0], false)
    result.add " END"

proc sourceSql(source: DataFrameSource): string =
  case source.kind
  of dfSourceTable:
    result = quoteIdentifier(source.name)
  of dfSourceCsv:
    result = "read_csv_auto(" & quoteSqlString(source.path) & ")"
  of dfSourceParquet:
    result = "read_parquet(" & quoteSqlString(source.path) & ")"
  of dfSourceSql:
    result = "(" & source.query & ")"

proc joinKindSql(kind: JoinKind): string =
  case kind
  of joinInner:
    result = "INNER JOIN"
  of joinLeft:
    result = "LEFT JOIN"
  of joinRight:
    result = "RIGHT JOIN"
  of joinFull:
    result = "FULL JOIN"
  of joinCross:
    result = "CROSS JOIN"

proc nextAlias(counter: var int): string =
  result = "q" & $counter
  inc counter

proc namedExprSql(expr: NamedExpr): string =
  result = exprSql(expr.expr, false) & " AS " & quoteIdentifier(expr.name)

proc compileSql(source: DataFrameSource; steps: openArray[DataFrameStep];
    counter: var int): string =
  result = "SELECT * FROM " & sourceSql(source)
  var pendingGroupKeys: seq[Expr]

  for step in steps:
    case step.kind
    of dfStepSelect:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let alias = nextAlias(counter)
      var selected: seq[string]
      for expr in step.exprs:
        selected.add exprSql(expr)
      result = "SELECT " & selected.joinStrings(", ") & " FROM (" & result &
        ") AS " & alias
    of dfStepFilter:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let alias = nextAlias(counter)
      requireAny(step.exprs, "filter predicates")
      result = "SELECT * FROM (" & result & ") AS " & alias & " WHERE " &
        exprSql(step.exprs[0], false)
    of dfStepMutate:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let alias = nextAlias(counter)
      var assignments: seq[string]
      for assignment in step.namedExprs:
        assignments.add namedExprSql(assignment)
      result = "SELECT " & alias & ".*, " & assignments.joinStrings(", ") &
        " FROM (" & result & ") AS " & alias
    of dfStepGroupBy:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError, "nested groupBy is not supported")
      pendingGroupKeys = step.exprs
    of dfStepSummarise:
      let alias = nextAlias(counter)
      var selected: seq[string]
      for key in pendingGroupKeys:
        selected.add exprSql(key, false)
      for aggregation in step.namedExprs:
        selected.add namedExprSql(aggregation)
      result = "SELECT " & selected.joinStrings(", ") & " FROM (" & result &
        ") AS " & alias
      if pendingGroupKeys.len > 0:
        var grouped: seq[string]
        for key in pendingGroupKeys:
          grouped.add exprSql(key, false)
        result.add " GROUP BY " & grouped.joinStrings(", ")
      pendingGroupKeys.setLen 0
    of dfStepArrange:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let alias = nextAlias(counter)
      var orderings: seq[string]
      for ordering in step.exprs:
        orderings.add exprSql(ordering)
      result = "SELECT * FROM (" & result & ") AS " & alias &
        " ORDER BY " & orderings.joinStrings(", ")
    of dfStepLimit:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let alias = nextAlias(counter)
      result = "SELECT * FROM (" & result & ") AS " & alias &
        " LIMIT " & $step.limitCount
    of dfStepJoin:
      if pendingGroupKeys.len > 0:
        raise newException(DataFrameError,
          "groupBy must be followed by summarise")
      let leftAlias = nextAlias(counter)
      let rightAlias = nextAlias(counter)
      let rightSql = compileSql(step.joinSource, step.joinSteps, counter)
      result = "SELECT * FROM (" & result & ") AS " & leftAlias & " " &
        joinKindSql(step.joinKind) & " (" & rightSql & ") AS " & rightAlias
      if step.joinKind != joinCross:
        if not step.hasJoinPredicate:
          raise newException(DataFrameError,
            "non-cross joins require a predicate")
        result.add " ON "
        result.add exprSql(step.joinPredicate, false)

  if pendingGroupKeys.len > 0:
    raise newException(DataFrameError, "groupBy must be followed by summarise")

proc toSql*(df: DataFrame): string =
  ## Emits DuckDB-compatible SQL for the lazy logical plan.
  var counter = 0
  result = compileSql(df.source, df.steps, counter)

proc sql*(df: DataFrame): string =
  ## Emits DuckDB-compatible SQL for the lazy logical plan.
  result = df.toSql()

proc collect*(df: DataFrame): DataFrameCollection =
  ## Records the SQL that a backend will execute when materialising `df`.
  result = DataFrameCollection(plan: df, sql: df.toSql())
