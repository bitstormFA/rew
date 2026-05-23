## DataFrame phase-1 expression engine and SQL emission tests.

import rew/dataframe

block sql_emits_grouped_pipeline:
  let query = fromTable("events")
    .filter(col("country") == lit("DE"))
    .mutate(named("net", col("gross") - col("discount")))
    .groupBy(col("user_id"))
    .summarise(
      named("totalNet", sum(col("net"))),
      named("rows", count()))
    .arrange(desc(col("totalNet")))
    .limit(100)
    .toSql()

  doAssert query ==
    "SELECT * FROM (SELECT * FROM (SELECT \"user_id\", " &
    "sum(\"net\") AS \"totalNet\", count(*) AS \"rows\" FROM " &
    "(SELECT q1.*, (\"gross\" - \"discount\") AS \"net\" FROM " &
    "(SELECT * FROM (SELECT * FROM \"events\") AS q0 WHERE " &
    "(\"country\" = 'DE')) AS q1) AS q2 GROUP BY \"user_id\") AS q3 " &
    "ORDER BY \"totalNet\" DESC) AS q4 LIMIT 100"

block sql_escapes_identifiers_and_literals:
  let query = fromTable("main.events")
    .filter(contains(col("name"), "O'Reilly"))
    .select(
      alias(year(col("created_at")), "year"),
      alias(col("name"), "displayName"))
    .toSql()

  doAssert query ==
    "SELECT date_part('year', \"created_at\") AS \"year\", " &
    "\"name\" AS \"displayName\" FROM (SELECT * FROM " &
    "(SELECT * FROM \"main\".\"events\") AS q0 WHERE " &
    "contains(\"name\", 'O''Reilly')) AS q1"

block join_pipeline_emits_nested_right_plan:
  let users = fromTable("users").select(col("id"), col("tier"))
  let query = fromTable("events")
    .join(users, on = col("user_id") == col("id"), kind = joinLeft)
    .select(col("user_id"), col("tier"))
    .toSql()

  doAssert query ==
    "SELECT \"user_id\", \"tier\" FROM (SELECT * FROM " &
    "(SELECT * FROM \"events\") AS q0 LEFT JOIN " &
    "(SELECT \"id\", \"tier\" FROM (SELECT * FROM \"users\") AS q2) " &
    "AS q1 ON (\"user_id\" = \"id\")) AS q3"

block raw_sql_source_is_explicit_escape_hatch:
  let query = sql("select 1 as x").filter(col("x") > lit(0)).sql()

  doAssert query ==
    "SELECT * FROM (SELECT * FROM (select 1 as x)) AS q0 WHERE (\"x\" > 0)"

block collect_records_phase1_query:
  let df = readParquet("events.parquet")
    .filter(startsWith(col("path"), "/api"))
    .limit(5)
  let collected = df.collect()

  doAssert collected.plan.steps.len == 2
  doAssert collected.sql == df.toSql()
  doAssert collected.sql ==
    "SELECT * FROM (SELECT * FROM (SELECT * FROM " &
    "read_parquet('events.parquet')) AS q0 WHERE " &
    "starts_with(\"path\", '/api')) AS q1 LIMIT 5"

block logical_plan_round_trip_preserves_pipeline:
  let df = readCsv("events.csv")
    .mutate(named("month", month(col("created_at"))))
    .select(col("month"))
  let plan = df.toLogicalPlan()
  let rebuilt = initDataFrame(plan)

  doAssert plan.source.kind == dfSourceCsv
  doAssert plan.steps.len == 2
  doAssert rebuilt.toSql() == df.toSql()

block expression_helpers_build_expected_nodes:
  let label = caseWhen(like(col("email"), "%@example.com"))
    .then(lit("internal"))
    .elseValue(lit("external"))
  let timestampParts = dayOfWeek(col("created_at")) + second(col("created_at"))
  let predicate = matches(col("email"), ".*@example\\.com")

  doAssert label.kind == exprCase
  doAssert label.args[0].literal.stringVal == "external"
  doAssert timestampParts.kind == exprBinary
  doAssert timestampParts.args[0].name == "dayOfWeek"
  doAssert predicate.kind == exprCall
  doAssert predicate.name == "matches"
