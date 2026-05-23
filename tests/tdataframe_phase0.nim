## DataFrame phase-0 public API scaffolding tests.

import rew
import rew/dataframe

block expression_dsl_builds_ast:
  let net = col("gross") - col("discount")
  doAssert net.kind == exprBinary
  doAssert net.name == "-"
  doAssert net.args.len == 2
  doAssert net.args[0].kind == exprColumn
  doAssert net.args[0].name == "gross"

  let flag = caseWhen(col("gross") > lit(100))
    .then(lit("large"))
    .otherwise(lit("small"))
  doAssert flag.kind == exprCase
  doAssert flag.branches.len == 1
  doAssert flag.args.len == 1
  doAssert flag.branches[0].condition.name == ">"

  let ordered = desc(sum(col("gross")))
  doAssert ordered.kind == exprSort
  doAssert ordered.direction == sortDescending
  doAssert ordered.args[0].name == "sum"

block lazy_pipeline_records_steps:
  let filtered = fromTable("events").filter(col("country") == lit("DE"))
  let mutated = filtered.mutate(named("net", col("gross") - col("discount")))
  let grouped = mutated.groupBy(col("userId"))
  let summarized = grouped.summarise(
    named("totalNet", sum(col("net"))), named("rows", count()))
  let arranged = summarized.arrange(desc(col("totalNet")))
  let pipeline = arranged.limit(10)

  doAssert pipeline.source.kind == dfSourceTable
  doAssert pipeline.source.name == "events"
  doAssert pipeline.steps.len == 6
  doAssert pipeline.steps[0].kind == dfStepFilter
  doAssert pipeline.steps[1].kind == dfStepMutate
  doAssert pipeline.steps[2].kind == dfStepGroupBy
  doAssert pipeline.steps[3].kind == dfStepSummarise
  doAssert pipeline.steps[4].kind == dfStepArrange
  doAssert pipeline.steps[5].kind == dfStepLimit
  doAssert pipeline.steps[5].limitCount == 10

block file_sources_are_lazy_values:
  let csv = readCsv("events.csv")
  let parquet = readParquet("events.parquet")

  doAssert csv.source.kind == dfSourceCsv
  doAssert csv.source.path == "events.csv"
  doAssert csv.steps.len == 0
  doAssert parquet.source.kind == dfSourceParquet
  doAssert parquet.source.path == "events.parquet"

block validation_errors_are_explicit:
  var raised = false
  try:
    discard fromTable("")
  except DataFrameError:
    raised = true
  doAssert raised

  raised = false
  try:
    discard fromTable("events").limit(-1)
  except DataFrameError:
    raised = true
  doAssert raised
