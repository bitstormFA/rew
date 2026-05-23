## DataFrame phase-2 DuckDB backend materialization tests.

import std/[os, strutils]
import rew
import rew/dataframe

block pinned_artifact_metadata_covers_supported_platforms:
  let linux = duckDbArtifactSpec("linux", "amd64")
  let mac = duckDbArtifactSpec("darwin", "arm64")
  let win64 = duckDbArtifactSpec("windows", "amd64")
  let winArm = duckDbArtifactSpec("windows", "aarch64")

  doAssert linux.platformName == "linux-amd64"
  doAssert linux.member == "libduckdb.so"
  doAssert mac.platformName == "osx-universal"
  doAssert mac.member == "libduckdb.dylib"
  doAssert win64.platformName == "windows-amd64"
  doAssert win64.member == "duckdb.dll"
  doAssert winArm.platformName == "windows-arm64"
  doAssert winArm.member == "duckdb.dll"
  for spec in [linux, mac, win64, winArm]:
    doAssert spec.url.startsWith("https://github.com/duckdb/duckdb/releases/download/v1.5.3/")
    doAssert spec.url.endsWith(".zip")
    doAssert spec.sha256.len == 64

block duckdb_collects_raw_sql_source:
  let rows = sql("select 1::integer as x, 2.5::double as y, 'a' as label")
    .collect()

  doAssert rows.sql == "SELECT * FROM (select 1::integer as x, 2.5::double as y, 'a' as label)"
  doAssert rows.rowCount == 1
  doAssert rows.schema.columns.len == 3
  doAssert rows.schema.columns[0].name == "x"
  doAssert rows.schema.columns[0].dtype == dtInt64
  doAssert rows.columns[0].values[0].intVal == 1
  doAssert rows.columns[1].values[0].floatVal == 2.5
  doAssert rows.columns[2].values[0].stringVal == "a"

block duckdb_executes_expression_pipeline:
  let rows = sql("""
      select * from (values
        ('DE', 10.0, 2.0),
        ('US', 20.0, 5.0),
        ('DE', 3.0, 1.0)
      ) as events(country, gross, discount)
    """)
    .filter(col("country") == lit("DE"))
    .mutate(named("net", col("gross") - col("discount")))
    .summarise(named("total", sum(col("net"))), named("rows", count()))
    .collect()

  doAssert rows.rowCount == 1
  doAssert rows.columns[0].schema.name == "total"
  doAssert rows.columns[0].values[0].floatVal == 10.0
  doAssert rows.columns[1].values[0].intVal == 2

block duckdb_nulls_are_marked_in_schema:
  let rows = sql("select null::integer as maybe_id").collect()

  doAssert rows.rowCount == 1
  doAssert rows.schema.columns[0].nullable
  doAssert rows.columns[0].values[0].kind == dfvNull

block duckdb_collects_csv_source:
  let path = getTempDir() / ("rew_dataframe_phase2_" &
    $getCurrentProcessId() & ".csv")
  writeFile(path, "x,y,label\n1,2.5,a\n3,4.5,b\n")
  try:
    let rows = readCsv(path)
      .filter(col("x") > lit(1))
      .select(col("x"), col("y"), col("label"))
      .collect()

    doAssert rows.rowCount == 1
    doAssert rows.columns[0].values[0].intVal == 3
    doAssert rows.columns[1].values[0].floatVal == 4.5
    doAssert rows.columns[2].values[0].stringVal == "b"
  finally:
    if fileExists(path):
      removeFile(path)
