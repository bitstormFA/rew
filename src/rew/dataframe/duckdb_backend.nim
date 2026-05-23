## DuckDB execution backend for DataFrames.

import std/strformat
import ../dtype
import ./core
import ./materialized
import ./duckdb_raw

proc mapDuckDbType(typ: DuckDbType): DType =
  if typ == DuckDbTypeBoolean:
    dtBool
  elif typ == DuckDbTypeTinyInt or typ == DuckDbTypeSmallInt or
      typ == DuckDbTypeInteger or typ == DuckDbTypeBigInt:
    dtInt64
  elif typ == DuckDbTypeUTinyInt or typ == DuckDbTypeUSmallInt or
      typ == DuckDbTypeUInteger or typ == DuckDbTypeUBigInt:
    dtUint64
  elif typ == DuckDbTypeFloat or typ == DuckDbTypeDouble or
      typ == DuckDbTypeDecimal:
    dtFloat64
  elif typ == DuckDbTypeVarchar:
    dtUint8
  else:
    raise newException(DataFrameError,
      "unsupported DuckDB result type id: " & $typ.typeId)

proc readValue(api: DuckDbApi; rawResult: var DuckDbResult; typ: DuckDbType;
    col, row: DuckDbIdx): DataValue =
  if api.valueIsNull(addr rawResult, col, row):
    return nullValue()
  if typ == DuckDbTypeBoolean:
    return boolValue(api.valueBoolean(addr rawResult, col, row))
  if typ == DuckDbTypeTinyInt or typ == DuckDbTypeSmallInt or
      typ == DuckDbTypeInteger or typ == DuckDbTypeBigInt:
    return intValue(api.valueInt64(addr rawResult, col, row))
  if typ == DuckDbTypeUTinyInt or typ == DuckDbTypeUSmallInt or
      typ == DuckDbTypeUInteger or typ == DuckDbTypeUBigInt:
    let value = api.valueUInt64(addr rawResult, col, row)
    if value > uint64(high(int64)):
      raise newException(DataFrameError,
        "unsigned DuckDB value does not fit int64 at column " & $col &
          ", row " & $row)
    return intValue(int64(value))
  if typ == DuckDbTypeFloat or typ == DuckDbTypeDouble or
      typ == DuckDbTypeDecimal:
    return floatValue(api.valueDouble(addr rawResult, col, row))
  if typ == DuckDbTypeVarchar:
    let raw = api.valueVarchar(addr rawResult, col, row)
    if raw == nil:
      return nullValue()
    result = stringValue($raw)
    api.free(raw)
    return
  raise newException(DataFrameError,
    "unsupported DuckDB result type id: " & $typ.typeId)

proc raiseResultError(api: DuckDbApi; result: var DuckDbResult;
    sql: string) {.noinline, noreturn.} =
  let msg = api.resultError(addr result)
  let detail = if msg == nil: "unknown DuckDB error" else: $msg
  raise newException(DataFrameError,
    &"DuckDB query failed: {detail}; SQL: {sql}")

proc executeDuckDb*(df: DataFrame): DataFrameCollection =
  ## Executes `df` against an in-memory DuckDB connection.
  let api = loadDuckDbApi()
  let sql = df.toSql()
  var db: DuckDbDatabase
  if not api.open(nil, addr db).ok:
    raise newException(DataFrameError, "DuckDB open failed")
  try:
    var conn: DuckDbConnection
    if not api.connect(db, addr conn).ok:
      raise newException(DataFrameError, "DuckDB connect failed")
    try:
      var rawResult: DuckDbResult
      if not api.query(conn, sql.cstring, addr rawResult).ok:
        try:
          api.raiseResultError(rawResult, sql)
        finally:
          api.destroyResult(addr rawResult)
      try:
        let colCount = int(api.columnCount(addr rawResult))
        let rowCount = int(api.rowCount(addr rawResult))
        result.plan = df
        result.sql = sql
        result.rowCount = rowCount
        result.columns = newSeq[DataFrameColumn](colCount)
        result.schema.columns = newSeq[ColumnSchema](colCount)

        var types = newSeq[DuckDbType](colCount)
        for c in 0 ..< colCount:
          let rawName = api.columnName(addr rawResult, DuckDbIdx(c))
          let typ = api.columnType(addr rawResult, DuckDbIdx(c))
          types[c] = typ
          let schema = ColumnSchema(
            name: if rawName == nil: "col" & $c else: $rawName,
            dtype: mapDuckDbType(typ),
            nullable: false)
          result.schema.columns[c] = schema
          result.columns[c] = DataFrameColumn(schema: schema,
            values: newSeq[DataValue](rowCount))

        for r in 0 ..< rowCount:
          for c in 0 ..< colCount:
            let value = api.readValue(rawResult, types[c], DuckDbIdx(c),
              DuckDbIdx(r))
            if value.kind == dfvNull:
              result.schema.columns[c].nullable = true
              result.columns[c].schema.nullable = true
            result.columns[c].values[r] = value
      finally:
        api.destroyResult(addr rawResult)
    finally:
      api.disconnect(addr conn)
  finally:
    api.close(addr db)
