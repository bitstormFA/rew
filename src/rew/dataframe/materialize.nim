## Public DataFrame materialization and typed batch bridge.

import std/tables
import ../dtype
import ../device
import ../tensor
import ../eager
import ../data/dataset
import ./core
import ./materialized
import ./duckdb_backend

proc collect*(df: DataFrame): DataFrameCollection =
  ## Executes a lazy DataFrame through the pinned DuckDB backend.
  executeDuckDb(df)

proc tensorFromColumn(column: DataFrameColumn; start, stop: int;
    device: Device; options: CollectOptions): Tensor =
  let count = stop - start
  if count < 0:
    raise newException(DataFrameError, "invalid collection slice")

  case column.schema.dtype
  of dtBool:
    var data = newSeq[bool](count)
    for i in 0 ..< count:
      let value = column.values[start + i]
      if value.kind == dfvNull:
        if options.allowNulls: data[i] = false
        else: raise newException(DataFrameError,
          "null value in boolean column: " & column.schema.name)
      elif value.kind == dfvBool:
        data[i] = value.boolVal
      else:
        raise newException(DataFrameError,
          "non-boolean value in column: " & column.schema.name)
    fromHost(device, data, [count])
  of dtInt64, dtUint64:
    if options.coerce == coerceNumericToFloat32:
      var data = newSeq[float32](count)
      for i in 0 ..< count:
        let value = column.values[start + i]
        if value.kind == dfvNull:
          if options.allowNulls: data[i] = 0'f32
          else: raise newException(DataFrameError,
            "null value in numeric column: " & column.schema.name)
        elif value.kind == dfvInt:
          data[i] = float32(value.intVal)
        elif value.kind == dfvFloat:
          data[i] = float32(value.floatVal)
        else:
          raise newException(DataFrameError,
            "non-numeric value in column: " & column.schema.name)
      fromHostF32(device, data, [count])
    else:
      var data = newSeq[int64](count)
      for i in 0 ..< count:
        let value = column.values[start + i]
        if value.kind == dfvNull:
          if options.allowNulls: data[i] = 0
          else: raise newException(DataFrameError,
            "null value in integer column: " & column.schema.name)
        elif value.kind == dfvInt:
          data[i] = value.intVal
        else:
          raise newException(DataFrameError,
            "non-integer value in column: " & column.schema.name)
      fromHost(device, data, [count])
  of dtFloat64, dtFloat32:
    if options.coerce == coerceNumericToFloat32:
      var data = newSeq[float32](count)
      for i in 0 ..< count:
        let value = column.values[start + i]
        if value.kind == dfvNull:
          if options.allowNulls: data[i] = 0'f32
          else: raise newException(DataFrameError,
            "null value in float column: " & column.schema.name)
        elif value.kind == dfvFloat:
          data[i] = float32(value.floatVal)
        elif value.kind == dfvInt:
          data[i] = float32(value.intVal)
        else:
          raise newException(DataFrameError,
            "non-float value in column: " & column.schema.name)
      fromHostF32(device, data, [count])
    else:
      var data = newSeq[float64](count)
      for i in 0 ..< count:
        let value = column.values[start + i]
        if value.kind == dfvNull:
          if options.allowNulls: data[i] = 0.0
          else: raise newException(DataFrameError,
            "null value in float column: " & column.schema.name)
        elif value.kind == dfvFloat:
          data[i] = value.floatVal
        elif value.kind == dfvInt:
          data[i] = float64(value.intVal)
        else:
          raise newException(DataFrameError,
            "non-float value in column: " & column.schema.name)
      fromHost(device, data, [count])
  else:
    raise newException(DataFrameError,
      "cannot convert column to tensor: " & column.schema.name)

proc collect*(df: DataFrame; device: Device;
    options: CollectOptions = initCollectOptions()): TensorFrame =
  ## Executes and transfers every supported column to `device`.
  let host = df.collect()
  result.plan = host.plan
  result.sql = host.sql
  result.schema = host.schema
  result.rowCount = host.rowCount
  result.columns = newSeq[Tensor](host.columns.len)
  for i, column in host.columns:
    result.columns[i] = tensorFromColumn(column, 0, host.rowCount, device,
      options)

func optionsForField(spec: BatchFieldSpec; options: CollectOptions):
    CollectOptions =
  result = options
  if spec.hasDType:
    case spec.dtype
    of dtFloat32:
      result.coerce = coerceNumericToFloat32
    else:
      result.coerce = coerceNone

proc tensorForField(host: DataFrameCollection; spec: BatchFieldSpec;
    start, stop: int; device: Device; options: CollectOptions): Tensor =
  if spec.columns.len == 0:
    raise newException(DataFrameError,
      "batch field has no mapped columns: " & spec.fieldName)
  let fieldOptions = optionsForField(spec, options)
  if spec.columns.len == 1:
    return tensorFromColumn(host.columns[host.requireColumn(spec.columns[0])],
      start, stop, device, fieldOptions)

  if spec.hasDType and spec.dtype != dtFloat32:
    raise newException(DataFrameError,
      "multi-column batch fields currently materialize as float32: " &
        spec.fieldName)
  var parts: seq[float32]
  let rows = stop - start
  for row in start ..< stop:
    for name in spec.columns:
      let col = host.columns[host.requireColumn(name)]
      let value = col.values[row]
      case value.kind
      of dfvInt:
        parts.add float32(value.intVal)
      of dfvFloat:
        parts.add float32(value.floatVal)
      of dfvNull:
        if fieldOptions.allowNulls: parts.add 0'f32
        else: raise newException(DataFrameError,
          "null value in mapped field: " & spec.fieldName)
      else:
        raise newException(DataFrameError,
          "non-numeric value in mapped field: " & spec.fieldName)
  fromHostF32(device, parts, [rows, spec.columns.len])

proc collectAs*[Batch](df: DataFrame; device: Device;
    mapping: BatchMapping = autoBatchMapping[Batch]();
    options: CollectOptions = initCollectOptions(coerceNumericToFloat32)):
    Batch =
  ## Collects all rows into one typed batch. Tensor fields are populated from
  ## matching DataFrame columns by default.
  let host = df.collect()
  let map = mapping.columnTable()
  for name, field in fieldPairs(result):
    when field is Tensor:
      if name notin map:
        raise newException(DataFrameError,
          "missing DataFrame mapping for batch Tensor field: " & name)
      field = tensorForField(host, map[name], 0, host.rowCount, device,
        options)

proc toDataset*[Batch](df: DataFrame; batchSize: int; device: Device;
    mapping: BatchMapping = autoBatchMapping[Batch]();
    options: CollectOptions = initCollectOptions(coerceNumericToFloat32)):
    Dataset[Batch] =
  ## Materializes a DataFrame as a re-entrant typed Dataset. Each epoch
  ## re-executes the lazy DuckDB query.
  if batchSize <= 0:
    raise newException(DataFrameError, "toDataset batchSize must be positive")
  result.source = proc(): iterator(): Batch =
    let captured = df
    let bs = batchSize
    let d = device
    let map = mapping
    let opts = options
    result = iterator(): Batch {.closure.} =
      let host = captured.collect()
      let table = map.columnTable()
      var start = 0
      while start < host.rowCount:
        let stop = min(start + bs, host.rowCount)
        var batch: Batch
        for name, field in fieldPairs(batch):
          when field is Tensor:
            if name notin table:
              raise newException(DataFrameError,
                "missing DataFrame mapping for batch Tensor field: " & name)
            field = tensorForField(host, table[name], start, stop, d, opts)
        yield batch
        start = stop
