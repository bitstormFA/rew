## Materialized DataFrame result types.

import std/tables
import ../dtype
import ../tensor
import ./core

type
  DataValueKind* = enum
    dfvNull,
    dfvBool,
    dfvInt,
    dfvFloat,
    dfvString

  DataValue* = object
    ## Host scalar value returned from DuckDB.
    kind*: DataValueKind
    boolVal*: bool
    intVal*: int64
    floatVal*: float64
    stringVal*: string

  ColumnSchema* = object
    ## Name and dtype for a materialized column.
    name*: string
    dtype*: DType
    nullable*: bool

  DataFrameSchema* = object
    ## Materialized result schema.
    columns*: seq[ColumnSchema]

  DataFrameColumn* = object
    ## Host-side materialized column.
    schema*: ColumnSchema
    values*: seq[DataValue]

  DataFrameCollection* = object
    ## Host-side result of collecting a DataFrame.
    plan*: DataFrame
    sql*: string
    schema*: DataFrameSchema
    columns*: seq[DataFrameColumn]
    rowCount*: int

  TensorFrame* = object
    ## Device-side columnar tensor result.
    plan*: DataFrame
    sql*: string
    schema*: DataFrameSchema
    columns*: seq[Tensor]
    rowCount*: int

  CoercePolicy* = enum
    coerceNone,
    coerceNumericToFloat32

  CollectOptions* = object
    ## Materialization policy for tensor conversion.
    coerce*: CoercePolicy
    allowNulls*: bool

  BatchFieldSpec* = object
    ## Mapping from a batch field to one or more DataFrame columns.
    fieldName*: string
    columns*: seq[string]

  BatchMapping* = object
    ## Explicit DataFrame column-to-batch mapping.
    fields*: seq[BatchFieldSpec]

func initCollectOptions*(coerce: CoercePolicy = coerceNone;
    allowNulls = false): CollectOptions =
  CollectOptions(coerce: coerce, allowNulls: allowNulls)

func boolValue*(value: bool): DataValue =
  DataValue(kind: dfvBool, boolVal: value)

func intValue*(value: int64): DataValue =
  DataValue(kind: dfvInt, intVal: value)

func floatValue*(value: float64): DataValue =
  DataValue(kind: dfvFloat, floatVal: value)

func stringValue*(value: string): DataValue =
  DataValue(kind: dfvString, stringVal: value)

func nullValue*(): DataValue =
  DataValue(kind: dfvNull)

proc columnIndex*(schema: DataFrameSchema; name: string): int =
  for i, col in schema.columns:
    if col.name == name:
      return i
  -1

proc requireColumn*(collection: DataFrameCollection; name: string): int =
  result = collection.schema.columnIndex(name)
  if result < 0:
    raise newException(DataFrameError,
      "column not found in materialized DataFrame: " & name)

proc requireColumn*(frame: TensorFrame; name: string): int =
  result = frame.schema.columnIndex(name)
  if result < 0:
    raise newException(DataFrameError,
      "column not found in tensor frame: " & name)

proc columnTable*(mapping: BatchMapping): Table[string, BatchFieldSpec] =
  for spec in mapping.fields:
    result[spec.fieldName] = spec

proc autoBatchMapping*[Batch](): BatchMapping =
  ## Maps every Tensor field in `Batch` to a DataFrame column with the same
  ## name. Non-tensor fields keep their default value.
  var proto: Batch
  for name, field in fieldPairs(proto):
    when field is Tensor:
      result.fields.add BatchFieldSpec(fieldName: name, columns: @[name])

