## Runtime-loaded DuckDB C API subset.

import std/[dynlib, strformat]
import ./artifacts

type
  DuckDbError* = object of CatchableError
    ## Raised when DuckDB cannot be loaded or a DuckDB call fails.

  DuckDbIdx* = uint64
  DuckDbType* = distinct uint32
  DuckDbState* = distinct cint

  DuckDbDatabase* = pointer
  DuckDbConnection* = pointer

  DuckDbColumn {.bycopy.} = object
    deprecatedData: pointer
    deprecatedNullmask: pointer
    deprecatedType: DuckDbType
    deprecatedName: cstring
    internalData: pointer

  DuckDbResult* {.bycopy.} = object
    deprecatedColumnCount*: DuckDbIdx
    deprecatedRowCount*: DuckDbIdx
    deprecatedRowsChanged*: DuckDbIdx
    deprecatedColumns*: ptr DuckDbColumn
    deprecatedErrorMessage*: cstring
    internalData*: pointer

  DuckDbApi* = object
    lib: LibHandle
    libraryPath*: string
    libraryVersion*: proc(): cstring {.cdecl.}
    open*: proc(path: cstring; outDatabase: ptr DuckDbDatabase): DuckDbState {.cdecl.}
    close*: proc(database: ptr DuckDbDatabase) {.cdecl.}
    connect*: proc(database: DuckDbDatabase; outConnection: ptr DuckDbConnection): DuckDbState {.cdecl.}
    disconnect*: proc(connection: ptr DuckDbConnection) {.cdecl.}
    query*: proc(connection: DuckDbConnection; query: cstring; outResult: ptr DuckDbResult): DuckDbState {.cdecl.}
    destroyResult*: proc(result: ptr DuckDbResult) {.cdecl.}
    columnCount*: proc(result: ptr DuckDbResult): DuckDbIdx {.cdecl.}
    rowCount*: proc(result: ptr DuckDbResult): DuckDbIdx {.cdecl.}
    columnName*: proc(result: ptr DuckDbResult; col: DuckDbIdx): cstring {.cdecl.}
    columnType*: proc(result: ptr DuckDbResult; col: DuckDbIdx): DuckDbType {.cdecl.}
    resultError*: proc(result: ptr DuckDbResult): cstring {.cdecl.}
    valueIsNull*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): bool {.cdecl.}
    valueBoolean*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): bool {.cdecl.}
    valueInt64*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): int64 {.cdecl.}
    valueUInt64*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): uint64 {.cdecl.}
    valueDouble*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): float64 {.cdecl.}
    valueVarchar*: proc(result: ptr DuckDbResult; col, row: DuckDbIdx): cstring {.cdecl.}
    free*: proc(ptrValue: pointer) {.cdecl.}

const
  DuckDbSuccess* = DuckDbState(0)
  DuckDbTypeInvalid* = DuckDbType(0)
  DuckDbTypeBoolean* = DuckDbType(1)
  DuckDbTypeTinyInt* = DuckDbType(2)
  DuckDbTypeSmallInt* = DuckDbType(3)
  DuckDbTypeInteger* = DuckDbType(4)
  DuckDbTypeBigInt* = DuckDbType(5)
  DuckDbTypeUTinyInt* = DuckDbType(6)
  DuckDbTypeUSmallInt* = DuckDbType(7)
  DuckDbTypeUInteger* = DuckDbType(8)
  DuckDbTypeUBigInt* = DuckDbType(9)
  DuckDbTypeFloat* = DuckDbType(10)
  DuckDbTypeDouble* = DuckDbType(11)
  DuckDbTypeVarchar* = DuckDbType(17)
  DuckDbTypeDecimal* = DuckDbType(19)

var loadedApi: DuckDbApi
var loaded = false

proc ok*(state: DuckDbState): bool =
  cast[cint](state) == cast[cint](DuckDbSuccess)

proc typeId*(typ: DuckDbType): int =
  int(cast[uint32](typ))

func `==`*(left, right: DuckDbType): bool =
  cast[uint32](left) == cast[uint32](right)

proc loadSymbol[T](lib: LibHandle; name: string): T =
  let p = symAddr(lib, name)
  if p == nil:
    raise newException(DuckDbError,
      "DuckDB library is missing required symbol: " & name)
  cast[T](p)

proc loadDuckDbApi*(): DuckDbApi =
  ## Loads the pinned DuckDB artifact and verifies its runtime version.
  if loaded:
    return loadedApi

  let path = resolveDuckDbLibraryPath()
  let lib = loadLib(path, globalSymbols = false)
  if lib == nil:
    raise newException(DuckDbError, "could not load DuckDB library: " & path)

  try:
    loadedApi = DuckDbApi(
      lib: lib,
      libraryPath: path,
      libraryVersion: loadSymbol[proc(): cstring {.cdecl.}](lib,
        "duckdb_library_version"),
      open: loadSymbol[proc(path: cstring; outDatabase: ptr DuckDbDatabase):
        DuckDbState {.cdecl.}](lib, "duckdb_open"),
      close: loadSymbol[proc(database: ptr DuckDbDatabase) {.cdecl.}](lib,
        "duckdb_close"),
      connect: loadSymbol[proc(database: DuckDbDatabase;
        outConnection: ptr DuckDbConnection): DuckDbState {.cdecl.}](lib,
        "duckdb_connect"),
      disconnect: loadSymbol[proc(connection: ptr DuckDbConnection) {.cdecl.}](
        lib, "duckdb_disconnect"),
      query: loadSymbol[proc(connection: DuckDbConnection; query: cstring;
        outResult: ptr DuckDbResult): DuckDbState {.cdecl.}](lib,
        "duckdb_query"),
      destroyResult: loadSymbol[proc(result: ptr DuckDbResult) {.cdecl.}](lib,
        "duckdb_destroy_result"),
      columnCount: loadSymbol[proc(result: ptr DuckDbResult): DuckDbIdx
        {.cdecl.}](lib, "duckdb_column_count"),
      rowCount: loadSymbol[proc(result: ptr DuckDbResult): DuckDbIdx {.cdecl.}](
        lib, "duckdb_row_count"),
      columnName: loadSymbol[proc(result: ptr DuckDbResult; col: DuckDbIdx):
        cstring {.cdecl.}](lib, "duckdb_column_name"),
      columnType: loadSymbol[proc(result: ptr DuckDbResult; col: DuckDbIdx):
        DuckDbType {.cdecl.}](lib, "duckdb_column_type"),
      resultError: loadSymbol[proc(result: ptr DuckDbResult): cstring
        {.cdecl.}](lib, "duckdb_result_error"),
      valueIsNull: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): bool {.cdecl.}](lib, "duckdb_value_is_null"),
      valueBoolean: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): bool {.cdecl.}](lib, "duckdb_value_boolean"),
      valueInt64: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): int64 {.cdecl.}](lib, "duckdb_value_int64"),
      valueUInt64: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): uint64 {.cdecl.}](lib, "duckdb_value_uint64"),
      valueDouble: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): float64 {.cdecl.}](lib, "duckdb_value_double"),
      valueVarchar: loadSymbol[proc(result: ptr DuckDbResult; col, row:
        DuckDbIdx): cstring {.cdecl.}](lib, "duckdb_value_varchar"),
      free: loadSymbol[proc(ptrValue: pointer) {.cdecl.}](lib, "duckdb_free"),
    )
    let version = $loadedApi.libraryVersion()
    if version != DuckDbVersionTag:
      unloadLib(lib)
      raise newException(DuckDbError,
        &"DuckDB version mismatch: required {DuckDbVersionTag}, loaded {version} from {path}")
    loaded = true
    result = loadedApi
  except CatchableError as e:
    unloadLib(lib)
    raise e
