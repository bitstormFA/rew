## DType — element types of a `Tensor`.
##
## Closed enum mapped 1:1 to the StableHLO/PJRT element-type space we expose
## in v1. Helpers convert to/from native Nim scalar types and report element
## sizes in bytes. Per the layered design, this module is dependency-free so
## both `pjrt/`-adjacent code (buffer, device) and the StableHLO emitter can
## use it without circularity.

type
  DType* = enum
    ## Element type of a `Tensor`. Order is stable; do not reorder without
    ## updating the StableHLO bytecode constants in `stablehlo/`.
    dtBool      ## 1-bit boolean (stored as 1 byte on host).
    dtInt8      ## signed 8-bit integer.
    dtInt16     ## signed 16-bit integer.
    dtInt32     ## signed 32-bit integer.
    dtInt64     ## signed 64-bit integer.
    dtUint8     ## unsigned 8-bit integer.
    dtUint16    ## unsigned 16-bit integer.
    dtUint32    ## unsigned 32-bit integer.
    dtUint64    ## unsigned 64-bit integer.
    dtFloat16   ## IEEE-754 binary16 (half precision).
    dtBFloat16  ## bfloat16.
    dtFloat32   ## IEEE-754 binary32 (single precision).
    dtFloat64   ## IEEE-754 binary64 (double precision).
    dtComplex64 ## Complex value with two float32 components.
    dtComplex128 ## Complex value with two float64 components.
    dtInt4      ## signed 4-bit integer (packed: 2 values per byte).
    dtUint4     ## unsigned 4-bit integer (packed: 2 values per byte).
    dtNF4       ## NormalFloat4 (packed: 2 values per byte).
    dtFloat8E4M3Fn ## Float8 E4M3 format.
    dtFloat8E5M2   ## Float8 E5M2 format.

func byteSize*(dt: DType): int =
  ## Returns the on-device byte size of one element of `dt`.
  ## Packed types (int4/uint4/nf4) report 1 byte (the minimum storage unit).
  case dt
  of dtBool, dtInt4, dtUint4, dtNF4, dtInt8, dtUint8,
     dtFloat8E4M3Fn, dtFloat8E5M2: 1
  of dtInt16, dtUint16, dtFloat16, dtBFloat16: 2
  of dtInt32, dtUint32, dtFloat32: 4
  of dtInt64, dtUint64, dtFloat64, dtComplex64: 8
  of dtComplex128: 16

func bitWidth*(dt: DType): int =
  ## Returns the StableHLO element bit width of `dt`.
  case dt
  of dtBool: 1
  of dtInt4, dtUint4, dtNF4: 4
  of dtInt8, dtUint8, dtFloat8E4M3Fn, dtFloat8E5M2: 8
  of dtInt16, dtUint16, dtFloat16, dtBFloat16: 16
  of dtInt32, dtUint32, dtFloat32: 32
  of dtInt64, dtUint64, dtFloat64, dtComplex64: 64
  of dtComplex128: 128

func name*(dt: DType): string =
  ## Returns the canonical user-facing dtype name (e.g. `"float32"`).
  case dt
  of dtBool: "bool"
  of dtInt4: "int4"
  of dtInt8: "int8"
  of dtInt16: "int16"
  of dtInt32: "int32"
  of dtInt64: "int64"
  of dtUint4: "uint4"
  of dtUint8: "uint8"
  of dtUint16: "uint16"
  of dtUint32: "uint32"
  of dtUint64: "uint64"
  of dtFloat16: "float16"
  of dtBFloat16: "bfloat16"
  of dtFloat32: "float32"
  of dtFloat64: "float64"
  of dtComplex64: "complex64"
  of dtComplex128: "complex128"
  of dtNF4: "nf4"
  of dtFloat8E4M3Fn: "float8_e4m3fn"
  of dtFloat8E5M2: "float8_e5m2"

func isFloat*(dt: DType): bool =
  ## True for floating-point dtypes.
  dt in {dtFloat16, dtBFloat16, dtFloat32, dtFloat64,
    dtFloat8E4M3Fn, dtFloat8E5M2, dtNF4}

func isSignedInt*(dt: DType): bool =
  ## True for signed integer dtypes.
  dt in {dtInt4, dtInt8, dtInt16, dtInt32, dtInt64}

func isUnsignedInt*(dt: DType): bool =
  ## True for unsigned integer dtypes.
  dt in {dtUint4, dtUint8, dtUint16, dtUint32, dtUint64}

func isComplex*(dt: DType): bool =
  ## True for complex floating-point dtypes.
  dt in {dtComplex64, dtComplex128}

func complexPartDType*(dt: DType): DType =
  ## Returns the real/imaginary component dtype of a complex dtype.
  case dt
  of dtComplex64: dtFloat32
  of dtComplex128: dtFloat64
  else: dt

func complexDType*(part: DType): DType =
  ## Returns the complex dtype with `part` as its component dtype.
  case part
  of dtFloat32: dtComplex64
  of dtFloat64: dtComplex128
  else: part

func dtypeOf*(t: typedesc[bool]): DType = dtBool
  ## Compile-time mapping from a native Nim scalar type to the matching
  ## `DType`. Used by `fromHost` and friends. Coverage is intentionally
  ## limited to the unambiguous Nim scalar types; `float16`/`bfloat16` have
  ## no native Nim type and must be selected explicitly.
func dtypeOf*(t: typedesc[int8]): DType = dtInt8
func dtypeOf*(t: typedesc[int16]): DType = dtInt16
func dtypeOf*(t: typedesc[int32]): DType = dtInt32
func dtypeOf*(t: typedesc[int64]): DType = dtInt64
func dtypeOf*(t: typedesc[uint8]): DType = dtUint8
func dtypeOf*(t: typedesc[uint16]): DType = dtUint16
func dtypeOf*(t: typedesc[uint32]): DType = dtUint32
func dtypeOf*(t: typedesc[uint64]): DType = dtUint64
func dtypeOf*(t: typedesc[float32]): DType = dtFloat32
func dtypeOf*(t: typedesc[float64]): DType = dtFloat64
