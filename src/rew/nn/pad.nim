## Padding layers — ReflectionPad2d.
##
## Reflection padding mirrors input values across the boundary.
## Composite implementation using slicing and concatenation.

import ../tensor
import ../ops/concat

proc reflectionPad2d*(x: Tensor; padding: array[4, int]): Tensor =
  ## Pads NHWC `x` with reflection padding.
  ## `padding` is `[padLeft, padRight, padTop, padBottom]` (torch convention).
  if x.shape.len != 4:
    raise newException(TensorError,
      "reflectionPad2d: expected NHWC rank-4, got " & $x.shape)
  let pL = padding[0]
  let pR = padding[1]
  let pT = padding[2]
  let pB = padding[3]
  if pL < 0 or pR < 0 or pT < 0 or pB < 0:
    raise newException(TensorError,
      "reflectionPad2d: padding must be non-negative")
  let n = x.shape[0]
  let h = x.shape[1]
  let w = x.shape[2]
  let c = x.shape[3]
  if pL >= w or pR >= w or pT >= h or pB >= h:
    raise newException(TensorError,
      "reflectionPad2d: padding must be smaller than spatial dims")
  # Pad width: reflect along axis 2
  var paddedW = x
  if pL > 0:
    let leftReflect = slice(x, [0, 0, pL, 0],
      [n, h, 1, c], [1, 1, -1, 1])
    paddedW = concat(@[leftReflect, paddedW], 2)
  if pR > 0:
    let rightReflect = slice(x, [0, 0, w - pR - 1, 0],
      [n, h, w - 1, c], [1, 1, -1, 1])
    paddedW = concat(@[paddedW, rightReflect], 2)
  # Pad height: reflect along axis 1
  var padded = paddedW
  if pT > 0:
    let topReflect = slice(paddedW, [0, pT, 0, 0],
      [n, 1, paddedW.shape[2], c], [1, -1, 1, 1])
    padded = concat(@[topReflect, padded], 1)
  if pB > 0:
    let hPadded = paddedW.shape[1]
    let botReflect = slice(paddedW, [0, hPadded - pB - 1, 0, 0],
      [n, hPadded - 1, paddedW.shape[2], c], [1, -1, 1, 1])
    padded = concat(@[padded, botReflect], 1)
  padded

type
  ReflectionPad2d* = object
    ## Reflection padding layer (value type).
    padding*: array[4, int]

proc initReflectionPad2d*(padding: array[4, int]): ReflectionPad2d =
  ## Constructs a ReflectionPad2d layer.
  ## `padding` is `[padLeft, padRight, padTop, padBottom]`.
  ReflectionPad2d(padding: padding)

proc forward*(layer: ReflectionPad2d; x: Tensor): Tensor =
  ## Applies reflection padding.
  reflectionPad2d(x, layer.padding)
