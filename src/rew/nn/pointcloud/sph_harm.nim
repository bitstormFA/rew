## Real spherical harmonics for 3D equivariant networks.
##
## Computes Y_l^m(x, y, z) for l = 0..lMax as polynomials in the
## normalized coordinates (x/r, y/r, z/r). Based on the real spherical
## harmonics with Condon-Shortley phase convention.
##
## All ops are trace-compatible composites of `mul`, `add`, `sqrt`,
## and `div` — no trig functions needed.

import std/math
import ../../tensor
import ../../ops/arith
import ../../ops/unary
import ../../ops/shape
import ../../ops/linalg
import ../../ops/concat
import ../../ops/literal

proc sphericalHarmonics*(xyz: Tensor; lMax: int): Tensor =
  ## Compute real spherical harmonics up to degree `lMax`.
  ##
  ## `xyz`: `[..., 3]` where the last dim is (x, y, z) coordinates.
  ## Returns: `[..., (lMax+1)^2]` — concatenated harmonics ordered by
  ## increasing l, and within each l by m = -l, -l+1, ..., l.
  ##
  ## For lMax = 1: 4 values  (Y_0^0, Y_1^{-1}, Y_1^0, Y_1^1)
  ## For lMax = 2: 9 values
  ## For lMax = 3: 16 values
  if xyz.shape.len == 0 or xyz.shape[^1] != 3:
    raise newException(TensorError,
      "sphericalHarmonics: last dim must be 3, got " & $xyz.shape)
  if lMax < 0 or lMax > 4:
    raise newException(TensorError,
      "sphericalHarmonics: lMax must be in [0, 4], got " & $lMax)
  # Extract x, y, z from the last dim.
  let rank = xyz.shape.len
  var allStarts = newSeq[int](rank)
  var allLimits = newSeq[int](rank)
  for i in 0 ..< rank:
    allLimits[i] = xyz.shape[i]
  var ones = newSeq[int](rank)
  for i in 0 ..< rank: ones[i] = 1
  allStarts[rank - 1] = 0
  allLimits[rank - 1] = 1
  let x = slice(xyz, allStarts, allLimits, ones)
  allStarts[rank - 1] = 1
  allLimits[rank - 1] = 2
  let y = slice(xyz, allStarts, allLimits, ones)
  allStarts[rank - 1] = 2
  allLimits[rank - 1] = 3
  let z = slice(xyz, allStarts, allLimits, ones)
  # Squeeze the trailing dim: [..., 1] → [...]
  let squeezeDim = rank - 1
  let x2 = squeeze(x, squeezeDim)
  let y2 = squeeze(y, squeezeDim)
  let z2 = squeeze(z, squeezeDim)
  # Helper: broadcast scalar to target shape.
  proc bc(v: float32; shape: openArray[int]): Tensor =
    broadcastTo(scalarF32(v), @shape, @[])
  # Compute r = sqrt(x^2 + y^2 + z^2).
  let r2 = add(add(mul(x2, x2), mul(y2, y2)), mul(z2, z2))
  let r = sqrt(r2)
  let eps = 1e-10'f32
  let rSafe = maximum(r, bc(eps, r.shape))
  let ooR = divide(bc(1'f32, rSafe.shape), rSafe)
  # Normalized coordinates.
  let nx = mul(x2, ooR)
  let ny = mul(y2, ooR)
  let nz = mul(z2, ooR)
  # Powers of normalized z.
  let z2v = mul(nz, nz)
  let z3v = mul(nz, z2v)
  # x^2, y^2, xy.
  let x2v = mul(nx, nx)
  let y2v = mul(ny, ny)
  let xy = mul(nx, ny)
  # ---- l = 0 ----
  var h0: Tensor
  if lMax >= 0:
    h0 = bc(0.28209479177387814'f32, r.shape)
  # ---- l = 1 ----
  var h1: seq[Tensor] = @[]
  if lMax >= 1:
    let c1 = 0.4886025119029199'f32
    h1 = @[mul(bc(c1, r.shape), ny),   # Y_1^{-1}
            mul(bc(c1, r.shape), nz),   # Y_1^{0}
            mul(bc(c1, r.shape), nx)]   # Y_1^{1}
  # ---- l = 2 ----
  var h2: seq[Tensor] = @[]
  if lMax >= 2:
    h2 = @[
      mul(bc(1.0925484305920792'f32, r.shape), xy),              # Y_2^{-2}
      mul(bc(1.0925484305920792'f32, r.shape), mul(ny, nz)),     # Y_2^{-1}
      mul(bc(0.31539156525252005'f32, r.shape),
        sub(mul(bc(3'f32, r.shape), z2v), bc(1'f32, r.shape))),  # Y_2^{0}
      mul(bc(1.0925484305920792'f32, r.shape), mul(nx, nz)),     # Y_2^{1}
      mul(bc(0.5462742152960396'f32, r.shape), sub(x2v, y2v)),   # Y_2^{2}
    ]
  # ---- l = 3 ----
  var h3: seq[Tensor] = @[]
  if lMax >= 3:
    let xyf = sub(mul(bc(3'f32, r.shape), x2v), y2v)
    h3 = @[
      mul(bc(0.5900435899266435'f32, r.shape),
        mul(ny, xyf)),                                             # Y_3^{-3}
      mul(bc(2.890611442640554'f32, r.shape),
        mul(mul(nx, ny), nz)),                                     # Y_3^{-2}
      mul(bc(0.4570457994644658'f32, r.shape),
        mul(ny, sub(mul(bc(5'f32, r.shape), z2v),
          bc(1'f32, r.shape)))),                                   # Y_3^{-1}
      mul(bc(0.3731763325901154'f32, r.shape),
        sub(mul(bc(5'f32, r.shape), z3v),
          mul(bc(3'f32, r.shape), nz))),                           # Y_3^{0}
      mul(bc(0.4570457994644658'f32, r.shape),
        mul(nx, sub(mul(bc(5'f32, r.shape), z2v),
          bc(1'f32, r.shape)))),                                   # Y_3^{1}
      mul(bc(1.445305721320277'f32, r.shape),
        mul(nz, sub(x2v, y2v))),                                   # Y_3^{2}
      mul(bc(0.5900435899266435'f32, r.shape),
        mul(nx, sub(x2v, mul(bc(3'f32, r.shape), y2v)))),          # Y_3^{3}
    ]
  # ---- l = 4 ----
  var h4: seq[Tensor] = @[]
  if lMax >= 4:
    let z2s = mul(bc(7'f32, r.shape), z2v)
    let x2mY2 = sub(x2v, y2v)
    h4 = @[
      mul(bc(2.5033429417967046'f32, r.shape),
        mul(mul(nx, ny), x2mY2)),                                  # Y_4^{-4}
      mul(bc(1.7701307697799304'f32, r.shape),
        mul(mul(ny, nz), sub(mul(bc(7'f32, r.shape), x2v), y2v))), # Y_4^{-3}
      mul(bc(0.9461746957575601'f32, r.shape),
        mul(xy, sub(z2s, bc(1'f32, r.shape)))),                    # Y_4^{-2}
      mul(bc(0.6690465435572892'f32, r.shape),
        mul(ny, mul(nz, sub(z2s, bc(3'f32, r.shape))))),            # Y_4^{-1}
      mul(bc(0.10578554691520431'f32, r.shape),
        add(sub(mul(bc(35'f32, r.shape), mul(z2v, z2v)),
          mul(bc(30'f32, r.shape), z2v)),
          bc(3'f32, r.shape))),                                     # Y_4^{0}
      mul(bc(0.6690465435572892'f32, r.shape),
        mul(nx, mul(nz, sub(z2s, bc(3'f32, r.shape))))),            # Y_4^{1}
      mul(bc(0.47308734787878004'f32, r.shape),
        mul(x2mY2, sub(z2s, bc(1'f32, r.shape)))),                  # Y_4^{2}
      mul(bc(1.7701307697799304'f32, r.shape),
        mul(mul(nx, nz), sub(x2v, mul(bc(7'f32, r.shape), y2v)))),  # Y_4^{3}
      mul(bc(0.6258357354491761'f32, r.shape),
        sub(mul(x2v, x2v),
          add(mul(bc(6'f32, r.shape), mul(x2v, y2v)), mul(y2v, y2v)))), # Y_4^{4}
    ]
  # Concatenate all harmonics along the last dim.
  var all: seq[Tensor] = @[]
  all.add unsqueeze(h0, rank - 1)
  for h in h1: all.add unsqueeze(h, rank - 1)
  for h in h2: all.add unsqueeze(h, rank - 1)
  for h in h3: all.add unsqueeze(h, rank - 1)
  for h in h4: all.add unsqueeze(h, rank - 1)
  concat(all, rank - 1)
