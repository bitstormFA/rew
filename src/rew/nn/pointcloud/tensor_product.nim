## Clebsch-Gordan tensor product for SE(3)-equivariant networks.
##
## Implements the fully-connected tensor product used in NequIP and
## e3nn: contracts input irreps along their m-dimensions via CG
## coefficients and mixes channels with learned weights.
##
## The CG coefficients are precomputed host-side and loaded as
## constantF32 tensors. The forward pass uses `einsum` for the
## m-dimension contraction.

import std/math
import ../../tensor
import ../../pytree
import ../../dtype
import ../../rng
import ../../ops/literal
import ../../ops/arith
import ../../ops/concat
import ../../ops/factory
import ../../ops/linalg
import ../../ops/shape
import ../init

proc numIrrepChannels(l: int): int =
  ## Number of m-dimensions for degree l: 2*l + 1.
  2 * l + 1

type
  Irreps* = seq[int]
    ## A list of l values representing the irreducible representations.
    ## Example: [0, 1, 2] means one scalar (l=0), one vector (l=1),
    ## one tensor (l=2).

  TensorProduct* = object
    ## Fully-connected tensor product with learned weights.
    ##
    ## For each path (l1, l2 → l3), mixes the input channels via a
    ## linear weight and contracts over m-dimensions with CG coefficients.
    ##
    ## Forward:
    ##   out_{l3,m3,c_out} = Σ_{l1,l2,c_in} W_{l1,l2,l3,c_in,c_out} *
    ##     Σ_{m1,m2} CG(l1,m1,l2,m2,l3,m3) * in1_{l1,m1,c_in} * in2_{l2,m2,c_in}
    weights*: Param[Tensor]   ## [totalPaths, maxPathChannels, outChannels]
    cgCoeffs*: Buffer[Tensor] ## [totalPaths, 2*l1+1, 2*l2+1, 2*l3+1]
    inIrreps*: Irreps
    outIrreps*: Irreps
    sharedIrreps*: Irreps   ## irrep list for input2 (shared channel dim)
    totalChannels*: int
    outChannels*: int

proc cgCoefficient(l1, l2, l3, m1, m2, m3: int): float32 =
  ## Compute the Clebsch-Gordan coefficient <l1 m1; l2 m2 | l3 m3>.
  ##
  ## Uses the Wigner 3-j symbol formulation:
  ##   CG = (-1)^(l1-l2+m3) * sqrt(2*l3+1) * Wigner3j(l1,l2,l3, m1,m2,-m3)
  ##
  ## Implemented via the Racah formula for the 3-j symbol.
  ##
  ## Returns 0 for invalid combinations.
  if m1 + m2 != m3:
    return 0'f32
  if abs(m1) > l1 or abs(m2) > l2 or abs(m3) > l3:
    return 0'f32
  if l3 < abs(l1 - l2) or l3 > l1 + l2:
    return 0'f32
  # Wigner 3-j symbol via Racah formula.
  func factorial(n: int): float64 =
    result = 1.0
    for i in 2 .. n:
      result *= float64(i)
  func fact(n: int): float64 =
    factorial(n)
  let j1 = float64(l1)
  let j2 = float64(l2)
  let j3 = float64(l3)
  let mm1 = float64(m1)
  let mm2 = float64(m2)
  let mm3 = float64(-m3)
  let a = j1 + j2 - j3
  let b = j1 - j2 + j3
  let c = -j1 + j2 + j3
  if a < 0 or b < 0 or c < 0:
    return 0'f32
  let tmin = max(max(0.0, -j3 + j2 - mm1), -j3 + j1 + mm2).int
  let tmax = min(min(a.int, (j1 - mm1).int), (j2 + mm2).int)
  var sum = 0.0
  for t in tmin .. tmax:
    let term = pow(-1.0, float64(t)) /
      (fact(t) * fact(int(a) - t) * fact(int(j1 - float64(t) - mm1)) *
       fact(int(j2 + float64(t) + mm2)) *
       fact(int(j3 - j2 + float64(t) + mm1)) *
       fact(int(j3 - j1 - float64(t) - mm2)))
    sum += term
  let prefactor = pow(-1.0, j1 - j2 - mm3) * sqrt(2.0 * j3 + 1.0)
  let triangle = sqrt(
    fact(int(a)) * fact(int(b)) * fact(int(c)) /
    fact(int(j1 + j2 + j3 + 1.0)))
  let mag = sqrt(
    fact(int(j1 + mm1)) * fact(int(j1 - mm1)) *
    fact(int(j2 + mm2)) * fact(int(j2 - mm2)) *
    fact(int(j3 + mm3)) * fact(int(j3 - mm3)))
  let cg = prefactor * triangle * mag * sum
  float32(cg)

proc initTensorProduct*(key: Key; inIrreps1, inIrreps2, outIrreps: Irreps;
    outChannels: int): TensorProduct =
  ## Construct a fully-connected tensor product.
  ##
  ## `inIrreps1`, `inIrreps2`: lists of l values for each input.
  ## `outIrreps`: list of l values for the output.
  ## `outChannels`: number of output channels.
  ##
  ## A weight matrix is learned for each (l1, l2 → l3) path.
  if inIrreps1.len == 0 or inIrreps2.len == 0 or outIrreps.len == 0:
    raise newException(TensorError,
      "initTensorProduct: irreps must be non-empty")
  if outChannels <= 0:
    raise newException(TensorError,
      "initTensorProduct: outChannels must be positive")
  # Collect valid paths.
  type Path = tuple[l1, l2, l3: int]
  var paths: seq[Path] = @[]
  for l1 in inIrreps1:
    for l2 in inIrreps2:
      for l3 in outIrreps:
        if l3 >= abs(l1 - l2) and l3 <= l1 + l2:
          paths.add (l1, l2, l3)
  if paths.len == 0:
    raise newException(TensorError,
      "initTensorProduct: no valid (l1, l2, l3) paths found")
  # Compute total input channels.
  var totalInCh1 = 0
  for l in inIrreps1: totalInCh1 += numIrrepChannels(l)
  var totalInCh2 = 0
  for l in inIrreps2: totalInCh2 += numIrrepChannels(l)
  # Build one padded CG coefficient tensor for all paths.
  var maxM1 = 0
  var maxM2 = 0
  var maxM3 = 0
  for path in paths:
    maxM1 = max(maxM1, numIrrepChannels(path.l1))
    maxM2 = max(maxM2, numIrrepChannels(path.l2))
    maxM3 = max(maxM3, numIrrepChannels(path.l3))
  var cgData = newSeq[float32](paths.len * maxM1 * maxM2 * maxM3)
  for p, path in paths:
    let d1 = numIrrepChannels(path.l1)
    let d2 = numIrrepChannels(path.l2)
    let d3 = numIrrepChannels(path.l3)
    for m1 in 0 ..< d1:
      for m2 in 0 ..< d2:
        for m3 in 0 ..< d3:
          let vm1 = m1 - path.l1
          let vm2 = m2 - path.l2
          let vm3 = m3 - path.l3
          let idx = ((p * maxM1 + m1) * maxM2 + m2) * maxM3 + m3
          cgData[idx] = cgCoefficient(path.l1, path.l2, path.l3,
            vm1, vm2, vm3)
  let cgAll = constantF32(@[paths.len, maxM1, maxM2, maxM3], cgData)
  # Initialize weights: one linear mix per path.
  # Simplified: single weight matrix for all paths.
  let weightData = normalF32(key, paths.len * outChannels,
    0'f32, 1'f32 / sqrt(float32(paths.len)))
  TensorProduct(
    weights: param(constantF32(@[paths.len, outChannels], weightData)),
    cgCoeffs: buffer(cgAll),
    inIrreps: inIrreps1,
    outIrreps: outIrreps,
    sharedIrreps: inIrreps2,
    totalChannels: totalInCh1,
    outChannels: outChannels,
  )

proc irrepOffsets(irreps: Irreps): seq[int] =
  result = newSeq[int](irreps.len)
  var offset = 0
  for i, l in irreps:
    result[i] = offset
    offset += numIrrepChannels(l)

proc totalIrrepChannels(irreps: Irreps): int =
  for l in irreps:
    result += numIrrepChannels(l)

proc prefixShape(t: Tensor): seq[int] =
  for i in 0 ..< t.shape.len - 1:
    result.add t.shape[i]

proc sliceFeature(x: Tensor; index: int): Tensor =
  let rank = x.shape.len
  var starts = newSeq[int](rank)
  var limits = x.shape
  var strides = newSeq[int](rank)
  for i in 0 ..< rank:
    strides[i] = 1
  starts[rank - 1] = index
  limits[rank - 1] = index + 1
  squeeze(slice(x, starts, limits, strides), rank - 1)

proc sliceWeight(weights: Tensor; path, outChannel: int): Tensor =
  reshape(slice(weights, [path, outChannel],
    [path + 1, outChannel + 1], [1, 1]), [])

proc sliceCg(cg: Tensor; path, m1, m2, m3: int): Tensor =
  reshape(slice(cg, [path, m1, m2, m3],
    [path + 1, m1 + 1, m2 + 1, m3 + 1], [1, 1, 1, 1]), [])

proc scaleByScalar(x, scalar: Tensor): Tensor =
  if x.shape.len == 0:
    return mul(x, scalar)
  var dims: seq[int] = @[]
  mul(x, broadcastTo(scalar, x.shape, dims))

proc forward*(tp: TensorProduct; x1, x2: Tensor): Tensor =
  ## Forward pass of the tensor product.
  ##
  ## `x1`: `[..., in1Ch]` where in1Ch = Σ (2*l1+1).
  ## `x2`: `[..., in2Ch]` where in2Ch = Σ (2*l2+1).
  ## Returns: `[..., outCh]` where outCh = Σ (2*l3+1) * outChannels.
  let rank1 = x1.shape.len
  let rank2 = x2.shape.len
  if rank1 != rank2:
    raise newException(TensorError,
      "TensorProduct.forward: rank mismatch")
  if rank1 == 0:
    raise newException(TensorError,
      "TensorProduct.forward: inputs must have a feature dimension")
  if x1.shape[0 ..< rank1 - 1] != x2.shape[0 ..< rank2 - 1]:
    raise newException(TensorError,
      "TensorProduct.forward: batch/prefix shape mismatch")
  if x1.dtype != dtFloat32 or x2.dtype != dtFloat32:
    raise newException(TensorError,
      "TensorProduct.forward: inputs must be float32")
  requireSameMode(x1, x2, "TensorProduct.forward")
  requireSameDevice(x1, x2, "TensorProduct.forward")

  let in1Channels = totalIrrepChannels(tp.inIrreps)
  let in2Channels = totalIrrepChannels(tp.sharedIrreps)
  if x1.shape[^1] != in1Channels:
    raise newException(TensorError,
      "TensorProduct.forward: x1 last dim " & $x1.shape[^1] &
        " does not match irreps size " & $in1Channels)
  if x2.shape[^1] != in2Channels:
    raise newException(TensorError,
      "TensorProduct.forward: x2 last dim " & $x2.shape[^1] &
        " does not match irreps size " & $in2Channels)

  let prefix = prefixShape(x1)
  let in1Offsets = irrepOffsets(tp.inIrreps)
  let in2Offsets = irrepOffsets(tp.sharedIrreps)
  let outOffsets = irrepOffsets(tp.outIrreps)
  let outFeatures = totalIrrepChannels(tp.outIrreps) * tp.outChannels
  var outputs = newSeq[Tensor](outFeatures)
  for i in 0 ..< outputs.len:
    outputs[i] = zeros(prefix, dtFloat32, x1.device)

  var path = 0
  for i1, l1 in tp.inIrreps:
    let d1 = numIrrepChannels(l1)
    for i2, l2 in tp.sharedIrreps:
      let d2 = numIrrepChannels(l2)
      for i3, l3 in tp.outIrreps:
        if l3 >= abs(l1 - l2) and l3 <= l1 + l2:
          let d3 = numIrrepChannels(l3)
          for m3 in 0 ..< d3:
            var contracted = zeros(prefix, dtFloat32, x1.device)
            for m1 in 0 ..< d1:
              let v1 = sliceFeature(x1, in1Offsets[i1] + m1)
              for m2 in 0 ..< d2:
                let v2 = sliceFeature(x2, in2Offsets[i2] + m2)
                let term = mul(v1, v2)
                contracted = add(contracted,
                  scaleByScalar(term, sliceCg(tp.cgCoeffs, path,
                    m1, m2, m3)))
            for outChannel in 0 ..< tp.outChannels:
              let outIndex = (outOffsets[i3] + m3) * tp.outChannels +
                outChannel
              outputs[outIndex] = add(outputs[outIndex],
                scaleByScalar(contracted,
                  sliceWeight(tp.weights, path, outChannel)))
          inc path

  var expanded: seq[Tensor] = @[]
  for part in outputs:
    expanded.add unsqueeze(part, prefix.len)
  concat(expanded, prefix.len)
