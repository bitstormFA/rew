## Pairwise distance computation for point clouds.
##
## Composites over existing arithmetic and linalg ops. All functions
## are trace-compatible.

import ../../tensor
import ../../ops/arith
import ../../ops/reduce
import ../../ops/shape
import ../../ops/linalg
import ../../ops/literal

proc pairwiseDistances2*(x: Tensor): Tensor =
  ## All-pairs squared Euclidean distances for a single point cloud.
  ##
  ## `x`: `[N, D]` coordinates. Returns `[N, N]` where result[i, j]
  ## is `||x_i - x_j||^2`.
  ##
  ## Uses the expansion:
  ##   ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T x_j
  if x.shape.len != 2:
    raise newException(TensorError,
      "pairwiseDistances2: expected [N, D], got " & $x.shape)
  let n = x.shape[0]
  let xSq = reduceSum(mul(x, x), @[1])  # [N]
  # x_i^T x_j via matmul: x @ x^T
  let xTx = matmul(x, transpose(x, @[1, 0]))  # [N, N]
  # Broadcast xSq to [N, N].
  let diagTerm = unsqueeze(xSq, 1)  # [N, 1]
  let diagB = broadcastTo(diagTerm, @[n, n], @[0, 1])
  let diagTB = broadcastTo(unsqueeze(xSq, 0), @[n, n], @[1, 0])
  # result = ||x||^2 + ||x||^2 - 2 * x^T x
  add(sub(diagB, mul(scalarF32(2'f32), xTx)), diagTB)

proc pairwiseDistances2*(x, y: Tensor): Tensor =
  ## Cross-set squared Euclidean distances.
  ##
  ## `x`: `[N, D]`, `y`: `[M, D]`. Returns `[N, M]`.
  if x.shape.len != 2 or y.shape.len != 2:
    raise newException(TensorError,
      "pairwiseDistances2: expected [N, D] and [M, D]")
  if x.shape[1] != y.shape[1]:
    raise newException(TensorError,
      "pairwiseDistances2: dimension mismatch (" &
        $x.shape[1] & " vs " & $y.shape[1] & ")")
  let n = x.shape[0]
  let m = y.shape[0]
  let xSq = reduceSum(mul(x, x), @[1])  # [N]
  let ySq = reduceSum(mul(y, y), @[1])  # [M]
  let xy = matmul(x, transpose(y, @[1, 0]))  # [N, M]
  let xSqB = broadcastTo(unsqueeze(xSq, 1), @[n, m], @[0, 1])
  let ySqB = broadcastTo(unsqueeze(ySq, 0), @[n, m], @[1, 0])
  add(sub(xSqB, mul(broadcastTo(scalarF32(2'f32), @[n, m], @[]), xy)), ySqB)
