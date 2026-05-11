## Global graph pooling — reduce node features to graph-level features.
##
## These are thin wrappers over the segment reduction ops. They
## aggregate all node features in each graph to produce a single
## graph-level representation.

import ../tensor
import ../ops/segment

proc globalAddPool*(x, batch: Tensor; numGraphs: int): Tensor =
  ## Sum pooling: sum node features per graph.
  ##
  ## `x` shape `[N_total, F]`, `batch` shape `[N_total]` int32 (graph
  ## assignment per node), `numGraphs` is the number of graphs.
  ## Returns `[numGraphs, F]`.
  segmentSum(x, batch, numGraphs)

proc globalMeanPool*(x, batch: Tensor; numGraphs: int): Tensor =
  ## Mean pooling: average node features per graph.
  segmentMean(x, batch, numGraphs)

proc globalMaxPool*(x, batch: Tensor; numGraphs: int): Tensor =
  ## Max pooling: per-channel max of node features per graph.
  segmentMax(x, batch, numGraphs)
