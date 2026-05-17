## DataSplits — a bundle of datasets for train/val/test/predict subsets.
##
## Users compose data manipulation flows with `rew/data` combinators, then
## group the resulting datasets here for the Trainer.

import std/options
import ../data/dataset
import ../data/transform
import ../tensor
import ../pytree
import ../eager
import ./runtime

type
  DataSplits*[T] = object
    train*: Dataset[T]
    val*: Option[Dataset[T]]
    test*: Option[Dataset[T]]
    predict*: Option[Dataset[T]]

func initDataSplits*[T](train: Dataset[T];
    val: Option[Dataset[T]] = none[Dataset[T]](),
    test: Option[Dataset[T]] = none[Dataset[T]](),
    predict: Option[Dataset[T]] = none[Dataset[T]]()): DataSplits[T] =
  ## Creates dataset splits from individual datasets.
  ## Only `train` is required; `val`, `test`, and `predict` are optional.
  DataSplits[T](train: train, val: val, test: test, predict: predict)

proc toDevice*[T](ds: Dataset[T]; runtime: Runtime): Dataset[T] =
  ## Maps every tensor leaf yielded by `ds` onto `runtime.device`.
  ds.map(proc(item: T): T =
    treeMap(item, proc(t: Tensor): Tensor = t.to(runtime.device)))
