## DataPipe — a bundle of Datasets for train/val/test/predict splits.
##
## No concept, no inheritance. Users compose pipelines with existing
## `rew/data` combinators and pass them to the Trainer.

import std/options
import ../data/dataset
import ../data/transform
import ../tensor
import ../pytree
import ../eager
import ./workbench

type
  DataPipe*[T] = object
    train*: Dataset[T]
    val*: Option[Dataset[T]]
    test*: Option[Dataset[T]]
    predict*: Option[Dataset[T]]

  DataSplits*[T] = DataPipe[T]
    ## Public high-level name for train/validation/test/predict datasets.

func initDataPipe*[T](train: Dataset[T];
    val: Option[Dataset[T]] = none[Dataset[T]](),
    test: Option[Dataset[T]] = none[Dataset[T]](),
    predict: Option[Dataset[T]] = none[Dataset[T]]()): DataPipe[T] =
  ## Creates a DataPipe from individual datasets.
  ## Only `train` is required; `val`, `test`, and `predict` are optional.
  DataPipe[T](train: train, val: val, test: test, predict: predict)

func initDataSplits*[T](train: Dataset[T];
    val: Option[Dataset[T]] = none[Dataset[T]](),
    test: Option[Dataset[T]] = none[Dataset[T]](),
    predict: Option[Dataset[T]] = none[Dataset[T]]()): DataSplits[T] =
  ## Creates dataset splits from individual datasets.
  initDataPipe(train, val, test, predict)

proc toDevice*[T](ds: Dataset[T]; runtime: Runtime): Dataset[T] =
  ## Maps every tensor leaf yielded by `ds` onto `runtime.device`.
  ds.map(proc(item: T): T =
    treeMap(item, proc(t: Tensor): Tensor = t.to(runtime.device)))
