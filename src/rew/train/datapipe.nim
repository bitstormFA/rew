## DataPipe — a bundle of Datasets for train/val/test/predict splits.
##
## No concept, no inheritance. Users compose pipelines with existing
## `rew/data` combinators and pass them to the Trainer.

import std/options
import ../data/dataset

type
  DataPipe*[T] = object
    train*: Dataset[T]
    val*: Option[Dataset[T]]
    test*: Option[Dataset[T]]
    predict*: Option[Dataset[T]]

func initDataPipe*[T](train: Dataset[T];
    val: Option[Dataset[T]] = none[Dataset[T]](),
    test: Option[Dataset[T]] = none[Dataset[T]](),
    predict: Option[Dataset[T]] = none[Dataset[T]]()): DataPipe[T] =
  ## Creates a DataPipe from individual datasets.
  ## Only `train` is required; `val`, `test`, and `predict` are optional.
  DataPipe[T](train: train, val: val, test: test, predict: predict)
