## Neural estimator wrappers over rew's typed training API.

import ../train
import ./core

type
  NeuralClassifier*[M, B, O, S] = object
    state*: TrainState[M, O, S]
    trainer*: Trainer
    fitted*: bool

  NeuralRegressor*[M, B, O, S] = object
    state*: TrainState[M, O, S]
    trainer*: Trainer
    fitted*: bool

func initNeuralClassifier*[M, B, O, S](state: TrainState[M, O, S];
    trainer: Trainer = initTrainer()): NeuralClassifier[M, B, O, S] =
  NeuralClassifier[M, B, O, S](state: state, trainer: trainer)

func initNeuralRegressor*[M, B, O, S](state: TrainState[M, O, S];
    trainer: Trainer = initTrainer()): NeuralRegressor[M, B, O, S] =
  NeuralRegressor[M, B, O, S](state: state, trainer: trainer)

proc fit*[M, B, O, S](model: NeuralClassifier[M, B, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]): NeuralClassifier[M, B, O, S] =
  ## Fits a neural classifier through `Trainer` and typed loss.
  result = model
  var trainer = result.trainer
  var state = result.state
  trainer.fit(state, data, loss)
  result.state = state
  result.trainer = trainer
  result.fitted = true

proc fit*[M, B, O, S](model: NeuralRegressor[M, B, O, S];
    data: DataSplits[B]; loss: LossFn[M, B]): NeuralRegressor[M, B, O, S] =
  ## Fits a neural regressor through `Trainer` and typed loss.
  result = model
  var trainer = result.trainer
  var state = result.state
  trainer.fit(state, data, loss)
  result.state = state
  result.trainer = trainer
  result.fitted = true

proc partialFit*[M, B, O, S](model: NeuralClassifier[M, B, O, S];
    data: DataSplits[B]; loss: LossFn[M, B];
    options: FitOptions = initFitOptions()): NeuralClassifier[M, B, O, S] =
  discard options
  model.fit(data, loss)

proc partialFit*[M, B, O, S](model: NeuralRegressor[M, B, O, S];
    data: DataSplits[B]; loss: LossFn[M, B];
    options: FitOptions = initFitOptions()): NeuralRegressor[M, B, O, S] =
  discard options
  model.fit(data, loss)
