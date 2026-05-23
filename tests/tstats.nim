## Classical stats and ML primitive tests.

import rew
import rew/dataframe

proc close(a, b: float64; eps = 1e-6): bool =
  abs(a - b) <= eps

let x: Matrix = @[
  @[1.0],
  @[2.0],
  @[3.0],
  @[4.0],
]
let y = @[2.0, 4.0, 6.0, 8.0]

block linear_regression_fits_closed_form:
  let model = initLinearRegression().fit(x, y)

  doAssert model.fitted
  doAssert close(model.coef[0], 2.0)
  doAssert close(model.intercept, 0.0)
  doAssert model.predict(@[@[5.0]])[0].close(10.0)

block dataframe_fit_uses_numeric_columns:
  let df = sql("""
      select * from (values
        (1.0::double, 2.0::double),
        (2.0::double, 4.0::double),
        (3.0::double, 6.0::double)
      ) as t(x, y)
    """)
  let model = initLinearRegression().fit(df, ["x"], "y")

  doAssert model.coef[0].close(2.0)

block ridge_and_lasso_have_estimator_state:
  let ridge = initRidge(alpha = 0.1).fit(x, y)
  let lasso = initLasso(alpha = 0.01, maxIter = 100).fit(x, y)

  doAssert ridge.fitted
  doAssert lasso.fitted
  doAssert ridge.coef.len == 1
  doAssert lasso.coef.len == 1

block logistic_regression_classifies_separable_data:
  let lx: Matrix = @[
    @[-2.0],
    @[-1.0],
    @[1.0],
    @[2.0],
  ]
  let ly = @[0.0, 0.0, 1.0, 1.0]
  let model = initLogisticRegression(lr = 0.5, maxIter = 300).fit(lx, ly)

  doAssert model.predict(lx) == @[0, 0, 1, 1]
  doAssert accuracy(@[0, 0, 1, 1], model.predict(lx)).close(1.0)

block standard_scaler_centers_columns:
  let scaler = initStandardScaler().fit(@[@[1.0, 10.0], @[3.0, 14.0]])
  let z = scaler.transform(@[@[1.0, 10.0], @[3.0, 14.0]])

  doAssert z[0][0].close(-1.0)
  doAssert z[1][1].close(1.0)

block pca_projects_to_requested_components:
  let pca = initPCA(1).fit(@[@[1.0, 1.0], @[2.0, 2.0], @[3.0, 3.0]])
  let projected = pca.transform(@[@[4.0, 4.0]])

  doAssert pca.components.len == 1
  doAssert projected.len == 1
  doAssert projected[0].len == 1

block metrics_splits_and_cross_validation:
  doAssert meanSquaredError(@[1.0, 2.0], @[1.0, 4.0]).close(2.0)
  doAssert rocAuc(@[0, 1, 0, 1], @[0.1, 0.9, 0.2, 0.8]).close(1.0)

  let split = trainTestSplit(@[1, 2, 3, 4], testSize = 0.5, seed = 7)
  doAssert split.train.len == 2
  doAssert split.test.len == 2

  let scores = crossValScore(initLinearRegression(), x, y, folds = 2)
  doAssert scores.len == 2
