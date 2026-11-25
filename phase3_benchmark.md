# Phase 3 Benchmark

## Summary Table

```text
               model  validation_mae  validation_rmse  validation_r2  test_mae  test_rmse  test_r2
        RandomForest          0.3552           0.5061         0.9579    0.3856     0.6608   0.9092
                 MLP          0.3530           0.5155         0.9563    0.3931     0.6395   0.9150
HistGradientBoosting          0.3124           0.4378         0.9685    0.3977     0.6954   0.8994
```

## Best Holdout Model

- Model: RandomForest
- Test MAE: 0.3856 hours
- Test RMSE: 0.6608 hours
- Test R2: 0.9092

## Notes

- All models were trained on the same batch-disjoint training split.
- Validation and test scores were computed on separate batch-disjoint sets.
- Stage-wise MAE was computed on the holdout test set using elapsed-time buckets.

- XGBoost was not benchmarked because the `xgboost` package is not installed in this environment.