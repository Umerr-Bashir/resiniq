# Phase 4 Conformal Prediction

## Setup

- Base model: `RandomForestRegressor`
- Training set: batch-disjoint `train` split
- Calibration set: batch-disjoint `validation` split
- Holdout set: untouched `test` split
- Nominal interval coverage: `90%` (`alpha = 0.10`)

## Conformal Method

Used split conformal prediction with symmetric absolute-residual intervals:

- Fit the model on the training split.
- Predict the calibration split.
- Compute absolute residuals on calibration rows.
- Use the finite-sample conformal quantile to get a single symmetric radius.
- Build intervals as `prediction ± radius`.

## Results

- Conformal radius: `0.7837` hours
- Average interval width: `1.5674` hours
- Test coverage: `0.8834`
- Holdout MAE: `0.3856` hours
- Holdout RMSE: `0.6608` hours
- Holdout R²: `0.9092`

## Stage-Wise Coverage

| Elapsed bucket | Coverage | Rows |
|---|---:|---:|
| `0-2h` | `0.7733` | `75` |
| `2-4h` | `0.8310` | `290` |
| `4-6h` | `0.9252` | `214` |
| `6-8h` | `0.9531` | `128` |
| `8h+` | `0.9692` | `65` |

## Notes

- Coverage is weaker in the earliest elapsed-stage buckets.
- Later-stage intervals are more reliable and mostly exceed the nominal target.
- This is still a synthetic-data demonstration, so the interval behavior should not be treated as industrial validation.

