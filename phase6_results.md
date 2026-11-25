# Phase 6: Tests and Results

## What Was Verified

- Dataset integrity checks in `validate.py`
- Batch-disjoint split integrity
- Model benchmarking on the shared split
- Conformal interval generation and coverage
- Python syntax compilation for the updated code
- Bundle loading and prediction smoke test

## Test Results

### 1. Split and dataset validation

Command:

```bash
python validate.py
```

Result:

- `500` batches
- batch size range: `3` to `25` rows
- `time_remaining` increases within a batch: `0`
- `final_visc` inconsistent batches: `0`
- `off_spec` inconsistent batches: `0`
- train / validation / test batch overlap: `0`

### 2. Benchmark comparison

Command:

```bash
python benchmark_models.py
```

Holdout results:

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| RandomForest | `0.3856` | `0.6608` | `0.9092` |
| MLP | `0.3931` | `0.6395` | `0.9150` |
| HistGradientBoosting | `0.3977` | `0.6954` | `0.8994` |

Best holdout MAE:

- `RandomForest`

### 3. Conformal prediction

Command:

```bash
python conformal.py
```

Result:

- Conformal radius: `0.7837` hours
- Average interval width: `1.5674` hours
- Test coverage: `0.8834`
- Target coverage: `0.9000`

Stage coverage on the holdout set:

- `0-2h`: `0.7733`
- `2-4h`: `0.8310`
- `4-6h`: `0.9252`
- `6-8h`: `0.9531`
- `8h+`: `0.9692`

### 4. Syntax checks

Command:

```bash
python -m py_compile app.py conformal.py split_data.py benchmark_models.py train_mlp.py save_model.py validate.py
```

Result:

- Passed without syntax errors.

### 5. Bundle smoke test

Command:

```bash
python - <<'PY'
import joblib
from split_data import X_test
bundle = joblib.load("resin_bundle.joblib")
model = bundle["model"]
print(model.predict(X_test.head(5)))
print(bundle["quantile_radius"])
PY
```

Result:

- Predictions returned successfully.
- Conformal radius loaded successfully.

## How To Run

1. Rebuild the conformal bundle:

```bash
python conformal.py
```

2. Launch the dashboard:

```bash
streamlit run app.py
```

## Remaining Limitations

- The data is synthetic, so performance is not industrial validation.
- The conformal interval undercovers slightly on the holdout set.
- Early elapsed-stage coverage is weaker than later-stage coverage.
- The OOD warning is range-based, not a true novelty detector.
- `xgboost` is not installed in this environment, so it was not benchmarked.

## Recommended Next Steps

1. Add a proper calibration-aware evaluation split report.
2. Investigate why early-stage conformal coverage is low.
3. Decide whether to keep RandomForest or select another model after more tuning.
4. Add automated regression tests for the split and bundle-loading path.

