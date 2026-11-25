# Phase 1 Audit: Dataset, Target, and Existing Pipeline

## Scope

Audited the current synthetic resin dataset and the existing training/inference scripts:

- `data/rows.csv`
- `split_data.py`
- `train_mlp.py`
- `evaluate_model.py`
- `baseline.py`
- `save_model.py`
- `app.py`
- `validate.py`
- `drift_test.py`

## Dataset Snapshot

- Rows: `5,217`
- Batches: `500`
- Columns: `19`
- Missing values: none
- Duplicate/identical row issues: none observed during the audit

### Columns

- `batch_id`
- `elapsed`
- `av_init`
- `temp_set`
- `catalyst`
- `glycerol_ratio`
- `purity_meas`
- `lab_av`
- `lab_av_prev`
- `lab_visc`
- `torque`
- `torque_slope`
- `temp_last`
- `temp_mean`
- `water_cum`
- `water_rate`
- `time_remaining`
- `final_visc`
- `off_spec`

Note:

- `final_visc` is constant within each batch and appears to be a post-batch outcome label.
- `off_spec` is also constant within each batch and is binary (`0`/`1`).
- Neither is used as an input feature in the current prediction pipeline.

## Target Definition

The target is `time_remaining`.

Key finding:

- Within every batch, `time_remaining + elapsed` is constant up to floating-point precision.
- That means the label is a batch-specific countdown to the endpoint, not an independent noisy measurement.
- The label is effectively derived from the batch endpoint time:
  - `time_remaining = endpoint_elapsed - elapsed`
  - where `endpoint_elapsed` is constant for a batch and `elapsed` does not start at zero in this dataset

Observed target stats:

- Min: `0.000154`
- Median: `2.687968`
- Mean: `3.205142`
- Max: `12.446963`

## Feature Audit

### Batch-constant features

These are constant within each batch:

- `av_init`
- `temp_set`
- `catalyst`
- `glycerol_ratio`

### Time-varying features

These change across rows within a batch:

- `elapsed`
- `purity_meas`
- `lab_av`
- `lab_av_prev`
- `lab_visc`
- `torque`
- `torque_slope`
- `temp_last`
- `temp_mean`
- `water_cum`
- `water_rate`

### Potential leakage review

No direct target columns are included in the feature set.

However, several feature names suggest they may encode current or rolling process state:

- `lab_av_prev`
- `temp_mean`
- `water_cum`
- `water_rate`
- `torque_slope`

Those are not necessarily leakage by themselves, but they should be treated carefully in later phases to ensure they are available at inference time using only information up to the current measurement.

## Existing Split Strategy

`split_data.py` already uses `GroupShuffleSplit` with `batch_id` as the grouping key.

Current behavior:

- First split: 85% train/validation, 15% test
- Second split: train/validation are split again by batch
- Batch overlap between train, validation, and test is prevented by design
- Actual split sizes in the current dataset:
  - Train: `3,665` rows across `350` batches
  - Validation: `780` rows across `75` batches
  - Test: `772` rows across `75` batches

Audit conclusion:

- The current split avoids the most serious leakage risk of having the same batch appear in multiple splits.
- This is a good start, but the current workflow does not yet preserve a dedicated calibration set for conformal prediction.

## Existing Modeling Pipeline

### Training

- `train_mlp.py` trains a `StandardScaler` + `MLPRegressor` pipeline.
- Scaling is fit inside the pipeline, so it is learned only on the training data used in `fit()`.

### Evaluation

- `evaluate_model.py` reports MAE, RMSE, and an actual-vs-predicted scatter plot.
- `baseline.py` computes a simple mean predictor as a baseline.

### Dashboard

- `app.py` loads `resin_model.joblib`
- It accepts the 15 feature inputs used in training
- It returns a point prediction and a simple estimated completion time

## Findings

1. The dataset is clean in the narrow sense: no missing values and no obvious schema problems.
2. The target is a synthetic countdown label that is mathematically tied to `elapsed` within each batch.
3. Batch-level grouping is already used in the split code, which avoids row-level leakage across splits.
4. The current pipeline does not yet provide:
   - a true holdout workflow with a separate calibration set
   - benchmarked model comparison
   - prediction intervals
   - metadata saved with the trained model
5. The biggest risk to watch in later phases is temporal leakage through feature availability, not just batch overlap.

## Phase 1 Conclusion

The current repository is a workable prototype, but it is still only a baseline demonstration.

What is solid already:

- grouped batch splitting
- simple model training
- clean synthetic data

What still needs improvement:

- clearer separation of train / validation / calibration / holdout
- stricter inference-time feature availability checks
- model benchmarking
- uncertainty estimation
- richer dashboard metadata and warnings
