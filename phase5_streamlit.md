# Phase 5 Streamlit Integration

## What Changed

- The dashboard now prefers `resin_bundle.joblib`, which includes:
  - the fitted model
  - the conformal interval radius
  - feature ordering
  - holdout metrics
  - training-range metadata
- The UI now shows:
  - point prediction
  - 90% prediction interval
  - completion window
  - model summary in the sidebar
  - detailed model metadata in an expander
- The app now warns when inputs fall:
  - outside the training range
  - near the edge of the training distribution

## Behavior

- If the conformal bundle is present, the dashboard shows uncertainty intervals.
- If only the older model file is present, it falls back to point prediction only.
- The dashboard remains a single-screen Streamlit app and does not add new dependencies.

## Notes

- The warnings are a simple range-based sanity check, not a full novelty detector.
- The interval is still the split-conformal interval calibrated on the validation split.
- This remains a synthetic-data demonstration and should not be treated as production validation.

