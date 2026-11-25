# Phase 7: Final Viscosity and Off-Spec Risk

## Capability Added

- Predicted final viscosity from the current batch measurements
- Estimated off-spec risk from the same batch state
- AI-generated process insight for industrial review
- Optional synthetic demo batch preload

## Model Setup

- Final viscosity: `RandomForestRegressor`
- Off-spec risk: `RandomForestClassifier`
- Training data: batch-disjoint training plus validation batches
- Evaluation data: untouched test batches

## Holdout Results

- Final viscosity MAE: `207.4 cP`
- Final viscosity RMSE: `262.5 cP`
- Off-spec accuracy: `0.759`
- Off-spec precision: `0.382`
- Off-spec recall: `0.403`
- Off-spec F1: `0.392`
- Off-spec ROC AUC: `0.819`

## Demo Behavior

- The dashboard now includes a one-click synthetic demo batch loader.
- The preloaded example is intended to show an in-progress batch with:
  - elapsed time around `4.0 h`
  - a remaining-time estimate
  - a final-viscosity forecast
  - an elevated off-spec signal

## Visuals

- Remaining time interval
- Final viscosity band chart against the synthetic training distribution
- Off-spec risk meter
- Model summary in the sidebar

## Notes

- These are experimental synthetic outputs only.
- The off-spec signal is a learned probability, not a production alarm.
- The process insight is explanatory text generated from the model outputs and should not be treated as operator guidance.

