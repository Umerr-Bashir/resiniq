from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from split_data import X_train, X_val, X_test, y_train, y_val, y_test

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = OUTPUT_DIR / "phase3_benchmark_results.csv"
SUMMARY_MD = OUTPUT_DIR / "phase3_benchmark.md"


@dataclass
class ModelSpec:
    name: str
    estimator: object
    scaled: bool = False


def stage_bucket(elapsed: pd.Series) -> pd.Series:
    bins = [0, 2, 4, 6, 8, np.inf]
    labels = ["0-2h", "2-4h", "4-6h", "6-8h", "8h+"]
    return pd.cut(elapsed, bins=bins, labels=labels, include_lowest=True, right=False)


def make_pipeline_or_model(spec: ModelSpec):
    if spec.scaled:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", spec.estimator),
            ]
        )
    return spec.estimator


def evaluate(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def main():
    models = [
        ModelSpec(
            name="MLP",
            scaled=True,
            estimator=MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=False,
                random_state=42,
            ),
        ),
        ModelSpec(
            name="RandomForest",
            estimator=RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                max_features=0.9,
                random_state=42,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            name="HistGradientBoosting",
            estimator=HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]

    if XGBRegressor is not None:
        models.append(
            ModelSpec(
                name="XGBoost",
                estimator=XGBRegressor(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=2,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
            )
        )

    rows = []
    stage_rows = []

    test_elapsed = X_test["elapsed"].reset_index(drop=True)
    test_stage = stage_bucket(test_elapsed)

    for spec in models:
        model = make_pipeline_or_model(spec)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_metrics = evaluate(y_val, val_pred)
        test_metrics = evaluate(y_test, test_pred)

        rows.append(
            {
                "model": spec.name,
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_r2": val_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )

        test_frame = pd.DataFrame(
            {
                "model": spec.name,
                "elapsed_stage": test_stage,
                "actual": y_test.reset_index(drop=True),
                "predicted": pd.Series(test_pred),
            }
        )
        test_frame["absolute_error"] = (test_frame["actual"] - test_frame["predicted"]).abs()
        stage_summary = (
            test_frame.groupby("elapsed_stage", observed=True)["absolute_error"]
            .agg(["mean", "count"])
            .reset_index()
        )
        stage_summary.insert(0, "model", spec.name)
        stage_rows.append(stage_summary)

    results = pd.DataFrame(rows).sort_values("test_mae").reset_index(drop=True)
    stage_results = pd.concat(stage_rows, ignore_index=True)

    results.to_csv(RESULTS_CSV, index=False)
    stage_results.to_csv(OUTPUT_DIR / "phase3_stage_mae.csv", index=False)

    best_model = results.iloc[0]

    md_lines = [
        "# Phase 3 Benchmark",
        "",
        "## Summary Table",
        "",
        "```text",
        results.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "## Best Holdout Model",
        "",
        f"- Model: {best_model['model']}",
        f"- Test MAE: {best_model['test_mae']:.4f} hours",
        f"- Test RMSE: {best_model['test_rmse']:.4f} hours",
        f"- Test R2: {best_model['test_r2']:.4f}",
        "",
        "## Notes",
        "",
        "- All models were trained on the same batch-disjoint training split.",
        "- Validation and test scores were computed on separate batch-disjoint sets.",
        "- Stage-wise MAE was computed on the holdout test set using elapsed-time buckets.",
    ]
    if XGBRegressor is None:
        md_lines.extend(
            [
                "",
                "- XGBoost was not benchmarked because the `xgboost` package is not installed in this environment.",
            ]
        )

    SUMMARY_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("Benchmark complete.")
    print(results.to_string(index=False))
    print("\nBest model:", best_model["model"])
    print("Results saved to:", RESULTS_CSV.name)
    print("Stage results saved to:", "phase3_stage_mae.csv")
    print("Summary saved to:", SUMMARY_MD.name)


if __name__ == "__main__":
    main()
