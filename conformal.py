from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from split_data import (
    FEATURES,
    GROUP_COL,
    RANDOM_STATE,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    train_batches,
    val_batches,
    test_batches,
)

OUTPUT_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = OUTPUT_DIR / "resin_bundle.joblib"
RESULTS_PATH = OUTPUT_DIR / "phase4_conformal_results.json"
STAGE_RESULTS_PATH = OUTPUT_DIR / "phase4_conformal_stage_coverage.csv"


@dataclass
class ConformalBundle:
    model_name: str
    alpha: float
    quantile_radius: float
    feature_names: list[str]
    model: object
    metadata: dict


def stage_bucket(elapsed: pd.Series) -> pd.Series:
    bins = [0, 2, 4, 6, 8, np.inf]
    labels = ["0-2h", "2-4h", "4-6h", "6-8h", "8h+"]
    return pd.cut(elapsed, bins=bins, labels=labels, include_lowest=True, right=False)


def conformal_radius(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    if n == 0:
        raise ValueError("Calibration scores are empty.")
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(scores)[rank - 1])


def evaluate_intervals(y_true, pred, radius):
    lower = pred - radius
    upper = pred + radius
    coverage = np.mean((y_true >= lower) & (y_true <= upper))
    width = float(np.mean(upper - lower))
    return lower, upper, coverage, width


def build_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def main():
    alpha = 0.10

    model = build_model()
    model.fit(X_train, y_train)

    calib_pred = model.predict(X_val)
    calib_scores = np.abs(y_val.values - calib_pred)
    radius = conformal_radius(calib_scores, alpha)

    test_pred = model.predict(X_test)
    lower, upper, coverage, width = evaluate_intervals(y_test.values, test_pred, radius)

    test_frame = pd.DataFrame(
        {
            "elapsed": X_test["elapsed"].reset_index(drop=True),
            "actual": y_test.reset_index(drop=True),
            "predicted": pd.Series(test_pred),
            "lower": pd.Series(lower),
            "upper": pd.Series(upper),
        }
    )
    test_frame["elapsed_stage"] = stage_bucket(test_frame["elapsed"])
    test_frame["covered"] = (test_frame["actual"] >= test_frame["lower"]) & (
        test_frame["actual"] <= test_frame["upper"]
    )

    stage_coverage = (
        test_frame.groupby("elapsed_stage", observed=True)
        .agg(
            coverage=("covered", "mean"),
            count=("covered", "size"),
            avg_width=("upper", lambda s: float((s - test_frame.loc[s.index, "lower"]).mean())),
            mae=("actual", lambda s: float(mean_absolute_error(s, test_frame.loc[s.index, "predicted"]))),
        )
        .reset_index()
    )

    bundle = ConformalBundle(
        model_name="RandomForest",
        alpha=alpha,
        quantile_radius=radius,
        feature_names=FEATURES,
        model=model,
        metadata={
            "random_state": RANDOM_STATE,
            "train_batches": len(train_batches),
            "calibration_batches": len(val_batches),
            "test_batches": len(test_batches),
            "train_rows": len(X_train),
            "calibration_rows": len(X_val),
            "test_rows": len(X_test),
            "coverage_target": 1 - alpha,
            "split_source": "batch-disjoint train/calibration/test",
            "model_mae_test": float(mean_absolute_error(y_test, test_pred)),
            "model_rmse_test": float(np.sqrt(mean_squared_error(y_test, test_pred))),
            "model_r2_test": float(r2_score(y_test, test_pred)),
            "feature_ranges": {
                feature: {
                    "min": float(X_train[feature].min()),
                    "max": float(X_train[feature].max()),
                    "p05": float(X_train[feature].quantile(0.05)),
                    "p95": float(X_train[feature].quantile(0.95)),
                }
                for feature in FEATURES
            },
        },
    )

    # Store as a plain dict so joblib can persist it easily.
    import joblib

    joblib.dump(asdict(bundle), BUNDLE_PATH)

    result = {
        "model_name": bundle.model_name,
        "alpha": alpha,
        "radius": radius,
        "target_coverage": 1 - alpha,
        "test_coverage": float(coverage),
        "average_interval_width": width,
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "test_r2": float(r2_score(y_test, test_pred)),
    }

    RESULTS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    stage_coverage.to_csv(STAGE_RESULTS_PATH, index=False)

    print("Conformal bundle saved to:", BUNDLE_PATH.name)
    print(json.dumps(result, indent=2))
    print("\nStage coverage:")
    print(stage_coverage.to_string(index=False))


if __name__ == "__main__":
    main()
