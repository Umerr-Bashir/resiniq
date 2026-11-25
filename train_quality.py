from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from split_data import FEATURES, load_dataset, make_splits


OUTPUT_PATH = Path(__file__).resolve().parent / "batch_quality_bundle.joblib"


def top_importances(model, feature_names, top_n=5):
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return []
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": values,
        }
    ).sort_values("importance", ascending=False)
    return importance.head(top_n).to_dict(orient="records")


def main():
    df = load_dataset()
    splits = make_splits(df)

    train_df = pd.concat([splits["train_df"], splits["val_df"]], axis=0).sort_index()
    test_df = splits["test_df"]

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_final_train = train_df["final_visc"]
    y_final_test = test_df["final_visc"]

    y_off_train = train_df["off_spec"]
    y_off_test = test_df["off_spec"]

    final_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=1,
    )
    final_model.fit(X_train, y_final_train)
    final_pred = final_model.predict(X_test)

    risk_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.9,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    risk_model.fit(X_train, y_off_train)
    risk_pred = risk_model.predict(X_test)
    risk_prob = risk_model.predict_proba(X_test)[:, 1]

    final_metrics = {
        "mae": float(mean_absolute_error(y_final_test, final_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_final_test, final_pred))),
    }

    risk_metrics = {
        "accuracy": float(accuracy_score(y_off_test, risk_pred)),
        "precision": float(precision_score(y_off_test, risk_pred, zero_division=0)),
        "recall": float(recall_score(y_off_test, risk_pred, zero_division=0)),
        "f1": float(f1_score(y_off_test, risk_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_off_test, risk_prob)),
    }

    final_quantiles = {
        "min": float(y_final_train.min()),
        "p05": float(y_final_train.quantile(0.05)),
        "p25": float(y_final_train.quantile(0.25)),
        "median": float(y_final_train.quantile(0.50)),
        "p75": float(y_final_train.quantile(0.75)),
        "p95": float(y_final_train.quantile(0.95)),
        "max": float(y_final_train.max()),
    }

    bundle = {
        "model_name": "RandomForest",
        "feature_names": FEATURES,
        "final_viscosity_model": final_model,
        "off_spec_model": risk_model,
        "final_viscosity_metrics": final_metrics,
        "off_spec_metrics": risk_metrics,
        "final_viscosity_quantiles": final_quantiles,
        "feature_importances": {
            "final_viscosity": top_importances(final_model, FEATURES),
            "off_spec": top_importances(risk_model, FEATURES),
        },
        "metadata": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_batches": int(train_df["batch_id"].nunique()),
            "test_batches": int(test_df["batch_id"].nunique()),
            "split_source": "batch-disjoint train+validation/test",
            "note": "Synthetic quality capability for demonstration only.",
        },
    }

    joblib.dump(bundle, OUTPUT_PATH)

    print("Saved batch quality bundle to:", OUTPUT_PATH.name)
    print("Final viscosity metrics:", final_metrics)
    print("Off-spec metrics:", risk_metrics)


if __name__ == "__main__":
    main()
