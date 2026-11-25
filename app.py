from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Resin Batch Intelligence",
    page_icon="test_tube",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(55, 84, 255, 0.18), transparent 28%),
            radial-gradient(circle at 85% 10%, rgba(0, 164, 138, 0.14), transparent 22%),
            linear-gradient(180deg, #f5f7fb 0%, #eef2f7 48%, #e9eef4 100%);
        color: #0f172a;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.4rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .hero-shell {
        border: 1px solid rgba(15, 23, 42, 0.08);
        background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(245,247,251,0.96));
        border-radius: 24px;
        padding: 1.4rem 1.5rem 1.2rem 1.5rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }

    .eyebrow {
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #475569;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.05;
        color: #0f172a;
        margin-bottom: 0.35rem;
    }

    .hero-copy {
        font-size: 1rem;
        line-height: 1.6;
        color: #334155;
        max-width: 55rem;
    }

    .scenario-chip {
        display: inline-block;
        padding: 0.42rem 0.7rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.06);
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.45rem;
        margin-top: 0.35rem;
    }

    .demo-banner {
        border-left: 4px solid #0f766e;
        background: linear-gradient(90deg, rgba(15, 118, 110, 0.12), rgba(15, 118, 110, 0.03));
        border-radius: 16px;
        padding: 0.95rem 1rem;
        margin: 0.9rem 0 1rem 0;
    }

    .demo-banner strong {
        color: #0f172a;
    }

    .section-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.85rem 1rem;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        color: #64748b;
        margin-bottom: 0.35rem;
        font-weight: 700;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.6rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .metric-subtext {
        color: #475569;
        font-size: 0.86rem;
        margin-top: 0.2rem;
    }

    .callout {
        background: linear-gradient(135deg, rgba(15,118,110,0.11), rgba(255,255,255,0.9));
        border: 1px solid rgba(15, 118, 110, 0.18);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        color: #0f172a;
    }

    .callout h4 {
        margin: 0 0 0.25rem 0;
        font-size: 1rem;
    }

    .callout p {
        margin: 0;
        color: #334155;
    }

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_time_artifact():
    bundle_path = Path("resin_bundle.joblib")
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        return {
            "model": bundle["model"],
            "feature_names": bundle["feature_names"],
            "alpha": bundle.get("alpha", 0.10),
            "quantile_radius": bundle.get("quantile_radius"),
            "metadata": bundle.get("metadata", {}),
            "model_name": bundle.get("model_name", "Unknown"),
        }

    legacy_paths = [
        Path("resin_model_v2.joblib"),
        Path("resin_model.joblib"),
    ]
    for path in legacy_paths:
        if path.exists():
            return {
                "model": joblib.load(path),
                "feature_names": [
                    "elapsed",
                    "av_init",
                    "temp_set",
                    "catalyst",
                    "glycerol_ratio",
                    "purity_meas",
                    "lab_av",
                    "lab_av_prev",
                    "lab_visc",
                    "torque",
                    "torque_slope",
                    "temp_last",
                    "temp_mean",
                    "water_cum",
                    "water_rate",
                ],
                "alpha": None,
                "quantile_radius": None,
                "metadata": {},
                "model_name": path.name,
            }

    raise FileNotFoundError(
        "No saved time model found. Expected resin_bundle.joblib, resin_model_v2.joblib, or resin_model.joblib."
    )


@st.cache_resource
def load_quality_artifact():
    bundle_path = Path("batch_quality_bundle.joblib")
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        return {
            "final_model": bundle["final_viscosity_model"],
            "risk_model": bundle["off_spec_model"],
            "feature_names": bundle["feature_names"],
            "final_metrics": bundle.get("final_viscosity_metrics", {}),
            "risk_metrics": bundle.get("off_spec_metrics", {}),
            "final_quantiles": bundle.get("final_viscosity_quantiles", {}),
            "feature_importances": bundle.get("feature_importances", {}),
            "metadata": bundle.get("metadata", {}),
        }

    return None


time_artifact = load_time_artifact()
quality_artifact = load_quality_artifact()

time_model = time_artifact["model"]
time_features = time_artifact["feature_names"]
time_radius = time_artifact["quantile_radius"]
time_alpha = time_artifact["alpha"]
time_metadata = time_artifact["metadata"]
time_model_name = time_artifact["model_name"]
feature_ranges = time_metadata.get("feature_ranges", {})


st.title("Resin Batch Intelligence")
st.markdown(
    """
    <div class="hero-shell">
      <div class="eyebrow">Synthetic resin operations demo</div>
      <div class="hero-title">AI batch review for reaction time, viscosity, and quality risk</div>
      <div class="hero-copy">
        This dashboard presents an experimental industrial AI workflow for an in-progress resin batch.
        It uses synthetic data honestly, keeps uncertainty visible, and frames each prediction as a decision aid rather than a production alarm.
      </div>
      <div style="margin-top: 0.8rem;">
        <span class="scenario-chip">Batch #RB-1042</span>
        <span class="scenario-chip">In progress</span>
        <span class="scenario-chip">Remaining time</span>
        <span class="scenario-chip">Final viscosity</span>
        <span class="scenario-chip">Off-spec risk</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="demo-banner">
      <strong>Example scenario:</strong> Resin Batch #RB-1042 is in progress. The demo uses synthetic data and is labeled as experimental.
    </div>
    """,
    unsafe_allow_html=True,
)

DEMO_VALUES = {
    "elapsed": 4.0,
    "av_init": 81.31962747007948,
    "temp_set": 230.27042199802904,
    "catalyst": 0.1357212495599913,
    "glycerol_ratio": 0.9475020052735784,
    "purity_meas": 0.9662766623189384,
    "lab_av": 37.21243155105216,
    "lab_av_prev": 40.9316487460312,
    "lab_visc": 549.9323767323907,
    "torque": 32.99850754875148,
    "torque_slope": 2.671204359195162,
    "temp_last": 229.25385195529907,
    "temp_mean": 229.83247200950848,
    "water_cum": 52.86976687852784,
    "water_rate": 7.018357342752523,
}

if st.button("Load synthetic demo batch", use_container_width=True):
    for key, value in DEMO_VALUES.items():
        st.session_state[f"input_{key}"] = value
    st.rerun()

with st.sidebar:
    st.header("Model Summary")
    st.write(f"Time model: {time_model_name}")
    if time_alpha is not None:
        st.write(f"Time interval coverage: {int((1 - time_alpha) * 100)}%")
    if "model_mae_test" in time_metadata:
        st.write(f"Time holdout MAE: {time_metadata['model_mae_test']:.3f} hours")
    if quality_artifact:
        st.write("Quality model: RandomForest")
        fm = quality_artifact["final_metrics"]
        rm = quality_artifact["risk_metrics"]
        if fm:
            st.write(f"Final viscosity MAE: {fm.get('mae', 0):.1f} cP")
        if rm:
            st.write(f"Off-spec ROC AUC: {rm.get('roc_auc', 0):.3f}")
    st.caption("Synthetic-data demonstration only.")


st.divider()
st.subheader("1. Remaining Reaction Time Prediction")
st.write(
    "Enter the latest measurements from an ongoing resin batch to estimate how much reaction time remains."
)

features = time_features
values = {}

with st.form("batch_prediction_form"):
    st.subheader("Current batch measurements")
    col1, col2, col3 = st.columns(3)

    for i, feature in enumerate(features):
        column = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
        with column:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0,
                format="%.4f",
                key=f"input_{feature}",
            )

    submitted = st.form_submit_button(
        "Generate Batch AI Review",
        type="primary",
        use_container_width=True,
    )


def make_warning_text(values_map):
    warnings = []
    for feature, value in values_map.items():
        limits = feature_ranges.get(feature)
        if not limits:
            continue
        if value < limits["min"] or value > limits["max"]:
            warnings.append(
                f"{feature.replace('_', ' ').title()} is outside the training range "
                f"({limits['min']:.4f} to {limits['max']:.4f})."
            )
        elif value < limits["p05"] or value > limits["p95"]:
            warnings.append(
                f"{feature.replace('_', ' ').title()} is near the edge of the training distribution."
            )
    return warnings


def render_viscosity_chart(predicted_value: float):
    if not quality_artifact:
        return
    q = quality_artifact["final_quantiles"]
    if not q:
        return

    bands = pd.DataFrame(
        [
            {"band": "p05-p95", "x": q["p05"], "x2": q["p95"], "y": 0},
            {"band": "p25-p75", "x": q["p25"], "x2": q["p75"], "y": 0},
        ]
    )
    rule = pd.DataFrame([{"value": predicted_value, "y": 0}])

    chart = (
        alt.Chart(bands)
        .mark_bar(size=28)
        .encode(
            x=alt.X("x:Q", title="Final viscosity (cP)"),
            x2="x2:Q",
            y=alt.Y("y:O", axis=None),
            color=alt.Color(
                "band:N",
                scale=alt.Scale(domain=["p05-p95", "p25-p75"], range=["#DCEBFF", "#8CB9FF"]),
                legend=None,
            ),
        )
        .properties(height=70)
    )
    chart = chart + alt.Chart(rule).mark_rule(color="#B00020", size=3).encode(x="value:Q", y="y:O")
    st.altair_chart(chart, use_container_width=True)


def render_metric_card(label: str, value: str, subtext: str, accent: str = "#0f172a"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-top: 4px solid {accent};">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if submitted:
    X = pd.DataFrame([values], columns=features)
    warnings = make_warning_text(values)

    try:
        remaining_pred = float(time_model.predict(X)[0])
        if time_radius is not None:
            remaining_lower = max(0.0, remaining_pred - time_radius)
            remaining_upper = max(0.0, remaining_pred + time_radius)
            lower_completion = datetime.now() + timedelta(hours=remaining_lower)
            upper_completion = datetime.now() + timedelta(hours=remaining_upper)
        else:
            remaining_lower = remaining_upper = None
            lower_completion = upper_completion = None

        st.divider()
        st.subheader("AI Prediction Results")

        if warnings:
            st.warning("\n".join(warnings))

        if remaining_pred > 0:
            completion_time = datetime.now() + timedelta(hours=remaining_pred)
            hours = int(remaining_pred)
            minutes = int((remaining_pred - hours) * 60)

            result1, result2 = st.columns(2)
            with result1:
                render_metric_card(
                    "Estimated Remaining Time",
                    f"{remaining_pred:.2f} hours",
                    "Conformal point estimate for the current batch state.",
                    accent="#0f766e",
                )
            with result2:
                render_metric_card(
                    "Expected Completion",
                    completion_time.strftime("%I:%M %p"),
                    "Based on the current local clock.",
                    accent="#1d4ed8",
                )

            if remaining_lower is not None and remaining_upper is not None:
                interval1, interval2 = st.columns(2)
                with interval1:
                    render_metric_card(
                        "90% Prediction Interval",
                        f"{remaining_lower:.2f} to {remaining_upper:.2f} hours",
                        "Split conformal interval from batch-disjoint calibration.",
                        accent="#7c3aed",
                    )
                with interval2:
                    render_metric_card(
                        "Completion Window",
                        f"{lower_completion.strftime('%I:%M %p')} - {upper_completion.strftime('%I:%M %p')}",
                        "Likely completion range if conditions stay similar.",
                        accent="#b45309",
                    )

            st.success(
                f"The model estimates approximately {hours} hours and {minutes} minutes remaining until the reaction endpoint."
            )

            if remaining_lower is not None and remaining_upper is not None:
                st.info(
                    "The interval comes from split conformal calibration on a batch-disjoint validation set."
                )
        else:
            st.warning(
                "The model predicts zero or negative remaining time. The batch may have reached its endpoint, but this requires confirmation."
            )

        st.divider()
        st.subheader("2. Final Viscosity and Off-Spec Risk")

        if quality_artifact:
            final_model = quality_artifact["final_model"]
            risk_model = quality_artifact["risk_model"]
            final_quantiles = quality_artifact["final_quantiles"]
            importances = quality_artifact["feature_importances"]

            final_pred = float(final_model.predict(X)[0])
            risk_prob = float(risk_model.predict_proba(X)[0][1])
            risk_label = "High" if risk_prob >= 0.60 else "Elevated" if risk_prob >= 0.25 else "Low"

            q75 = final_quantiles.get("p75", final_pred)
            q95 = final_quantiles.get("p95", final_pred)

            left, right = st.columns(2)
            with left:
                render_metric_card(
                    "Predicted Final Viscosity",
                    f"{final_pred:,.0f} cP",
                    "Regression forecast for the synthetic endpoint quality.",
                    accent="#7c3aed",
                )
                render_viscosity_chart(final_pred)
                st.caption(
                    f"Training distribution reference: 75th percentile {q75:,.0f} cP, 95th percentile {q95:,.0f} cP."
                )
            with right:
                render_metric_card(
                    "Off-Spec Risk",
                    risk_label,
                    "Classifier output mapped to an industrial risk call.",
                    accent="#dc2626" if risk_label != "Low" else "#0f766e",
                )
                st.progress(min(max(risk_prob, 0.0), 1.0))
                st.caption(f"Estimated off-spec probability: {risk_prob:.0%}")

            if importances:
                top_quality_features = importances.get("off_spec", [])[:3]
                if top_quality_features:
                    st.write("Most influential signals in the risk model:")
                    st.dataframe(
                        pd.DataFrame(top_quality_features),
                        hide_index=True,
                        use_container_width=True,
                    )

            if risk_label in ("Elevated", "High") or final_pred >= q95:
                insight = (
                    "The current batch is progressing toward its expected endpoint. "
                    "However, the predicted final viscosity is approaching the upper specification limit. "
                    "Review the current reaction conditions and monitor subsequent laboratory measurements."
                )
            elif risk_prob >= 0.35:
                insight = (
                    "The batch appears to be progressing normally, but the quality model sees a moderate off-spec signal. "
                    "Continue monitoring temperature, torque, and the next lab update."
                )
            else:
                insight = (
                    "The batch is on track and the predicted final viscosity remains comfortably within the observed synthetic operating band."
                )

            st.info("Process insight")
            st.write(insight)

        else:
            st.warning("The batch quality model bundle is not available yet.")

        st.info(
            "Demonstration based on synthetic training data. Predictions have not been validated against actual industrial batch records."
        )

        with st.expander("Model details", expanded=False):
            st.write(f"Time model: {time_model_name}")
            if time_alpha is not None:
                st.write(f"Time interval nominal coverage: {int((1 - time_alpha) * 100)}%")
            if "model_mae_test" in time_metadata:
                st.write(f"Time holdout MAE: {time_metadata['model_mae_test']:.3f} hours")
            if "model_r2_test" in time_metadata:
                st.write(f"Time holdout R2: {time_metadata['model_r2_test']:.3f}")
            st.write(
                f"Batch-disjoint split: {time_metadata.get('split_source', 'unknown')}"
            )
            if quality_artifact:
                st.write(f"Final viscosity holdout MAE: {quality_artifact['final_metrics'].get('mae', 0):.1f} cP")
                st.write(f"Off-spec holdout ROC AUC: {quality_artifact['risk_metrics'].get('roc_auc', 0):.3f}")
                st.write(
                    f"Quality split: {quality_artifact['metadata'].get('split_source', 'unknown')}"
                )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()
st.caption("AI Resin Batch Intelligence | Experimental Demonstration")
