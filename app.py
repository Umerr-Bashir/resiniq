from datetime import datetime, timedelta, timezone
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score

from split_data import load_dataset, make_splits

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

    section[data-testid="stMain"],
    section[data-testid="stMain"] * {
        color: #0f172a !important;
    }

    section[data-testid="stMain"] .stCaption,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"],
    section[data-testid="stMain"] .stMarkdown,
    section[data-testid="stMain"] .stMarkdown p,
    section[data-testid="stMain"] .stMarkdown strong,
    section[data-testid="stMain"] .stAlert,
    section[data-testid="stMain"] .stAlert p,
    section[data-testid="stMain"] .stAlert div,
    section[data-testid="stMain"] .stAlert li {
        color: #0f172a !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.4rem;
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
    .input-label {
        margin: 0.05rem 0 0.35rem 0;
        color: #0f172a;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    div[data-testid="stNumberInput"] label {
        color: #0f172a !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #0f172a;
        color: #f8fafc;
        border: 1px solid #0f172a;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.14);
        transition: all 0.2s ease;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #111827;
        color: #f8fafc;
        border-color: #111827;
        box-shadow: 0 14px 28px rgba(15, 23, 42, 0.18);
        filter: none;
    }

    div[data-testid="stButton"] > button:active,
    div[data-testid="stFormSubmitButton"] > button:active {
        background: #0b1220;
        color: #f8fafc;
        border-color: #0b1220;
    }

    div[data-testid="stButton"] > button:focus-visible,
    div[data-testid="stFormSubmitButton"] > button:focus-visible {
        outline: 2px solid #38bdf8;
        outline-offset: 2px;
    }
    div[data-testid="stFormSubmitButton"] > button {
        background: #b91c1c;
        color: #ffffff;
        border: 1px solid #b91c1c;
        transition: all 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #991b1b;
        color: #ffffff;
        border-color: #991b1b;
        filter: none;
    }

    div[data-testid="stFormSubmitButton"] > button:active {
        background: #7f1d1d;
        color: #ffffff;
        border-color: #7f1d1d;
    }
    [data-testid="stHeader"] {
        background: #f8fafc;
        color: #0f172a;
        border-bottom: 1px solid rgba(15, 23, 42, 0.08);
    }

    [data-testid="stHeader"] * {
        color: #0f172a !important;
    }

    [data-testid="stToolbar"] {
        background: #f8fafc;
        color: #0f172a;
    }

    [data-testid="stToolbar"] * {
        color: #0f172a !important;
    }
     /* Dark GitHub icon in the top bar */
    [data-testid="stHeader"] svg,
    [data-testid="stHeader"] img,
    [data-testid="stToolbar"] svg,
    [data-testid="stToolbar"] img {
        filter: brightness(0) saturate(100%);
    }

    /* If the icon is inside a link/button, keep it dark there too */
    [data-testid="stHeader"] a svg,
    [data-testid="stHeader"] a img,
    [data-testid="stToolbar"] a svg,
    [data-testid="stToolbar"] a img {
        filter: brightness(0) saturate(100%);
    }

    /* Fork button with dark border */
    button:has(span:contains("Fork")),
    [data-testid="stButton"] button:has(span:contains("Fork")) {
        border: 1px solid #00000 !important;
        color: #00000 !important;
        background: transparent !important;
    }

    /* Fallback for buttons that don't support :has() well */
    [data-testid="stButton"] button {
        border-color: #0f172a;
    }
    [data-testid="stHeader"] svg,
    [data-testid="stHeader"] img,
    [data-testid="stToolbar"] svg,
    [data-testid="stToolbar"] img {
        filter: brightness(0) saturate(100%) !important;
        opacity: 1 !important;
    }

    [data-testid="stHeader"] a,
    [data-testid="stToolbar"] a {
        color: #000000 !important;
    }
</style>
    """,
    unsafe_allow_html=True,
)


def sync_browser_clock():
    st.html(
        """
        <script>
        (function() {
          const updateClock = () => {
            const params = new URLSearchParams(window.location.search);
            params.set("browser_now_ms", String(Date.now()));
            params.set("browser_tz_offset_min", String(new Date().getTimezoneOffset()));
            const nextUrl = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
            window.history.replaceState({}, "", nextUrl);
          };

          updateClock();

          if (!window.__browserClockSyncInstalled) {
            window.__browserClockSyncInstalled = true;
            const refresh = () => updateClock();
            ["click", "change", "input", "keydown", "pointerdown"].forEach((eventName) => {
              document.addEventListener(eventName, refresh, true);
            });
          }
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def get_browser_now() -> datetime:
    browser_now_ms = st.query_params.get("browser_now_ms")
    browser_tz_offset_min = st.query_params.get("browser_tz_offset_min")

    if browser_now_ms is not None and browser_tz_offset_min is not None:
        try:
            utc_now = datetime.fromtimestamp(float(browser_now_ms) / 1000.0, tz=timezone.utc)
            browser_tz = timezone(timedelta(minutes=-int(float(browser_tz_offset_min))))
            return utc_now.astimezone(browser_tz)
        except (TypeError, ValueError):
            pass

    return datetime.now().astimezone()


FINAL_VISC_SPEC_LOWER = 1600.0
FINAL_VISC_SPEC_UPPER = 3200.0
FINAL_VISC_SPEC_MARGIN = (FINAL_VISC_SPEC_UPPER - FINAL_VISC_SPEC_LOWER) * 0.15


FEATURE_LABELS = {
    "elapsed": "Elapsed time (h)",
    "av_init": "Initial acid value (mg KOH/g)",
    "temp_set": "Set temperature (°F)",
    "catalyst": "Catalyst dose (fraction)",
    "glycerol_ratio": "Glycerol ratio (ratio)",
    "purity_meas": "Purity, lab (fraction)",
    "lab_av": "Acid value, lab (mg KOH/g)",
    "lab_av_prev": "Previous acid value, lab (mg KOH/g)",
    "lab_visc": "Viscosity, lab (cP)",
    "torque": "Torque (synthetic units)",
    "torque_slope": "Torque slope (synthetic units/h)",
    "temp_last": "Last temperature (°F)",
    "temp_mean": "Mean temperature (°F)",
    "water_cum": "Water cumulative (synthetic units)",
    "water_rate": "Water rate (synthetic units/h)",
}


def feature_display_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def classify_viscosity(predicted_value: float) -> tuple[str, str]:
    if predicted_value < FINAL_VISC_SPEC_LOWER:
        return (
            "below lower limit",
            f"Below the lower specification limit of {FINAL_VISC_SPEC_LOWER:,.0f} cP.",
        )
    if predicted_value <= FINAL_VISC_SPEC_LOWER + FINAL_VISC_SPEC_MARGIN:
        return (
            "near lower limit",
            f"Near the lower specification limit of {FINAL_VISC_SPEC_LOWER:,.0f} cP.",
        )
    if predicted_value >= FINAL_VISC_SPEC_UPPER:
        return (
            "above upper limit",
            f"Above the upper specification limit of {FINAL_VISC_SPEC_UPPER:,.0f} cP.",
        )
    if predicted_value >= FINAL_VISC_SPEC_UPPER - FINAL_VISC_SPEC_MARGIN:
        return (
            "near upper limit",
            f"Near the upper specification limit of {FINAL_VISC_SPEC_UPPER:,.0f} cP.",
        )
    return (
        "within spec",
        f"Within the {FINAL_VISC_SPEC_LOWER:,.0f} to {FINAL_VISC_SPEC_UPPER:,.0f} cP specification band.",
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


@st.cache_data(show_spinner=False)
def load_quality_evaluation():
    if not quality_artifact:
        return None

    df = load_dataset()
    splits = make_splits(df)
    train_df = pd.concat([splits["train_df"], splits["val_df"]], axis=0).sort_index()
    test_df = splits["test_df"]

    X_test = test_df[time_features]
    y_final_test = test_df["final_visc"]
    y_off_test = test_df["off_spec"]

    final_model = quality_artifact["final_model"]
    risk_model = quality_artifact["risk_model"]

    final_pred = final_model.predict(X_test)
    risk_prob = risk_model.predict_proba(X_test)[:, 1]
    risk_alarm = (risk_prob >= 0.5).astype(int)
    baseline_pred = np.full(len(y_final_test), float(train_df["final_visc"].mean()))

    return {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_batches": int(train_df["batch_id"].nunique()),
        "test_batches": int(test_df["batch_id"].nunique()),
        "final_mae": float(mean_absolute_error(y_final_test, final_pred)),
        "final_rmse": float(np.sqrt(mean_squared_error(y_final_test, final_pred))),
        "baseline_mae": float(mean_absolute_error(y_final_test, baseline_pred)),
        "risk_auc": float(roc_auc_score(y_off_test, risk_prob)),
        "risk_precision": float(precision_score(y_off_test, risk_alarm, zero_division=0)),
        "risk_recall": float(recall_score(y_off_test, risk_alarm, zero_division=0)),
        "alarm_threshold": 0.5,
    }


time_artifact = load_time_artifact()
quality_artifact = load_quality_artifact()

time_model = time_artifact["model"]
time_features = time_artifact["feature_names"]
time_radius = time_artifact["quantile_radius"]
time_alpha = time_artifact["alpha"]
time_metadata = time_artifact["metadata"]
time_model_name = time_artifact["model_name"]
feature_ranges = time_metadata.get("feature_ranges", {})


sync_browser_clock()

st.title("Resin Batch Intelligence")
st.markdown(
    """
    <div class="hero-shell">
      <div class="eyebrow">Synthetic resin operations</div>
      <div class="hero-title">Batch review for reaction time, viscosity, and quality risk</div>
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
            st.markdown(f'<div class="input-label">{feature_display_label(feature)}</div>', unsafe_allow_html=True)
            values[feature] = st.number_input(
                feature_display_label(feature),
                value=0.0,
                format="%.4f",
                key=f"input_{feature}",
                label_visibility="collapsed",
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
                f"{feature_display_label(feature)} is outside the training range "
                f"({limits['min']:.4f} to {limits['max']:.4f})."
            )
        elif value < limits["p05"] or value > limits["p95"]:
            warnings.append(
                f"{feature_display_label(feature)} is near the edge of the training distribution."
            )
    return warnings


def render_viscosity_chart(predicted_value: float):
    if not quality_artifact:
        return
    q = quality_artifact["final_quantiles"]
    if not q:
        return

    status, _ = classify_viscosity(predicted_value)
    marker_color = "#0f766e" if status == "within spec" else "#b45309" if "near" in status else "#b91c1c"
    chart_min = min(q["min"], FINAL_VISC_SPEC_LOWER) - 100
    chart_max = max(q["max"], FINAL_VISC_SPEC_UPPER) + 100

    observed_range = pd.DataFrame([{"x": q["min"], "x2": q["max"], "y": 0}])
    spec_band = pd.DataFrame([{"x": FINAL_VISC_SPEC_LOWER, "x2": FINAL_VISC_SPEC_UPPER, "y": 0}])
    limit_rules = pd.DataFrame(
        [
            {"value": FINAL_VISC_SPEC_LOWER, "label": "Lower limit", "y": 0},
            {"value": FINAL_VISC_SPEC_UPPER, "label": "Upper limit", "y": 0},
        ]
    )
    marker = pd.DataFrame([{"value": predicted_value, "y": 0}])

    base = alt.Chart(observed_range).mark_bar(size=24, cornerRadius=8, color="#E5E7EB").encode(
        x=alt.X("x:Q", title="Final viscosity (cP)", scale=alt.Scale(domain=[chart_min, chart_max])),
        x2="x2:Q",
        y=alt.Y("y:O", axis=None),
    )
    spec = alt.Chart(spec_band).mark_bar(size=24, cornerRadius=8, color="#8FD6B3").encode(
        x="x:Q",
        x2="x2:Q",
        y=alt.Y("y:O", axis=None),
    )
    limits = alt.Chart(limit_rules).mark_rule(color="#64748b", strokeDash=[6, 4], strokeWidth=1.5).encode(
        x="value:Q",
        y="y:O",
        tooltip=[alt.Tooltip("label:N", title="Limit"), alt.Tooltip("value:Q", title="cP", format=",.0f")],
    )
    pred = alt.Chart(marker).mark_rule(color=marker_color, strokeWidth=4).encode(x="value:Q", y="y:O")
    pred_point = alt.Chart(marker).mark_point(color=marker_color, filled=True, size=160, shape="diamond").encode(
        x="value:Q",
        y="y:O",
        tooltip=[alt.Tooltip("value:Q", title="Prediction", format=",.0f")],
    )

    chart = (base + spec + limits + pred + pred_point).properties(height=92).configure_view(strokeOpacity=0).configure_axis(grid=False)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Grey bar = observed training range, green bar = 1,600 to 3,200 cP spec band, diamond = model prediction.")


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
    browser_now = get_browser_now()

    try:
        remaining_pred = float(time_model.predict(X)[0])
        if time_radius is not None:
            remaining_lower = max(0.0, remaining_pred - time_radius)
            remaining_upper = max(0.0, remaining_pred + time_radius)
            lower_completion = browser_now + timedelta(hours=remaining_lower)
            upper_completion = browser_now + timedelta(hours=remaining_upper)
        else:
            remaining_lower = remaining_upper = None
            lower_completion = upper_completion = None

        st.divider()
        st.subheader("AI Prediction Results")

        if warnings:
            st.warning("\n".join(warnings))

        if remaining_pred > 0:
            completion_time = browser_now + timedelta(hours=remaining_pred)
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

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.divider()
        st.subheader("2. Final Viscosity and Off-Spec Risk")

        if quality_artifact:
            evaluation = load_quality_evaluation()
            final_model = quality_artifact["final_model"]
            risk_model = quality_artifact["risk_model"]
            importances = quality_artifact["feature_importances"]

            final_pred = float(final_model.predict(X)[0])
            risk_prob = float(risk_model.predict_proba(X)[0][1])
            risk_label = "High" if risk_prob >= 0.60 else "Elevated" if risk_prob >= 0.25 else "Low"

            left, right = st.columns(2)
            with left:
                render_metric_card(
                    "Predicted Final Viscosity",
                    f"{final_pred:,.0f} cP",
                    f"Forecast against the {FINAL_VISC_SPEC_LOWER:,.0f} to {FINAL_VISC_SPEC_UPPER:,.0f} cP specification band.",
                    accent="#7c3aed",
                )
                render_viscosity_chart(final_pred)
                viscosity_state, viscosity_message = classify_viscosity(final_pred)
                st.caption(f"{viscosity_message} Prediction status: {viscosity_state}.")
            with right:
                render_metric_card(
                    "Off-Spec Risk",
                    risk_label,
                    "Classifier output mapped to an alarm probability.",
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

            if viscosity_state in ("below lower limit", "above upper limit"):
                insight = (
                    f"The current batch is progressing toward its expected endpoint. {viscosity_message} "
                    "That is an out-of-spec forecast, so review the current reaction conditions and monitor the next lab sample closely."
                )
            elif viscosity_state in ("near lower limit", "near upper limit"):
                insight = (
                    f"The current batch is progressing toward its expected endpoint. {viscosity_message} "
                    "The forecast is still within spec, but it is close enough to the boundary that the next lab check matters."
                )
            elif risk_prob >= 0.35:
                insight = (
                    f"The current batch is progressing toward its expected endpoint. {viscosity_message} "
                    "The viscosity forecast is acceptable, but the quality model still sees a moderate off-spec signal. Continue monitoring temperature, torque, and the next lab update."
                )
            else:
                insight = (
                    f"The current batch is progressing toward its expected endpoint. {viscosity_message} "
                    "The batch looks on track from both the viscosity forecast and the off-spec probability."
                )

            st.info("Process insight")
            st.write(insight)

        else:
            st.warning("The batch quality model bundle is not available yet.")

        st.info(
            "Demonstration based on synthetic training data. Predictions have not been validated against actual industrial batch records."
        )

        with st.expander("Model details", expanded=False):
            if evaluation:
                st.markdown(
                    f"""
                    - **Data set:** {evaluation['train_rows'] + evaluation['test_rows']:,} synthetic rows across {evaluation['train_batches'] + evaluation['test_batches']:,} batches.
                    - **How it was generated:** the app is built from synthetic batch snapshots, where each row captures the current process state plus the downstream labels for remaining time, final viscosity, and off-spec risk.
                    - **Split discipline:** whole batches stayed together during training and testing, so the held-out batches were never seen during fitting.
                    - **Regression result:** final viscosity MAE {evaluation['final_mae']:.1f} cP versus a naive mean baseline of {evaluation['baseline_mae']:.1f} cP, with RMSE {evaluation['final_rmse']:.1f} cP.
                    - **Classification result:** ROC AUC {evaluation['risk_auc']:.3f}. At the {evaluation['alarm_threshold']:.2f} alarm threshold, precision is {evaluation['risk_precision']:.3f} and recall is {evaluation['risk_recall']:.3f}.
                    - **What AUC means:** 0.819 means the classifier ranks an off-spec case above an in-spec case about 82% of the time; it does not describe how noisy one alarm threshold will be.
                    - **Limitations:** synthetic-only data, no plant validation, and threshold performance is sensitive to class balance and the chosen alarm cutoff.
                    """
                )
            else:
                st.write("Evaluation metrics are unavailable.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.divider()
st.caption("AI Resin Batch Intelligence | Experimental Demonstration")
