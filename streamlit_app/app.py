import sys
import os

# Fix import paths
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from src.features import prepare_single_session_features
from src.decision import decide_action

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Behavioral Anomaly Detection",
    layout="wide"
)

st.title("🚀 Behavioral Anomaly Detection System")
st.write("User-Specific Behavioral Intelligence")

# =========================================================
# LOAD MODEL + SCALER
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("src/models/isolation_forest.pkl")


@st.cache_resource
def load_scaler():
    return joblib.load("src/models/scaler.pkl")


model = load_model()
scaler = load_scaler()

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.header("🧠 User Session Input")

user_id = st.sidebar.selectbox(
    "Select User",
    [1, 2, 3, 4, 5]
)

typing_speed = st.sidebar.slider(
    "Typing Speed",
    50,
    400,
    180
)

avg_key_delay = st.sidebar.slider(
    "Average Key Delay",
    0.01,
    1.0,
    0.12
)

click_rate = st.sidebar.slider(
    "Click Rate",
    0.1,
    10.0,
    2.5
)

mouse_speed = st.sidebar.slider(
    "Mouse Speed",
    100,
    2000,
    400
)

session_duration = st.sidebar.slider(
    "Session Duration",
    10,
    1000,
    300
)

idle_time = st.sidebar.slider(
    "Idle Time",
    0,
    300,
    20
)

# =========================================================
# SIMULATE ATTACK BUTTON
# =========================================================

if st.sidebar.button("🚨 Simulate Attack"):

    typing_speed = 350
    avg_key_delay = 0.9
    click_rate = 9.0
    mouse_speed = 1800
    session_duration = 30
    idle_time = 0

    st.sidebar.warning("Attack values loaded!")

# =========================================================
# CREATE INPUT DATAFRAME
# =========================================================

input_data = pd.DataFrame([{
    "user_id": user_id,
    "typing_speed": typing_speed,
    "avg_key_delay": avg_key_delay,
    "click_rate": click_rate,
    "mouse_speed": mouse_speed,
    "session_duration": session_duration,
    "idle_time": idle_time
}])

# =========================================================
# FEATURE ENGINEERING
# =========================================================

try:
    features = prepare_single_session_features(input_data)

except Exception as e:
    st.error(f"Feature Engineering Error: {e}")
    st.stop()

# =========================================================
# SCALE FEATURES
# =========================================================

try:
    scaled_features = scaler.transform(features)

except Exception as e:
    st.error(f"Scaling Error: {e}")
    st.stop()

# =========================================================
# PREDICTION
# =========================================================

try:
    anomaly_score = model.decision_function(
        scaled_features
    )[0]

    # Normalize score to risk (0 → 1)
    risk_score = 1 - ((anomaly_score + 1) / 2)

    # Clamp values
    risk_score = max(0, min(1, risk_score))

except Exception as e:
    st.error(f"Prediction Error: {e}")
    st.stop()

# =========================================================
# DECISION SYSTEM
# =========================================================

action = decide_action(risk_score)

# =========================================================
# MAIN OUTPUT
# =========================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Risk Analysis")

    st.metric(
        "Risk Score",
        f"{risk_score:.2f}"
    )

    st.progress(float(risk_score))

with col2:

    st.subheader("🚦 Recommended Action")

    if action == "ALLOW":
        st.success(action)

    elif action == "MONITOR":
        st.warning(action)

    elif action == "WARN":
        st.error(action)

    else:
        st.error("🚨 REAUTHENTICATE")

# =========================================================
# USER BASELINE COMPARISON
# =========================================================

st.subheader("📈 User Baseline Comparison")

baseline_data = {
    "Feature": [
        "Typing Speed",
        "Avg Key Delay",
        "Click Rate",
        "Mouse Speed"
    ],
    "Current Session": [
        typing_speed,
        avg_key_delay,
        click_rate,
        mouse_speed
    ],
    "Typical User Average": [
        180,
        0.12,
        2.5,
        400
    ]
}

baseline_df = pd.DataFrame(baseline_data)

st.dataframe(
    baseline_df,
    use_container_width=True
)

# =========================================================
# FEATURE VISUALIZATION
# =========================================================

st.subheader("📉 Behavioral Features")

chart_data = pd.DataFrame({
    "Feature": [
        "Typing Speed",
        "Click Rate",
        "Mouse Speed",
        "Idle Time"
    ],
    "Value": [
        typing_speed,
        click_rate,
        mouse_speed,
        idle_time
    ]
})

st.bar_chart(
    chart_data.set_index("Feature")
)

# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

st.subheader("🔍 Explainability (SHAP)")

try:

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        scaled_features
    )

    st.write(
        "Feature contributions to anomaly detection:"
    )

    fig, ax = plt.subplots()

    shap.summary_plot(
        shap_values,
        scaled_features,
        show=False
    )

    st.pyplot(fig)

except Exception as e:

    st.warning(
        f"SHAP visualization unavailable: {e}"
    )

# =========================================================
# SESSION HISTORY (SIMULATED)
# =========================================================

st.subheader("🕒 Session Risk History")

history_df = pd.DataFrame({
    "risk_score": [
        0.12,
        0.20,
        0.35,
        0.50,
        risk_score
    ]
})

st.line_chart(history_df)

# =========================================================
# FINAL JSON OUTPUT
# =========================================================

st.subheader("📦 API-style Output")

output = {
    "user_id": int(user_id),
    "risk_score": round(risk_score, 2),
    "action": action,
    "message": (
        "Behavior deviates from normal pattern"
        if risk_score > 0.6
        else "Behavior appears normal"
    )
}

st.json(output)