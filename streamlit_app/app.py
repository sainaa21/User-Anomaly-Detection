import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Behavioral Anomaly Detection",
    layout="wide"
)

st.title("🚀 Behavioral Anomaly Detection System")
st.write("User-Specific Behavioral Intelligence")

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

if st.sidebar.button("🚨 Simulate Attack"):

    typing_speed = 350
    avg_key_delay = 0.9
    click_rate = 9.0
    mouse_speed = 1800
    session_duration = 30
    idle_time = 0

    st.sidebar.warning("Attack values loaded!")

input_data = {
    "user_id": int(user_id),
    "typing_speed": float(typing_speed),
    "avg_key_delay": float(avg_key_delay),
    "click_rate": float(click_rate),
    "mouse_speed": float(mouse_speed),
    "session_duration": float(session_duration),
    "idle_time": float(idle_time)
}

try:

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    result = response.json()

    risk_score = result["risk_score"]

    action = result["action"]

except Exception as e:

    st.error(f"API Connection Error: {e}")
    st.stop()

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

st.subheader("🔍 Explainability")

if risk_score > 0.8:

    st.error(
        "Strong behavioral anomaly detected"
    )

elif risk_score > 0.6:

    st.warning(
        "Behavior deviates from normal pattern"
    )

else:

    st.success(
        "Behavior appears normal"
    )

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