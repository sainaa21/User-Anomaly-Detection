import joblib
import pandas as pd

# load model
model = joblib.load("src/models/isolation_forest.pkl")

# load scaler
scaler = joblib.load("src/models/scaler.pkl")


def predict_risk(features_df):

    scaled = scaler.transform(features_df)

    score = model.decision_function(scaled)[0]

    # convert anomaly score → risk score
    risk_score = round(1 - ((score + 0.5) / 1.0), 2)

    # clamp between 0 and 1
    risk_score = max(0, min(risk_score, 1))

    return risk_score