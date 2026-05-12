from fastapi import FastAPI

from .schemas import SessionData
from .predictor import predict_risk
from .features import prepare_single_session_features
from .decision import decide_action

app = FastAPI(
    title="Behavioral Anomaly Detection API"
)


@app.get("/")
def home():

    return {
        "message": "API is running"
    }


@app.post("/predict")
def predict(data: SessionData):

    try:

        # feature engineering
        features_df = prepare_single_session_features(data)

        # risk prediction
        risk_score = predict_risk(features_df)

        # action
        action = decide_action(risk_score)

        return {
            "risk_score": risk_score,
            "action": action
        }

    except Exception as e:

        return {
            "error": str(e)
        }