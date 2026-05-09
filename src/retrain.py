import pandas as pd

from sklearn.ensemble import IsolationForest
from joblib import dump

from logger import logger
def retrain_model():

    df = pd.read_csv("data/live_sessions.csv")

    features = [

        "typing_speed",
        "avg_key_delay",
        "click_rate",
        "mouse_speed",
        "session_duration",
        "idle_time"
    ]

    X = df[features]

    model = IsolationForest(

        contamination=0.05,
        random_state=42
    )

    model.fit(X)

    dump(model, "src/models/isolation_forest.pkl")
    logger.info("Model retrained successfully")
    print("Model retrained")