from trusted_sessions import is_trusted_session
from collect_new_sessions import store_session
from drift_detection import detect_drift
from retrain import retrain_model

from datetime import datetime
from logger import logger

import pandas as pd


def process_session(session_data):

    trusted = is_trusted_session(
        session_data["risk_score"]
    )

    if trusted:

        # Store trusted session
        store_session(session_data)

        print("Trusted session accepted")

        logger.info("Trusted session stored")

        # Drift Detection

        new_mean = session_data["typing_speed"]

        drift = detect_drift(
            session_data["user_id"],
            new_mean
        )

        if drift:

            print("Behavior drift detected")

            logger.warning(
                f"Behavior drift detected for user {session_data['user_id']}"
            )

        # Automatic Retraining

        df = pd.read_csv("data/live_sessions.csv")

        if len(df) % 100 == 0:

            retrain_model()

            logger.info(
                "Automatic retraining triggered"
            )

    else:

        print("Suspicious session rejected")

        logger.warning(
            "Suspicious session rejected"
        )



# TEST SESSION

session = {

    "user_id": 1,
    "typing_speed": 185,
    "avg_key_delay": 0.11,
    "click_rate": 2.5,
    "mouse_speed": 400,
    "session_duration": 320,
    "idle_time": 20,
    "risk_score": 0.12,
    "timestamp": datetime.now()
}

process_session(session)