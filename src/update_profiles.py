import pandas as pd
import json

from logger import logger


def update_user_profiles():

    df = pd.read_csv("data/live_sessions.csv")

    profiles = {}

    users = df["user_id"].unique()

    for user in users:

        user_df = df[df["user_id"] == user]

        profiles[str(user)] = {

            "typing_speed_mean":
                float(user_df["typing_speed"].mean()),

            "typing_speed_std":
                float(user_df["typing_speed"].std()),

            "click_rate_mean":
                float(user_df["click_rate"].mean()),

            "click_rate_std":
                float(user_df["click_rate"].std()),

            "mouse_speed_mean":
                float(user_df["mouse_speed"].mean()),

            "mouse_speed_std":
                float(user_df["mouse_speed"].std())
        }

    with open("data/user_profiles.json", "w") as f:

        json.dump(profiles, f, indent=4)

    logger.info("User profiles updated")

    print("Profiles updated")