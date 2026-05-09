import json
from logger import logger


def detect_drift(user_id, new_typing_mean):

    with open("data/user_profiles.json", "r") as f:

        profiles = json.load(f)

    old_mean = profiles[str(user_id)]["typing_speed_mean"]

    difference = abs(new_typing_mean - old_mean)

    if difference > 30:

        logger.warning(
            f"Behavior drift detected for user {user_id}"
        )

        return True

    return False