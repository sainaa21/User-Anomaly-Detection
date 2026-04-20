import numpy as np
import pandas as pd

# ensures same random data every time
np.random.seed(42)

NUM_USERS = 20


def generate_user_profile():
    """
    Create base behavior for each user
    """
    return {
        "typing_speed": np.random.normal(180, 15),
        "avg_key_delay": np.random.normal(0.12, 0.01),
        "click_rate": np.random.normal(2.5, 0.3),
        "mouse_speed": np.random.normal(400, 40),
        "session_duration": np.random.normal(300, 40),
        "idle_time": np.random.normal(20, 4),
    }


def generate_normal_sessions(user_id, profile, n=50):
    """
    Generate sessions close to user's normal behavior
    """
    data = {
        "user_id": [user_id] * n,

        "typing_speed": np.random.normal(profile["typing_speed"], 10, n),
        "avg_key_delay": np.random.normal(profile["avg_key_delay"], 0.01, n),
        "click_rate": np.random.normal(profile["click_rate"], 0.3, n),
        "mouse_speed": np.random.normal(profile["mouse_speed"], 30, n),
        "session_duration": np.random.normal(profile["session_duration"], 30, n),
        "idle_time": np.random.normal(profile["idle_time"], 3, n),

        "label": np.zeros(n)  # normal
    }

    return pd.DataFrame(data)


def generate_anomaly_sessions(user_id, profile, n=10):
    """
    Generate sessions that do NOT match user's behavior
    """
    data = {
        "user_id": [user_id] * n,

        "typing_speed": np.random.uniform(50, 400, n),
        "avg_key_delay": np.random.uniform(0.01, 0.5, n),
        "click_rate": np.random.uniform(0.1, 10, n),
        "mouse_speed": np.random.uniform(50, 1000, n),
        "session_duration": np.random.uniform(10, 1000, n),
        "idle_time": np.random.uniform(0, 200, n),

        "label": np.ones(n)  # anomaly
    }

    return pd.DataFrame(data)


if __name__ == "__main__":

    all_data = []

    for user_id in range(NUM_USERS):
        profile = generate_user_profile()

        normal_df = generate_normal_sessions(user_id, profile)
        anomaly_df = generate_anomaly_sessions(user_id, profile)

        all_data.append(normal_df)
        all_data.append(anomaly_df)

    df = pd.concat(all_data, ignore_index=True)

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # save to CSV
    df.to_csv("data/raw_sessions.csv", index=False)

    print("✅ User-specific dataset generated!")
    print(df.head())