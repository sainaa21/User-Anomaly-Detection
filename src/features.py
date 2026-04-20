import pandas as pd

def load_data():
    df1 = pd.read_csv("data/raw_sessions.csv")
    df2 = pd.read_csv("data/real_user_sessions.csv")
    df2["label"]=0
    
    df = pd.concat([df1, df2], ignore_index=True)
    return df


def clean_data(df):
    df = df.dropna()
    
    # remove impossible values
    df = df[df["typing_speed"] > 0]
    df = df[df["click_rate"] >= 0]
    
    return df


def basic_features(df):
    
    # typing consistency
    df["typing_consistency"] = 1 / (df["avg_key_delay"] + 1e-5)
    
    # activity ratio
    df["activity_ratio"] = df["session_duration"] / (df["idle_time"] + 1)
    
    return df


def compute_user_stats(df):
    
    user_stats = df.groupby("user_id").agg({
        "typing_speed": ["mean", "std"],
        "click_rate": ["mean", "std"],
        "mouse_speed": ["mean", "std"]
    })
    
    user_stats.columns = ["_".join(col) for col in user_stats.columns]
    
    return user_stats


def merge_user_stats(df, user_stats):
    df = df.merge(user_stats, on="user_id", how="left")
    return df


def deviation_features(df):
    
    # deviation from mean
    df["typing_dev"] = abs(df["typing_speed"] - df["typing_speed_mean"])
    df["click_dev"] = abs(df["click_rate"] - df["click_rate_mean"])
    
    # z-score (standardized deviation)
    df["typing_z"] = df["typing_dev"] / (df["typing_speed_std"] + 1e-5)
    df["click_z"] = df["click_dev"] / (df["click_rate_std"] + 1e-5)
    
    return df


def behavior_drift(df):
    
    df["drift_score"] = (
        df["typing_z"] +
        df["click_z"]
    ) / 2
    
    return df


def select_features(df):
    
    features = [
        "typing_speed",
        "click_rate",
        "mouse_speed",
        "typing_consistency",
        "activity_ratio",
        "typing_dev",
        "click_dev",
        "typing_z",
        "click_z",
        "drift_score"
    ]
    
    return df[["user_id"] + features +["label"]]


def save_features(df):
    df.to_csv("data/features.csv", index=False)


def run_pipeline():
    
    df = load_data()
    df = clean_data(df)
    df = basic_features(df)
    
    user_stats = compute_user_stats(df)
    df = merge_user_stats(df, user_stats)
    
    df = deviation_features(df)
    df = behavior_drift(df)
    
    df = select_features(df)
    save_features(df)

if __name__ == "__main__":
    run_pipeline()