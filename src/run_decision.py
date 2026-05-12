import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from decision import (
    decide_action,
    execute_action,
    log_decision
)

from explainability import generate_explanation


# =========================
# LOAD DATA
# =========================

df = pd.read_csv("data/features_final.csv")

print(df.columns)
# =========================
# NORMALIZE SCORES
# =========================

scaler = MinMaxScaler()

df[["ae_score_norm"]] = scaler.fit_transform(
    df[["ae_score"]]
)

df[["iso_score_norm"]] = scaler.fit_transform(
    df[["iso_score"]]
)


# =========================
# CALCULATE FINAL RISK SCORE
# =========================

df["risk_score"] = (
    0.5 * df["ae_score_norm"] +
    0.5 * df["iso_score_norm"]
)


# =========================
# DECISION SYSTEM
# =========================

df["action"] = df["risk_score"].apply(decide_action)


# =========================
# TRUSTED SESSIONS
# =========================

df["is_trusted"] = df["risk_score"] < 0.3

trusted_df = df[df["is_trusted"]]

print(f"\n✅ Trusted sessions: {len(trusted_df)}")


# =========================
# ACTION DISTRIBUTION
# =========================

print("\n📊 Action Distribution:\n")
print(df["action"].value_counts())


# =========================
# FEATURES USED FOR SHAP
# =========================

feature_columns = [
    "user_id",
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


# =========================
# GENERATE EXPLANATIONS
# =========================

df["explanation"] = df.apply(
    lambda row: generate_explanation(
        pd.DataFrame([row[feature_columns]])
    ),
    axis=1
)


# =========================
# SAVE FINAL OUTPUT
# =========================

df.to_csv("data/final_output.csv", index=False)


# =========================
# PRINT RESULTS
# =========================

print("\n✅ Decision system executed successfully!\n")

print(
    df[
        [
            "risk_score",
            "action",
            "explanation"
        ]
    ].head(10)
)


# =========================
# EXECUTE ACTIONS + LOGS
# =========================

for _, row in df.head(10).iterrows():

    log_decision(
        row["risk_score"],
        row["action"]
    )

    execute_action(
        row["action"]
    )