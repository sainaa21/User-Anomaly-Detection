import pandas as pd

# Load final dataset
df = pd.read_csv("data/features_final.csv")


df["risk_score"] = 0.5 * df["ae_score"] + 0.5 * df["iso_score"]
def decide_action(risk):
    if risk < 0.3:
        return "ALLOW"
    elif risk < 0.6:
        return "MONITOR"
    elif risk < 0.8:
        return "WARN"
    else:
        return "REAUTHENTICATE"


# Apply decision
df["action"] = df["risk_score"].apply(decide_action)

df.to_csv("data/final_output.csv", index=False)

print("✅ Decision system applied and saved!")

print(df[["risk_score", "action"]].head(10))