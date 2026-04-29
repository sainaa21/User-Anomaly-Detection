import pandas as pd
from src.decision import decide_action, generate_explanation, execute_action, log_decision

df = pd.read_csv("data/features_final.csv")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

df[["ae_score_norm"]] = scaler.fit_transform(df[["ae_score"]])
df[["iso_score_norm"]] = scaler.fit_transform(df[["iso_score"]])

df["risk_score"] = 0.5 * df["ae_score_norm"] + 0.5 * df["iso_score_norm"]
   
df["action"] = df["risk_score"].apply(decide_action)
print("\n📊 Action Distribution:\n")
print(df["action"].value_counts())

df["explanation"] = df.apply(lambda row: generate_explanation(row, row["risk_score"]), axis=1)

df.to_csv("data/final_output.csv", index=False)

print("✅ Decision system executed successfully!\n")
print(df[["risk_score", "action", "explanation"]].head(10))

for _, row in df.head(10).iterrows():
    log_decision(row["risk_score"], row["action"])
    execute_action(row["action"])