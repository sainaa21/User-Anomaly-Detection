import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv("data/features.csv")

model = joblib.load("src/models/isolation_forest.pkl")
scaler = joblib.load("src/models/scaler.pkl")

X = df.drop(columns=["user_id", "label"], errors="ignore")
X_scaled = scaler.transform(X)

scores = model.decision_function(X_scaled)

risk_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())

df["risk_score"] = risk_scores

threshold = 0.7

df["pred"] = (df["risk_score"] > threshold).astype(int)

print(classification_report(df["label"], df["pred"]))
print(df[["user_id", "risk_score"]].head().to_dict(orient="records"))