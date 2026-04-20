import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.models.isolation_forest import train_isolation_forest
import joblib

df = pd.read_csv("data/features.csv")

X = df.drop(columns=["user_id", "label"], errors='ignore')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Shape:", X.shape)
print(X.head())

model = train_isolation_forest(X_scaled)

scores = model.decision_function(X_scaled)

risk_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())

df["risk_score"] = risk_scores

joblib.dump(model, "src/models/isolation_forest.pkl")
joblib.dump(scaler, "src/models/scaler.pkl")