import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/features_with_ae.csv")

X = df.drop(columns=["label"])
y = df["label"]

# Train only on normal data
X_train = X[y == 0]
X_test = X

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
iso_model = IsolationForest(
    n_estimators=200,
    contamination=0.1,
    random_state=42
)

iso_model.fit(X_train)

# Get scores
iso_scores = -iso_model.decision_function(X_test)

# Add to dataframe
df["iso_score"] = iso_scores

# Save back
df.to_csv("data/features_final.csv", index=False)

print("✅ iso_score added and saved!")