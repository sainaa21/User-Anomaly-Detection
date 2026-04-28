import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# -------------------------------
# Create results folder
# -------------------------------
os.makedirs("results", exist_ok=True)

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("data/features.csv")

# Isolation Forest input (NO label)
X_if = df.drop(columns=["user_id", "label"], errors='ignore')

# Autoencoder input (MATCH training)
X_ae = df.drop(columns=["user_id"], errors='ignore')

y_true = df["label"]

# -------------------------------
# Load scaler + scale IF input
# -------------------------------
scaler = joblib.load("src/models/scaler.pkl")
X_if_scaled = scaler.transform(X_if)

# -------------------------------
# 1. Isolation Forest
# -------------------------------
model_if = joblib.load("src/models/isolation_forest.pkl")
scores_if = model_if.decision_function(X_if_scaled)
risk_if = -scores_if

# -------------------------------
# 2. Autoencoder
# -------------------------------
autoencoder = load_model("src/models/autoencoder.h5", compile=False)

X_pred = autoencoder.predict(X_ae)
risk_ae = np.mean((X_ae - X_pred) ** 2, axis=1)

# -------------------------------
# 3. Normalize both
# -------------------------------
scaler_if = MinMaxScaler()
risk_if_scaled = scaler_if.fit_transform(
    np.array(risk_if).reshape(-1, 1)
).flatten()

scaler_ae = MinMaxScaler()
risk_ae_scaled = scaler_ae.fit_transform(
    risk_ae.values.reshape(-1, 1)
).flatten()

# -------------------------------
# 4. Combine BOTH (HYBRID MODEL 🔥)
# -------------------------------
risk_final = 0.6 * risk_ae_scaled + 0.4 * risk_if_scaled

df["risk_score"] = risk_final

# -------------------------------
# 5. Threshold tuning
# -------------------------------
thresholds = [0.3, 0.5, 0.7, 0.9]

print("\nThreshold Results:\n")

results = []

for t in thresholds:
    y_pred = (risk_final > t).astype(int)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    results.append({
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(f"Threshold: {t}")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
    print("-" * 30)

# -------------------------------
# 6. Best threshold
# -------------------------------
best = max(results, key=lambda x: x["f1"])
print("\nBest Threshold:", best)

# -------------------------------
# 7. Save results
# -------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("results/threshold_tuning.csv", index=False)

# -------------------------------
# 8. Visualization (IMPORTANT 🔥)
# -------------------------------
plt.figure()

plt.hist(risk_final[y_true == 0], bins=50, alpha=0.5, label="Normal")
plt.hist(risk_final[y_true == 1], bins=50, alpha=0.5, label="Anomaly")

plt.axvline(best["threshold"], linestyle='--', label="Best Threshold")

plt.legend()
plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Frequency")

plt.savefig("results/risk_distribution.png")
plt.close()

# -------------------------------
# 9. Debug check
# -------------------------------
print("\nSample Outputs:")
print("IF risk:", risk_if_scaled[:5])
print("AE risk:", risk_ae_scaled[:5])
print("Final risk:", risk_final[:5])

from sklearn.metrics import roc_curve, auc

# -------------------------------
# ROC Curve
# -------------------------------
fpr, tpr, roc_thresholds = roc_curve(y_true, risk_final)
roc_auc = auc(fpr, tpr)

print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random model line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.savefig("results/roc_curve.png")
plt.close()