import pandas as pd

from sklearn.metrics import classification_report, roc_auc_score


# Load dataset (make sure this file has BOTH iso_score + ae_score)
df = pd.read_csv("data/features_final.csv")
print(df.columns)

y_true = df["label"]


# -------------------------------
# Isolation Forest Evaluation
# -------------------------------

# Simple threshold (you can tune later in Stage 5)
iso_pred = (df["iso_score"] > 0.5).astype(int)

print("\n🔵 Isolation Forest Results")
print(classification_report(y_true, iso_pred))

iso_auc = roc_auc_score(y_true, df["iso_score"])
print("ROC-AUC:", iso_auc)


# -------------------------------
# Autoencoder Evaluation
# -------------------------------

# Threshold = mean (simple baseline)
ae_threshold = df["ae_score"].mean()
ae_pred = (df["ae_score"] > ae_threshold).astype(int)

print("\n🟣 Autoencoder Results")
print(classification_report(y_true, ae_pred))

ae_auc = roc_auc_score(y_true, df["ae_score"])
print("ROC-AUC:", ae_auc)


# -------------------------------
# Save Comparison
# -------------------------------

results = pd.DataFrame({
    "Model": ["Isolation Forest", "Autoencoder"],
    "ROC-AUC": [iso_auc, ae_auc]
})

results.to_csv("results/model_comparison.csv", index=False)

print("\n✅ Model comparison saved to results/model_comparison.csv")