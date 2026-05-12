import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("src/models/isolation_forest.pkl")

df = pd.read_csv("data/features_final.csv")


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

X = df[feature_columns]


explainer = shap.TreeExplainer(model)

sample = X.iloc[[0]]

shap_values = explainer.shap_values(sample)


# =========================
# SUMMARY PLOT
# =========================

shap.summary_plot(
    shap_values,
    sample,
    show=False
)

plt.savefig("results/shap_summary.png")

plt.close()


# =========================
# WATERFALL PLOT
# =========================

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value[0],
        data=sample.iloc[0],
        feature_names=sample.columns.tolist()
    ),
    show=False
)

plt.savefig("results/shap_waterfall.png")

plt.close()


print("SHAP plots saved successfully!")