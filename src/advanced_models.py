import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/features.csv")

X = df.drop(columns=["label"])   # label = 0 (normal), 1 (anomaly)

# Train only on NORMAL data for anomaly detection
X_train = X[df["label"] == 0]

# Test on full dataset
X_test = X
y_test = df["label"]