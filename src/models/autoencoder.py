import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense


# Load data
df = pd.read_csv("data/features.csv")

# Split features and label
X = df.drop(columns=["label"])
y = df["label"]

# Train only on normal data
X_train = X[y == 0]
X_test = X

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build Autoencoder
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)

autoencoder.compile(optimizer="adam", loss="mse")

# Train
autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predict on test data
reconstructions = autoencoder.predict(X_test)

# Calculate reconstruction error
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Save scores
df["ae_score"] = mse

# Save updated file
df.to_csv("data/features_with_ae.csv", index=False)

print("✅ Autoencoder scores added and saved!")