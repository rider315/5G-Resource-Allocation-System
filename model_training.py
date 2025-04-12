import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import os
import pickle  # Added to fix NameError

# Load data
df = pd.read_csv("datasets/5g_data.csv")
X = df[["bandwidth", "latency", "power", "snr", "interference"]].values
y = df[["spectrum", "allocated_power"]].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Output: spectrum and allocated_power
])

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate
loss = model.evaluate(X_test_scaled, y_test)
print(f"Test loss: {loss}")

# Save model and scaler
if not os.path.exists("saved_models"):
    os.makedirs("saved_models")
model.save("saved_models/5g_model.h5")  # HDF5 format (legacy warning)
with open("saved_models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully")