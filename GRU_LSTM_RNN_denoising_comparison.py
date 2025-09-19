#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 18:45:44 2025
@author: Sahar Jahani

Description:
This script compares RNN, LSTM, and GRU models for denoising a noisy sine wave.
Each model is trained multiple times and evaluated based on average RMSE and standard deviation.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Generate noisy sine wave signal
n = 1500
t = np.linspace(0, 20 * np.pi, n)
noise = np.random.normal(0, 0.5, n)
true = np.sin(t)
output = true + noise

# Hyperparameters
window_size = 20       # number of time steps per input sequence
epoch = 20             # number of training epochs
batch_size = 32        # mini-batch size
itr = 7                # number of times each model is trained (for averaging)

# Normalize data
scaler = MinMaxScaler()
output_scaled = scaler.fit_transform(output.reshape(-1, 1))
true_scaled = scaler.transform(true.reshape(-1, 1))

# Function to prepare data for sequence models
def data_batch(output, true, window_size):
    X, y = [], []
    for i in range(window_size, len(output)):
        X.append(output[i - window_size: i])
        y.append(np.mean(true[i - window_size: i]))  # average target in window
    X = np.array(X)
    y = np.array(y)
    return X, y

# Prepare input/output pairs
X, y = data_batch(output_scaled, true_scaled, window_size)

# Train/test split
split = int(0.7 * len(X))
trainX, trainy = X[:split], y[:split]
testX, testy = X[split:], y[split:]

# Convert test labels back to original scale for RMSE
testy = scaler.inverse_transform(testy.reshape(-1, 1))

# Define models to compare
models = {"RNN": SimpleRNN, "GRU": GRU, "LSTM": LSTM }

out = {}   # Store averaged predictions
rmse = {}  # Store RMSE + std per model

# Train each model multiple times
for model_name, layer in models.items():
    print(f"\nTraining {model_name} model...")

    predictions = []  # store predictions from each run

    for run in range(itr):
        print(f"  Run {run+1}/{itr}")
        
        model = Sequential()
        model.add(layer(20, input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(trainX, trainy, epochs=epoch, batch_size=batch_size, verbose=0)

        y_pred = model.predict(testX, verbose=0)
        predictions.append(y_pred)

    # Stack and average predictions from all runs
    preds_stack = np.stack(predictions, axis=0)  # shape: (itr, samples, 1)
    avg_pred = np.mean(preds_stack, axis=0)
    avg_pred = scaler.inverse_transform(avg_pred)  # convert back to original scale
    out[model_name] = avg_pred

    # Compute RMSE + STD across all runs
    rmses = [np.sqrt(mean_squared_error(testy, scaler.inverse_transform(p))) for p in predictions]
    rmse[model_name] = (np.mean(rmses), np.std(rmses))
    print(f"{model_name} RMSE: {rmse[model_name][0]:.4f} ± {rmse[model_name][1]:.4f}")

# Plot predictions vs true signal
plt.figure(figsize=(12, 6))
plt.plot(t[-len(testy):], out["GRU"], label="GRU")
plt.plot(t[-len(testy):], out["LSTM"], label="LSTM")
plt.plot(t[-len(testy):], out["RNN"], label="RNN")
plt.plot(t[-len(testy):], testy, label="True", color="black", linewidth=2)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('Amplitude', fontsize=14)
plt.title('Comparison of GRU, LSTM, and RNN in Signal Denoising', fontsize=15, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
