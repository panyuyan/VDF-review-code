#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM travel time forecasting with 80% train / 20% test split (by day).

Outputs:
 - Evaluation metrics for all test days
 - Per-day comparison Excel files (predicted & actual)
 - Per-day prediction plots
 - Per-day time series Excel files (day-only, with date)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import random
import math

# ====================== 0) Configuration ======================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_LSTM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

target_column = 'tt_obs_min'
seq_len = 10
batch_size = 32
num_epochs = 100
lr = 1e-3
hidden_size = 64
num_layers = 1
use_standardize = True

# ====================== 1) Reproducibility ======================
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(2024)

# ====================== 2) Read daily data ======================
all_files = sorted([
    f for f in os.listdir(DATA_DIR_INPUT)
    if f.endswith('.xlsx') and not f.startswith('~$')
])

daily_data = []
for file in all_files:
    fp = os.path.join(DATA_DIR_INPUT, file)
    df = pd.read_excel(fp)
    cols_norm = {c: c.strip().lower() for c in df.columns}
    lower_to_orig = {v: k for k, v in cols_norm.items()}
    if target_column.lower() in cols_norm.values():
        col = lower_to_orig[target_column.lower()]
    else:
        raise ValueError(f"Column '{target_column}' not found in {file}")
    series = df[col].values
    daily_data.append(series)

lengths = [len(day) for day in daily_data]
if len(set(lengths)) != 1:
    raise ValueError(f"Inconsistent number of data points per day: {lengths}")
daily_data = np.array(daily_data, dtype=float)
n_days, points_per_day = daily_data.shape

train_days = math.floor(0.8 * n_days)
test_days = n_days - train_days
print(f"Total {n_days} days → Train {train_days} days, Test {test_days} days")

# ====================== 3) Dataset class ======================
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, arr, seq_len=10):
        self.data = torch.tensor(arr, dtype=torch.float32).view(-1, 1)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return x, y

# ====================== 4) LSTM model ======================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ====================== 5) Train model (first 80%) ======================
train_series = daily_data[:train_days].flatten()
if use_standardize:
    mu, sigma = np.mean(train_series), np.std(train_series) or 1.0
    train_series_norm = (train_series - mu) / sigma
else:
    mu, sigma = 0.0, 1.0
    train_series_norm = train_series

train_loader = DataLoader(SeqDataset(train_series_norm, seq_len), batch_size=batch_size, shuffle=True)
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Train MSE: {total_loss / len(train_loader.dataset):.6f}")

# ====================== 6) Evaluate last 20% days ======================
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-9
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100 if mask.any() else np.nan
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

results = []
plot_dir = OUTPUT_DIR / "plots_last20"
plot_dir.mkdir(parents=True, exist_ok=True)
timeseries_dir = OUTPUT_DIR / "timeseries_last20"
timeseries_dir.mkdir(parents=True, exist_ok=True)

for i, day_idx in enumerate(range(train_days, n_days)):
    test_series = daily_data[day_idx]
    concat = np.concatenate([daily_data[day_idx-1], test_series]) if day_idx > 0 else test_series
    concat_norm = (concat - mu) / sigma
    test_dataset = SeqDataset(concat_norm, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    preds_norm = []
    model.eval()
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb).cpu().numpy().ravel()[0]
            preds_norm.append(pred)

    preds = np.array(preds_norm) * sigma + mu
    preds_last = preds[-points_per_day:]  # Only keep the last (current day)
    y_true = test_series

    # ====== Save per-day results (288 rows) with date column ======
    file_name = all_files[day_idx]
    day_label = os.path.splitext(file_name)[0]

    ts_df = pd.DataFrame({
        "Date": [day_label] * points_per_day,
        "TimeStep": np.arange(points_per_day),
        "Predicted_tt_min": preds_last,
        "Actual_tt_min": y_true
    })
    ts_path = timeseries_dir / f"{day_label}_timeseries_dayonly.xlsx"
    ts_df.to_excel(ts_path, index=False)

    # ====== Evaluate ======
    mae, rmse, mape, r2 = evaluate(y_true, preds_last)
    results.append({
        "DayIndex": day_idx + 1,
        "Date": day_label,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE(%)": mape,
        "R2": r2
    })

    # ====== Plot ======
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="Actual tt")
    plt.plot(preds_last, "--", label="Predicted tt")
    plt.title(f"LSTM Prediction vs Actual ({day_label})")
    plt.xlabel("Time step")
    plt.ylabel("Travel time (min)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_path = plot_dir / f"{day_label}_prediction.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"{day_label}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, R2={r2:.3f}")

# ====================== 7) Export metrics ======================
metrics_df = pd.DataFrame(results)
metrics_path = OUTPUT_DIR / "evaluation_metrics_LSTM_last20days.xlsx"
metrics_df.to_excel(metrics_path, index=False)
print(f"\n✅ All evaluation metrics saved to: {metrics_path}")
print(f"✅ Plots saved to: {plot_dir}")
print(f"✅ Time-series (day-only) saved to: {timeseries_dir}")
