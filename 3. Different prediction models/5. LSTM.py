#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
from pathlib import Path

# ===== Adjustable parameters =====
# ============ Input & Output Directory Configuration ============
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
OUTPUT_DIR = BASE_DIR / "3. Different Models" / "Output_LSTM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

target_column = 'tt_obs_min'
train_days = 80
test_days = 1
seq_len = 10
batch_size = 32
num_epochs = 100
lr = 1e-3
hidden_size = 50
num_layers = 1
use_standardize = True     # whether to standardize the target (based on train mean/std)

# ===== Reproducibility =====
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(2024)

# ===== Read daily "travel time (tt_obs_min)" curves =====
all_files = sorted([
    f for f in os.listdir(DATA_DIR_INPUT)
    if f.endswith('.xlsx') and not f.startswith('~$')
])



daily_data = []
for file in all_files:
    fp = os.path.join(DATA_DIR_INPUT, file)

    try:
        df = pd.read_excel(fp)
    except Exception as e:
        raise RuntimeError(f'Failed to read {fp}: {e}')

    # compatible with case/space differences in column names
    cols_norm = {c: c.strip().lower() for c in df.columns}
    lower_to_orig = {v: k for k, v in cols_norm.items()}
    if target_column.lower() in cols_norm.values():
        col = lower_to_orig[target_column.lower()]
    else:
        raise ValueError(f"Column '{target_column}' not found in file {file} (case-insensitive). Existing columns: {list(df.columns)}")

    series = df[col].values
    daily_data.append(series)

if not daily_data:
    raise ValueError(f"No Excel files with column '{target_column}' found in path {DATA_DIR_INPUT}")


# Ensure same number of points per day
lengths = [len(day) for day in daily_data]
if len(set(lengths)) != 1:
    raise ValueError(f"Inconsistent number of data points per day: {lengths}, please check files")

daily_data = np.array(daily_data, dtype=float)  # shape = (days, points per day)
n_days, points_per_day = daily_data.shape
print(f"Total days: {n_days}, Points per day: {points_per_day}")

if train_days + test_days > n_days:
    raise ValueError(f"train_days({train_days}) + test_days({test_days}) exceeds total days ({n_days})")

# ===== Flatten into one sequence and split train/test (by day) =====
train_series = daily_data[:train_days].flatten()
test_series = daily_data[train_days:train_days + test_days].flatten()

# ===== Standardization (use only train statistics) =====
if use_standardize:
    mu = np.mean(train_series)
    sigma = np.std(train_series) if np.std(train_series) > 0 else 1.0
    train_series_norm = (train_series - mu) / sigma
    # Concatenate last part of train + test to ensure continuity for sliding window
    test_concat = np.concatenate([train_series[-seq_len:], test_series])
    test_concat_norm = (test_concat - mu) / sigma
else:
    mu, sigma = 0.0, 1.0
    train_series_norm = train_series.copy()
    test_concat = np.concatenate([train_series[-seq_len:], test_series])
    test_concat_norm = test_concat.copy()

# ===== Dataset =====
class SeqDataset(Dataset):
    def __init__(self, arr, seq_len=10):
        self.seq_len = seq_len
        self.data = torch.tensor(arr, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return len(self.data) - self.seq_len
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y

train_dataset = SeqDataset(train_series_norm, seq_len=seq_len)
# Test dataset uses concatenation of "last seq_len of train + test days" for rolling prediction
test_dataset = SeqDataset(test_concat_norm, seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ===== LSTM model =====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)          # out: [B, T, H]
        out = self.fc(out[:, -1, :])   # take last time step
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===== Training =====
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d} | Train MSE: {running_loss / len(train_dataset):.6f}")

# ===== Prediction (evaluate only the last day) =====
model.eval()
preds_norm = []
with torch.no_grad():
    for Xb, _ in test_loader:
        Xb = Xb.to(device)
        pred = model(Xb).cpu().numpy().ravel()[0]
        preds_norm.append(pred)

# Restore scale
preds_norm = np.array(preds_norm, dtype=float)
preds_full = preds_norm * sigma + mu  # predictions for concatenated sequence (len = len(test_concat_norm) - seq_len)

# Take only the "last day" predictions: should match points_per_day
preds_lastday = preds_full[-points_per_day:]
y_true = test_series  # original scale (minutes)

# ===== Evaluation =====
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # avoid divide by zero: compute MAPE only where y_true != 0
    mask = np.abs(y_true) > 1e-9
    if mask.any():
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, mape, r2

mae, rmse, mape, r2 = evaluate(y_true, preds_lastday)

print("LSTM travel time prediction results (first 80 days training, last 1 day testing):")
print(f"MAE: {mae:.4f} min, RMSE: {rmse:.4f} min, MAPE: {mape:.2f}%, R²: {r2:.4f}")

# ===== Export evaluation metrics =====
metrics_df = pd.DataFrame({
    "Target": [target_column],
    "Model": [f"LSTM(seq_len={seq_len}, hidden_size={hidden_size})"],
    "TrainDays": [train_days],
    "TestDays": [test_days],
    "MAE(min)": [mae],
    "RMSE(min)": [rmse],
    "MAPE(%)": [mape],
    "R2": [r2]
})

metrics_path = OUTPUT_DIR / "evaluation_metrics_LSTM_tt_lastday.xlsx"

metrics_df.to_excel(metrics_path, index=False)
print(f"Evaluation metrics saved to: {metrics_path}")

# ===== Export time-step prediction comparison =====
compare_df = pd.DataFrame({
    "timestep": np.arange(points_per_day),
    "tt_true_min": y_true,
    "tt_pred_min": preds_lastday
})
compare_path = OUTPUT_DIR / "prediction_vs_actual_tt_lastday.xlsx"

compare_df.to_excel(compare_path, index=False)
print(f"Prediction comparison saved to: {compare_path}")

# ===== Plot =====
plot_dir = OUTPUT_DIR / "prediction_plots_LSTM_tt_lastday"
plot_dir.mkdir(parents=True, exist_ok=True)


plt.figure(figsize=(10, 5))
plt.plot(np.arange(points_per_day), y_true, label="Actual tt (min)")
plt.plot(np.arange(points_per_day), preds_lastday, "--", label="Predicted tt (min)")
plt.xlabel("Time Step (5-min interval)", fontsize=12)
plt.ylabel("Travel time (min)", fontsize=12)
plt.title("LSTM Prediction vs Actual Travel Time (Last Day)", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_path = os.path.join(plot_dir, "last_day_tt_prediction.png")
plt.savefig(save_path, dpi=300)
plt.show()
plt.close()
print(f"Prediction plot saved to: {save_path}")
