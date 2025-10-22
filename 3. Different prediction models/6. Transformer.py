#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer travel time forecasting with 80% train / 20% test split (by day).

Outputs:
 - Per-day prediction plots (last 20%)
 - Per-day time-series Excel files (day-only, Predicted + Actual, with Date)
 - Evaluation metrics for all test days
"""

import os, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ====================== Configuration ======================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_Transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

target_column = 'tt_obs_min'
seq_len = 10
batch_size = 32
num_epochs = 100
lr = 1e-3
model_dim = 64
num_heads = 4
num_layers = 2
dropout = 0.1
use_standardize = True
print_every = 5

# ====================== Reproducibility ======================
def set_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(2024)

# ====================== Dataset / Model ======================
class SeqDataset(Dataset):
    def __init__(self, arr, seq_len=10):
        self.seq_len = seq_len
        self.data = torch.tensor(arr, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, i):
        x = self.data[i:i+self.seq_len]
        y = self.data[i+self.seq_len]
        return x, y

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()  # <-- Fixed line
        self.input_proj = nn.Linear(input_size, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)
    def forward(self, x):
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        h = self.encoder(h)
        return self.fc(h[:, -1, :])

# ====================== Main workflow ======================
def main():
    # ===== 1) Read data =====
    files = sorted([f for f in os.listdir(DATA_DIR_INPUT) if f.endswith('.xlsx') and not f.startswith('~$')])
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {DATA_DIR_INPUT}")
    daily_data = []
    for fp in files:
        df = pd.read_excel(DATA_DIR_INPUT / fp)
        cols = {c.lower().strip(): c for c in df.columns}
        if target_column.lower() not in cols:
            raise ValueError(f"'{target_column}' not found in {fp}")
        col = cols[target_column.lower()]
        daily_data.append(df[col].values)
    lengths = [len(d) for d in daily_data]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent day lengths: {lengths}")
    daily_data = np.array(daily_data, dtype=float)
    n_days, points_per_day = daily_data.shape
    print(f"Total {n_days} days; {points_per_day} points/day")

    # ===== 2) Split 80:20 =====
    train_days = math.floor(0.8 * n_days)
    test_days = n_days - train_days
    print(f"Train {train_days} days, Test {test_days} days")

    train_series = daily_data[:train_days].flatten()

    # ===== 3) Standardization =====
    if use_standardize:
        mu, sigma = train_series.mean(), train_series.std() or 1.0
    else:
        mu, sigma = 0.0, 1.0
    z = lambda x: (x - mu) / sigma
    iz = lambda x: x * sigma + mu
    train_series_norm = z(train_series)

    # ===== 4) Train Model =====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTimeSeries(1, model_dim, num_heads, num_layers, dropout, seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(SeqDataset(train_series_norm, seq_len), batch_size=batch_size, shuffle=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)
        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE {total_loss / len(train_loader.dataset):.6f}")

    # ===== 5) Evaluate last 20% days =====
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

    for day_idx in range(train_days, n_days):
        test_series = daily_data[day_idx]
        concat = np.concatenate([daily_data[day_idx-1], test_series]) if day_idx > 0 else test_series
        concat_norm = z(concat)

        try:
            X_test_np = np.lib.stride_tricks.sliding_window_view(concat_norm, window_shape=seq_len)
        except Exception:
            X_test_np = np.array([concat_norm[i:i+seq_len] for i in range(len(concat_norm)-seq_len+1)])
        X_test_t = torch.tensor(X_test_np[:, :, None], dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            preds_norm = model(X_test_t).cpu().numpy().ravel()
        preds = iz(preds_norm)

        preds_last = preds[-points_per_day:]
        y_true = test_series
        file_name = files[day_idx]
        day_label = os.path.splitext(file_name)[0]

        # ===== Save per-day Excel =====
        ts_df = pd.DataFrame({
            "Date": [day_label] * points_per_day,
            "TimeStep": np.arange(points_per_day),
            "Predicted_tt_min": preds_last,
            "Actual_tt_min": y_true
        })
        ts_path = timeseries_dir / f"{day_label}_timeseries_dayonly.xlsx"
        ts_df.to_excel(ts_path, index=False)

        # ===== Metrics =====
        mae, rmse, mape, r2 = evaluate(y_true, preds_last)
        results.append({
            "DayIndex": day_idx + 1,
            "Date": day_label,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE(%)": mape,
            "R2": r2
        })

        # ===== Plot =====
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="Actual tt")
        plt.plot(preds_last, "--", label="Predicted tt")
        plt.title(f"Transformer Prediction vs Actual ({day_label})")
        plt.xlabel("Time step")
        plt.ylabel("Travel time (min)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{day_label}_prediction.png", dpi=300)
        plt.close()

        print(f"{day_label}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%, R2={r2:.3f}")

    # ===== 6) Export all metrics =====
    metrics_df = pd.DataFrame(results)
    metrics_path = OUTPUT_DIR / "evaluation_metrics_Transformer_last20days.xlsx"
    metrics_df.to_excel(metrics_path, index=False)
    print(f"\nAll evaluation metrics saved to: {metrics_path}")
    print(f"Plots saved to: {plot_dir}")
    print(f"Per-day time series saved to: {timeseries_dir}")

if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    main()
