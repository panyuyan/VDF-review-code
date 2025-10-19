#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

# ====================== Configuration ======================
# Data and script are in the same directory; prioritize I-405 daily files; fallback if needed
# ============ Input & Output Directory Configuration ============
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
OUTPUT_DIR = BASE_DIR / "3. Different Models" / "Output_Transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

file_patterns   = [
    'CA_I405_bottleneck_13.74_*.xlsx',
    'bpr_timeseries_*.xlsx'
]
csv_fallback    = DATA_DIR_INPUT / 'bpr_daily_timeseries.csv'
raw_daily_glob  = str(DATA_DIR_INPUT / '*.xlsx')

target_column   = 'tt_obs_min'         # target column (minutes)
LENGTH_KM_FOR_FALLBACK = 0.23          # when no tt_obs_min, compute from speed: tt = L/v * 60
SPEED_COL_CANDS = ['Speed', 'speed', 'v', 'v_kmh']
TT_COL_CANDS    = [target_column, 'Travel time', 'Travel time (min)', 'TT', 'tt', 'tt_obs_min']

# Training parameters
train_days    = 80
test_days     = 1
seq_len       = 10
batch_size    = 256
num_epochs    = 200
lr            = 1e-3
model_dim     = 64
num_heads     = 4
num_layers    = 2
dropout       = 0.1
use_standardize = True
early_stop_patience = 10
print_every   = 2

# ====================== Reproducibility ======================
def set_seed(seed=2024):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(2024)

# ====================== Reading utilities ======================
def read_tt_from_one_excel(fp, target_col='tt_obs_min', L_km=0.23):
    """Prefer reading tt_obs_min; otherwise compute from Speed as tt=L/v*60 (min). Only read required columns for speed."""
    df_head = pd.read_excel(fp, nrows=0)
    cols = list(df_head.columns)
    lower = [c.strip().lower() for c in cols]

    # 1) Travel time column
    for c in TT_COL_CANDS:
        if c in cols:
            s = pd.read_excel(fp, usecols=[c])[c].astype(float).values
            return s
        if c.lower() in lower:
            real = cols[lower.index(c.lower())]
            s = pd.read_excel(fp, usecols=[real])[real].astype(float).values
            return s

    # 2) Speed column -> tt
    for c in SPEED_COL_CANDS:
        if c in cols:
            v = pd.read_excel(fp, usecols=[c])[c].astype(float).values
            v = np.clip(v, 1e-6, None)
            return (L_km / v) * 60.0
        if c.lower() in lower:
            real = cols[lower.index(c.lower())]
            v = pd.read_excel(fp, usecols=[real])[real].astype(float).values
            v = np.clip(v, 1e-6, None)
            return (L_km / v) * 60.0

    raise ValueError(f"{os.path.basename(fp)} missing {TT_COL_CANDS} or {SPEED_COL_CANDS}, cannot construct {target_col}")

def load_daily_series():
    """Return daily_data: np.ndarray[days, T], and list of day_names. Automatically fallback and align lengths."""
    daily_data, day_names = [], []

    # 1) Sequentially match file_patterns
    for pat in file_patterns:
        files = sorted(glob.glob(str(DATA_DIR_INPUT / pat)))

        if files:
            for fp in files:
                try:
                    y = read_tt_from_one_excel(fp, target_column, LENGTH_KM_FOR_FALLBACK)
                    daily_data.append(y)
                    name = os.path.splitext(os.path.basename(fp))[0]
                    day_names.append(name)
                except Exception as e:
                    print(f"[skip] {os.path.basename(fp)} -> {e}")
            if daily_data:
                print(f"[source] {len(daily_data)} days: from pattern {pat}")
                break

    # 2) CSV fallback
    if not daily_data and os.path.exists(csv_fallback):
        ts = pd.read_csv(csv_fallback)
        # Case-insensitive fix
        if target_column not in ts.columns and target_column.lower() in [c.lower() for c in ts.columns]:
            real = ts.columns[[c.lower() for c in ts.columns].index(target_column.lower())]
            ts = ts.rename(columns={real: target_column})
        if 'day' not in ts.columns:
            raise ValueError(f"{csv_fallback} missing 'day' column; cannot group by day.")
        for day, g in ts.groupby('day', sort=True):
            daily_data.append(g[target_column].astype(float).values)
            day_names.append(str(day))
        print(f"[source] {len(daily_data)} days: from {os.path.basename(csv_fallback)}")

    # 3) Raw xlsx fallback (all in same directory)
    if not daily_data:
        raw_files = sorted(glob.glob(raw_daily_glob))
        raw_files = [f for f in raw_files if f.lower().endswith('.xlsx')
                     and 'metric' not in f.lower() and 'prediction' not in f.lower()
                     and 'timeseries_all' not in f.lower()]
        if not raw_files:
            raise FileNotFoundError("No available data files found: please provide daily xlsx or bpr_daily_timeseries.csv")
        for fp in raw_files:
            try:
                y = read_tt_from_one_excel(fp, target_column, LENGTH_KM_FOR_FALLBACK)
                daily_data.append(y)
                day_names.append(os.path.splitext(os.path.basename(fp))[0])
            except Exception as e:
                print(f"[skip] {os.path.basename(fp)} -> {e}")
        if not daily_data:
            raise FileNotFoundError("No *.xlsx in the directory contain tt_obs_min or Speed columns.")
        print(f"[source] {len(daily_data)} days: from raw *.xlsx (current directory)")

    # Align lengths (truncate to shortest if inconsistent)
    lengths = list(map(len, daily_data))
    T = min(lengths)
    if len(set(lengths)) != 1:
        print(f"[warn] Inconsistent day lengths: {lengths} → unified to shortest T={T}")
    daily_data = np.asarray([d[:T] for d in daily_data], dtype=float)
    day_names = day_names[:len(daily_data)]
    return daily_data, day_names

# ====================== Dataset / Model ======================
class SeqDataset(Dataset):
    def __init__(self, arr, seq_len=10):
        self.seq_len = seq_len
        self.data = torch.tensor(arr, dtype=torch.float32).view(-1, 1)
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, i):
        x = self.data[i:i+self.seq_len]   # [T, 1]
        y = self.data[i+self.seq_len]     # [1]
        return x, y

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size=1, model_dim=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.input_proj = nn.Linear(input_size, model_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, model_dim))
        enc_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, 1)
    def forward(self, x):
        # x: [B, T, 1]
        h = self.input_proj(x) + self.pos_emb[:, :x.size(1), :]
        h = self.encoder(h)
        return self.fc(h[:, -1, :])  # [B, 1]

def make_loader(ds, batch_size, want_workers):
    """DataLoader auto fallback: switch to single-process if multi-process fails."""
    pin = torch.cuda.is_available()
    try:
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=want_workers, pin_memory=pin, persistent_workers=(want_workers > 0)
        )
    except Exception as e:
        print(f"[WARN] DataLoader multiprocessing failed ({e}), downgraded to single process.")
        return DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=False,
            num_workers=0, pin_memory=pin, persistent_workers=False
        )

# ====================== Main workflow ======================
def main():
    # Read data
    t0 = time.time()
    daily_data, day_names = load_daily_series()
    n_days, points_per_day = daily_data.shape
    print(f"Total days: {n_days}, Points per day: {points_per_day}")
    if train_days + test_days > n_days:
        raise ValueError(f"train_days({train_days}) + test_days({test_days}) exceeds total days ({n_days})")

    # Split
    train_series = daily_data[:train_days].flatten()
    test_series  = daily_data[train_days:train_days + test_days].flatten()

    # Standardize
    if use_standardize:
        mu = float(train_series.mean())
        sigma = float(train_series.std()) if train_series.std() > 0 else 1.0
    else:
        mu, sigma = 0.0, 1.0
    z  = lambda x: (x - mu) / sigma
    iz = lambda x: x * sigma + mu

    train_series_norm = z(train_series)
    test_concat_norm  = z(np.concatenate([train_series[-seq_len:], test_series]))

    # Dataset / loader
    train_dataset = SeqDataset(train_series_norm, seq_len=seq_len)
    if len(train_dataset) <= 0:
        raise ValueError(f"Training sample count is 0. Please check seq_len={seq_len} and train sequence length {len(train_series_norm)}.")

    num_workers_desired = max(1, os.cpu_count() // 2)
    train_loader = make_loader(train_dataset, batch_size, num_workers_desired)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTimeSeries(1, model_dim, num_heads, num_layers, dropout, seq_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # Early stopping: use tail of training as pseudo-validation
    val_len = max(points_per_day, len(train_dataset)//20)  # ≥ one day or 5%
    val_idx0 = max(0, len(train_dataset) - val_len)
    val_X = torch.stack([train_dataset[i][0] for i in range(val_idx0, len(train_dataset))]).to(device)
    val_y = torch.stack([train_dataset[i][1] for i in range(val_idx0, len(train_dataset))]).to(device)

    best_val = float('inf'); patience = early_stop_patience; no_improve = 0
    t1 = time.time()
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0; nseen = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = model(Xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = Xb.size(0); total_loss += loss.item() * bs; nseen += bs

        # Validation
        model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            val_pred = model(val_X)
            val_loss = criterion(val_pred, val_y).item()

        if epoch % print_every == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE {total_loss/nseen:.6f} | Val MSE {val_loss:.6f}")

        if val_loss + 1e-9 < best_val:
            best_val = val_loss; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"No val improvement for {patience} rounds, early stopped at epoch {epoch}.")
                break

    if 'best_state' in locals():
        model.load_state_dict(best_state)

    print(f"Training time: {time.time()-t1:.1f}s  (Device: {device}, AMP={torch.cuda.is_available()})")

    # Vectorized test inference
    from numpy.lib.stride_tricks import sliding_window_view
    X_test_np = sliding_window_view(test_concat_norm, window_shape=seq_len)  # [N, T]
    X_test_t  = torch.tensor(X_test_np[:, :, None], dtype=torch.float32).to(device)  # [N, T, 1]

    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        preds_norm = model(X_test_t).cpu().numpy().ravel()

    preds_full = iz(preds_norm)
    preds_lastday = preds_full[-points_per_day:]
    y_true = test_series

    # Evaluation
    mae = mean_absolute_error(y_true, preds_lastday)
    rmse = np.sqrt(mean_squared_error(y_true, preds_lastday))
    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_true[mask]-preds_lastday[mask]) / y_true[mask])) * 100) if mask.any() else np.nan
    r2 = r2_score(y_true, preds_lastday)

    print("Transformer predicted travel time (last day)")
    print(f"MAE: {mae:.4f} min, RMSE: {rmse:.4f} min, MAPE: {mape:.2f}%, R²: {r2:.4f}")

    # Output
    out_metrics = OUTPUT_DIR / "evaluation_metrics_Transformer_tt_lastday.xlsx"


    pd.DataFrame({
        "Target": [target_column],
        "Model": [f"Transformer(seq_len={seq_len}, dim={model_dim}, heads={num_heads}, layers={num_layers})"],
        "TrainDays": [train_days], "TestDays": [test_days],
        "MAE(min)": [mae], "RMSE(min)": [rmse], "MAPE(%)": [mape], "R2": [r2]
    }).to_excel(out_metrics, index=False)
    print(f"Evaluation metrics saved to {out_metrics}")

    out_compare = OUTPUT_DIR / "prediction_vs_actual_tt_Transformer_lastday.xlsx"

    pd.DataFrame({
        "timestep": np.arange(points_per_day),
        "tt_true_min": y_true,
        "tt_pred_min": preds_lastday
    }).to_excel(out_compare, index=False)
    print(f"Prediction comparison saved to {out_compare}")

    # Plot
    plot_dir = OUTPUT_DIR / "prediction_plots_Transformer_tt_lastday"
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / "last_day_tt_prediction.png"

    time_steps = np.arange(points_per_day)
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, y_true, label="Actual tt (min)")
    plt.plot(time_steps, preds_lastday, "--", label="Predicted tt (min)")
    plt.xlabel("Time Step (5-min interval)")
    plt.ylabel("Travel time (min)")
    plt.title("Transformer Prediction vs Actual Travel Time (Last Day)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    
    plt.savefig(save_path, dpi=300); plt.close()
    print(f"Prediction plot saved to {save_path}")

if __name__ == "__main__":
    # Some optional stable settings for macOS/conda
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    main()

