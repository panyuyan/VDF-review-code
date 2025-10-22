#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physics-Guided LSTM (PG-LSTM) — compatible with old date formats
✅ Now outputs results for the entire last 20% test days (not just the last one)
Each test day will have:
  - daily Excel: prediction_vs_actual_day_<MMDD>.xlsx
  - daily plot:  plots_PG_LSTM_last20/day_<MMDD>.png
Also saves a summary metrics Excel across all test days.
"""

import os, re, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============ Output Directory ============
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "Output_PINN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR}")

# ===================== Config =====================
OBS_DIR    = "./1. observed"
PHY_DIR    = "./2. physical"
INDEX_PATH = "./3. index/Distribution_4_month_13.74.xlsx"
FILE_PREFIX    = "CA_I405_bottleneck_13.74_"

SEG_LEN_MILES  = 0.23
SPEED_UNIT     = "mph"
TIME_COL_CANDS = ["time","Time","timestamp","Timestamp"]

seq_len = 10
batch_size = 32
num_epochs = 100
lr = 1e-3
hidden_size = 64
num_layers = 1
lambda_phys_q = 0.20
lambda_phys_ff= 0.10
use_y_standardize = True
seed = 2024

# ===================== Reproducibility =================
def set_seed(seed=2024):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(seed)

# ===================== Helper functions =====================
def list_days(dir_path, prefix):
    return sorted([f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(".xlsx")])

def extract_mmdd(name):
    """Extract date like _0415 or _20230415 → '0415'."""
    m = re.search(r"_(\d{4,8})", name)
    if not m: return None
    s = m.group(1)
    return s[-4:]

def to_mmdd_from_day(x):
    """Robustly extract MMDD from mixed date formats."""
    if pd.isna(x): return None
    s = str(x).strip()
    # numeric 0415, 415, 20230415
    if re.search(r"\d", s):
        digits = re.sub(r"\D", "", s)
        if len(digits) >= 4:
            return digits[-4:]
        if len(digits) == 3:  # e.g. 415 → 0415
            return digits.zfill(4)
    # e.g. 4/15 or 4-15
    m = re.search(r"(\d{1,2})[\/\-. ](\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):02d}{int(m.group(2)):02d}"
    # e.g. Apr 15
    m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s*(\d{1,2})", s, re.I)
    if m:
        months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                  "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
        mon = months[m.group(1)[:3].lower()]
        return f"{mon:02d}{int(m.group(2)):02d}"
    return None

def freeflow_time_from_speed(speed, unit=SPEED_UNIT):
    v = np.clip(speed, 1e-6, None)
    if unit.lower() in ["mph"]:
        return SEG_LEN_MILES / v * 60.0
    else:
        L_km = SEG_LEN_MILES * 1.60934
        return L_km / v * 60.0

def build_time_features(times):
    s = pd.Series(times)
    s_dt = pd.to_datetime(s, errors="coerce")
    if s_dt.notna().any():
        hours = s_dt.dt.hour + s_dt.dt.minute/60.0
    else:
        hours = np.arange(len(s))*5/60.0
    rad = 2*np.pi*hours/24.0
    return np.column_stack([np.sin(rad), np.cos(rad)])

def mape_safe(y_true, y_pred):
    mask = np.abs(y_true) > 1e-9
    return np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))*100 if mask.any() else np.nan

# ===================== Load index =====================
idx_df = pd.read_excel(INDEX_PATH)
colmap = {c.strip().lower(): c for c in idx_df.columns}
col_day = colmap.get("day")
col_weekday = colmap.get("weekday")
idx_df["date_key"] = idx_df[col_day].apply(to_mmdd_from_day)
wk_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
idx_df["Weekday_idx"] = idx_df[col_weekday].astype(str).str.lower().map(wk_map).fillna(0).astype(int)
idx_by_mmdd = {r["date_key"]: {"Weekday":r["Weekday_idx"]} for _, r in idx_df.iterrows() if pd.notna(r["date_key"])}

# ===================== Gather days =====================
obs_days = list_days(OBS_DIR, FILE_PREFIX)
phy_days = list_days(PHY_DIR, FILE_PREFIX)
obs_map = {extract_mmdd(f): f for f in obs_days}
phy_map = {extract_mmdd(f): f for f in phy_days}
common_mmdd = sorted(set(obs_map.keys()) & set(phy_map.keys()) & set(idx_by_mmdd.keys()))
if not common_mmdd:
    raise RuntimeError("No overlapping days across observed/, physical/, and index.xlsx")

# ===================== Build tensors =====================
X_days, y_days, meta_days = [], [], []
for mmdd in common_mmdd:
    df_obs = pd.read_excel(os.path.join(OBS_DIR, obs_map[mmdd]))
    df_phy = pd.read_excel(os.path.join(PHY_DIR, phy_map[mmdd]))
    flow, speed, density, queue = df_obs["Flow per hour"].values, df_obs["Speed"].values, df_obs["Density"].values, df_obs["Queue"].values
    tt = df_obs["tt_obs_min"].values
    tf = freeflow_time_from_speed(speed)
    queue_obs, queue_fit, mu_hat, flow_obs = df_phy["queue_obs"].values, df_phy["queue_fit"].values, df_phy["mu_hat"].values, df_phy["flow_obs"].values
    T = min(len(flow), len(queue_fit), len(mu_hat))
    flow, speed, density, queue, tt, tf = flow[:T], speed[:T], density[:T], queue[:T], tt[:T], tf[:T]
    queue_obs, queue_fit, mu_hat, flow_obs = queue_obs[:T], queue_fit[:T], mu_hat[:T], flow_obs[:T]
    time_feats = build_time_features(np.arange(T))
    weekday = idx_by_mmdd[mmdd]["Weekday"]
    weekday_feats = np.repeat(np.eye(7)[weekday].reshape(1,-1), T, axis=0)
    X = np.column_stack([flow, speed, density, queue, queue_obs, queue_fit, mu_hat, flow_obs, tf, time_feats, weekday_feats])
    mu_safe = np.clip(mu_hat, 1e-6, None)
    delay_ref = 60.0 * (queue_fit / mu_safe)
    cong_mask = queue_fit > np.percentile(queue_fit, 60)
    X_days.append(X); y_days.append(tt); meta_days.append({"tf": tf, "delay_ref": delay_ref, "cong_mask": cong_mask})

points_per_day = X_days[0].shape[0]
n_days = len(X_days)
train_days = int(np.floor(0.8 * n_days))
test_days = n_days - train_days
print(f"Total days: {n_days}, Train: {train_days}, Test: {test_days}")

# ===================== Flatten & split =====================
def flatten(X_list): return np.vstack(X_list)
def concat_meta(key): return np.concatenate([m[key] for m in meta_days])
X_train = flatten(X_days[:train_days]); y_train = np.concatenate(y_days[:train_days])
tf_train = concat_meta("tf")[:len(y_train)]; delay_ref_train = concat_meta("delay_ref")[:len(y_train)]; cong_train = concat_meta("cong_mask")[:len(y_train)]
X_test = flatten(X_days[train_days:]); y_test = np.concatenate(y_days[train_days:])
tf_test = concat_meta("tf")[len(y_train):]; delay_ref_test = concat_meta("delay_ref")[len(y_train):]; cong_test = concat_meta("cong_mask")[len(y_train):]
test_mmdd = common_mmdd[train_days:]

# ===================== Standardization =====================
y_mu, y_std = y_train.mean(), y_train.std() if y_train.std()>0 else 1.0
y_train_norm = (y_train - y_mu) / y_std; y_test_norm = (y_test - y_mu) / y_std

# ===================== Dataset & Model =====================
class SlidingSeq(Dataset):
    def __init__(self, X, y_norm, tf, dref, cong, seq_len):
        self.X, self.y, self.tf, self.dref, self.cong = X, y_norm, tf, dref, cong; self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, i):
        return (torch.tensor(self.X[i:i+self.seq_len], dtype=torch.float32),
                torch.tensor([self.y[i+self.seq_len]], dtype=torch.float32),
                torch.tensor([self.tf[i+self.seq_len]], dtype=torch.float32),
                torch.tensor([self.dref[i+self.seq_len]], dtype=torch.float32),
                torch.tensor([1.0 if self.cong[i+self.seq_len] else 0.0], dtype=torch.float32))
train_loader = DataLoader(SlidingSeq(X_train, y_train_norm, tf_train, delay_ref_train, cong_train, seq_len), batch_size=batch_size, shuffle=True)

class PGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__(); self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1); self.softplus = nn.Softplus()
    def forward(self, x_seq, tf_min):
        out, _ = self.lstm(x_seq); h = out[:, -1, :]; delay = self.softplus(self.fc(h)); tt_pred = tf_min + delay; return tt_pred, delay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PGLSTM(X_train.shape[1], hidden_size, num_layers).to(device)
mse = nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=lr)

# ===================== Train =====================
model.train()
for epoch in range(1, num_epochs+1):
    total = 0
    for xb, yb, tfb, drefb, congb in train_loader:
        xb, yb, tfb, drefb, congb = xb.to(device), yb.to(device), tfb.to(device), drefb.to(device), congb.to(device)
        opt.zero_grad()
        tt_pred, delay = model(xb, tfb)
        y_pred_norm = (tt_pred - y_mu) / y_std
        loss_pred = mse(y_pred_norm, yb)
        loss_phys_q = mse(delay, torch.clamp(drefb, 0.0, 120.0))
        loss_phys_ff = ((delay.squeeze(1)*(1.0-congb.squeeze(1)))**2).mean()
        loss = loss_pred + lambda_phys_q*loss_phys_q + lambda_phys_ff*loss_phys_ff
        loss.backward(); opt.step(); total += loss.item()
    if epoch==1 or epoch%5==0:
        print(f"Epoch {epoch:03d} | Loss: {total/len(train_loader):.6f}")

# ===================== Evaluate all test days =====================
model.eval()
plot_dir = OUTPUT_DIR / "plots_PG_LSTM_last20"; plot_dir.mkdir(exist_ok=True)
metrics_all = []
idx0 = 0
for i, mmdd in enumerate(test_mmdd):
    X_day, y_day = X_days[train_days+i], y_days[train_days+i]
    tf_day, dref_day, cong_day = meta_days[train_days+i]["tf"], meta_days[train_days+i]["delay_ref"], meta_days[train_days+i]["cong_mask"]
    X_tail = X_train[-seq_len:, :]
    X_input = np.vstack([X_tail, X_day]); tf_input = np.concatenate([tf_train[-seq_len:], tf_day])
    preds = []
    with torch.no_grad():
        for j in range(len(X_input)-seq_len):
            xb = torch.tensor(X_input[j:j+seq_len], dtype=torch.float32).unsqueeze(0).to(device)
            tfb = torch.tensor([tf_input[j+seq_len]], dtype=torch.float32).unsqueeze(0).to(device)
            tt_pred, _ = model(xb, tfb)
            preds.append(tt_pred.cpu().numpy().ravel()[0])
    preds = np.array(preds[-len(y_day):])
    mae, rmse = mean_absolute_error(y_day, preds), np.sqrt(mean_squared_error(y_day, preds))
    mape, r2 = mape_safe(y_day, preds), r2_score(y_day, preds)
    metrics_all.append({"Day":mmdd,"MAE":mae,"RMSE":rmse,"MAPE(%)":mape,"R2":r2})
    # save daily excel
    pd.DataFrame({"timestep":np.arange(len(y_day)),"tt_true_min":y_day,"tt_pred_min":preds}).to_excel(
        OUTPUT_DIR / f"prediction_vs_actual_day_{mmdd}.xlsx", index=False)
    # plot
    plt.figure(figsize=(10,5))
    plt.plot(y_day, label="Actual tt (min)")
    plt.plot(preds, "--", label="Predicted tt (min)")
    plt.xlabel("Time step (5-min)"); plt.ylabel("Travel time (min)")
    plt.title(f"PG-LSTM Prediction vs Actual — Day {mmdd}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(plot_dir / f"day_{mmdd}.png", dpi=300); plt.close()

# ===================== Summary =====================
metrics_df = pd.DataFrame(metrics_all)
metrics_df.loc["Mean"] = metrics_df.mean(numeric_only=True)
metrics_df.to_excel(OUTPUT_DIR / "evaluation_metrics_PG_LSTM_summary.xlsx", index=False)
print("✅ Saved: daily plots & Excel for last 20%, plus summary metrics.")
