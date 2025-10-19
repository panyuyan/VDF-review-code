#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Physical-Guided LSTM for Travel Time (tt_obs_min) — MILES/MPH version

Folders:
- ./1. observed/   daily xlsx: ['Flow per hour','Speed','Density','Queue','tt_obs_min','time'(optional)]
- ./2. physical/   daily xlsx: ['queue_obs','queue_fit','mu_hat','flow_obs']
- ./3. index/Distribution_4_month_13.74.xlsx with columns: Day, t0, t2, t3, Weekday, ...

Physics:
- tf = L / speed * 60  (minutes)
- Here L = 0.23 miles, Speed is assumed in MPH (change SPEED_UNIT if needed)
- Output: delay >= 0 via softplus; tt_pred = tf + delay_pred >= tf
- Physics losses:
  (A) queue-delay consistency: delay_pred ~ 60 * queue_fit / mu_hat
  (B) outside [t0, t3], delay -> 0

Artifacts:
- evaluation_metrics_PG_LSTM.xlsx
- prediction_vs_actual_lastday.xlsx
- ./prediction_plots_PG_LSTM_lastday/last_day_tt.png
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

# ============ Output Directory Configuration ============
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/
OUTPUT_DIR = BASE_DIR / "Output_PINN"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR}")

# ===================== User Config =====================
OBS_DIR    = "./1. observed"
PHY_DIR    = "./2. physical"
INDEX_PATH = "./3. index/Distribution_4_month_13.74.xlsx"

FILE_PREFIX    = "CA_I405_bottleneck_13.74_"
SEG_LEN_MILES  = 0.23                # <<< L = 0.23 mile
SPEED_UNIT     = "mph"               # 'mph' (default) or 'kmh'
TIME_COL_CANDS = ["time","Time","timestamp","Timestamp"]

target_column = "tt_obs_min"
train_days    = 80
test_days     = 1
seq_len       = 10
batch_size    = 32
num_epochs    = 100
lr            = 1e-3
hidden_size   = 64
num_layers    = 1
lambda_phys_q = 0.20
lambda_phys_ff= 0.10
use_y_standardize = True
seed = 2024

# ===================== Reproducibility =================
def set_seed(seed=2024):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(seed)

# ===================== Helpers =========================
def list_days(dir_path, prefix):
    return sorted([f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith(".xlsx")])

def extract_mmdd(name):
    m = re.search(r"_(\d{4})", name)
    return m.group(1) if m else None

def find_time_column(df):
    for c in TIME_COL_CANDS:
        if c in df.columns: return c
    return None

def build_time_features(time_series):
    """time_series: pandas Series/list of timestamps or strings 'HH:MM'."""
    s = pd.Series(time_series)
    if np.issubdtype(s.dtype, np.datetime64):
        hours = s.dt.hour + s.dt.minute/60.0
    else:
        s_dt = pd.to_datetime(s, errors="coerce")
        if s_dt.notna().all():
            hours = s_dt.dt.hour + s_dt.dt.minute/60.0
        else:
            def to_hr(x):
                try:
                    h, m = str(x).strip().split(":")[:2]
                    return int(h) + int(m)/60.0
                except Exception:
                    return np.nan
            hours = s.apply(to_hr).astype(float)
            if np.any(np.isnan(hours)):
                hours = pd.Series(np.arange(len(s))*5/60.0)  # fallback: 5-min grid
    rad = 2*np.pi*hours/24.0
    return np.column_stack([np.sin(rad), np.cos(rad)])

def freeflow_time_from_speed(speed, unit=SPEED_UNIT):
    """Return minutes. Speed in mph (default) or km/h."""
    v = np.clip(speed, 1e-6, None)
    if unit.lower() in ["mph"]:
        return SEG_LEN_MILES / v * 60.0
    elif unit.lower() in ["kmh", "km/h"]:
        L_km = SEG_LEN_MILES * 1.60934
        return L_km / v * 60.0
    else:
        raise ValueError(f"Unknown speed unit: {unit}")

def mape_safe(y_true, y_pred):
    mask = np.abs(y_true) > 1e-9
    return np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask]))*100 if mask.any() else np.nan

# ---------- Index utilities ----------
def to_mmdd_from_day(x):
    s = str(x).strip()
    try:
        v = int(s); m = v // 100; d = v % 100
        if 1 <= m <= 12 and 1 <= d <= 31:
            return f"{m:02d}{d:02d}"
    except Exception:
        pass
    m = re.search(r"(\d{1,2})[/-\. ]?(\d{1,2})", s)
    if m:
        return f"{int(m.group(1)):02d}{int(m.group(2)):02d}"
    return None

def norm_hhmm(s):
    if pd.isna(s): return None
    s = str(s).strip()
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}" if m else None

def idx_from_str(target, ts):
    """Find index in ts (Series/list/Index) that matches 'HH:MM' target."""
    if target is None:
        return None
    s = pd.Series(ts)
    s_dt = pd.to_datetime(s, errors="coerce")
    if s_dt.notna().any():
        hhmm = s_dt.dt.strftime("%H:%M")
    else:
        hhmm = s.astype(str).str.extract(r"(\d{1,2}:\d{2})", expand=False)
    where = np.where(hhmm.values == target)[0]
    return int(where[0]) if len(where) > 0 else None

def norm_colmap(df):
    return {re.sub(r"[\s_]+","", c.strip().lower()): c for c in df.columns}

# ===================== Load Index ======================
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
idx_df = pd.read_excel(INDEX_PATH)

norm = {c.strip().lower(): c for c in idx_df.columns}
col_day     = norm.get("day", None)
col_t0      = norm.get("t0", None)
col_t2      = norm.get("t2", None)
col_t3      = norm.get("t3", None)
col_weekday = norm.get("weekday", None)
need = [col_day, col_t0, col_t2, col_t3, col_weekday]
if any(c is None for c in need):
    raise ValueError(f"Index file missing one or more required columns: Day, t0, t2, t3, or Weekday. Existing columns: {list(idx_df.columns)}")

idx_df["date_key"]    = idx_df[col_day].apply(to_mmdd_from_day)
idx_df["t0_time_str"] = idx_df[col_t0].apply(norm_hhmm)
idx_df["t2_time_str"] = idx_df[col_t2].apply(norm_hhmm)
idx_df["t3_time_str"] = idx_df[col_t3].apply(norm_hhmm)

wk_map = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,"friday":4,"saturday":5,"sunday":6}
idx_df["Weekday_idx"] = idx_df[col_weekday].astype(str).str.lower().map(wk_map).fillna(0).astype(int)

idx_by_mmdd = {r["date_key"]:
               {"t0_time": r["t0_time_str"], "t2_time": r["t2_time_str"],
                "t3_time": r["t3_time_str"], "Weekday": r["Weekday_idx"]}
               for _, r in idx_df.iterrows() if pd.notna(r["date_key"])}

# ===================== Gather Day Files =================
obs_days = list_days(OBS_DIR, FILE_PREFIX)
phy_days = list_days(PHY_DIR, FILE_PREFIX)
obs_map = {extract_mmdd(f): f for f in obs_days}
phy_map = {extract_mmdd(f): f for f in phy_days}
common_mmdd = sorted(set(obs_map.keys()) & set(phy_map.keys()) & set(idx_by_mmdd.keys()))
if len(common_mmdd) == 0:
    raise RuntimeError("No overlapping days across observed/, physical/, and index.xlsx")

# ===================== Build per-day tensors =================
X_days, y_days, meta_days = [], [], []

for mmdd in common_mmdd:
    # ---- observed
    obs_fp = os.path.join(OBS_DIR, obs_map[mmdd])
    df_obs = pd.read_excel(obs_fp)
    m_obs = norm_colmap(df_obs)
    def oget(name):
        key = re.sub(r"[\s_]+","", name.strip().lower())
        if key in m_obs: return df_obs[m_obs[key]].values
        raise ValueError(f"{obs_fp} missing column: {name}")

    flow    = oget("Flow per hour")
    speed   = oget("Speed")
    density = oget("Density")
    queue   = oget("Queue")
    tt      = oget("tt_obs_min")

    tcol = find_time_column(df_obs)
    if tcol is not None:
        times = df_obs[tcol]
        try: times = pd.to_datetime(times)
        except Exception: pass
    else:
        times = pd.date_range("00:00", periods=len(df_obs), freq="5min")

    time_feats = build_time_features(pd.Series(times))
    tf = freeflow_time_from_speed(speed, SPEED_UNIT)  # <<< MPH/MILES

    # ---- physical
    phy_fp = os.path.join(PHY_DIR, phy_map[mmdd])
    df_phy = pd.read_excel(phy_fp)
    m_phy = norm_colmap(df_phy)
    def pget(name):
        key = re.sub(r"[\s_]+","", name.strip().lower())
        if key in m_phy: return df_phy[m_phy[key]].values
        raise ValueError(f"{phy_fp} missing column: {name}")

    queue_obs = pget("queue_obs")
    queue_fit = pget("queue_fit")
    mu_hat    = pget("mu_hat")
    flow_obs  = pget("flow_obs")

    # ---- align
    T = min(len(flow), len(queue_fit), len(mu_hat), len(time_feats))
    flow, speed, density, queue, tt, tf = flow[:T], speed[:T], density[:T], queue[:T], tt[:T], tf[:T]
    queue_obs, queue_fit, mu_hat, flow_obs = queue_obs[:T], queue_fit[:T], mu_hat[:T], flow_obs[:T]
    time_feats = time_feats[:T]

    # ---- congestion window from index
    row = idx_by_mmdd[mmdd]
    t0_str, t3_str = row["t0_time"], row["t3_time"]
    i0 = idx_from_str(t0_str, times)
    i3 = idx_from_str(t3_str, times)

    cong_mask = np.zeros(T, dtype=bool)
    if i0 is not None and i3 is not None and i3 > i0:
        cong_mask[i0:i3+1] = True
    else:
        cong_mask = queue_fit > np.percentile(queue_fit, 60)

    weekday = int(row["Weekday"])
    weekday_onehot = np.eye(7)[weekday % 7]
    weekday_feats = np.repeat(weekday_onehot.reshape(1,-1), T, axis=0)

    # ---- features & target
    X = np.column_stack([
        flow, speed, density, queue,               # observed
        queue_obs, queue_fit, mu_hat, flow_obs,    # physical
        tf,                                        # explicit tf (mph+mile)
        time_feats,                                # hour_sin, hour_cos
        weekday_feats                              # weekday one-hot
    ])
    y = tt.astype(float)

    # reference delay (min): 60 * queue_fit / mu_hat
    mu_safe = np.clip(mu_hat, 1e-6, None)
    delay_ref = 60.0 * (queue_fit / mu_safe)

    meta = {"tf": tf.astype(float),
            "delay_ref": delay_ref.astype(float),
            "cong_mask": cong_mask.astype(bool)}

    X_days.append(X); y_days.append(y); meta_days.append(meta)

# ---- checks
lengths = [Xi.shape[0] for Xi in X_days]
if len(set(lengths)) != 1:
    raise ValueError(f"Per-day lengths not equal: {lengths}")
points_per_day = lengths[0]
n_days = len(X_days)

# ===================== Train/Test split =================
if train_days + test_days > n_days:
    raise ValueError(f"train_days({train_days})+test_days({test_days}) > {n_days}")

X_train = np.array(X_days[:train_days]);    y_train = np.array(y_days[:train_days]);    meta_train = meta_days[:train_days]
X_test  = np.array(X_days[train_days:train_days+test_days]); y_test  = np.array(y_days[train_days:train_days+test_days]); meta_test  = meta_days[train_days:train_days+test_days]

# flatten to long
X_train_long = X_train.reshape(-1, X_train.shape[-1])
y_train_long = y_train.reshape(-1)
tf_train_long = np.concatenate([m["tf"] for m in meta_train])
delay_ref_train_long = np.concatenate([m["delay_ref"] for m in meta_train])
cong_train_long = np.concatenate([m["cong_mask"] for m in meta_train])

# add tail for test continuity
X_tail = X_train_long[-seq_len:, :]
y_tail = y_train_long[-seq_len:]
tf_tail = tf_train_long[-seq_len:]
delay_ref_tail = delay_ref_train_long[-seq_len:]
cong_tail = cong_train_long[-seq_len:]

X_test_long = np.vstack([X_tail, X_test.reshape(-1, X_test.shape[-1])])
y_test_long = np.concatenate([y_tail, y_test.reshape(-1)])
tf_test_long = np.concatenate([tf_tail, np.concatenate([m["tf"] for m in meta_test])])
delay_ref_test_long = np.concatenate([delay_ref_tail, np.concatenate([m["delay_ref"] for m in meta_test])])
cong_test_long = np.concatenate([cong_tail, np.concatenate([m["cong_mask"] for m in meta_test])])

# ===================== Target standardization =================
if use_y_standardize:
    y_mu = y_train_long.mean()
    y_std = y_train_long.std() if y_train_long.std()>0 else 1.0
else:
    y_mu, y_std = 0.0, 1.0

y_train_norm = (y_train_long - y_mu) / y_std
y_test_concat_norm = (y_test_long - y_mu) / y_std

# ===================== Dataset =========================
class SlidingSeq(Dataset):
    def __init__(self, X, y_norm, tf_min, delay_ref_min, cong_mask, seq_len):
        self.X = X.astype(np.float32)
        self.y = y_norm.astype(np.float32)
        self.tf = tf_min.astype(np.float32)
        self.dref = delay_ref_min.astype(np.float32)
        self.cong = cong_mask.astype(bool)
        self.seq_len = seq_len
    def __len__(self):
        return len(self.X) - self.seq_len
    def __getitem__(self, i):
        sl = slice(i, i+self.seq_len)
        x_seq = self.X[sl, :]
        y_target = self.y[i+self.seq_len]
        tf_now   = self.tf[i+self.seq_len]
        dref_now = self.dref[i+self.seq_len]
        cong_now = self.cong[i+self.seq_len]
        return (torch.from_numpy(x_seq),
                torch.tensor([y_target]),
                torch.tensor([tf_now]),
                torch.tensor([dref_now]),
                torch.tensor([1.0 if cong_now else 0.0]))

train_ds = SlidingSeq(X_train_long, y_train_norm, tf_train_long, delay_ref_train_long, cong_train_long, seq_len)
test_ds  = SlidingSeq(X_test_long,  y_test_concat_norm, tf_test_long, delay_ref_test_long, cong_test_long, seq_len)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=1, shuffle=False)

# ===================== Model ===========================
class PGLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
    def forward(self, x_seq, tf_min):
        out, _ = self.lstm(x_seq)
        h_last = out[:, -1, :]
        raw_delay = self.fc(h_last)
        delay_pos = self.softplus(raw_delay)
        tt_pred_min = tf_min + delay_pos
        return tt_pred_min, delay_pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train_long.shape[-1]
model = PGLSTM(input_size, hidden_size, num_layers).to(device)
mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===================== Train ===========================
model.train()
for epoch in range(1, num_epochs+1):
    total = 0.0
    for xb, yb_norm, tfb, drefb, congb in train_loader:
        xb = xb.to(device); tfb = tfb.to(device)
        yb_norm = yb_norm.to(device); drefb = drefb.to(device); congb = congb.to(device)

        optimizer.zero_grad()
        tt_pred_min, delay_pos = model(xb, tfb)
        y_pred_norm = (tt_pred_min - y_mu) / y_std

        loss_pred = mse(y_pred_norm, yb_norm)
        drefb_clamped = torch.clamp(drefb, 0.0, 120.0)
        loss_phys_q = mse(delay_pos, drefb_clamped)

        outside = (1.0 - congb)
        loss_phys_ff = (delay_pos.squeeze(1) * outside.squeeze(1)).pow(2).mean() if outside.sum() > 0 else torch.tensor(0.0, device=device)

        loss = loss_pred + lambda_phys_q*loss_phys_q + lambda_phys_ff*loss_phys_ff
        loss.backward()
        optimizer.step()
        total += loss.item()*xb.size(0)

    if epoch==1 or epoch%5==0:
        print(f"Epoch {epoch:03d} | Loss: {total/len(train_ds):.6f}")

# ===================== Predict last-day only ==================
model.eval()
preds_min = []
with torch.no_grad():
    for xb, _, tfb, _, _ in test_loader:
        xb = xb.to(device); tfb = tfb.to(device)
        tt_pred_min, _ = model(xb, tfb)
        preds_min.append(tt_pred_min.cpu().numpy().ravel()[0])

preds_min = np.array(preds_min)
preds_lastday = preds_min[-points_per_day:]
y_true_lastday = y_test.reshape(-1)[-points_per_day:]

# ===================== Metrics & Save ==================
mae  = mean_absolute_error(y_true_lastday, preds_lastday)
rmse = np.sqrt(mean_squared_error(y_true_lastday, preds_lastday))
mape = mape_safe(y_true_lastday, preds_lastday)
r2   = r2_score(y_true_lastday, preds_lastday)

print("PG-LSTM (physics-guided, mph+miles) — last-day evaluation")
print(f"MAE: {mae:.4f} min | RMSE: {rmse:.4f} min | MAPE: {mape:.2f}% | R²: {r2:.4f}")

pd.DataFrame({
    "Model": ["PG-LSTM (mph+miles)"],
    "TrainDays": [train_days],
    "TestDays": [test_days],
    "SeqLen": [seq_len],
    "Hidden": [hidden_size],
    "MAE(min)": [mae],
    "RMSE(min)": [rmse],
    "MAPE(%)": [mape],
    "R2": [r2],
    "lambda_phys_q":[lambda_phys_q],
    "lambda_phys_ff":[lambda_phys_ff]
}).to_excel(OUTPUT_DIR / "evaluation_metrics_PG_LSTM.xlsx", index=False)


pd.DataFrame({
    "timestep": np.arange(points_per_day),
    "tt_true_min": y_true_lastday,
    "tt_pred_min": preds_lastday
}).to_excel(OUTPUT_DIR / "prediction_vs_actual_lastday.xlsx", index=False)


plot_dir = OUTPUT_DIR / "prediction_plots_PG_LSTM_lastday"
plot_dir.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(10,5))
plt.plot(y_true_lastday, label="Actual tt (min)")
plt.plot(preds_lastday, "--", label="Predicted tt (min)")
plt.xlabel("Time step (5-min)")
plt.ylabel("Travel time (min)")
plt.title("PG-LSTM vs Actual (Last Day)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(plot_dir / "last_day_tt.png", dpi=300)
plt.close()
print("Saved: evaluation_metrics_PG_LSTM.xlsx, prediction_vs_actual_lastday.xlsx, plots/")
