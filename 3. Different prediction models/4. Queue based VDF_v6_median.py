#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Queue-based γ fitting with strict 80:20 split and MEAN aggregation.

- Sort by _#### in filename; keep only successfully-fitted days as "valid".
- TRAIN_DAYS = floor(0.8 * len(valid_days)).
- Aggregate params (mu, gamma) by MEAN over the FIRST TRAIN_DAYS valid days.
- Plot:
  * Train days (first 80% valid): 2 lines -> Observed + Calibrated (daily fit)
  * Test days  (last 20% valid):  2 lines -> Observed + Aggregated Predicted (mean μ,γ)

Exports:
  - tt_estimate_gamma_from_tt_daily.xlsx
  - tt_estimate_gamma_from_tt_full.xlsx
  - oos_eval_summary.csv
  - timeseries_day<TESTDAY>_oos_pred.csv
  - plots_gamma_from_tt/agg_plot_<day>.png
  - timeseries_last_day_<day>.csv
"""

import os, re, glob, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 0) Basic Configuration ============
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_Queue based"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = OUTPUT_DIR / "plots_gamma_from_tt"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

OUT_DAILY_SIMPLE_XLSX = "tt_estimate_gamma_from_tt_daily.xlsx"
OUT_EVAL_FULL_XLSX    = "tt_estimate_gamma_from_tt_full.xlsx"
OUT_OOS_SUMMARY_CSV   = "oos_eval_summary.csv"

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

# Columns
COL_FLOW  = "Flow per hour"
COL_SPEED = "Speed"
COL_QUEUE = "Queue"
TT_CANDIDATES = ["Travel time", "Travel time (min)", "TT", "tt", "tt_obs_min"]

# Units & constants
# (If your Flow/Queue are not per-lane, normalize before using.)
V_CO_KMH  = 70.0
V_F_KMH   = 70.0
L_KM      = 0.23
TT_CO_MIN = (L_KM / V_CO_KMH) * 60.0
TT_FF_MIN = (L_KM / V_F_KMH) * 60.0

STEP_MIN      = 5   # minutes per step (for time axis if no timestamps)
MIN_Q_POINTS  = 5
EPS_QUEUE     = 1e-6
MIN_RUN_EXIT  = 3

plt.rcParams['font.family'] = ['Times New Roman']

# ============ 1) Utilities ============
def infer_day4_from_filename(path: str):
    m = re.search(r"_(\d{4})\.xlsx?$", os.path.basename(path))
    return m.group(1) if m else None

def infer_day_label(path: str):
    tok = infer_day4_from_filename(path)
    return tok if tok else os.path.splitext(os.path.basename(path))[0]

def safe_mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[m] - yhat[m]))) if m.any() else np.nan

def safe_rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((yhat[m] - y[m])**2))) if m.any() else np.nan

def safe_mape(y, yhat, eps=1e-6):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat) & (np.abs(y) > eps)
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100.0) if m.any() else np.nan

def find_tt_obs_column(df: pd.DataFrame):
    for c in TT_CANDIDATES:
        if c in df.columns:
            return c
    return None

def get_tt_obs_from_df(df: pd.DataFrame):
    col = find_tt_obs_column(df)
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").values, col
    if COL_SPEED not in df.columns:
        raise KeyError("No observed TT column and no Speed column to derive TT.")
    spd = np.clip(pd.to_numeric(df[COL_SPEED], errors="coerce").values, 1e-6, None)
    return (L_KM / spd) * 60.0, "[from speed]"

def coerce_queue_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def detect_t0_t3(q, eps=0.0, min_run_exit=3):
    q = np.asarray(q, float)
    pos = np.flatnonzero(q > eps)
    if pos.size == 0:
        return None, None
    t0 = int(pos[0])
    j = t0 + 1
    while j <= len(q) - min_run_exit:
        if np.all(q[j:j+min_run_exit] <= eps):
            t3 = j - 1
            while t3 > t0 and q[t3] <= eps:
                t3 -= 1
            return int(t0), int(t3)
        j += 1
    return int(t0), int(pos[-1])

def fit_gamma_from_tt_nooffset(t_hour, tt_obs_min, t0_idx, t3_idx, mu_const):
    """
    Least-squares cubic fit on [t0, t3]:
        tt - TT_CO_MIN ≈ (γ/(3μ)) * (t - t0)^2 * (t3 - t) * 60
    Solve α = γ/(3μ) by:
        α = (Z^T * Y) / (Z^T * Z), with Z = ((t - t0)^2 (t3 - t)) * 60, Y = tt - TT_CO_MIN
        then γ = 3 μ α
    """
    if (t0_idx is None) or (t3_idx is None) or not np.isfinite(mu_const) or mu_const <= 0:
        return np.nan
    seg = slice(t0_idx, t3_idx + 1)
    t_seg = np.asarray(t_hour[seg], float)
    y_obs = np.asarray(tt_obs_min[seg], float)
    m = np.isfinite(t_seg) & np.isfinite(y_obs)
    if m.sum() < 3:
        return np.nan
    t0h, t3h = float(t_hour[t0_idx]), float(t_hour[t3_idx])
    Z = ((t_seg - t0h) ** 2) * (t3h - t_seg) * 60.0
    Y = y_obs - TT_CO_MIN
    mm = np.isfinite(Z) & np.isfinite(Y) & (Z > 0)
    if mm.sum() < 2:
        return np.nan
    denom = float(np.dot(Z[mm], Z[mm]))
    if denom <= 0:
        return np.nan
    alpha = float(np.dot(Z[mm], Y[mm]) / denom)
    alpha = max(0.0, alpha)
    return 3.0 * mu_const * alpha

# ============ 2) Read files (sorted) & daily fitting ============
files = sorted(
    glob.glob(FILE_GLOB),
    key=lambda p: int(infer_day4_from_filename(p)) if infer_day4_from_filename(p) else 10**9
)
if not files:
    raise FileNotFoundError(f"No files found: {FILE_GLOB}")

daily_rows = []   # valid days only
ts_blocks  = []   # valid days only (store t0, t3)
valid_days = []   # list of day labels (sorted) for 80:20 split

for fp in files:
    day = infer_day_label(fp)
    df  = pd.read_excel(fp)

    if (COL_FLOW not in df.columns) or (COL_QUEUE not in df.columns):
        continue

    try:
        tt_obs, _ = get_tt_obs_from_df(df)
    except Exception as e:
        warnings.warn(f"{fp} cannot get observed TT: {e}")
        continue

    flow = pd.to_numeric(df[COL_FLOW], errors="coerce").to_numpy(float)
    q    = coerce_queue_series(df[COL_QUEUE]).to_numpy(float)
    n    = len(df)
    if np.isfinite(q).sum() < MIN_Q_POINTS:
        continue

    t0_idx, t3_idx = detect_t0_t3(q, eps=EPS_QUEUE, min_run_exit=MIN_RUN_EXIT)
    if t0_idx is None or t3_idx is None or t0_idx >= t3_idx:
        continue

    t_hour = np.arange(n, dtype=float) * (STEP_MIN / 60.0)

    seg_flow = flow[t0_idx:t3_idx+1]
    mu_const = float(np.nanmedian(seg_flow[np.isfinite(seg_flow)])) if np.isfinite(seg_flow).any() else np.nan

    gamma_hat = fit_gamma_from_tt_nooffset(t_hour, tt_obs, t0_idx, t3_idx, mu_const)

    # in-sample (daily-calibrated) prediction for this day
    tt_cal = np.full(n, TT_FF_MIN, dtype=float)
    if np.isfinite(mu_const) and np.isfinite(gamma_hat) and mu_const > 0:
        Z_all = (t_hour - t_hour[t0_idx])**2 * (t_hour[t3_idx] - t_hour)
        w_all = (gamma_hat / (3.0 * mu_const)) * Z_all * 60.0
        w_all = np.maximum(0.0, w_all)
        tt_cal[t0_idx:t3_idx+1] = TT_CO_MIN + w_all[t0_idx:t3_idx+1]

    mae  = safe_mae(tt_obs, tt_cal)
    rmes = safe_rmse(tt_obs, tt_cal)
    mape = safe_mape(tt_obs, tt_cal)

    daily_rows.append({
        "day": day, "mu": mu_const, "gamma": gamma_hat,
        "MAE": mae, "RMSE": rmes, "MAPE_%": mape,
        "t0": t0_idx, "t3": t3_idx, "n": n
    })
    ts_blocks.append(pd.DataFrame({
        "day": day,
        "idx": np.arange(n, dtype=int),
        "tt_obs": tt_obs,
        "tt_cal": tt_cal,   # calibrated with daily μ,γ
        "t0": t0_idx,
        "t3": t3_idx
    }))
    valid_days.append(day)

print(f"[INFO] Successfully fitted valid days: {len(valid_days)}")

daily_df = pd.DataFrame(daily_rows)
ts_all   = pd.concat(ts_blocks, ignore_index=True) if ts_blocks else pd.DataFrame()

# ============ 3) Export daily & full Excel ============
if not daily_df.empty:
    with pd.ExcelWriter(OUTPUT_DIR / OUT_DAILY_SIMPLE_XLSX, engine="openpyxl") as w:
        daily_df.sort_values("day").to_excel(w, index=False, sheet_name="daily_summary")
    print(f"[OK] Exported: {OUT_DAILY_SIMPLE_XLSX}")

if not daily_df.empty and not ts_all.empty:
    with pd.ExcelWriter(OUTPUT_DIR / OUT_EVAL_FULL_XLSX, engine="openpyxl") as w:
        daily_df.sort_values("day").to_excel(w, index=False, sheet_name="daily_summary")
        ts_all.head(1000).to_excel(w, index=False, sheet_name="timeseries_sample")
    print(f"[OK] Exported: {OUT_EVAL_FULL_XLSX}")

# ============ 4) Strict 80:20 split over VALID days + MEAN aggregation ============
oos_records = []

if len(valid_days) < 5:
    print(f"[INFO] Not enough valid days ({len(valid_days)}) for 80:20 split.")
else:
    TOTAL_VALID = len(valid_days)
    TRAIN_DAYS  = int(np.floor(0.8 * TOTAL_VALID))
    TRAIN_SET   = set(valid_days[:TRAIN_DAYS])
    TEST_SET    = set(valid_days[TRAIN_DAYS:])

    train_df = daily_df[daily_df["day"].isin(TRAIN_SET)].sort_values("day")
    mu_agg    = float(np.nanmedian(train_df["mu"].values)) if not train_df.empty else np.nan
    gamma_agg = float(np.nanmedian(train_df["gamma"].values)) if not train_df.empty else np.nan
    print(f"\n[OOS] TRAIN_DAYS={TRAIN_DAYS}/{TOTAL_VALID} | Aggregated (mean) μ={mu_agg:.2f}, γ={gamma_agg:.4g}")

    # Loop over ALL valid days to make plots per rule; compute OOS metrics for TEST_SET
    for day in valid_days:
        ts_day = ts_all[ts_all["day"] == day].copy()
        if ts_day.empty:
            continue

        x = ts_day["idx"].to_numpy(int)
        n = len(ts_day)
        t0_idx = int(ts_day["t0"].iloc[0]) if np.isfinite(ts_day["t0"].iloc[0]) else None
        t3_idx = int(ts_day["t3"].iloc[0]) if np.isfinite(ts_day["t3"].iloc[0]) else None

        # Build aggregated prediction (needed for TEST plots and OOS metrics)
        t_hour = np.arange(n, dtype=float) * (STEP_MIN / 60.0)
        tt_pred_agg = np.full(n, TT_FF_MIN, dtype=float)
        if (t0_idx is not None) and (t3_idx is not None) and np.isfinite(mu_agg) and np.isfinite(gamma_agg) and (mu_agg > 0):
            Z = (t_hour - t_hour[t0_idx])**2 * (t_hour[t3_idx] - t_hour)
            w = (gamma_agg / (3.0 * mu_agg)) * Z * 60.0
            w = np.maximum(0.0, w)
            tt_pred_agg[t0_idx:t3_idx+1] = TT_CO_MIN + w[t0_idx:t3_idx+1]

        is_train = (day in TRAIN_SET)

        # ----- Plot per-day according to rule -----
        plt.figure(figsize=(9, 4), dpi=200)
        if is_train:
            # Train: 2 lines -> Observed + Calibrated (daily)
            plt.plot(x, ts_day["tt_obs"].to_numpy(float), label="Observed TT", linewidth=1.8)
            plt.plot(x, ts_day["tt_cal"].to_numpy(float), label="Calibrated TT (daily)", linewidth=1.2)
            plt.title(f"Train Day {day}: Observed vs Calibrated")
        else:
            # Test: 2 lines -> Observed + Aggregated Predicted (mean μ,γ)
            obs = ts_day["tt_obs"].to_numpy(float)
            plt.plot(x, obs, label="Observed TT", linewidth=1.8)
            plt.plot(x, tt_pred_agg, label="Aggregated Predicted TT (mean params)", linewidth=1.6, color="#F4A7A7")
            plt.title(f"Test Day {day}: Observed vs Aggregated Prediction\nμ̄={mu_agg:.0f}, γ̄={gamma_agg:.3g}")

        plt.xlabel("Index (5-min steps)")
        plt.ylabel("Travel Time (min)")
        plt.legend()
        plt.tight_layout()
        out_png = PLOT_DIR / f"agg_plot_{day}.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[PLOT] Saved {out_png.name}")

        # ----- OOS metrics + per-day CSV for TEST_SET -----
        if not is_train:
            obs = ts_day["tt_obs"].to_numpy(float)
            L = min(len(obs), len(tt_pred_agg))
            mae = safe_mae(obs[:L], tt_pred_agg[:L])
            r   = safe_rmse(obs[:L], tt_pred_agg[:L])
            mp  = safe_mape(obs[:L], tt_pred_agg[:L])

            oos_records.append({
                "train_days": TRAIN_DAYS,
                "test_day": day,
                "mu_agg_mean": mu_agg,
                "gamma_agg_mean": gamma_agg,
                "OOS_MAE_TT": mae,
                "OOS_RMSE_TT": r,
                "OOS_MAPE_TT_%": mp
            })

            # Save per-test-day OOS timeseries
            out_oos_csv = OUTPUT_DIR / f"timeseries_day{day}_oos_pred.csv"
            pd.DataFrame({
                "day": day,
                "idx": x[:L],
                "TT_obs_min": obs[:L],
                "TT_pred_oos_min": tt_pred_agg[:L]
            }).to_csv(out_oos_csv, index=False)
            print(f"[OOS CSV] Saved {out_oos_csv.name} | MAE={mae:.3f}, RMSE={r:.3f}, MAPE={mp:.2f}%")

# ============ 5) Export OOS summary & last valid day CSV ============
if oos_records:
    pd.DataFrame(oos_records).to_csv(OUTPUT_DIR / OUT_OOS_SUMMARY_CSV, index=False)
    print(f"[OK] Exported {OUT_OOS_SUMMARY_CSV}")

if not ts_all.empty and valid_days:
    last_day = valid_days[-1]
    last_ts = ts_all[ts_all["day"] == last_day][["idx", "tt_obs", "tt_cal"]].copy()
    out_csv_last = OUTPUT_DIR / f"timeseries_last_day_{last_day}.csv"
    last_ts.to_csv(out_csv_last, index=False, encoding="utf-8-sig")
    print(f"[OK] Exported last-day time series: {out_csv_last.name}")
