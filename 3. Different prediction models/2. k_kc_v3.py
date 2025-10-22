#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.options.display.float_format = '{:.3f}'.format
np.set_printoptions(precision=3, suppress=True)

# ================= 0) Basic Configuration =================
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_BPR_density"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

# Column names
COL_FLOW    = "Flow per hour"
COL_SPEED   = "Speed"
COL_DENSITY = "Density"
COL_TT_OBS  = None
TT_CANDIDATES = ["Travel time", "Travel time (min)", "TT", "tt", "tt_obs_min"]

# Constants
VF_KMH = 70.0
L_KM   = 0.23
KC     = 30.0
T_FREE_MIN = (L_KM / VF_KMH) * 60.0

# Grid search settings
ALPHA0, BETA0 = 0.15, 4.0
COARSE_ALPHA = (max(0.01, ALPHA0/5), min(0.8, ALPHA0*5), 35)
COARSE_BETA  = (max(0.5,  BETA0-3),  BETA0+3,             35)
REFINE_ALPHA_HALFSPAN = 0.10
REFINE_BETA_HALFSPAN  = 1.00
REFINE_STEPS = 41

# Output and plotting
PARAM_AGG = "median"
TRIM_ALPHA = 0.10
OUT_DAILY_CSV = "bpr_daily_params_and_metrics.csv"
OUT_TS_CSV    = "bpr_daily_timeseries.csv"
OUT_OOS_SUM   = "bpr_oos_eval_summary.csv"
SAVE_DAILY_PLOTS = True
PLOT_DIR = OUTPUT_DIR / "plots_bpr_daily_calib"
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = ['Times New Roman']

# ================= 1) Utility Functions =================
def infer_day_from_filename(path):
    m = re.search(r"_(\d{4})\.xlsx?$", os.path.basename(path))
    return f"{m.group(1)[:2]}-{m.group(1)[2:]}" if m else os.path.basename(path)

def infer_day4_from_filename(path):
    m = re.search(r"_(\d{4})\.xlsx?$", os.path.basename(path))
    return m.group(1) if m else None  

def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]))) if mask.any() else np.nan

def safe_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_pred[mask] - y_true[mask])**2))) if mask.any() else np.nan

def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    return float(np.mean(np.abs((y_true[mask]-y_pred[mask]) / y_true[mask])) * 100.0) if mask.any() else np.nan

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2 or np.allclose(y_true[mask], y_true[mask].mean()): return np.nan
    try: return float(r2_score(y_true[mask], y_pred[mask]))
    except Exception: return np.nan

def bpr_tt_from_k(k_veh_per_km, alpha, beta, kc=KC, t_free_min=T_FREE_MIN):
    x = np.clip(np.asarray(k_veh_per_km, dtype=float) / kc, 0, None)
    return t_free_min * (1.0 + alpha * (x ** beta))

def find_tt_obs_column(df):
    if COL_TT_OBS and COL_TT_OBS in df.columns:
        return COL_TT_OBS
    for c in TT_CANDIDATES:
        if c in df.columns: return c
    return None

def get_tt_obs_from_df(df):
    col = find_tt_obs_column(df)
    if col:
        return df[col].astype(float).values, col
    else:
        if COL_SPEED not in df.columns:
            raise KeyError("Observed travel time column not found, and Speed column is missing — cannot compute observed TT.")
        return (L_KM / np.clip(df[COL_SPEED].astype(float).values, 1e-6, None) * 60.0), f"[from {COL_SPEED}]"

def get_density_series(df):
    if COL_DENSITY in df.columns:
        k = pd.to_numeric(df[COL_DENSITY], errors="coerce").astype(float).values
    else:
        if (COL_FLOW not in df.columns) or (COL_SPEED not in df.columns):
            raise KeyError("Density column missing, and cannot compute from flow/speed (requires both Flow and Speed).")
        q = pd.to_numeric(df[COL_FLOW], errors="coerce").astype(float).values
        v = np.clip(pd.to_numeric(df[COL_SPEED], errors="coerce").astype(float).values, 1e-6, None)
        k = q / v
    return k

def calibrate_alpha_beta_for_day(k_veh_per_km, tt_obs_min,
                                 coarse_alpha=COARSE_ALPHA, coarse_beta=COARSE_BETA,
                                 refine_da=REFINE_ALPHA_HALFSPAN, refine_db=REFINE_BETA_HALFSPAN,
                                 refine_steps=REFINE_STEPS):
    y = np.asarray(tt_obs_min, dtype=float)
    x = np.clip(np.asarray(k_veh_per_km, dtype=float) / KC, 0, None)

    # Coarse search
    a_grid = np.linspace(*coarse_alpha)
    b_grid = np.linspace(*coarse_beta)
    Xbeta = {b: x**b for b in b_grid}
    best = {"mae": np.inf, "a": None, "b": None}
    for a in a_grid:
        for b in b_grid:
            yhat = T_FREE_MIN * (1.0 + a * Xbeta[b])
            mae = safe_mae(y, yhat)
            if mae < best["mae"]:
                best.update({"mae": mae, "a": float(a), "b": float(b)})

    # Refined search
    a2 = np.linspace(max(0.005, best["a"] - refine_da), best["a"] + refine_da, refine_steps)
    b2 = np.linspace(max(0.25, best["b"] - refine_db), best["b"] + refine_db, refine_steps)
    best2 = {"mae": np.inf, "a": None, "b": None}
    for a in a2:
        for b in b2:
            yhat = T_FREE_MIN * (1.0 + a * (x ** b))
            mae = safe_mae(y, yhat)
            if mae < best2["mae"]:
                best2.update({"mae": mae, "a": float(a), "b": float(b)})

    yhat_best = T_FREE_MIN * (1.0 + best2["a"] * (x ** best2["b"]))
    metrics = {
        "MAE_min":  best2["mae"],
        "RMSE_min": safe_rmse(y, yhat_best),
        "MAPE_%":   safe_mape(y, yhat_best),
        "R2":       safe_r2(y, yhat_best)
    }
    return best2["a"], best2["b"], yhat_best, metrics

def agg_params(alpha_list, beta_list, method="median", trim_alpha=0.10):
    a = pd.to_numeric(alpha_list, errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(beta_list,  errors="coerce").to_numpy(dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0: return np.nan, np.nan
    if method == "median":
        return float(np.median(a)), float(np.median(b))
    elif method == "trimmed_mean":
        def tmean(x, p):
            x = np.sort(x); n = len(x); k = int(np.floor(p*n))
            x2 = x[k:n-k] if n-2*k > 0 else x
            return float(np.mean(x2))
        return tmean(a, trim_alpha), tmean(b, trim_alpha)
    return float(np.median(a)), float(np.median(b))

# ================= 2) Read and calibrate =================
files = sorted(glob.glob(FILE_GLOB),
               key=lambda p: int(infer_day4_from_filename(p)) if infer_day4_from_filename(p) else 10**9)
if not files:
    raise FileNotFoundError(f"No files found: {FILE_GLOB}")

daily_rows, timeseries_rows = [], []

for fp in files:
    day = infer_day_from_filename(fp)
    df = pd.read_excel(fp)

    tt_obs, _ = get_tt_obs_from_df(df)
    k_km = get_density_series(df)
    a_star, b_star, tt_hat, m = calibrate_alpha_beta_for_day(k_km, tt_obs)

    daily_rows.append({
        "day": day, "n": len(df),
        "alpha": a_star, "beta": b_star,
        **m
    })

    ts = pd.DataFrame({
        "day": day, "idx": np.arange(len(df)),
        "k_veh_per_km": k_km,
        "tt_obs_min": tt_obs,
        "tt_bpr_min": tt_hat
    })
    if "time" in df.columns: ts["time"] = df["time"]
    timeseries_rows.append(ts)

    if SAVE_DAILY_PLOTS:
        x_axis = ts["time"] if "time" in ts.columns else ts["idx"]
        plt.figure(figsize=(8,3.6))
        plt.plot(x_axis, ts["tt_obs_min"], label="Observed")
        plt.plot(x_axis, ts["tt_bpr_min"], label="BPR(k/KC) fitted")
        plt.xlabel("Time" if "time" in ts.columns else "Index (5-min steps)")
        plt.ylabel("Travel time (min)")
        plt.title(f"{day}: α={a_star:.3f}, β={b_star:.3f}")
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOT_DIR / f"daily_tt_{day.replace('/', '-')}.png", dpi=300)
        plt.close()

daily_df = pd.DataFrame(daily_rows)
ts_all = pd.concat(timeseries_rows, ignore_index=True)

daily_df.to_csv(OUTPUT_DIR / OUT_DAILY_CSV, index=False)
ts_all.to_csv(OUTPUT_DIR / OUT_TS_CSV, index=False)
print(f"Exported: {OUT_DAILY_CSV}, {OUT_TS_CSV}")

# ================= 5) 8:2 Split Evaluation =================
TOTAL_DAYS = len(daily_df)
oos_records = []

if TOTAL_DAYS >= 5:
    TRAIN_DAYS = int(np.floor(0.8 * TOTAL_DAYS))
    TEST_DAYS  = TOTAL_DAYS - TRAIN_DAYS
    print(f"\n=== Aggregating α, β from first {TRAIN_DAYS} days ===")

    a_agg, b_agg = agg_params(daily_df["alpha"].iloc[:TRAIN_DAYS],
                              daily_df["beta"].iloc[:TRAIN_DAYS],
                              method=PARAM_AGG, trim_alpha=TRIM_ALPHA)
    print(f"Aggregated α={a_agg:.4f}, β={b_agg:.4f}")

    for i, test_row in enumerate(daily_df.iloc[TRAIN_DAYS:].itertuples(index=False)):
        test_day = test_row.day
        ts_test = ts_all[ts_all["day"] == test_day].copy()
        if ts_test.empty: continue

        k = pd.to_numeric(ts_test["k_veh_per_km"], errors="coerce").values
        tt_obs = pd.to_numeric(ts_test["tt_obs_min"], errors="coerce").values
        mask = np.isfinite(k) & np.isfinite(tt_obs)
        k, tt_obs = k[mask], tt_obs[mask]
        tt_pred = bpr_tt_from_k(k, a_agg, b_agg)

        mae, rmse, mape = safe_mae(tt_obs, tt_pred), safe_rmse(tt_obs, tt_pred), safe_mape(tt_obs, tt_pred)
        oos_records.append({
            "train_days": TRAIN_DAYS, "test_day": test_day,
            "alpha_agg": a_agg, "beta_agg": b_agg,
            "OOS_MAE_TT": mae, "OOS_RMSE_TT": rmse, "OOS_MAPE_TT_%": mape
        })

        out_ts = OUTPUT_DIR / f"bpr_timeseries_day{re.sub('[^0-9A-Za-z-]', '', test_day)}_agg_pred.csv"
        pd.DataFrame({"day": test_day, "k_veh_per_km": k,
                      "tt_obs_min": tt_obs, "tt_pred_agg_min": tt_pred}).to_csv(out_ts, index=False)
        print(f"[EVAL] {test_day}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")

        # Plot comparison for test day
        plt.figure(figsize=(9,4))
        x_plot = ts_test["time"] if "time" in ts_test.columns else ts_test["idx"]
        plt.plot(x_plot, tt_obs, label="Observed", linewidth=1.8)
        plt.plot(x_plot, ts_test["tt_bpr_min"], label="BPR fitted (daily α,β)", linewidth=1.2)
        plt.plot(x_plot, tt_pred, label="BPR aggregated (α,β)", linewidth=1.2)
        plt.xlabel("Time" if "time" in ts_test.columns else "Index (5-min steps)")
        plt.ylabel("Travel time (min)")
        plt.title(f"Travel Time Comparison — {test_day}\n"
                  f"Daily α={test_row.alpha:.3f}, β={test_row.beta:.3f} | "
                  f"Aggregated α={a_agg:.3f}, β={b_agg:.3f}\n"
                  f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOT_DIR / f"tt_compare_{test_day.replace('/', '-')}.png", dpi=300)
        plt.close()

    pd.DataFrame(oos_records).to_csv(OUTPUT_DIR / OUT_OOS_SUM, index=False)
    print(f"\nExported aggregated evaluation summary: {OUT_OOS_SUM}")

else:
    print(f"[EVAL] Not enough valid days ({TOTAL_DAYS}) for 8:2 evaluation.")
