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

# ================= 0) Basic Configuration =================
# The script is located in: Dataset/3. Different Models/
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/

# Input and output directories
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

# Output folder (same level as the script)
OUTPUT_DIR = BASE_DIR / "3. Different Models" / "Output_BPR_density"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")


# Column names (modify according to your file)
COL_FLOW    = "Flow per hour"   # veh/h (equivalent veh/h per 5-min interval)
COL_SPEED   = "Speed"           # km/h (used if no observed TT column)
COL_DENSITY = "Density"         # veh/km (if missing, will be computed as flow/speed)
COL_TT_OBS  = None              # If observed TT (min) exists, specify the column name here; otherwise leave None
TT_CANDIDATES = ["Travel time", "Travel time (min)", "TT", "tt", "tt_obs_min"]

# Fixed constants
VF_KMH = 70.0       # Free-flow speed (km/h)
L_KM   = 0.23       # Segment length (km)
KC     = 30.0       # Density threshold (veh/km)
T_FREE_MIN = (L_KM / VF_KMH) * 60.0  # Free-flow travel time (minutes)

# Initial reference and search range (two stages: coarse then fine; objective = MAE)
ALPHA0, BETA0 = 0.15, 4.0
COARSE_ALPHA = (max(0.01, ALPHA0/5), min(0.8, ALPHA0*5), 35)  # approx. [0.03, 0.75]
COARSE_BETA  = (max(0.5,  BETA0-3),  BETA0+3,             35) # approx. [1, 7]
REFINE_ALPHA_HALFSPAN = 0.10   # ±0.10 around coarse-search optimum
REFINE_BETA_HALFSPAN  = 1.00   # ±1.00 around coarse-search optimum
REFINE_STEPS = 41

# OOS extrapolation settings
N_TRAIN = 80                     # Aggregate parameters for first N_TRAIN days → predict day N_TRAIN+1 (if insufficient, use N-1→N)
PARAM_AGG = "median"             # "median" or "trimmed_mean"
TRIM_ALPHA = 0.10                # Trimming ratio for trimmed_mean

# Output and plotting
OUT_DAILY_CSV = "bpr_daily_params_and_metrics.csv"
OUT_TS_CSV    = "bpr_daily_timeseries.csv"
OUT_OOS_SUM   = "bpr_oos_eval_summary.csv"
SPLIT_DIR     = "bpr_daily_timeseries_excel"       # Folder: one Excel per day
ONEBOOK_XLSX  = "bpr_daily_timeseries_all.xlsx"    # Optional: one file with multiple sheets
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

# —— BPR(density-based)：x = k / KC ——
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
        # Compute observed TT (minutes) from speed
        return (L_KM / np.clip(df[COL_SPEED].astype(float).values, 1e-6, None) * 60.0), f"[from {COL_SPEED}]"

def get_density_series(df):
    """Prefer density column; otherwise compute k = q / v (veh/h ÷ km/h → veh/km)"""
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
    """Two-stage grid search minimizing MAE; returns a*, b*, yhat*, metrics (x = k/KC)"""
    y = np.asarray(tt_obs_min, dtype=float)
    x = np.clip(np.asarray(k_veh_per_km, dtype=float) / KC, 0, None)

    # 1) Coarse search
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

    # 2) Fine search
    a_lo = max(0.005, best["a"] - refine_da); a_hi = best["a"] + refine_da
    b_lo = max(0.25, best["b"] - refine_db);  b_hi = best["b"] + refine_db
    a2 = np.linspace(a_lo, a_hi, refine_steps)
    b2 = np.linspace(b_lo, b_hi, refine_steps)
    Xbeta2 = {b: x**b for b in b2}
    best2 = {"mae": np.inf, "a": None, "b": None}
    for a in a2:
        for b in b2:
            yhat = T_FREE_MIN * (1.0 + a * Xbeta2[b])
            mae = safe_mae(y, yhat)
            if mae < best2["mae"]:
                best2.update({"mae": mae, "a": float(a), "b": float(b)})

    # Final prediction and evaluation metrics
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

# ================= 2) Read data and calibrate day by day =================
files = sorted(glob.glob(FILE_GLOB),
               key=lambda p: int(infer_day4_from_filename(p)) if infer_day4_from_filename(p) else 10**9)
if not files:
    raise FileNotFoundError(f"No files found: {FILE_GLOB}")

print(f"Constants: vf={VF_KMH} km/h, L={L_KM} km, tf={T_FREE_MIN:.6f} min, KC={KC} veh/km")
print(f"Initial guess: alpha0={ALPHA0}, beta0={BETA0}\n")

daily_rows = []
timeseries_rows = []
days_in_order = []  # Record processing order for OOS

for fp in files:
    day = infer_day_from_filename(fp)
    days_in_order.append(day)
    df  = pd.read_excel(fp)

    # Observed travel time (TT)
    tt_obs, obs_source = get_tt_obs_from_df(df)

    # Observed density k (prefer density column, otherwise use flow/speed)
    k_km = get_density_series(df)

    # Calibrate α, β (for the day, using x = k/KC)
    a_star, b_star, tt_hat, m = calibrate_alpha_beta_for_day(k_km, tt_obs)

    # Daily-level results
    daily_rows.append({
        "day": day, "n": len(df),
        "alpha": a_star, "beta": b_star,
        "vf_kmh": VF_KMH, "KC": KC, "L_km": L_KM, "tf_min": T_FREE_MIN,
        **m
    })

    # Time-series results
    base = {
        "day": day,
        "idx": np.arange(len(df)),
        "k_veh_per_km": k_km,
        "x_k_over_KC": np.clip(k_km / KC, 0, None),
        "tt_obs_min": tt_obs,
        "tt_bpr_kKC_min": tt_hat
    }
    if COL_FLOW in df.columns:
        base["q_vph"] = pd.to_numeric(df[COL_FLOW], errors="coerce").astype(float).values
    if COL_SPEED in df.columns:
        base["v_kmh"] = pd.to_numeric(df[COL_SPEED], errors="coerce").astype(float).values

    ts = pd.DataFrame(base)
    if "time" in df.columns:
        ts["time"] = df["time"]
    timeseries_rows.append(ts)

    # Optional: daily plots
    if SAVE_DAILY_PLOTS:
        x_axis = ts["time"] if "time" in ts.columns else ts["idx"]
        plt.figure(figsize=(8,3.6))
        plt.plot(x_axis, ts["tt_obs_min"].values, label="Observed")
        plt.plot(x_axis, ts["tt_bpr_kKC_min"].values, label="BPR(k/KC) fitted")
        plt.xlabel("Time" if "time" in ts.columns else "Index (5-min steps)")
        plt.ylabel("Travel time (min)")
        plt.title(f"I-405 Daily Travel Time — {day}\n"
                  f"alpha={a_star:.4f}, beta={b_star:.4f} | "
                  f"MAE={m['MAE_min']:.2f}, RMSE={m['RMSE_min']:.2f}")
        plt.legend(); plt.tight_layout()
        out_png = Path(PLOT_DIR) / f"daily_tt_{day.replace('/', '-')}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()

daily_df = pd.DataFrame(daily_rows)
ts_all   = pd.concat(timeseries_rows, ignore_index=True) if timeseries_rows else pd.DataFrame()

# ================= 3) Export summary =================
daily_df.sort_values("day").to_csv(OUTPUT_DIR / OUT_DAILY_CSV, index=False)
ts_all.to_csv(OUTPUT_DIR / OUT_TS_CSV, index=False)
print(f"Exported: {OUT_DAILY_CSV}, {OUT_TS_CSV}")

# ================= 4) Split by day into multiple Excel files =================
split_dir = OUTPUT_DIR / SPLIT_DIR
split_dir.mkdir(parents=True, exist_ok=True)

base_cols = ["day", "idx", "k_veh_per_km", "x_k_over_KC", "tt_obs_min", "tt_bpr_kKC_min"]
extra_cols = []
if "time" in ts_all.columns: extra_cols.append("time")
if "q_vph" in ts_all.columns: extra_cols.append("q_vph")
if "v_kmh" in ts_all.columns: extra_cols.append("v_kmh")
cols_order = (["day"] + extra_cols + [c for c in base_cols if c not in (["day"] + extra_cols)])

for day, g in ts_all.groupby("day", sort=True):
    safe_day = str(day).replace("/", "-")
    out_path = split_dir / f"bpr_timeseries_{safe_day}.xlsx"
    g[cols_order].to_excel(out_path, index=False)

print(f"Exported by day to folder: {split_dir.resolve()}")

# ====== Optional: one Excel file with multiple sheets (one sheet per day) ======
make_one_workbook = True
if make_one_workbook:
    onebook_path = OUTPUT_DIR / ONEBOOK_XLSX
    with pd.ExcelWriter(onebook_path, engine="openpyxl", mode="w") as writer:
        for day, g in ts_all.groupby("day", sort=True):
            sheet = str(day).replace("/", "-")[:31]  # Excel sheet name max length 31 chars
            g[cols_order].to_excel(writer, index=False, sheet_name=sheet)
    print(f"Also created a multi-sheet file: {onebook_path.resolve()}")

# =================5) OOS: aggregate first N_TRAIN days → predict day N_TRAIN+1 (or last day) =================
oos_records = []
if len(days_in_order) >= 2:
    if len(days_in_order) >= N_TRAIN + 1:
        train_days = days_in_order[:N_TRAIN]
        test_day   = days_in_order[N_TRAIN]       
    else:
        train_days = days_in_order[:-1]        
        test_day   = days_in_order[-1]               # last day

    train_slice = daily_df[daily_df["day"].isin(train_days)]
    if train_slice.empty or (test_day not in daily_df["day"].values):
        print("[OOS] Training or test day missing, skipping OOS.")
    else:
        a_agg, b_agg = agg_params(train_slice["alpha"], train_slice["beta"],
                                  method=PARAM_AGG, trim_alpha=TRIM_ALPHA)

        ts_last = ts_all[ts_all["day"] == test_day].copy()
        if ts_last.empty or not np.isfinite(a_agg) or not np.isfinite(b_agg):
            print("[OOS] Test-day time series empty or invalid aggregate parameters, skipping plots and metrics.")
            mae = rm = mp = np.nan
            y_oos = np.array([])
        else:
            k_last = pd.to_numeric(ts_last["k_veh_per_km"], errors="coerce").to_numpy(float)
            y_obs = pd.to_numeric(ts_last["tt_obs_min"], errors="coerce").to_numpy(float)
            y_oos_full = bpr_tt_from_k(k_last, a_agg, b_agg)
            L = min(len(y_obs), len(y_oos_full))
            y_obs = y_obs[:L]; y_oos = y_oos_full[:L]
            mae = safe_mae(y_obs, y_oos)
            rm  = safe_rmse(y_obs, y_oos)
            mp  = safe_mape(y_obs, y_oos)

            # Write hourly OOS table
            out_ts = OUTPUT_DIR / f"timeseries_day{re.sub('[^0-9A-Za-z-]', '', test_day)}_oos_pred.csv"
            pd.DataFrame({
                "day": test_day,
                "idx": np.arange(L),
                "k_veh_per_km": k_last[:L],
                "tt_obs_min": y_obs,
                "tt_pred_oos_min": y_oos
            }).to_csv(out_ts, index=False)

        oos_records.append({
            "N_TRAIN": min(N_TRAIN, len(train_days)),
            "param_agg": PARAM_AGG,
            "test_day": test_day,
            "alpha_agg": a_agg,
            "beta_agg": b_agg,
            "OOS_MAE_TT": mae,
            "OOS_RMSE_TT": rm,
            "OOS_MAPE_TT_%": mp
        })

        # ================= 6) “Last-day” comparison plot (Observed vs In-sample vs OOS)  =================
        try:
            if ts_last.empty:
                print(f"[PLOT] Last day {test_day} has no time-series data, skipping plot.")
            else:
                # Choose x-axis: prefer time, otherwise index
                if "time" in ts_last.columns and ts_last["time"].notna().any():
                    x_last = ts_last["time"].values
                    x_label = "Time"
                else:
                    x_last = ts_last["idx"].values
                    x_label = "Index (5-min steps)"

                y_obs = pd.to_numeric(ts_last["tt_obs_min"], errors="coerce").to_numpy(float)
                y_fit = pd.to_numeric(ts_last["tt_bpr_kKC_min"], errors="coerce").to_numpy(float)

                # Align OOS sequence length
                if 'y_oos' in locals() and y_oos.size:
                    Lmin = min(len(x_last), len(y_obs), len(y_fit), len(y_oos))
                    x_plot = x_last[:Lmin]
                    y_obs  = y_obs[:Lmin]
                    y_fit  = y_fit[:Lmin]
                    y_oos  = y_oos[:Lmin]
                else:
                    Lmin = min(len(x_last), len(y_obs), len(y_fit))
                    x_plot = x_last[:Lmin]
                    y_obs  = y_obs[:Lmin]
                    y_fit  = y_fit[:Lmin]
                    y_oos  = None

                # Add OOS metrics in title (if available)
                title_extra = ""
                if y_oos is not None and y_oos.size and np.isfinite(mae) and np.isfinite(rm) and np.isfinite(mp):
                    title_extra = f" | OOS MAE={mae:.2f} RMSE={rm:.2f} MAPE={mp:.1f}%"

                plt.figure(figsize=(9,4))
                plt.plot(x_plot, y_obs, label="Observed", linewidth=1.8)
                if np.isfinite(y_fit).any():
                    plt.plot(x_plot, y_fit, label="BPR(k/KC) fitted (in-sample)", linewidth=1.2)
                if y_oos is not None and np.isfinite(y_oos).any():
                    plt.plot(x_plot, y_oos, label="BPR(k/KC) OOS (agg α,β)", linewidth=1.2)

                plt.title(f"Travel Time Comparison — {test_day}{title_extra}")
                plt.xlabel(x_label); plt.ylabel("Travel time (min)")
                plt.legend(); plt.tight_layout()
                out_png = Path(PLOT_DIR) / f"oos_tt_compare_{re.sub('[^0-9A-Za-z-]', '', test_day)}.png"
                plt.savefig(out_png, dpi=300); plt.close()
                print(f"[PLOT] Generated last-day comparison plot: {out_png.resolve()}")
        except Exception as e:
            print(f"[ERR][PLOT] Failed to generate last-day comparison plot: {e}")

# Write OOS summary
pd.DataFrame(oos_records).to_csv(OUTPUT_DIR / OUT_OOS_SUM, index=False)
print(f"Exported: {(OUTPUT_DIR / OUT_OOS_SUM).resolve()}")

# ================= End =================
print(f"Plot directory (if enabled): {Path(PLOT_DIR).resolve()}")