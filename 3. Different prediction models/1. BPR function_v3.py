#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, re, warnings
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


# The script is located in: Dataset/3. Different Models/
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/

# Input and output directories
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

# Output folder (same level as the script)
OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_BPR_flow"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")


# Column names (modify according to your file)
COL_FLOW   = "Flow per hour"   # veh/h (equivalent veh/h per 5 min)
COL_SPEED  = "Speed"           # km/h (used if no observed TT column)
COL_TT_OBS = None              # If travel time (min) exists, specify here; otherwise None
TT_CANDIDATES = ["Travel time", "Travel time (min)", "TT", "tt", "tt_obs_min"]

# Constants (modify as needed)
VF_KMH = 70.0       # free-flow speed km/h
CA_VPH = 1750.0     # capacity veh/h
L_KM   = 0.23       # length km
T_FREE_MIN = (L_KM / VF_KMH) * 60.0  # free-flow travel time (minutes)

# Two-stage grid search（coarse first, then fine；objective=MAE）
ALPHA0, BETA0 = 0.15, 4.0
COARSE_ALPHA = (max(0.01, ALPHA0/5), min(0.8, ALPHA0*5), 35)  # [0.03, 0.75]
COARSE_BETA  = (max(0.5,  BETA0-3),  BETA0+3,             35) # [1, 7]
REFINE_ALPHA_HALFSPAN = 0.10
REFINE_BETA_HALFSPAN  = 1.00
REFINE_STEPS = 41
TRIM_ALPHA = 0.10


# Output and plotting
OUT_DAILY_CSV = "bpr_daily_params_and_metrics.csv"
OUT_TS_CSV    = "bpr_daily_timeseries.csv"
OUT_OOS_SUM   = "bpr_oos_eval_summary.csv"
SPLIT_DIR     = "bpr_daily_timeseries_excel"
ONEBOOK_XLSX  = "bpr_daily_timeseries_all.xlsx"
SAVE_DAILY_PLOTS = True
PLOT_DIR = OUTPUT_DIR / "plots_bpr_daily_calib"
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
plt.rcParams['font.family'] = ['Times New Roman']

# ================= 1) Utility Functions =================
def infer_day_token(path):
    """Extract 4-digit numbers (e.g., 0731) from the filename; if none, return filename without extension"""
    m = re.search(r"_(\d{4})\.xlsx?$", os.path.basename(path))
    return m.group(1) if m else os.path.splitext(os.path.basename(path))[0]

def pretty_day(day_token):
    """Convert '0731' -> '07-31'; if not a 4-digit number, return as is"""
    s = re.sub(r"\D", "", str(day_token).strip())
    return f"{s[:2]}-{s[2:]}" if len(s) == 4 else str(day_token)

def norm_day4(x):
    """Normalize to a 4-digit string ('731' or 731 -> '0731')"""
    s = re.sub(r"\D", "", str(x).strip())
    return f"{int(s):04d}" if s else ""

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

def bpr_tt(flow_vph, alpha, beta, cap_vph=CA_VPH, t_free_min=T_FREE_MIN):
    x = np.clip(np.asarray(flow_vph, dtype=float) / cap_vph, 0, None)
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
            raise KeyError("Observed travel time column not found, and missing speed column — unable to compute observed TT.")
        return (L_KM / np.clip(df[COL_SPEED].astype(float).values, 1e-6, None) * 60.0), f"[from {COL_SPEED}]"

def calibrate_alpha_beta_for_day(q_vph, tt_obs_min,
                                 coarse_alpha=COARSE_ALPHA, coarse_beta=COARSE_BETA,
                                 refine_da=REFINE_ALPHA_HALFSPAN, refine_db=REFINE_BETA_HALFSPAN,
                                 refine_steps=REFINE_STEPS):
    """Two-stage grid search minimizing MAE; returns a*, b*, yhat*, and metrics"""
    y = np.asarray(tt_obs_min, dtype=float)
    x = np.clip(np.asarray(q_vph, dtype=float) / CA_VPH, 0, None)

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

    # Final prediction and metrics
    yhat_best = T_FREE_MIN * (1.0 + best2["a"] * (x ** best2["b"]))
    metrics = {
        "MAE_min":  best2["mae"],
        "RMSE_min": safe_rmse(y, yhat_best),
        "MAPE_%":   safe_mape(y, yhat_best),
        "R2":       safe_r2(y, yhat_best)
    }
    return best2["a"], best2["b"], yhat_best, metrics

def agg_params(alpha_list, beta_list, method="median", trim_alpha=0.10):
    a = np.asarray(alpha_list, float); a = a[np.isfinite(a)]
    b = np.asarray(beta_list,  float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, np.nan
    if method == "median":
        return float(np.median(a)), float(np.median(b))
    elif method == "trimmed_mean":
        def tmean(x, p):
            x = np.sort(x); n = len(x); k = int(np.floor(p*n))
            x2 = x[k:n-k] if n-2*k > 0 else x
            return float(np.mean(x2))
        return tmean(a, trim_alpha), tmean(b, trim_alpha)
    else:
        return float(np.median(a)), float(np.median(b))

# ================= 2) Read data and calibrate day by day =================
files = sorted(glob.glob(FILE_GLOB),
               key=lambda p: int(norm_day4(infer_day_token(p))) if norm_day4(infer_day_token(p)).isdigit() else 10**9)
if not files:
    raise FileNotFoundError(f"File not found: {FILE_GLOB}")

print(f"Constants: vf={VF_KMH} km/h, ca={CA_VPH} veh/h, L={L_KM} km, tf={T_FREE_MIN:.6f} min")
print(f"Initial guess: alpha0={ALPHA0}, beta0={BETA0}\n")

daily_rows = []
timeseries_rows = []

for fp in files:
    day_tok = infer_day_token(fp)       # '0731'
    day4    = norm_day4(day_tok)        # 4 
    day     = pretty_day(day4)          # '07-31'
    df  = pd.read_excel(fp)

    if COL_FLOW not in df.columns:
        print(f"[WARN] {fp} missing column: {COL_FLOW}, skipped")
        continue

    q_vph   = pd.to_numeric(df[COL_FLOW], errors="coerce").values
    try:
        tt_obs, obs_source = get_tt_obs_from_df(df)
    except Exception as e:
        print(f"[WARN] {fp} failed to get observed TT: {e}, skipped")
        continue

    # Remove invalid entries
    mask = np.isfinite(q_vph) & np.isfinite(tt_obs)
    q_vph = q_vph[mask]; tt_obs = tt_obs[mask]
    if q_vph.size == 0:
        print(f"[INFO] {fp} has 0 valid samples, skipped.")
        continue

    # Calibrate α, β (for the current day)
    a_star, b_star, tt_hat, m = calibrate_alpha_beta_for_day(q_vph, tt_obs)

    # Daily results
    daily_rows.append({
        "day": day, "day4": day4, "n": int(len(q_vph)),
        "alpha": a_star, "beta": b_star,
        "vf_kmh": VF_KMH, "ca_vph": CA_VPH, "L_km": L_KM, "tf_min": T_FREE_MIN,
        **m
    })

    # Time-series results (original order)
    ts = pd.DataFrame({
        "day": day, "day4": day4,
        "idx": np.arange(len(df)),
        "q_vph": pd.to_numeric(df[COL_FLOW], errors="coerce").values,
        "tt_obs_min": pd.to_numeric(tt_obs, errors="coerce").astype(float),
    })
    tt_hat_full = bpr_tt(ts["q_vph"].values, a_star, b_star)
    ts["tt_bpr_min"] = tt_hat_full

    if "time" in df.columns:
        ts["time"] = df["time"]
    timeseries_rows.append(ts)

    # Optional: plotting (visualize metrics using aligned sequences)
    if SAVE_DAILY_PLOTS:
        x = ts["time"] if "time" in ts.columns else ts["idx"]
        plt.figure(figsize=(8,3.6))
        plt.plot(x, ts["tt_obs_min"].values, label="Observed")
        plt.plot(x, ts["tt_bpr_min"].values, label="BPR (fitted)")
        plt.xlabel("Time" if "time" in ts.columns else "Index (5-min steps)")
        plt.ylabel("Travel time (min)")
        plt.title(f"I-405 Daily Travel Time — {day}\n"
                  f"alpha={a_star:.4f}, beta={b_star:.4f} | "
                  f"MAE={m['MAE_min']:.2f}, RMSE={m['RMSE_min']:.2f}")
        plt.legend(); plt.tight_layout()
        fn = f"daily_tt_{day4}.png"
        plt.savefig(Path(PLOT_DIR) / fn, dpi=300)
        plt.close()

if len(daily_rows) == 0:
    raise RuntimeError("No successfully calibrated days found, cannot continue.")

daily_df = pd.DataFrame(daily_rows).sort_values("day4").reset_index(drop=True)
ts_all   = pd.concat(timeseries_rows, ignore_index=True) if timeseries_rows else pd.DataFrame()

# ================= 3) Export summary =================
daily_df.to_csv(OUTPUT_DIR / OUT_DAILY_CSV, index=False)
ts_all.to_csv(OUTPUT_DIR / OUT_TS_CSV, index=False)
print(f"Exported:{OUT_DAILY_CSV}, {OUT_TS_CSV}")

# ================= 4) Split by day into multiple Excel files =================
split_dir = OUTPUT_DIR / SPLIT_DIR
split_dir.mkdir(parents=True, exist_ok=True)

base_cols = ["day", "day4", "idx", "q_vph", "tt_obs_min", "tt_bpr_min"]
cols_order = (["day", "day4", "time"] + base_cols[2:]) if "time" in ts_all.columns else base_cols

for day_key, g in ts_all.groupby("day4", sort=True):
    safe_day = str(day_key)
    out_path = split_dir / f"bpr_timeseries_{safe_day}.xlsx"
    g = g.sort_values("idx")
    g[cols_order].to_excel(out_path, index=False)

print(f"Exported by day to folder: {split_dir.resolve()}")

# ====== Optional: one Excel file with multiple sheets (one per day) ======
make_one_workbook = True
if make_one_workbook and not ts_all.empty:
    onebook_path = OUTPUT_DIR / ONEBOOK_XLSX
    with pd.ExcelWriter(onebook_path, engine="openpyxl", mode="w") as writer:
        for day_key, g in ts_all.groupby("day4", sort=True):
            sheet = str(day_key)[:31]
            g_sorted = g.sort_values("idx")
            g_sorted[cols_order].to_excel(writer, index=False, sheet_name=sheet)
    print(f"Also generated single Excel file with multiple sheets：{onebook_path.resolve()}")

# ================= 5) 8:2 Split Evaluation =================
TRAIN_DAYS = 57  # 80% training data
if len(daily_df) <= TRAIN_DAYS:
    print(f"[EVAL] Not enough valid days (need > {TRAIN_DAYS})")
else:
    # aggregation
    a_agg, b_agg = agg_params(daily_df["alpha"].iloc[:TRAIN_DAYS],
                              daily_df["beta"].iloc[:TRAIN_DAYS],
                              method="median", trim_alpha=TRIM_ALPHA)
    print(f"\n=== Aggregated α, β from first {TRAIN_DAYS} days ===")
    print(f"alpha_agg={a_agg:.4f}, beta_agg={b_agg:.4f}\n")

    oos_records = []

    #20% testing data
    TEST_TAIL_DAYS = 16
    test_slice = daily_df.iloc[-TEST_TAIL_DAYS:]
    for test_row in test_slice.itertuples(index=False):

        test_day4 = str(test_row.day4)
        test_day = pretty_day(test_day4)
        ts_test = ts_all[ts_all["day4"].astype(str) == test_day4].copy()
        if ts_test.empty:
            print(f"[WARN] No time-series data for day {test_day4}, skipped.")
            continue

        q = pd.to_numeric(ts_test["q_vph"], errors="coerce").values
        tt_obs = pd.to_numeric(ts_test["tt_obs_min"], errors="coerce").values
        mask = np.isfinite(q) & np.isfinite(tt_obs)
        q, tt_obs = q[mask], tt_obs[mask]
        if not len(q):
            print(f"[WARN] Empty data for day {test_day4}, skipped.")
            continue

        tt_pred = bpr_tt(q, a_agg, b_agg)

        # error
        mae = safe_mae(tt_obs, tt_pred)
        rmse = safe_rmse(tt_obs, tt_pred)
        mape = safe_mape(tt_obs, tt_pred)

        # results
        oos_records.append({
            "train_days": TRAIN_DAYS,
            "test_day": test_day4,
            "alpha_agg": a_agg,
            "beta_agg": b_agg,
            "OOS_MAE_TT": mae,
            "OOS_RMSE_TT": rmse,
            "OOS_MAPE_TT_%": mape
        })

        out_ts = OUTPUT_DIR / f"bpr_timeseries_day{test_day4}_agg_pred.csv"
        pd.DataFrame({
            "day": test_day,
            "day4": test_day4,
            "q_vph": q,
            "tt_obs_min": tt_obs,
            "tt_pred_agg_min": tt_pred
        }).to_csv(out_ts, index=False)

        print(f"[EVAL] {test_day4}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")

    # output
    pd.DataFrame(oos_records).to_csv(OUTPUT_DIR / "bpr_agg_eval_summary.csv", index=False)
    print(f"\nExported aggregated evaluation summary: bpr_agg_eval_summary.csv")
    # ===== The remaining 20% of α and β are the aggregated parameters of the first 80%. =====
    for i in range(len(daily_df) - TEST_TAIL_DAYS, len(daily_df)):
        daily_df.loc[i, "alpha"] = a_agg
        daily_df.loc[i, "beta"]  = b_agg

    daily_df.to_csv(OUTPUT_DIR / OUT_DAILY_CSV, index=False)
    print(f"Updated bpr_daily_params_and_metrics.csv with fixed α,β for last {TEST_TAIL_DAYS} days.")


# ================= End =================
print(f"Plot directory (if enabled): {Path(PLOT_DIR).resolve()}")

# ================= 6) Plot travel time comparison for all days (Observed vs Fitted vs Aggregated) =================
print(f"\n[Plot] Generating comparison plots for all {len(daily_df)} days...")

try:
    for i, test_row in enumerate(daily_df.itertuples(index=False)):
        test_day4 = str(test_row.day4)
        test_day  = pretty_day(test_day4)
        ts_day = ts_all[ts_all["day4"].astype(str) == test_day4].copy()
        if ts_day.empty:
            print(f"[WARN][PLOT] No data for {test_day4}, skipped.")
            continue

        oos_csv = OUTPUT_DIR / f"bpr_timeseries_day{test_day4}_agg_pred.csv"
        ts_oos = pd.read_csv(oos_csv) if oos_csv.exists() else pd.DataFrame()

        # 横轴
        if "time" in ts_day.columns and ts_day["time"].notna().any():
            x_plot = ts_day["time"]
            x_lab = "Time"
        else:
            x_plot = ts_day["idx"] if "idx" in ts_day.columns else np.arange(len(ts_day))
            x_lab = "Index (5-min steps)"

        # three curve
        y_obs = pd.to_numeric(ts_day.get("tt_obs_min"), errors="coerce").to_numpy()
        y_fit = pd.to_numeric(ts_day.get("tt_bpr_min"), errors="coerce").to_numpy()
        y_oos = pd.to_numeric(ts_oos.get("tt_pred_agg_min"), errors="coerce").to_numpy() if not ts_oos.empty else None

        L = min([len(y_obs), len(y_fit)] + ([len(y_oos)] if y_oos is not None else []))
        x_plot = np.asarray(x_plot)[:L]
        y_obs, y_fit = y_obs[:L], y_fit[:L]
        if y_oos is not None:
            y_oos = y_oos[:L]

        # error
        if y_oos is not None:
            mae = safe_mae(y_obs, y_oos)
            rmse = safe_rmse(y_obs, y_oos)
            mape = safe_mape(y_obs, y_oos)
        else:
            mae = rmse = mape = np.nan

        # plot
        plt.figure(figsize=(9, 4))
        plt.plot(x_plot, y_obs, label="Observed", linewidth=1.8)
        plt.plot(x_plot, y_fit, label="BPR fitted (daily α,β)", linewidth=1.2)

        is_test_day = i >= len(daily_df) - TEST_TAIL_DAYS

        if is_test_day and y_oos is not None:
            plt.plot(x_plot, y_oos, label="BPR aggregated (α,β)", linewidth=1.2)

        alpha_daily = float(test_row.alpha)
        beta_daily  = float(test_row.beta)

        # title
        if is_test_day and y_oos is not None:
            title = (
                f"Travel Time Comparison — {test_day}\n"
                f"Daily α={alpha_daily:.3f}, β={beta_daily:.3f} | "
                f"Aggregated α={a_agg:.3f}, β={b_agg:.3f}\n"
                f"MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%"
            )

        else:
            title = (
                f"Travel Time Comparison — {test_day}\n"
                f"Daily α={alpha_daily:.3f}, β={beta_daily:.3f}"
            )
        plt.title(title)

        plt.xlabel(x_lab)
        plt.ylabel("Travel time (min)")
        plt.legend()
        plt.tight_layout()

        out_png = Path(PLOT_DIR) / f"tt_compare_{test_day4}.png"
        plt.savefig(out_png, dpi=300)
        plt.close()
        print(f"[PLOT] Generated: {out_png.name}")

    print(f"\n[Plot] All {len(daily_df)} daily comparison plots saved in {PLOT_DIR.resolve()}")
except Exception as e:
    print(f"[ERR][PLOT] Failed to generate daily comparison plots: {e}")


