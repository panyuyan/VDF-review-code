#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, glob, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============ 0) Basic Configuration ============
# The script is located in: Dataset/3. Different Models/
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/

# Input and output directories
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

# Output folder (same level as the script)
OUTPUT_DIR = BASE_DIR / "3. Different Models" / "Output_Queue based"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")


COL_FLOW    = "Flow per hour"   # veh/h (if per lane, then veh/h/ln)
COL_SPEED   = "Speed"           # km/h (used only when no observed TT)
COL_QUEUE   = "Queue"           # Queue, used to identify t0, t3
COL_TT_OBS  = None              # If observed TT (minutes) column exists, specify it here; otherwise set to None
TT_CANDIDATES = ["Travel time", "Travel time (min)", "TT", "tt", "tt_obs_min"]

# — Unit/Scale Consistency — #
FLOW_IS_PER_LANE   = True   # Whether the Flow column is “per lane” (veh/h/ln)
QUEUE_IS_PER_LANE  = True   # Whether the Queue column is “per lane” (veh/ln)
NUM_LANES          = 2

# Speed and length (units: km/h and km)
V_CO_KMH  = 70.0   # Baseline congested speed v_co
V_F_KMH   = 70.0   # Free-flow speed v_f
L_KM      = 0.23
TT_CO_MIN = (L_KM / V_CO_KMH) * 60.0  # L/v_co (minutes)
TT_FF_MIN = (L_KM / V_F_KMH)  * 60.0  # L/v_f (minutes)

# Congestion identification
EPS_QUEUE     = 1e-6
MIN_RUN_EXIT  = 3
MIN_Q_POINTS  = 5

# Time step (if no timestamps, default is 5 minutes per step; only for building the time axis)
STEP_MIN      = 5  # min

# μ estimation
MU_USE_MEDIAN = True  # For congested segment μ, use median (True) or mean (False)

# Output
OUT_DAILY_SIMPLE_XLSX = "tt_estimate_gamma_from_tt_daily.xlsx"
OUT_EVAL_FULL_XLSX    = "tt_estimate_gamma_from_tt_full.xlsx"
SAVE_DAILY_PLOTS      = True
PLOT_DIR = OUTPUT_DIR / "plots_gamma_from_tt"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = ['Times New Roman']

# ============ 1) Utility Functions ============
def infer_day_from_filename(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_([A-Za-z0-9\-]+)$", name)
    return m.group(1) if m else name

def safe_mae(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.abs(y[m]-yhat[m]))) if m.any() else np.nan

def safe_rmse(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.sqrt(np.mean((yhat[m]-y[m])**2))) if m.any() else np.nan

def safe_mape(y, yhat, eps=1e-6):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat) & (np.abs(y) > eps)
    return float(np.mean(np.abs((y[m]-yhat[m])/y[m]))*100) if m.any() else np.nan

def safe_r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 2 or np.allclose(y[m], y[m].mean()): return np.nan
    try: return float(r2_score(y[m], yhat[m]))
    except Exception: return np.nan

def safe_me(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(yhat[m]-y[m])) if m.any() else np.nan

def safe_p90_ae(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    m = np.isfinite(y) & np.isfinite(yhat)
    return float(np.percentile(np.abs(yhat[m]-y[m]), 90)) if m.any() else np.nan

def find_tt_obs_column(df):
    if COL_TT_OBS and COL_TT_OBS in df.columns: return COL_TT_OBS
    for c in TT_CANDIDATES:
        if c in df.columns: return c
    return None

def get_tt_obs_from_df(df):
    """Return observed travel time (minutes); if no column exists, derive it using L/v (minutes)."""
    col = find_tt_obs_column(df)
    if col:
        return pd.to_numeric(df[col], errors="coerce").values, col
    if COL_SPEED not in df.columns:
        raise KeyError("Observed TT column not found, and speed column missing—cannot compute observed TT.")
    spd = np.clip(pd.to_numeric(df[COL_SPEED], errors="coerce").values, 1e-6, None)  # km/h
    return (L_KM/spd*60.0), f"[from {COL_SPEED}]"

def coerce_queue_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(",", "", regex=False)
    s2 = s2.str.replace(r"[^\d\.\-\+eE]", "", regex=True)
    return pd.to_numeric(s2, errors="coerce")

def detect_t0_t3(q, eps=0.0, min_run_exit=3):
    q = np.asarray(q, float); n = len(q)
    if n == 0: return None, None
    pos = np.flatnonzero(q > eps)
    if pos.size == 0: return None, None
    t0 = int(pos[0]); j = t0 + 1
    while j <= n - min_run_exit:
        if np.all(q[j:j+min_run_exit] <= eps):
            t3 = j - 1
            while t3 > t0 and q[t3] <= eps: t3 -= 1
            return int(t0), int(t3)
        j += 1
    return int(t0), int(pos[-1])

# ====== Fit γ based on observed TT (cubic version) ======
def fit_gamma_from_tt_nooffset(t_hour, tt_obs_min, t0_idx, t3_idx, mu_const_vph_per_lane):
    """
    Least-squares cubic fit for the congested segment [t0, t3]:
        tt_obs_min - TT_CO_MIN ≈ α * Z(t), where α = γ / (3μ)
        Z(t) = (t - t0)^2 * (t3 - t) * 60    (t in “hours”, multiplied by 60 → “minutes”)
    Closed-form solution: α = sum(Z*Y) / sum(Z^2), γ = 3 μ α
    """
    
    if (t0_idx is None) or (t3_idx is None) or not np.isfinite(mu_const_vph_per_lane) or mu_const_vph_per_lane <= 0:
        return np.nan

    seg = slice(t0_idx, t3_idx + 1)
    t_seg = np.asarray(t_hour[seg], float)
    y_obs = np.asarray(tt_obs_min[seg], float)

    m = np.isfinite(t_seg) & np.isfinite(y_obs)
    if m.sum() < 3:
        return np.nan

    t0h = float(t_hour[t0_idx])
    t3h = float(t_hour[t3_idx])

    Z = ((t_seg - t0h)**2 * (t3h - t_seg)) * 60.0  
    Y = y_obs - TT_CO_MIN

    mm = np.isfinite(Z) & np.isfinite(Y) & (Z > 0)
    if mm.sum() < 2:
        return np.nan

    denom = float(np.dot(Z[mm], Z[mm]))
    if denom <= 0:
        return np.nan
    alpha = float(np.dot(Z[mm], Y[mm]) / denom)
    alpha = max(0.0, alpha)  
    gamma_hat = 3.0 * mu_const_vph_per_lane * alpha
    return gamma_hat

def eval_segmented(y, yhat, t0, t3):
    y = np.asarray(y, float); yh = np.asarray(yhat, float)
    n = len(y)
    def pack(mask):
        return {
            "N": int(mask.sum()),
            "MAE":    safe_mae(y[mask], yh[mask]),
            "RMSE":   safe_rmse(y[mask], yh[mask]),
            "MAPE_%": safe_mape(y[mask], yh[mask]),
            "R2":     safe_r2(y[mask], yh[mask]),
            "ME":     safe_me(y[mask], yh[mask]),
            "P90AE":  safe_p90_ae(y[mask], yh[mask]),
        }
    mask_all = np.isfinite(y) & np.isfinite(yh)
    overall = pack(mask_all)
    if (t0 is not None) and (t3 is not None) and (0 <= t0 <= t3 < n):
        in_mask = np.zeros(n, bool); in_mask[t0:t3+1] = True
        in_seg  = pack(mask_all & in_mask)
        out_seg = pack(mask_all & ~in_mask)
    else:
        nanpack = {k: np.nan for k in ["N","MAE","RMSE","MAPE_%","R2","ME","P90AE"]}
        in_seg, out_seg = nanpack.copy(), nanpack.copy()
    return overall, in_seg, out_seg

# ============ 2) Daily Processing ============
files = sorted(glob.glob(FILE_GLOB))
if not files:
    raise FileNotFoundError(f"No files found: {FILE_GLOB}")

daily_simple_rows, daily_full_rows, ts_rows = [], [], []

for fp in files:
    day = infer_day_from_filename(fp)
    df  = pd.read_excel(fp)

    # Required column check
    if COL_FLOW not in df.columns:
        warnings.warn(f"{fp} missing column: {COL_FLOW}, skipped.");  continue
    if COL_QUEUE not in df.columns:
        warnings.warn(f"{fp} missing column: {COL_QUEUE} (used for identifying t0, t3), skipped.");  continue

    # Observed TT (from file; if not available, use fallback L/v)
    try:
        tt_obs_min, tt_obs_source = get_tt_obs_from_df(df)  # minutes
    except Exception as e:
        warnings.warn(f"{fp} failed to obtain observed TT: {e}, skipped.");  continue

    # Other columns
    flow_per_unit = pd.to_numeric(df[COL_FLOW], errors="coerce").to_numpy(float)  # veh/h or veh/h/ln
    q_raw         = coerce_queue_series(df[COL_QUEUE]).to_numpy(float)            
    n = len(df)

    # —— Normalize to “per lane” scale —— #
    if FLOW_IS_PER_LANE and (not QUEUE_IS_PER_LANE):
        q_for_seg = q_raw / max(1, NUM_LANES); flow_for_mu = flow_per_unit
    elif (not FLOW_IS_PER_LANE) and QUEUE_IS_PER_LANE:
        q_for_seg = q_raw * max(1, NUM_LANES); flow_for_mu = flow_per_unit
    else:
        q_for_seg = q_raw.copy(); flow_for_mu = flow_per_unit.copy()

    if not FLOW_IS_PER_LANE: flow_for_mu = flow_for_mu / max(1, NUM_LANES)
    if not QUEUE_IS_PER_LANE: q_for_seg = q_for_seg / max(1, NUM_LANES)

    # Congested segment identification (using per-lane Queue)
    if np.isfinite(q_for_seg).sum() < MIN_Q_POINTS:
        warnings.warn(f"{fp} has too few valid Queue points ({np.isfinite(q_for_seg).sum()}<{MIN_Q_POINTS}), skipped.");  continue
    t0_idx, t3_idx = detect_t0_t3(q_for_seg, eps=EPS_QUEUE, min_run_exit=MIN_RUN_EXIT)
    if t0_idx is None or t3_idx is None or t0_idx > t3_idx:
        warnings.warn(f"{fp} failed to identify valid t0/t3, treated as no congestion segment.")
        t0_idx, t3_idx = None, None

    # Time axis: hours
    t_hour = np.arange(n, dtype=float) * (STEP_MIN / 60.0)

    # μ: median/mean of observed flow (per lane) in congested segment
    if (t0_idx is not None) and (t3_idx is not None):
        flow_seg = flow_for_mu[t0_idx:t3_idx+1]
        if MU_USE_MEDIAN:
            mu_const = float(np.nanmedian(flow_seg[np.isfinite(flow_seg)])) if np.isfinite(flow_seg).any() else np.nan
        else:
            mu_const = float(np.nanmean(flow_seg[np.isfinite(flow_seg)])) if np.isfinite(flow_seg).any() else np.nan
    else:
        mu_const = np.nan
  
    
  # === Fit γ using observed TT (cubic form) ===
    gamma_hat = fit_gamma_from_tt_nooffset(t_hour, tt_obs_min, t0_idx, t3_idx, mu_const)

    # Generate predictions: Non-congested = L/v_f; Congested = L/v_co + (γ/(3μ))*((t-t0)^2 (t3-t))
    tt_pred_min = np.full(n, TT_FF_MIN, dtype=float)  # Default: non-congested
    w_fit_full  = np.zeros(n, dtype=float)            # Waiting time w(t) (minutes), default 0
    if (t0_idx is not None) and (t3_idx is not None) and np.isfinite(gamma_hat) and np.isfinite(mu_const) and mu_const > 0:
        t0h = float(t_hour[t0_idx]); t3h = float(t_hour[t3_idx])
        seg = slice(t0_idx, t3_idx+1)
        t_seg = t_hour[seg]
        Z_seg = (t_seg - t0h)**2 * (t3h - t_seg)             
        w_fit_min = (gamma_hat / (3.0 * mu_const)) * Z_seg * 60.0  
        w_fit_min = np.maximum(0.0, w_fit_min)               
        tt_pred_min[seg] = TT_CO_MIN + w_fit_min
        w_fit_full[seg]  = w_fit_min

    # Evaluation
    overall_eval, in_eval, out_eval = eval_segmented(tt_obs_min, tt_pred_min, t0_idx, t3_idx)

    # —— Summary —— #
    cong_len = int(t3_idx - t0_idx + 1) if (t0_idx is not None and t3_idx is not None) else 0
    daily_simple_rows.append({
        "day": day, "n": n,
        "L_km": L_KM, "v_f_kmh": V_F_KMH, "v_co_kmh": V_CO_KMH,
        "tt_ff_min": TT_FF_MIN, "tt_co_min": TT_CO_MIN,
        "mu_const_vph_per_lane": mu_const,
        "gamma_from_tt": gamma_hat,
        "MAE_min": overall_eval["MAE"], "RMSE_min": overall_eval["RMSE"],
        "MAPE_%": overall_eval["MAPE_%"], "R2": overall_eval["R2"],
    })
    daily_full_rows.append({
        "day": day, "n": n,
        "L_km": L_KM, "v_f_kmh": V_F_KMH, "v_co_kmh": V_CO_KMH,
        "tt_ff_min": TT_FF_MIN, "tt_co_min": TT_CO_MIN,
        "mu_const_vph_per_lane": mu_const,
        "gamma_from_tt": gamma_hat,
        "t0_idx": t0_idx, "t3_idx": t3_idx, "congested_len": cong_len,
        # overall
        "MAE": overall_eval["MAE"], "RMSE": overall_eval["RMSE"], "MAPE_%": overall_eval["MAPE_%"],
        "R2": overall_eval["R2"], "ME": overall_eval["ME"], "P90AE": overall_eval["P90AE"], "N_eval": overall_eval["N"],
        # in-seg
        "MAE_in": in_eval["MAE"], "RMSE_in": in_eval["RMSE"], "MAPE_in_%": in_eval["MAPE_%"],
        "R2_in": in_eval["R2"], "ME_in": in_eval["ME"], "P90AE_in": in_eval["P90AE"], "N_in": in_eval["N"],
        # out-seg
        "MAE_out": out_eval["MAE"], "RMSE_out": out_eval["RMSE"], "MAPE_out_%": out_eval["MAPE_%"],
        "R2_out": out_eval["R2"], "ME_out": out_eval["ME"], "P90AE_out": out_eval["P90AE"], "N_out": out_eval["N"],
    })

    # —— Time series (export & plot)——
    ts = pd.DataFrame({
        "day": day,
        "idx": np.arange(n),
        "tt_obs_min": tt_obs_min,
        "tt_pred_min": tt_pred_min,
        "w_min": w_fit_full,             
        "is_congested_seg": False
    })
    if (t0_idx is not None) and (t3_idx is not None):
        ts.loc[t0_idx:t3_idx, "is_congested_seg"] = True
    ts_rows.append(ts)

    # —— Daily plots —— #
    if SAVE_DAILY_PLOTS:
        x = ts["idx"].values
        plt.figure(figsize=(9, 4))
        plt.plot(x, ts["tt_obs_min"], label="Observed TT", linewidth=1.6)
        plt.plot(x, ts["tt_pred_min"],
                 label=r"Pred: out = L/$v_f$ ; in = L/$v_{co}$ + $\gamma/(3\mu)(t-t_0)^2(t_3-t)$",
                 linewidth=1.8)
        if (t0_idx is not None) and (t3_idx is not None):
            plt.axvspan(t0_idx, t3_idx, alpha=0.12, label="[t0,t3]")
        ttl = (f"{day} | v_f={V_F_KMH:.1f} km/h, v_co={V_CO_KMH:.1f} km/h, "
               f"μ≈{mu_const:.0f} vph/ln, γ={gamma_hat:.3g} | MAE={overall_eval['MAE']:.2f} min")
        plt.title(ttl); plt.xlabel("Index (5-min steps)"); plt.ylabel("Travel time (min)")
        h,l = plt.gca().get_legend_handles_labels(); by = dict(zip(l,h)); plt.legend(by.values(), by.keys())
        plt.tight_layout(); plt.savefig(PLOT_DIR/f"daily_tt_{day.replace('/', '-')}.png", dpi=300, bbox_inches="tight"); plt.close()

# ============ 3) Export (daily and overall + last day time series)============
daily_simple_df = pd.DataFrame(daily_simple_rows) if daily_simple_rows else pd.DataFrame()
daily_full_df   = pd.DataFrame(daily_full_rows)   if daily_full_rows   else pd.DataFrame()
ts_all          = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()

if not daily_simple_df.empty:
    cols = ["day","n","L_km","v_f_kmh","v_co_kmh","tt_ff_min","tt_co_min",
            "mu_const_vph_per_lane","gamma_from_tt","MAE_min","RMSE_min","MAPE_%","R2"]
    with pd.ExcelWriter(Path(OUTPUT_DIR)/OUT_DAILY_SIMPLE_XLSX, engine="openpyxl", mode="w") as w:
        daily_simple_df[cols].to_excel(w, index=False, sheet_name="daily_params_and_metrics")
    print(f"[OK] Exported: {(Path(OUTPUT_DIR)/OUT_DAILY_SIMPLE_XLSX).resolve()}")

if not daily_full_df.empty and not ts_all.empty:
    mask_all = np.isfinite(ts_all["tt_obs_min"]) & np.isfinite(ts_all["tt_pred_min"])
    mask_in  = mask_all & (ts_all["is_congested_seg"] == True)
    mask_out = mask_all & (ts_all["is_congested_seg"] == False)
    def pack(mask, scope):
        return {
            "scope": scope,
            "MAE":  safe_mae(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "RMSE": safe_rmse(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "MAPE_%": safe_mape(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "R2":   safe_r2(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "ME":   safe_me(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "P90AE":safe_p90_ae(ts_all.loc[mask,"tt_obs_min"], ts_all.loc[mask,"tt_pred_min"]),
            "N_eval": int(mask.sum())
        }
    overall_df = pd.DataFrame([
        pack(mask_all, "ALL_DAYS_OVERALL"),
        pack(mask_in,  "ALL_DAYS_IN_SEG"),
        pack(mask_out, "ALL_DAYS_OUT_SEG")
    ])
    with pd.ExcelWriter(Path(OUTPUT_DIR)/OUT_EVAL_FULL_XLSX, engine="openpyxl", mode="w") as w:
        daily_full_df.sort_values("day").to_excel(w, index=False, sheet_name="daily_summary")
        overall_df.to_excel(w, index=False, sheet_name="overall_summary")
        ts_all.head(1000).to_excel(w, index=False, sheet_name="timeseries_sample")
    print(f"[OK] Exported: {(Path(OUTPUT_DIR)/OUT_EVAL_FULL_XLSX).resolve()}")

# —— New: Export “last day” observed and predicted time series (CSV)——
if ts_rows:
    last_ts = ts_rows[-1].copy()
    last_day_str = str(last_ts["day"].iloc[0]).replace("/", "-")
    out_csv_last = Path(OUTPUT_DIR) / f"timeseries_last_day_{last_day_str}.csv"
    # Export only core columns: index, observed TT, predicted TT, waiting time
    last_ts[["idx", "tt_obs_min", "tt_pred_min", "w_min"]].to_csv(
        out_csv_last, index=False, encoding="utf-8-sig"
    )
    print(f"[OK] Exported last-day time series: {out_csv_last.resolve()}")
else:
    print("[INFO] No available time-series data; last-day export skipped.")
