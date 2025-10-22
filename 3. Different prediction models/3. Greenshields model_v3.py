# -*- coding: utf-8 -*-
"""
Batch daily Greenshields FD calibration with robust fitting & cleaning
+ Safe metrics
+ 80:20 OOS forecasting with day normalization & visualization

Greenshields: v(k) = vf * (1 - k / k_jam)

Outputs:
  - daily_params_metrics_multi.csv
  - daily_timeseries_outputs_multi.csv
  - oos_eval_summary.csv
  - timeseries_day<TESTDAY>_oos_pred.csv
  - fd_daily_plots/dens_speed/<day>_dens_speed.png   (optional)
  - fd_daily_plots/tt_compare/tt_compare_<day>.png   (OOS days)
  - fd_daily_plots/tt_compare_all/tt_compare_all_<day>.png  (ALL days)
"""
from pathlib import Path
import os, re, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============ 0) Basic Configuration ============
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")
OUTPUT_DIR = BASE_DIR / "3. Different prediction models" / "Output_FD_Greenshields"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

# ========================= User settings =========================
LEN_MI = 0.23
SAVE_PLOTS = True
PLOT_DIR_FULL = OUTPUT_DIR / "fd_daily_plots"

VMIN_MPH = 5.0
DENS_QTRIM = (0.01, 0.99)
SPEED_QTRIM = (0.01, 0.99)
TT_CAP_MIN = 120.0
NMIN_AFTER_FILTER = 40

VF_MIN, VF_MAX = 20.0, 120.0
KJ_MIN, KJ_MAX = 60.0, 400.0

TIME_COL_CANDS = ["Timestamp","DateTime","Datetime","Time","Date","date","datetime"]

OOS_PARAM_AGG = "median"   # or "trimmed_mean"
TRIM_ALPHA = 0.1

plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'

# ========================= Helpers =========================
def gs_speed_of_density(k, vf, k_jam):
    k = np.asarray(k, dtype=float)
    return vf * np.maximum(0.0, 1.0 - k / max(k_jam, 1e-6))

def _residuals(p, k, v_obs):
    vf, kj = p
    return gs_speed_of_density(k, vf, kj) - v_obs

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > eps)
    if mask.sum() == 0: return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def _pick_time_col(df):
    for c in TIME_COL_CANDS:
        if c in df.columns: return c
    return None

def _clean_dataframe_for_fit(g):
    v = pd.to_numeric(g['Speed'], errors='coerce').to_numpy()
    k = pd.to_numeric(g['Density'], errors='coerce').to_numpy()
    mask = np.isfinite(v) & np.isfinite(k) & (v > 0) & (k > 0)
    v, k = v[mask], k[mask]
    keep = v >= VMIN_MPH
    v, k = v[keep], k[keep]
    if v.size == 0:
        return v, k, {"keep_rate_%": 0.0, "trimmed": True, "note": "all dropped by VMIN"}
    def _trim(x, qpair):
        ql, qh = np.nanquantile(x, qpair[0]), np.nanquantile(x, qpair[1])
        return (x >= ql) & (x <= qh)
    m = _trim(k, DENS_QTRIM) & _trim(v, SPEED_QTRIM)
    v2, k2 = v[m], k[m]
    keep_rate = 100.0 * v2.size / max(1, v.size)
    return v2, k2, {"keep_rate_%": keep_rate, "trimmed": (v2.size != v.size)}

def _safe_tt(v):
    v = np.maximum(v, 1e-6)
    tt = (LEN_MI / v) * 60.0
    return np.clip(tt, None, TT_CAP_MIN)

def _agg_params(vf_list, kj_list, method="median", trim_alpha=0.1):
    vf_arr = np.asarray(vf_list, float); kj_arr = np.asarray(kj_list, float)
    vf_arr, kj_arr = vf_arr[np.isfinite(vf_arr)], kj_arr[np.isfinite(kj_arr)]
    if len(vf_arr)==0 or len(kj_arr)==0: return np.nan, np.nan
    if method=="median": return np.median(vf_arr), np.median(kj_arr)
    # trimmed mean
    def tmean(x, a):
        x = np.sort(x); n = len(x); cut = int(np.floor(a*n))
        x2 = x[cut:n-cut] if n-2*cut > 0 else x
        return float(np.mean(x2))
    return tmean(vf_arr, trim_alpha), tmean(kj_arr, trim_alpha)

def _norm_day(x):
    s = re.sub(r"\D", "", str(x))
    return f"{int(s):04d}" if s else ""

# ========================= Load & Fit per-day =========================
files = sorted(glob.glob(str(FILE_GLOB)))
if not files: raise FileNotFoundError(f"No files found: {FILE_GLOB}")
print(f"Found {len(files)} input files.")

daily_rows, ts_rows = [], []
for fp in files:
    m = re.search(r"_(\d{4})", os.path.basename(fp))
    day_label = m.group(1) if m else os.path.basename(fp)

    try:
        g = pd.read_excel(fp)
    except Exception as e:
        print(f"[WARN] Failed to read {fp}: {e}")
        continue

    req = ['Speed','Flow per hour','Density']
    miss = [c for c in req if c not in g.columns]
    if miss:
        print(f"[WARN] {os.path.basename(fp)} missing columns: {miss} - skipped.")
        continue

    time_col = _pick_time_col(g)
    if time_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g[time_col] = pd.to_datetime(g[time_col], errors="coerce")

    # cleaning
    v_fit, k_fit, diag = _clean_dataframe_for_fit(g)
    if len(v_fit) < NMIN_AFTER_FILTER:
        print(f"[INFO] Skip {day_label}: after cleaning kept {len(v_fit)} (<{NMIN_AFTER_FILTER}).")
        continue

    # robust fitting
    vf0 = float(np.nanpercentile(v_fit,95)) if np.isfinite(np.nanpercentile(v_fit,95)) else max(30.0, np.nanmax(v_fit))
    kj0 = max(1.25*np.nanmax(k_fit), 120.0)
    p0 = [np.clip(vf0, VF_MIN, VF_MAX), np.clip(kj0, KJ_MIN, KJ_MAX)]

    res = least_squares(_residuals, p0, args=(k_fit, v_fit),
                        bounds=([VF_MIN, KJ_MIN],[VF_MAX, KJ_MAX]),
                        loss="soft_l1", f_scale=2.0, max_nfev=20000)

    if not res.success:
        print(f"[WARN] {day_label} fit failed: {res.message}")
        continue

    vf, k_jam = res.x
    dens_all = pd.to_numeric(g["Density"], errors="coerce").to_numpy()
    v_all = gs_speed_of_density(dens_all, vf, k_jam)
    tt_obs = _safe_tt(pd.to_numeric(g["Speed"], errors="coerce"))
    tt_cal = _safe_tt(v_all)
    L = min(len(tt_obs), len(tt_cal))
    mae, r, mp = mean_absolute_error(tt_obs[:L], tt_cal[:L]), rmse(tt_obs[:L], tt_cal[:L]), mape(tt_obs[:L], tt_cal[:L])

    daily_rows.append({"day":day_label,"vf":vf,"k_jam":k_jam,"MAE_TT":mae,"RMSE_TT":r,"MAPE_TT_%":mp})

    ts_rows.append(pd.DataFrame({
        "day":day_label,
        "Density_obs_veh_per_mi_ln": dens_all[:L],
        "Speed_obs_mph": pd.to_numeric(g["Speed"], errors="coerce").to_numpy()[:L],
        "TT_obs_min": tt_obs[:L],
        "TT_cal_min": tt_cal[:L],
        "time": g[time_col].values[:L] if time_col else np.arange(L)
    }))

    # optional: density-speed plot
    if SAVE_PLOTS and np.isfinite(vf) and np.isfinite(k_jam) and len(k_fit) > 1:
        os.makedirs(PLOT_DIR_FULL / "dens_speed", exist_ok=True)
        xk = np.linspace(max(1e-3, np.nanmin(k_fit)), max(5.0, np.nanmax(k_fit)), 200)
        plt.figure(figsize=(7,5), dpi=200)
        plt.scatter(dens_all, pd.to_numeric(g["Speed"], errors="coerce").to_numpy(),
                    s=8, facecolors='none', edgecolors='r', label='Observed')
        plt.plot(xk, gs_speed_of_density(xk, vf, k_jam), lw=2.0, label='Greenshields fit')
        plt.xlabel('Density (veh/mi/ln)'); plt.ylabel('Speed (mph)')
        plt.title(f'Density vs Speed - {day_label}\n'
                  f'vf={vf:.1f}, k_jam={k_jam:.1f} | TT RMSE={r:.2f} min')
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOT_DIR_FULL / "dens_speed" / f"{day_label}_dens_speed.png", dpi=240)
        plt.close()

# ========================= Save daily calibration =========================
daily_df = pd.DataFrame(daily_rows).sort_values("day",ignore_index=True)
daily_df.to_csv(OUTPUT_DIR/"daily_params_metrics_multi.csv",index=False)

if ts_rows:
    daily_ts = pd.concat(ts_rows,ignore_index=True)
    daily_ts.to_csv(OUTPUT_DIR/"daily_timeseries_outputs_multi.csv",index=False)
else:
    print("No per-timestamp outputs written (no successful fits).")
print("Saved daily calibration results.")

# ========================= 80:20 OOS Evaluation =========================
oos_records = []
TOTAL = len(daily_df)
if TOTAL >= 5 and ts_rows:
    TRAIN = int(np.floor(0.8*TOTAL))
    print(f"\n=== Aggregating first {TRAIN} days for OOS ===")
    vf_agg, kj_agg = _agg_params(daily_df["vf"][:TRAIN], daily_df["k_jam"][:TRAIN],
                                 method=OOS_PARAM_AGG, trim_alpha=TRIM_ALPHA)
    print(f"Aggregated vf={vf_agg:.3f}, k_jam={kj_agg:.3f}")

    valid_ts = pd.read_csv(OUTPUT_DIR/"daily_timeseries_outputs_multi.csv", dtype={"day":str})
    valid_ts["day_norm"] = valid_ts["day"].map(_norm_day)

    # loop over the 20% test days
    for test_row in daily_df.iloc[TRAIN:].itertuples(index=False):
        test_day = _norm_day(test_row.day)
        ts_test = valid_ts[valid_ts["day_norm"]==test_day].copy()
        if ts_test.empty: 
            print(f"[INFO] No timeseries for test day {test_day}, skip.")
            continue

        k = pd.to_numeric(ts_test["Density_obs_veh_per_mi_ln"],errors="coerce").to_numpy(float)
        tt_obs = pd.to_numeric(ts_test["TT_obs_min"],errors="coerce").to_numpy(float)
        v_pred = gs_speed_of_density(k, vf_agg, kj_agg)
        tt_pred = _safe_tt(v_pred)
        L = min(len(tt_obs), len(tt_pred))
        mae, r, mp = mean_absolute_error(tt_obs[:L],tt_pred[:L]), rmse(tt_obs[:L],tt_pred[:L]), mape(tt_obs[:L],tt_pred[:L])

        oos_records.append({"train_days":TRAIN,"test_day":test_day,"vf_agg":vf_agg,"k_jam_agg":kj_agg,
                            "OOS_MAE_TT":mae,"OOS_RMSE_TT":r,"OOS_MAPE_TT_%":mp})

        pd.DataFrame({
            "day":test_day,
            "Density_obs_veh_per_mi_ln":k[:L],
            "TT_obs_min":tt_obs[:L],
            "TT_pred_oos_min":tt_pred[:L]
        }).to_csv(OUTPUT_DIR/f"timeseries_day{test_day}_oos_pred.csv",index=False)
        print(f"[EVAL] {test_day}: MAE={mae:.3f}, RMSE={r:.3f}, MAPE={mp:.2f}%")

        # per-test-day comparison plot
        if SAVE_PLOTS:
            os.makedirs(PLOT_DIR_FULL / "tt_compare", exist_ok=True)
            if "time" in ts_test.columns and ts_test["time"].notna().any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_axis = pd.to_datetime(ts_test["time"], errors="coerce")[:L]
                x_label = "Time"
            else:
                x_axis = np.arange(L); x_label = "Index"
            plt.figure(figsize=(9,4),dpi=200)
            plt.plot(x_axis, tt_obs[:L], label="Observed TT", linewidth=1.8)
            if "TT_cal_min" in ts_test.columns:
                plt.plot(x_axis, pd.to_numeric(ts_test["TT_cal_min"], errors="coerce").to_numpy(float)[:L],
                         label="Calibrated TT (daily)", linewidth=1.2)
            plt.plot(x_axis, tt_pred[:L], label="OOS Predicted TT (aggregated)", linewidth=1.2)
            plt.xlabel(x_label); plt.ylabel("Travel Time (min)")
            plt.title(f"TT Comparison - {test_day}\nAgg vf={vf_agg:.2f}, k_jam={kj_agg:.1f} | "
                      f"MAE={mae:.2f}, RMSE={r:.2f}, MAPE={mp:.1f}%")
            plt.legend(); plt.tight_layout()
            plt.savefig(PLOT_DIR_FULL/"tt_compare"/f"tt_compare_{test_day}.png",dpi=250)
            plt.close()
            print(f"[PLOT] Saved tt_compare_{test_day}.png")

    pd.DataFrame(oos_records).to_csv(OUTPUT_DIR/"oos_eval_summary.csv",index=False)
    print("\nExported OOS summary: oos_eval_summary.csv")

# ========================= Plot TT Comparison for ALL Days =========================
if SAVE_PLOTS:
    print("\n[Plot] Generating TT comparison plots for ALL days...")
    try:
        df_ts_all = pd.read_csv(OUTPUT_DIR/"daily_timeseries_outputs_multi.csv", dtype={"day": str})
        df_ts_all["day_norm"] = df_ts_all["day"].map(_norm_day)
        os.makedirs(PLOT_DIR_FULL / "tt_compare_all", exist_ok=True)

        # Define train/test sets (using day_norm comparison)
        train_days_set = set(daily_df.iloc[:TRAIN]["day"].map(_norm_day))
        test_days_set  = set(daily_df.iloc[TRAIN:]["day"].map(_norm_day))

        for day_id in sorted(df_ts_all["day_norm"].unique()):
            ts_day = df_ts_all[df_ts_all["day_norm"] == day_id].copy()
            if ts_day.empty:
                continue

            k_day = pd.to_numeric(ts_day["Density_obs_veh_per_mi_ln"], errors="coerce").to_numpy(float)
            tt_obs_day = pd.to_numeric(ts_day["TT_obs_min"], errors="coerce").to_numpy(float)
            tt_cal_day = pd.to_numeric(ts_day["TT_cal_min"], errors="coerce").to_numpy(float)
            L = min(len(tt_obs_day), len(tt_cal_day))

            # 仅对“测试集的天”计算/绘制 aggregated OOS
            draw_agg = (day_id in test_days_set) and np.isfinite(vf_agg) and np.isfinite(kj_agg)
            if draw_agg:
                tt_pred_oos_day = _safe_tt(gs_speed_of_density(k_day, vf_agg, kj_agg))
                L = min(L, len(tt_pred_oos_day))

            # 时间轴
            if "time" in ts_day.columns and ts_day["time"].notna().any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_axis = pd.to_datetime(ts_day["time"], errors="coerce")[:L]
                x_label = "Time"
            else:
                x_axis = np.arange(L); x_label = "Index (5-min steps)"

            # 绘图
            plt.figure(figsize=(9,4), dpi=200)
            plt.plot(x_axis, tt_obs_day[:L], label="Observed TT", linewidth=1.8)
            plt.plot(x_axis, tt_cal_day[:L], label="Calibrated TT (daily)", linewidth=1.2)
            if draw_agg:
                plt.plot(x_axis, tt_pred_oos_day[:L], label="Aggregated TT (vf_agg,k_jam_agg)", linewidth=1.2)

            plt.xlabel(x_label); plt.ylabel("Travel Time (min)")
            if draw_agg:
                title = f"Travel Time Comparison - Day {day_id}\nAggregated vf={vf_agg:.2f}, k_jam={kj_agg:.1f} (OOS)"
            else:
                title = f"Travel Time Comparison - Day {day_id} (Train day: no OOS line)"
            plt.title(title)
            plt.legend(); plt.tight_layout()
            out_png = PLOT_DIR_FULL / "tt_compare_all" / f"tt_compare_all_{day_id}.png"
            plt.savefig(out_png, dpi=250)
            plt.close()
            print(f"[PLOT] Saved TT comparison for day {day_id} (agg line: {'yes' if draw_agg else 'no'})")

    except Exception as e:
        print(f"[WARN][PLOT_ALL] Failed to generate all-day comparison plots: {e}")
