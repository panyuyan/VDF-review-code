# -*- coding: utf-8 -*-
"""
Batch daily Greenshields FD calibration with robust fitting & cleaning
+ Safe metrics (zero-length protection)
+ OOS forecasting (N_TRAIN -> N_TRAIN+1) with day normalization & CSV/Excel fallback

Greenshields: v(k) = vf * (1 - k / k_jam)

Outputs:
  - daily_params_metrics_multi.csv
  - daily_timeseries_outputs_multi.csv
  - oos_eval_summary.csv
  - timeseries_day<TESTDAY>_oos_pred.csv
  - fd_daily_plots/<day>_dens_speed.png  (optional)
"""
from pathlib import Path
import os, re, glob, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============ 0) Basic Configuration ============
# The script is located in: Dataset/3. Different Models/
BASE_DIR = Path(__file__).resolve().parent.parent  # Go up to Dataset/

# Input and output directories
DATA_DIR_INPUT = BASE_DIR / "1. Input data"
FILE_GLOB = os.path.join(DATA_DIR_INPUT, "CA_I405_bottleneck_*.xlsx")

# Output folder (same level as the script)
OUTPUT_DIR = BASE_DIR / "3. Different Models" / "Output_FD_Greenshields"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input folder:  {DATA_DIR_INPUT}")
print(f"Output folder: {OUTPUT_DIR}")

# ========================= User settings =========================

LEN_MI        = 0.23  # segment length [mile]

SAVE_PLOTS    = True
PLOT_DIR      = "fd_daily_plots"
PLOT_DIR_FULL = OUTPUT_DIR / PLOT_DIR 

# Cleaning & safeguards
VMIN_MPH          = 5.0                 # drop speeds below this
DENS_QTRIM        = (0.01, 0.99)        # keep density within [1%, 99%]
SPEED_QTRIM       = (0.01, 0.99)        # keep speed within [1%, 99%]
TT_CAP_MIN        = 120.0               # cap TT to 120 min for metrics/plots
NMIN_AFTER_FILTER = 40                  # minimal samples needed after cleaning

# Fitting bounds (adjust per corridor)
VF_MIN, VF_MAX = 20.0, 120.0            # mph
KJ_MIN, KJ_MAX = 60.0, 400.0            # veh/mi/ln

# Time column candidates
TIME_COL_CANDS = ["Timestamp","DateTime","Datetime","Time","Date","date","datetime"]

# OOS control
N_TRAIN        = 80
OOS_PARAM_AGG  = "median"               # "median" or "trimmed_mean"
TRIM_ALPHA     = 0.1                    # used only when OOS_PARAM_AGG == "trimmed_mean"

# Matplotlib font config
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font', family='Times New Roman')
plt.rcParams['mathtext.fontset'] = 'stix'


# ========================= Helpers & model =========================
def gs_speed_of_density(k, vf, k_jam):
    k = np.asarray(k, dtype=float)
    k_jam = max(k_jam, 1e-9)
    return vf * np.maximum(0.0, 1.0 - k / k_jam)

def _residuals(p, k, v_obs):
    vf, kj = p
    v_pred = gs_speed_of_density(k, vf, kj)
    return v_pred - v_obs

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
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

    m_d = _trim(k, DENS_QTRIM)
    m_v = _trim(v, SPEED_QTRIM)
    m = m_d & m_v

    v2, k2 = v[m], k[m]
    keep_rate = 100.0 * v2.size / max(1, v.size)
    return v2, k2, {"keep_rate_%": keep_rate, "trimmed": (v2.size != v.size), "note": ""}

def _safe_tt(v):
    v = np.maximum(v, 1e-6)
    tt = (LEN_MI / v) * 60.0
    return np.clip(tt, None, TT_CAP_MIN)

def _agg_params_for_oos(vf_list, kj_list, method="median", trim_alpha=0.1):
    vf_arr = np.asarray(vf_list, float); vf_arr = vf_arr[np.isfinite(vf_arr)]
    kj_arr = np.asarray(kj_list, float); kj_arr = kj_arr[np.isfinite(kj_arr)]
    if len(vf_arr) == 0 or len(kj_arr) == 0:
        return np.nan, np.nan
    if method == "median":
        return np.median(vf_arr), np.median(kj_arr)
    elif method == "trimmed_mean":
        def trimmed_mean(x, alpha):
            x = np.sort(x); n = len(x); cut = int(np.floor(alpha*n))
            x2 = x[cut:n-cut] if n-2*cut > 0 else x
            return np.mean(x2)
        return trimmed_mean(vf_arr, trim_alpha), trimmed_mean(kj_arr, trim_alpha)
    else:
        return np.median(vf_arr), np.median(kj_arr)

def _norm_day(x):
    """Normalize day to 4-digit string: '731' or 731 -> '0731'."""
    s = str(x).strip()
    s = re.sub(r"\D", "", s)
    if s == "": return ""
    return f"{int(s):04d}"


# ========================= Collect files =========================
files = sorted(glob.glob(str(FILE_GLOB)))
def sort_key(p):
    m = re.search(r"_(\d{4})\.xlsx$", os.path.basename(p))
    return int(m.group(1)) if m else 999999
files = sorted(files, key=sort_key)

if not files:
    raise FileNotFoundError(f"No files found with pattern: {FILE_GLOB}")

print(f"Found {len(files)} input files:")
for f in files:
    print(f"  - {os.path.basename(f)}")
    
daily_rows, ts_rows = [], []

# ========================= Process each day file =========================
for fp in files:
    day_id = os.path.splitext(os.path.basename(fp))[0]
    m = re.search(r"_(\d{4})$", day_id)
    day_label = m.group(1) if m else day_id

    try:
        g = pd.read_excel(fp)
    except Exception as e:
        print(f"[WARN] Failed to read {fp}: {e}")
        continue

    req = ['Speed','Flow per hour','Density']
    miss = [c for c in req if c not in g.columns]
    if miss:
        print(f"[WARN] {fp} missing columns: {miss} — skipped.")
        continue

    time_col = _pick_time_col(g)
    if time_col is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g[time_col] = pd.to_datetime(g[time_col], errors='coerce')

    # observed TT (min), with cap for stability
    if 'tt_obs_min' in g.columns:
        tt_obs_raw = pd.to_numeric(g['tt_obs_min'], errors='coerce').to_numpy()
        tt_obs = np.clip(tt_obs_raw, None, TT_CAP_MIN)
    else:
        v_obs_all = pd.to_numeric(g['Speed'], errors='coerce').to_numpy()
        tt_obs = _safe_tt(v_obs_all)

    # Cleaning for fitting
    v_fit, k_fit, diag = _clean_dataframe_for_fit(g)
    n_after = int(v_fit.size)
    n_total = int(len(g))

    if n_after < NMIN_AFTER_FILTER:
        print(f"[INFO] Day {day_label}: after cleaning {n_after}/{n_total} kept (<{NMIN_AFTER_FILTER}), skip fit & metrics.")
        daily_rows.append({
            'day': day_label, 'file': os.path.basename(fp),
            'vf': np.nan, 'k_jam': np.nan, 'kc': np.nan,
            'speed_at_capacity': np.nan, 'capacity': np.nan,
            'MAE_TT': np.nan, 'RMSE_TT': np.nan, 'MAPE_TT_%': np.nan,
            'n_points': n_total, 'n_after_filter': n_after,
            'keep_rate_%': diag.get("keep_rate_%", 0.0),
            'trimmed': diag.get("trimmed", False),
            'note': f"insufficient samples after cleaning; {diag.get('note','')}"
        })
        continue

    # Robust fitting
    vf0 = float(np.nanpercentile(v_fit, 95)) if np.isfinite(np.nanpercentile(v_fit, 95)) else max(30.0, np.nanmax(v_fit))
    kj0 = max(1.25 * np.nanmax(k_fit), 120.0)
    p0 = np.array([np.clip(vf0, VF_MIN, VF_MAX), np.clip(kj0, KJ_MIN, KJ_MAX)], dtype=float)

    res = least_squares(
        fun=_residuals, x0=p0, args=(k_fit, v_fit),
        bounds=([VF_MIN, KJ_MIN], [VF_MAX, KJ_MAX]),
        loss='soft_l1', f_scale=2.0, max_nfev=20000
    )

    if not res.success:
        vf, k_jam = np.nan, np.nan
        kc = v_cap = q_cap = np.nan
        dens_all = pd.to_numeric(g['Density'], errors='coerce').to_numpy()
        v_cal_all = gs_speed_of_density(dens_all, 1.0, 1.0) * np.nan
        note = f"fit failed: {res.message}"
    else:
        vf, k_jam = res.x
        kc = 0.5 * k_jam
        v_cap = 0.5 * vf
        q_cap = 0.25 * vf * k_jam
        dens_all = pd.to_numeric(g['Density'], errors='coerce').to_numpy()
        v_cal_all = gs_speed_of_density(dens_all, vf, k_jam)
        note = res.message

    # TT metrics with zero-length protection
    tt_cal = _safe_tt(v_cal_all)
    L = min(len(tt_obs), len(tt_cal))

    if L == 0:
        print(f"[WARN] Day {day_label}: zero-length after align. Set metrics to NaN and skip timeseries block.")
        mae_tt = rmse_tt_val = mape_tt_val = np.nan
        create_ts_block = False
    else:
        tt_obs_use = tt_obs[:L]
        tt_cal_use = tt_cal[:L]
        mae_tt = mean_absolute_error(tt_obs_use, tt_cal_use)
        rmse_tt_val = rmse(tt_obs_use, tt_cal_use)
        mape_tt_val = mape(tt_obs_use, tt_cal_use)
        create_ts_block = True

    daily_rows.append({
        'day': day_label, 'file': os.path.basename(fp),
        'vf': vf, 'k_jam': k_jam, 'kc': kc,
        'speed_at_capacity': v_cap, 'capacity': q_cap,
        'MAE_TT': mae_tt, 'RMSE_TT': rmse_tt_val, 'MAPE_TT_%': mape_tt_val,
        'n_points': n_total, 'n_after_filter': n_after,
        'keep_rate_%': diag.get("keep_rate_%", np.nan),
        'trimmed': diag.get("trimmed", False),
        'note': note if isinstance(note, str) else ("ok" if res.success else "fit failed")
    })

    if create_ts_block:
        ts_block = pd.DataFrame({
            'day': day_label,
            'file': os.path.basename(fp),
            'time': g[time_col] if time_col is not None else pd.NaT,
            'Density_obs_veh_per_mi_ln': pd.to_numeric(g['Density'], errors='coerce').values[:L],
            'Speed_obs_mph': pd.to_numeric(g['Speed'], errors='coerce').values[:L],
            'Speed_cal_mph': v_cal_all[:L],
            'TT_obs_min': tt_obs_use,
            'TT_cal_min': tt_cal_use,
            'Flow_obs_veh_per_h_ln': pd.to_numeric(g['Flow per hour'], errors='coerce').values[:L],
            'vf': vf, 'k_jam': k_jam, 'kc': kc
        })
        ts_rows.append(ts_block)
    else:
        print(f"[INFO] Day {day_label}: skip timeseries block due to zero samples.")

    if SAVE_PLOTS and create_ts_block:
        os.makedirs(PLOT_DIR_FULL / "tt_series", exist_ok=True)

        if (time_col is not None) and (g[time_col].notna().any()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_axis = pd.to_datetime(g[time_col], errors='coerce').values[:L]
            x_label = 'Time'
        else:
            x_axis = np.arange(L)
            x_label = 'Index'

        plt.figure(figsize=(9, 5), dpi=200)
        plt.plot(x_axis, tt_obs_use, label='Observed TT (min)', linewidth=1.8)
        plt.plot(x_axis, tt_cal_use, label='Calibrated TT (min)', linewidth=1.8)
        plt.xlabel(x_label); plt.ylabel('Travel Time (min)')
        plt.title(f'TT Time Series — day {day_label}')
        plt.legend(); plt.tight_layout()
        out_png_ts = PLOT_DIR_FULL / "tt_series" / f"{day_label}_tt_series.png"
        plt.savefig(out_png_ts, dpi=220)
        plt.close()


    # optional plot
    if SAVE_PLOTS and np.isfinite(vf) and np.isfinite(k_jam) and len(k_fit) > 1:
        os.makedirs(PLOT_DIR_FULL / "dens_speed", exist_ok=True)
        xk = np.linspace(max(1e-3, np.nanmin(k_fit)), max(5.0, np.nanmax(k_fit)), 200)
        plt.figure(figsize=(7,5))
        plt.scatter(pd.to_numeric(g['Density'], errors='coerce').values,
                    pd.to_numeric(g['Speed'], errors='coerce').values,
                    s=8, facecolors='none', edgecolors='r', label='Observed')
        plt.plot(xk, gs_speed_of_density(xk, vf, k_jam), lw=2.2, label='Greenshields (robust)')
        keep_pct = diag.get("keep_rate_%", np.nan)
        plt.xlabel('Density (veh/mi/ln)'); plt.ylabel('Speed (mph)')
        plt.title(f'Density vs Speed day {day_label}\n'
                  f'vf={vf:.1f}, k_jam={k_jam:.1f} | TT RMSE={rmse_tt_val if np.isfinite(rmse_tt_val) else np.nan:.2f} min | keep={keep_pct:.1f}%')
        plt.legend(); plt.tight_layout()
        plt.savefig(PLOT_DIR_FULL / "dens_speed" / f'{day_label}_dens_speed.png', dpi=220)
        plt.close()

# ========================= Save per-day results =========================
daily_params_metrics = pd.DataFrame(daily_rows).sort_values('day', ignore_index=True)
daily_params_metrics.to_csv(OUTPUT_DIR/'daily_params_metrics_multi.csv', index=False)

if ts_rows:
    daily_timeseries_outputs = pd.concat(ts_rows, ignore_index=True)
    daily_timeseries_outputs.to_csv(OUTPUT_DIR/'daily_timeseries_outputs_multi.csv', index=False)

    tt_obs_all = daily_timeseries_outputs['TT_obs_min'].to_numpy(dtype=float)
    tt_cal_all = daily_timeseries_outputs['TT_cal_min'].to_numpy(dtype=float)
    mask_all = np.isfinite(tt_obs_all) & np.isfinite(tt_cal_all)
    n_eff = int(mask_all.sum())

    if n_eff == 0:
        print("[WARN] Overall metrics: no valid pairs; set to NaN.")
        overall_mae = overall_rmse = overall_mape = np.nan
    else:
        overall_mae = mean_absolute_error(tt_obs_all[mask_all], tt_cal_all[mask_all])
        overall_rmse = rmse(tt_obs_all[mask_all], tt_cal_all[mask_all])
        overall_mape = mape(tt_obs_all[mask_all], tt_cal_all[mask_all])

    print("=== Overall in-sample TT metrics (capped) ===")
    print(f"MAE = {overall_mae:.3f} min | RMSE = {overall_rmse:.3f} min | MAPE = {overall_mape:.2f}%")
else:
    print("No per-timestamp outputs written (no successful fits).")

print("\nWrote daily_params_metrics_multi.csv and daily_timeseries_outputs_multi.csv.")

# ========================= OOS: N_TRAIN -> N_TRAIN+1 =========================
oos_records = []
if len(daily_params_metrics) >= N_TRAIN + 1:
    # only successful fitted days
    valid = daily_params_metrics[np.isfinite(daily_params_metrics['vf']) & np.isfinite(daily_params_metrics['k_jam'])].copy()
    valid = valid.sort_values('day', ignore_index=True)

    if len(valid) >= N_TRAIN + 1:
        train_slice = valid.iloc[:N_TRAIN]
        test_row    = valid.iloc[N_TRAIN]
        test_day    = _norm_day(test_row['day'])

        # aggregate params on training days
        vf_train = train_slice['vf'].to_numpy(float)
        kj_train = train_slice['k_jam'].to_numpy(float)
        vf_agg, kj_agg = _agg_params_for_oos(vf_train, kj_train, method=OOS_PARAM_AGG, trim_alpha=TRIM_ALPHA)

        # ---------- load test day timeseries: CSV first, fallback to raw Excel ----------
        ts_test = None
        try:
            df_ts_all = pd.read_csv(OUTPUT_DIR/'daily_timeseries_outputs_multi.csv', dtype={'day': str})
            df_ts_all['day_norm'] = df_ts_all['day'].map(_norm_day)
            ts_test = df_ts_all[df_ts_all['day_norm'] == test_day].copy()
        except Exception as e:
            print(f"[INFO][OOS] Read daily_timeseries_outputs_multi.csv failed: {e}. Will fallback to raw Excel.")

        if ts_test is None or ts_test.empty or 'Density_obs_veh_per_mi_ln' not in ts_test.columns:
            test_file = valid.iloc[N_TRAIN]['file']
            raw_path = os.path.join(DATA_DIR_INPUT, test_file)
            print(f"[INFO][OOS] Timeseries for day {test_day} not found in CSV. Fallback to raw: {raw_path}")
            g = pd.read_excel(raw_path)

            time_col = _pick_time_col(g)
            if time_col is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    g[time_col] = pd.to_datetime(g[time_col], errors='coerce')

            dens  = pd.to_numeric(g.get('Density'), errors='coerce').to_numpy()
            speed = pd.to_numeric(g.get('Speed'),   errors='coerce').to_numpy()

            if 'tt_obs_min' in g.columns:
                tt_obs_raw = pd.to_numeric(g['tt_obs_min'], errors='coerce').to_numpy()
            else:
                tt_obs_raw = (LEN_MI / np.maximum(speed, 1e-6)) * 60.0
            tt_obs = np.clip(tt_obs_raw, None, TT_CAP_MIN)

            ts_test = pd.DataFrame({
                'day': test_day,
                'file': os.path.basename(test_file),
                'time': g[time_col] if time_col is not None else pd.NaT,
                'Density_obs_veh_per_mi_ln': dens,
                'TT_obs_min': tt_obs
            })

        # filter usable density
        ts_test = ts_test.copy()
        ts_test['Density_obs_veh_per_mi_ln'] = pd.to_numeric(ts_test['Density_obs_veh_per_mi_ln'], errors='coerce')
        mask_k = np.isfinite(ts_test['Density_obs_veh_per_mi_ln'])
        ts_test = ts_test[mask_k]

        out_oos_ts = OUTPUT_DIR/ f"timeseries_day{test_day}_oos_pred.csv"

        # ---------- compute predictions & metrics ----------
        if (len(ts_test) == 0) or (not np.isfinite(vf_agg)) or (not np.isfinite(kj_agg)):
            print(f"[WARN][OOS] Day {test_day}: no usable samples OR invalid (vf,kj). Metrics=NaN.")
            ts_test['Speed_pred_oos_mph'] = []
            ts_test['TT_pred_oos_min']    = []
            ts_test.to_csv(out_oos_ts, index=False)
            mae = r = mp = np.nan
        else:
            k_test = ts_test['Density_obs_veh_per_mi_ln'].to_numpy(float)
            v_pred = gs_speed_of_density(k_test, vf_agg, kj_agg)
            tt_pred = _safe_tt(v_pred)

            if 'TT_obs_min' in ts_test.columns:
                tt_obs_test = pd.to_numeric(ts_test['TT_obs_min'], errors='coerce').to_numpy(float)
            else:
                spd = pd.to_numeric(ts_test.get('Speed_obs_mph'), errors='coerce').to_numpy(float) if 'Speed_obs_mph' in ts_test.columns else np.full_like(tt_pred, np.nan)
                tt_obs_test = (LEN_MI / np.maximum(spd, 1e-6)) * 60.0
                tt_obs_test = np.clip(tt_obs_test, None, TT_CAP_MIN)

            L = min(len(tt_obs_test), len(tt_pred))
            tt_obs_use = tt_obs_test[:L]
            tt_pred_use = tt_pred[:L]
            mask_pair = np.isfinite(tt_obs_use) & np.isfinite(tt_pred_use)

            if mask_pair.sum() == 0:
                print(f"[WARN][OOS] Day {test_day}: zero valid TT pairs. Metrics=NaN.")
                mae = r = mp = np.nan
            else:
                mae = mean_absolute_error(tt_obs_use[mask_pair], tt_pred_use[mask_pair])
                r   = rmse(tt_obs_use[mask_pair], tt_pred_use[mask_pair])
                mp  = mape(tt_obs_use[mask_pair], tt_pred_use[mask_pair])

            ts_test = ts_test.iloc[:L].copy()
            ts_test['Speed_pred_oos_mph'] = v_pred[:L]
            ts_test['TT_pred_oos_min']    = tt_pred[:L]
            ts_test.to_csv(out_oos_ts, index=False)

        print(f"\n=== OOS ({N_TRAIN}->{N_TRAIN+1}) using {OOS_PARAM_AGG} params ===")
        print(f"Train days head: {train_slice['day'].tolist()[:3]} ... tail: {train_slice['day'].tolist()[-3:]}")
        print(f"Test day: {test_day} | vf_agg={vf_agg if np.isfinite(vf_agg) else np.nan:.3f}, k_jam_agg={kj_agg if np.isfinite(kj_agg) else np.nan:.3f}")
        print(f"OOS TT: MAE={mae if np.isfinite(mae) else np.nan:.3f} min | "
              f"RMSE={r if np.isfinite(r) else np.nan:.3f} min | "
              f"MAPE={mp if np.isfinite(mp) else np.nan:.2f}%")

        oos_records.append({
            'N_TRAIN': N_TRAIN,
            'param_agg': OOS_PARAM_AGG,
            'test_day': test_day,
            'vf_agg': vf_agg,
            'k_jam_agg': kj_agg,
            'OOS_MAE_TT': mae,
            'OOS_RMSE_TT': r,
            'OOS_MAPE_TT_%': mp,
            'train_first_day': str(_norm_day(train_slice.iloc[0]['day'])),
            'train_last_day':  str(_norm_day(train_slice.iloc[-1]['day'])),
        })
    else:
        print("[OOS] Not enough valid fitted days for OOS. Need at least N_TRAIN+1.")
else:
    print("[OOS] Not enough days for OOS. Need at least N_TRAIN+1.")

# save OOS summary
if oos_records:
    pd.DataFrame(oos_records).to_csv(OUTPUT_DIR/ 'oos_eval_summary.csv', index=False)
    print(f"Wrote {(OUTPUT_DIR / 'oos_eval_summary.csv').resolve()}")

else:
    pd.DataFrame(columns=[
        'N_TRAIN','param_agg','test_day','vf_agg','k_jam_agg',
        'OOS_MAE_TT','OOS_RMSE_TT','OOS_MAPE_TT_%',
        'train_first_day','train_last_day'
    ]).to_csv( OUTPUT_DIR/'oos_eval_summary.csv', index=False)
    print(f"Wrote empty {(OUTPUT_DIR / 'oos_eval_summary.csv').resolve()} (no OOS run).")
# ========================= Plot: last day's TT comparison =========================
try:
    # 1) Read the full time-series output and determine the “last day”
    df_ts_all = pd.read_csv(OUTPUT_DIR/'daily_timeseries_outputs_multi.csv', dtype={'day': str})
    # Normalize the 'day' field to ensure consistent sorting
    def _norm_day_local(x):
        s = str(x).strip()
        s = re.sub(r"\D", "", s)
        if s == "": return ""
        return f"{int(s):04d}"
    df_ts_all['day_norm'] = df_ts_all['day'].map(_norm_day_local)

    if df_ts_all.empty:
        raise RuntimeError("daily_timeseries_outputs_multi.csv is empty; cannot plot.")

    # Select the last day
    last_day_norm = sorted(df_ts_all['day_norm'].unique())[-1]
    df_last = df_ts_all[df_ts_all['day_norm'] == last_day_norm].copy()

    # 2) Handle the time axis: prefer the 'time' column
    if 'time' in df_last.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df_last['time'] = pd.to_datetime(df_last['time'], errors='coerce')
        # If all are NaT, use the index instead
        if df_last['time'].isna().all():
            x_axis = np.arange(len(df_last))
            x_label = 'Index'
        else:
            x_axis = df_last['time']
            x_label = 'Time'
    else:
        x_axis = np.arange(len(df_last))
        x_label = 'Index'

    # 3) Base curves: observed and calibrated (in-sample)
    tt_obs = pd.to_numeric(df_last.get('TT_obs_min'), errors='coerce').to_numpy()
    tt_cal = pd.to_numeric(df_last.get('TT_cal_min'), errors='coerce').to_numpy()

    # 4) If OOS prediction exists (CSV with the same last-day name), overlay it
    oos_csv = OUTPUT_DIR/ f"timeseries_day{last_day_norm}_oos_pred.csv"
    tt_oos = None
    if os.path.exists(oos_csv):
        df_oos = pd.read_csv(oos_csv)
        # Align the length with in-sample data
        n = min(len(df_last), len(df_oos))
        if n > 0:
            df_oos = df_oos.iloc[:n].copy()
            tt_oos = pd.to_numeric(df_oos.get('TT_pred_oos_min'), errors='coerce').to_numpy()
            # If in-sample uses a time axis, try to align with OOS time; otherwise use x_axis
            if x_label == 'Time' and 'time' in df_oos.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t2 = pd.to_datetime(df_oos['time'], errors='coerce')
                if not t2.isna().all():
                    x_axis = t2  # Using OOS time axis is fine; usually identical to in-sample
        else:
            tt_oos = None

    # 5) Plot
    plt.figure(figsize=(9, 5), dpi=200)
    # Observed
    if tt_obs is not None and len(tt_obs) > 0:
        plt.plot(x_axis, tt_obs[:len(x_axis)], label='Observed TT (min)', linewidth=1.8)
    # Calibrated
    if tt_cal is not None and len(tt_cal) > 0:
        plt.plot(x_axis, tt_cal[:len(x_axis)], label='Calibrated (in-sample) TT (min)', linewidth=1.8)
    # OOS
    if tt_oos is not None:
        plt.plot(x_axis, tt_oos[:len(x_axis)], label='OOS Predicted TT (min)', linewidth=1.8, linestyle='--')

    plt.xlabel(x_label)
    plt.ylabel('Travel Time (min)')
    title_day = last_day_norm
    plt.title(f'Last Day TT Comparison — day {title_day}')
    plt.legend()
    plt.tight_layout()
    out_png = OUTPUT_DIR/'tt_last_day_compare.png'
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"Wrote {out_png}")
except Exception as e:
    print(f"[WARN] Failed to plot last day TT comparison: {e}")

