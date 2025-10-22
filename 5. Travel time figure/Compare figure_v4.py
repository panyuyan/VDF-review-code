# -*- coding: utf-8 -*-
"""
Travel Time Plotting & Evaluation (Line + Bar Combo Chart, Fixed Order)

Fixed order from left to right:
Density-based, Queue-based, FD-based, LSTM, Transformer, PINN

Unit:
All original values in “hours” are converted to “minutes” after reading.

Outputs:
1. Time-series comparison line chart
2. Performance metric bar chart + line chart
3. Combined chart: MAE/RMSE bars + MAPE line
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================== User Settings ==================
EXCEL_FILE  = "Summary_v2.xlsx"
SHEET_NAME  = 0
OBS_COL     = "tt_obs_min"
PRED_PREFIX = "tt_pred_min"

# Output directories
PLOT_DIR, METRIC_DIR = "plots", "metrics"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)

# Plot style
TICK_SIZE  = 16
LABEL_SIZE = 16   # Axis label font size
plt.rc('font', family='Times New Roman')
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE

COLOR_OBS = "#64B5F6"
PRED_PALETTE = [
    "#F4A7A7", "#8CD17D", "#B699E6", "#76B7B2", "#F1CE63",
    "#F28E2B", "#E15759", "#59A14F", "#4E79A7", "#9C755F"
]

# ======== Display Names (for legends and x-axis labels) ========
DISPLAY_NAMES = {
    "DENSITY":     "Density-based",
    "QUEUE":       "Queue-based",
    "FD":          "FD-based",
    "LSTM":        "LSTM",
    "TRANSFORMER": "Transformer",
    "PINN":        "PINN",
}

def save_png(fig, filename):
    fig.savefig(os.path.join(PLOT_DIR, filename + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)

# ================== Read Data ==================
if not os.path.isfile(EXCEL_FILE):
    raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}\nFiles in current directory:\n" + "\n".join(os.listdir(".")))
try:
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
except ValueError as e:
    with pd.ExcelFile(EXCEL_FILE) as xf:
        sheets = xf.sheet_names
    raise ValueError(f"Failed to read the sheet: {e}\nAvailable sheets: {sheets}")

if OBS_COL not in df.columns:
    raise ValueError(f"Observation column '{OBS_COL}' not found. Available columns: {list(df.columns)}")

all_pred_cols = [c for c in df.columns if c.startswith(PRED_PREFIX)]
if not all_pred_cols:
    raise ValueError(f"No columns found starting with '{PRED_PREFIX}'. Available columns: {list(df.columns)}")

# ---- Unit conversion: hours → minutes (for observed and predicted columns) ----
df[OBS_COL] = df[OBS_COL].astype(float) * 60.0
for col in all_pred_cols:
    df[col] = df[col].astype(float) * 60.0

# ================== Fixed Order & Alias Matching ==================
METHOD_ORDER_CANON = ["DENSITY", "QUEUE", "FD", "LSTM", "TRANSFORMER", "PINN"]
ALIASES = {
    "DENSITY":      ["density", "min_density"],
    "QUEUE":        ["queue", "min_queue"],
    "FD":           ["fd"],
    "LSTM":         ["lstm"],
    "TRANSFORMER":  ["transformer"],
    "PINN":         ["pinn"],
}

def find_col_for_method(method_key):
    for suffix in ALIASES[method_key]:
        col = f"{PRED_PREFIX}_{suffix}"
        if col in all_pred_cols:
            return col
    return None

# [(method_key, colname), ...] keep fixed order
ordered_pairs, missing = [], []
for m in METHOD_ORDER_CANON:
    col = find_col_for_method(m)
    if col is None:
        missing.append(m)
    else:
        ordered_pairs.append((m, col))
if missing:
    print("The following methods were not found in the table and will be skipped: " + ", ".join(missing))

# ================== Metric Computation ==================
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.mean(np.abs((y_pred - y_true) / denom)) * 100.0

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    return mae, rmse, r2, mape

metrics_rows = []
for method_key, col in ordered_pairs:
    mae, rmse, r2, mape = eval_metrics(df[OBS_COL].values, df[col].values)
    metrics_rows.append({
        "Method": DISPLAY_NAMES[method_key],
        "MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape
    })
metrics_df = pd.DataFrame(metrics_rows)
metrics_df[["MAE", "RMSE", "R²", "MAPE"]] = metrics_df[["MAE", "RMSE", "R²", "MAPE"]].astype(float).round(3)

# Save metrics
metrics_df.to_csv(
    os.path.join(METRIC_DIR, "tt_metrics.csv"),
    index=False, encoding="utf-8-sig", float_format="%.3f"
)
with pd.ExcelWriter(os.path.join(METRIC_DIR, "tt_metrics.xlsx")) as w:
    metrics_df.to_excel(w, index=False, sheet_name="metrics")

# ================== Time-Series Comparison ==================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.plot(df[OBS_COL].values, label="Observed", linewidth=2.6, alpha=0.98, color=COLOR_OBS)
for i, (method_key, col) in enumerate(ordered_pairs):
    color = PRED_PALETTE[i % len(PRED_PALETTE)]
    ax.plot(df[col].values, linestyle="--", linewidth=1.9, alpha=0.95,
            label=DISPLAY_NAMES[method_key], color=color)
ax.set_title("Observed vs. Predicted Travel Time", fontsize=18)
ax.set_xlabel("Time (5-min interval)", fontsize=LABEL_SIZE)
ax.set_ylabel("Travel Time (min)", fontsize=LABEL_SIZE)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#ax.grid(True, alpha=0.25)
for spine in ax.spines.values():
    spine.set_visible(True); spine.set_linewidth(1.0)
leg = ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9,
                fontsize=10, ncol=2, borderpad=0.6, labelspacing=0.5, handlelength=1.6)
leg.get_frame().set_linewidth(0.8)
ax.tick_params(axis='both', which='both', labelsize=TICK_SIZE)
plt.tight_layout(); save_png(fig, "tt_comparison")

# ================== Metric Bar Plot ==================
def plot_metric_bar(df_metrics, metric, save_name):
    labels = df_metrics["Method"].tolist()
    values = df_metrics[metric].values
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = [PRED_PALETTE[i % len(PRED_PALETTE)] for i in range(len(values))]
    ax.bar(x, values, width=0.6, alpha=0.95, color=colors, edgecolor="#333333", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel(metric, fontsize=LABEL_SIZE)
    ax.set_title(f"{metric} by Method", fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax.grid(axis='y', alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0)
    for r, v in enumerate(values):
        ax.text(x[r], v, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    ax.tick_params(axis='y', which='both', labelsize=TICK_SIZE)
    plt.tight_layout(); save_png(fig, save_name)

# ================== Metric Line Plot ==================
def plot_metric_line(df_metrics, metric, save_name):
    labels = df_metrics["Method"].tolist()
    x = np.arange(len(labels))
    y = df_metrics[metric].values
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.plot(x, y, marker='o', linewidth=2.0, alpha=0.95, color="#5A6ACF", label=metric)
    for i, v in enumerate(y):
        ax.scatter([i], [v], s=36, zorder=3, color=PRED_PALETTE[i % len(PRED_PALETTE)])
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=TICK_SIZE)
    ax.set_ylabel(metric, fontsize=LABEL_SIZE)
    ax.set_title(f"{metric} Trend across Methods", fontsize=16)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    #ax.grid(True, alpha=0.25)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0)
    leg = ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
    leg.get_frame().set_linewidth(0.8)
    ax.tick_params(axis='y', which='both', labelsize=TICK_SIZE)
    plt.tight_layout(); save_png(fig, save_name)

# ===== Combined Chart (MAE/RMSE bars + MAPE line) =====
def plot_metric_combo(df_metrics, save_name):
    labels = df_metrics["Method"].tolist()
    x = np.arange(len(labels))
    mae  = df_metrics["MAE"].values
    rmse = df_metrics["RMSE"].values
    mape = df_metrics["MAPE"].values

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    width = 0.36
    bars1 = ax.bar(x - width/2, mae,  width=width, label="MAE",
                   color=PRED_PALETTE[0], edgecolor="#333333", linewidth=0.5, alpha=0.95)
    bars2 = ax.bar(x + width/2, rmse, width=width, label="RMSE",
                   color=PRED_PALETTE[1], edgecolor="#333333", linewidth=0.5, alpha=0.95)

    ax.set_ylabel("MAE / RMSE (min)", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=TICK_SIZE)
    #ax.grid(axis='y', alpha=0.25)

    # Right axis: MAPE (%)
    ax2 = ax.twinx()
    line = ax2.plot(x, mape, marker='o', linewidth=2.0, alpha=0.95,
                    label="MAPE (%)", color="#5A6ACF")[0]
    ax2.set_ylabel("MAPE (%)", fontsize=LABEL_SIZE)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}%"))

    # Add annotations
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
                ha='center', va='bottom', fontsize=10)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{b.get_height():.2f}",
                ha='center', va='bottom', fontsize=10)
    for xi, yi in zip(x, mape):
        ax2.text(xi, yi, f"{yi:.2f}%", ha='center', va='bottom', fontsize=10)

    # Legend inside figure
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = [line], ["MAPE (%)"]
    leg = ax.legend(handles1 + handles2, labels1 + labels2,
                    loc="upper right", frameon=True, facecolor="white", framealpha=0.9,
                    fontsize=10, ncol=2, borderpad=0.6, labelspacing=0.5, handlelength=1.6)
    leg.get_frame().set_linewidth(0.8)

    ax.tick_params(axis='both', which='both', labelsize=TICK_SIZE)
    ax2.tick_params(axis='both', which='both', labelsize=TICK_SIZE)

    ax.set_title("Model Performance Comparison", fontsize=18)
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0)
    for spine in ax2.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0)
    plt.tight_layout(); save_png(fig, save_name)

# ===== Generate Charts =====
for m in ["MAE", "RMSE", "MAPE"]:
    plot_metric_bar(metrics_df, m, f"metric_{m}_bar")
    plot_metric_line(metrics_df, m, f"metric_{m}_line")
plot_metric_combo(metrics_df, "metric_mae_rmse_mape_combo")

print("\nSaved metrics to:")
print(" -", os.path.join(METRIC_DIR, "tt_metrics.csv"))
print(" -", os.path.join(METRIC_DIR, "tt_metrics.xlsx"))
print("Saved plots to:", os.path.abspath(PLOT_DIR))
