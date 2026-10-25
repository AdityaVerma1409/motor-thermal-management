"""
3_plot_dashboard.py
===================
Generates publication-ready figures for your electric motor thermal management project:
  1. figure1_model_benchmark.png   -> 3-panel comprehensive benchmark:
                                      - Panel A: 1-Step 5-Fold CV Accuracy (MAE & RMSE) with baselines
                                      - Panel B: Single-Prediction Inference Latency (us / prediction)
                                      - Panel C: Multi-Step Recursive Trajectory Error (1 to 30 min forecast)
  2. figure2_advisory_forecast.png -> Real-time 30-min recursive trajectory (Uncooled vs Cooled vs Limit)
  3. figure3_coolant_paradox.png   -> Physical coolant monotonicity check (pure KNN failure vs monotonic HistGBM)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib
import warnings
warnings.filterwarnings("ignore")

# Clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


# -----------------------------------------------------------------------------
# FIGURE 1: 3-PANEL COMPREHENSIVE BENCHMARK
# -----------------------------------------------------------------------------
def plot_model_comparison(results_csv="model_comparison_results.csv", traj_csv="trajectory_forecast_results.csv"):
    df_cv = pd.read_csv(results_csv)
    df_traj = pd.read_csv(traj_csv)
    
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    
    # -------------------------------------------------------------------------
    # Panel A: 1-Step 5-Fold Cross-Validation Accuracy
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    labels = ["No-Change\nBaseline", "Ridge\nRegression", "KNN (k=15,\n35k uniform)", "HistGBM\n(Modern Tree)"]
    x = np.arange(len(labels))
    width = 0.35
    
    # Flexible column resolution
    maes = df_cv["MAE (deg C)"].values if "MAE (deg C)" in df_cv else df_cv["MAE_mean"].values
    rmses = df_cv["RMSE (deg C)"].values if "RMSE (deg C)" in df_cv else df_cv["RMSE_mean"].values
    lats = df_cv["Latency_per_pred_us"].values if "Latency_per_pred_us" in df_cv else df_cv["Latency_us_mean"].values
    
    bars1 = ax1.bar(x - width/2, maes, width, label="MAE (deg C)", color="#1f77b4", edgecolor="black", alpha=0.85)
    bars2 = ax1.bar(x + width/2, rmses, width, label="RMSE (deg C)", color="#ff7f0e", edgecolor="black", alpha=0.85)
    
    # Value annotations on bars
    for b in bars1:
        ax1.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    for b in bars2:
        ax1.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
        
    ax1.set_ylabel("Error (deg C) - Lower is Better", fontweight="bold")
    ax1.set_title("A. 1-Step (5s) 5-Fold Cross-Validation Error", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, max(rmses) * 1.25)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------------------
    # Panel B: Single-Prediction Inference Latency (us / query)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    # Replace 0.0 with 0.05 for log scale display if needed, or bar plot
    lat_colors = ["#7f7f7f", "#2ca02c", "#d62728", "#9467bd"]
    bars_lat = ax2.bar(x, lats, width=0.5, color=lat_colors, edgecolor="black", alpha=0.85)
    
    for i, b in enumerate(bars_lat):
        val = lats[i]
        txt = f"{val:.1f} us" if val >= 0.1 else "< 0.1 us"
        ax2.annotate(txt, (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
        
    ax2.set_ylabel("Latency (microseconds / prediction)", fontweight="bold")
    ax2.set_title("B. Single-Prediction Inference Latency (ECU Feasibility)", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, max(lats) * 1.25)
    ax2.axhline(5e6, color="red", linestyle=":", label="5s Cycle Budget (5,000,000 us)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.annotate("5-second loop budget = 5,000,000 us\nAll models execute in < 0.001% of cycle",
                 xy=(0.04, 0.75), xycoords="axes fraction", fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#999999"))

    # -------------------------------------------------------------------------
    # Panel C: Multi-Step Recursive Trajectory Error vs Horizon (1 to 30 min)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    
    def parse_mean(val):
        if isinstance(val, (int, float)):
            return float(val)
        return float(str(val).split("+/-")[0].replace("C", "").strip())
    
    times = [1, 5, 10, 30]
    nc_col = [c for c in df_traj.columns if "No Change" in c or "No-Change" in c][0]
    ridge_col = [c for c in df_traj.columns if "Ridge" in c][0]
    knn_col = [c for c in df_traj.columns if "KNN" in c][0]
    hgbm_col = [c for c in df_traj.columns if "HistGradient" in c or "HistGBM" in c][0]

    nc_err = np.array([parse_mean(v) for v in df_traj[nc_col].values])
    ridge_err = np.array([parse_mean(v) for v in df_traj[ridge_col].values])
    knn_err = np.array([parse_mean(v) for v in df_traj[knn_col].values])
    hgbm_err = np.array([parse_mean(v) for v in df_traj[hgbm_col].values])
    
    ax3.plot(times, nc_err, marker="o", color="#7f7f7f", linestyle="--", linewidth=2, label=f"Predict No-Change Baseline (drifts to {nc_err[-1]:.1f} deg C)")
    ax3.plot(times, ridge_err, marker="s", color="#1f77b4", linestyle="-.", linewidth=2, label=f"Ridge Regression (Baseline: {ridge_err[-1]:.2f} deg C @ 30 min)")
    ax3.plot(times, knn_err, marker="^", color="#d62728", linestyle=":", linewidth=2, label=f"KNN (k=15, 35k uniform: {knn_err[-1]:.2f} deg C @ 30 min)")
    ax3.plot(times, hgbm_err, marker="D", color="#2ca02c", linestyle="-", linewidth=2.5, label=f"HistGradientBoosting (Ensemble: {hgbm_err[-1]:.2f} deg C @ 30 min)")
    
    # Annotate end points
    ax3.annotate(f"{nc_err[-1]:.1f} deg C", (times[-1], nc_err[-1]), xytext=(5, 0), textcoords="offset points", fontweight="bold", color="#555555")
    ax3.annotate(f"{ridge_err[-1]:.2f} deg C", (times[-1], ridge_err[-1]), xytext=(5, 3), textcoords="offset points", fontweight="bold", color="#1f77b4")
    ax3.annotate(f"{knn_err[-1]:.2f} deg C", (times[-1], knn_err[-1]), xytext=(5, -10), textcoords="offset points", fontweight="bold", color="#d62728")
    ax3.annotate(f"{hgbm_err[-1]:.2f} deg C", (times[-1], hgbm_err[-1]), xytext=(5, -5), textcoords="offset points", fontweight="bold", color="#2ca02c")
    
    ax3.set_xlabel("Forecast Horizon (Minutes)", fontweight="bold", fontsize=11)
    ax3.set_ylabel("Recursive Trajectory MAE (deg C)", fontweight="bold", fontsize=11)
    ax3.set_title("C. Multi-Step Recursive Trajectory Forecast Error Across Time Horizons (Held-Out Test Profiles)", fontweight="bold", fontsize=12)
    ax3.set_xticks(times)
    ax3.set_xticklabels(["1 min\n(12 steps)", "5 min\n(60 steps)", "10 min\n(120 steps)", "30 min\n(360 steps)"])
    ax3.set_xlim(0, 34)
    ax3.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax3.grid(True, linestyle="--", alpha=0.5)

    filename = "figure1_model_benchmark.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filename}")


# -----------------------------------------------------------------------------
# FIGURE 2: 30-MINUTE ADVISORY FORECAST (UNCOOLED VS COOLED)
# -----------------------------------------------------------------------------
def plot_advisory_forecast():
    prep = importlib.import_module("1_model_comparison")
    adv_module = importlib.import_module("2_thermal_advisor")
    
    df_thinned = prep.load_and_preprocess_data("measures_v2.csv", thin_factor=10)
    advisor = adv_module.MotorThermalAdvisor(dt=5.0, horizon_min=30.0, safety_margin_sec=60.0)
    advisor.fit(df_thinned)
    
    # Pick a high-load profile state
    sample = df_thinned[
        (df_thinned["motor_speed"] > 4000) & (df_thinned["torque"] > 35)
    ].iloc[0].to_dict()
    
    t_limit = 72.0
    adv = advisor.compute_advisory(sample, t_limit=t_limit, cool_drop=14.0)
    
    time_min = np.arange(len(adv["t1_trajectory"])) * 5.0 / 60.0  # In minutes
    
    plt.figure(figsize=(10, 5.5))
    
    # Plot uncooled curve
    plt.plot(time_min, adv["t1_trajectory"], color="#d62728", linestyle="--", linewidth=2.5, label="T1: Uncooled Forecast")
    
    # Plot cooled curve
    plt.plot(time_min, adv["t5_cooled_trajectory"], color="#1f77b4", linewidth=2.5, label="T5: Cooled Trajectory")
    
    # Plot safety threshold line
    plt.axhline(y=t_limit, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Safety Limit ({t_limit:.1f} deg C)")
    
    # Highlight cooling window (T3 to T3+T4)
    if adv["t3_start_seconds"] is not None and adv["t4_duration_seconds"] > 0:
        t_start_min = adv["t3_start_seconds"] / 60.0
        t_end_min = (adv["t3_start_seconds"] + adv["t4_duration_seconds"]) / 60.0
        plt.axvspan(t_start_min, t_end_min, color="#17becf", alpha=0.25, label=f"Active Cooling Window ({adv['t4_duration_seconds']:.0f}s)")
    
    plt.title("30-Minute Motor Thermal Forecast & Real-Time Cooling Advisory", fontsize=13, fontweight="bold")
    plt.xlabel("Time Horizon (Minutes)", fontsize=11, fontweight="bold")
    plt.ylabel("Rotor Magnet Temperature (deg C)", fontsize=11, fontweight="bold")
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Info annotation box
    info_text = (
        f"Operating: {sample['motor_speed']:.0f} RPM, {sample['torque']:.1f} Nm\n"
        f"T2 (Limit Crossing): {adv['t2_seconds']/60:.1f} min\n" if adv['t2_seconds'] else "T2: Not Reached\n"
    )
    if adv['t3_start_seconds']:
        info_text += f"T3 (Start Cooling): {adv['t3_start_seconds']/60:.1f} min\n"
        info_text += f"T4 (Cool Duration): {adv['t4_duration_seconds']/60:.1f} min\n"
    info_text += f"T5 (Cooled Peak): {adv['t5_peak']:.1f} deg C"
    
    plt.annotate(
        info_text, xy=(0.03, 0.65), xycoords="axes fraction",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#999999", alpha=0.9)
    )
    
    filename = "figure2_advisory_forecast.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filename}")


# -----------------------------------------------------------------------------
# FIGURE 3: THE COOLANT SENSITIVITY CHECK (WHY KNN FAILS VS HISTGBM)
# -----------------------------------------------------------------------------
def plot_coolant_sensitivity():
    prep = importlib.import_module("1_model_comparison")
    adv_module = importlib.import_module("2_thermal_advisor")
    
    df_thinned = prep.load_and_preprocess_data("measures_v2.csv", thin_factor=10)
    advisor = adv_module.MotorThermalAdvisor(dt=5.0)
    advisor.fit(df_thinned)
    
    # Test point
    sample = df_thinned.iloc[100].to_dict()
    coolant_sweep = np.linspace(15.0, 45.0, 20)
    
    knn_preds = []
    gbm_preds = []
    
    for c in coolant_sweep:
        x = np.array([[
            sample["pm"], c, sample["ambient"],
            sample["u_d"], sample["u_q"], sample["i_d"], sample["i_q"],
            sample["motor_speed"], sample["torque"]
        ]])
        knn_preds.append(advisor.knn_model.predict(x)[0])
        gbm_preds.append(advisor.model.predict(x)[0])
        
    plt.figure(figsize=(8.5, 5))
    plt.plot(coolant_sweep, knn_preds, label="K-Nearest Neighbors (KNN)", color="#d95f02", linestyle="--", marker="o")
    plt.plot(coolant_sweep, gbm_preds, label="HistGradientBoosting (Monotonic)", color="#2ca02c", linewidth=2.5, marker="s")
    
    plt.title("Coolant Sensitivity Sanity Check (Physical Monotonicity)", fontsize=13, fontweight="bold")
    plt.xlabel("Simulated Coolant Temperature (deg C)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted 5-Second Heating Delta_T (deg C)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    filename = "figure3_coolant_paradox.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {filename}")


if __name__ == "__main__":
    print("=" * 75)
    print("  GENERATING PUBLICATION-READY PROJECT FIGURES")
    print("=" * 75)
    plot_model_comparison()
    plot_advisory_forecast()
    plot_coolant_sensitivity()
    print("\n[SUCCESS] All 3 project figures generated and saved as high-resolution PNGs!")
