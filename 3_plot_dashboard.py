"""
3_plot_dashboard.py
===================
Generates 3 publication-ready visual figures for your project presentation,
report, and slides:
  1. figure1_model_benchmark.png   -> Compares MAE, RMSE, and R2 of Ridge vs KNN vs HistGBM
  2. figure2_advisory_forecast.png -> The 30-min trajectory: Uncooled vs Cooled vs Limit Line
  3. figure3_coolant_paradox.png   -> Visualizes why pure KNN gets confused by coolant
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib

# Clean scientific style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


# -----------------------------------------------------------------------------
# FIGURE 1: MODEL ACCURACY COMPARISON
# -----------------------------------------------------------------------------
def plot_model_comparison(results_csv="model_comparison_results.csv"):
    df = pd.read_csv(results_csv)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    width = 0.28
    
    rects1 = ax1.bar(x - width, df["MAE (deg C)"], width, label="MAE (°C)", color="#2b5c8f")
    rects2 = ax1.bar(x, df["RMSE (deg C)"], width, label="RMSE (°C)", color="#d95f02")
    
    ax2 = ax1.twinx()
    rects3 = ax2.bar(x + width, df["R2 Score"], width, label="R² Score", color="#2ca02c")
    
    ax1.set_ylabel("Error (°C) - Lower is Better", fontweight="bold")
    ax2.set_ylabel("R² Score - Higher is Better", fontweight="bold")
    ax1.set_title("Model Comparison on Held-Out Test Profiles (Electric Motor Dataset)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model"], rotation=12, ha="right", fontweight="bold")
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    # Save figure
    filename = "figure1_model_benchmark.png"
    plt.savefig(filename, dpi=300)
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
    plt.axhline(y=t_limit, color="#ff7f0e", linestyle=":", linewidth=2, label=f"Safety Limit ({t_limit:.1f}°C)")
    
    # Highlight cooling window (T3 to T3+T4)
    if adv["t3_start_seconds"] is not None and adv["t4_duration_seconds"] > 0:
        t_start_min = adv["t3_start_seconds"] / 60.0
        t_end_min = (adv["t3_start_seconds"] + adv["t4_duration_seconds"]) / 60.0
        plt.axvspan(t_start_min, t_end_min, color="#17becf", alpha=0.25, label=f"Active Cooling Window ({adv['t4_duration_seconds']:.0f}s)")
    
    plt.title("30-Minute Motor Thermal Forecast & Real-Time Cooling Advisory", fontsize=14, fontweight="bold")
    plt.xlabel("Time Horizon (Minutes)", fontsize=12, fontweight="bold")
    plt.ylabel("Rotor Magnet Temperature (°C)", fontsize=12, fontweight="bold")
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
    info_text += f"T5 (Cooled Peak): {adv['t5_peak']:.1f} °C"
    
    plt.annotate(
        info_text, xy=(0.03, 0.65), xycoords="axes fraction",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#999999", alpha=0.9)
    )
    
    filename = "figure2_advisory_forecast.png"
    plt.savefig(filename, dpi=300)
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
        
    plt.figure(figsize=(8, 4.8))
    plt.plot(coolant_sweep, knn_preds, label="K-Nearest Neighbors (KNN)", color="#d95f02", linestyle="--", marker="o")
    plt.plot(coolant_sweep, gbm_preds, label="HistGradientBoosting (Monotonic)", color="#2ca02c", linewidth=2.5, marker="s")
    
    plt.title("Coolant Sensitivity Sanity Check (Physical Monotonicity)", fontsize=13, fontweight="bold")
    plt.xlabel("Simulated Coolant Temperature (°C)", fontsize=11, fontweight="bold")
    plt.ylabel("Predicted 5-Second Heating ΔT (°C)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    filename = "figure3_coolant_paradox.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"[SAVED] {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("  GENERATING PUBLICATION-READY PROJECT FIGURES")
    print("=" * 70)
    plot_model_comparison()
    plot_advisory_forecast()
    plot_coolant_sensitivity()
    print("\n[SUCCESS] All 3 project figures generated and saved as high-resolution PNGs!")
