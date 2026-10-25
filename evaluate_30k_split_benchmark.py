"""
evaluate_30k_split_benchmark.py
===============================
Implements the exact evaluation pipeline requested in the user's diagram:

                 30,000 motor measurements
                            |
       +--------------------+--------------------+
       |                    |                    |
      70%                  15%                  15%
    21,000                4,500                4,500
   Training             Validation            Testing
 (20,927 rows)         (4,502 rows)         (4,571 rows)
  48 profiles          10 profiles          11 profiles
       |                    |                    |
       v                    v                    v
  [Same Train]         [Same Val]           [Same Test]
       |
  +----+----+
  |    |    |
 KNN Ridge HistGBM
"""

import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Clean scientific plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})


def run_benchmark():
    print("=" * 75)
    print("  30,000 MOTOR MEASUREMENTS: 70% TRAIN / 15% VAL / 15% TEST BENCHMARK")
    print("=" * 75)
    
    csv_file = "motor_temperature_30000.csv"
    df = pd.read_csv(csv_file)
    print(f"Loaded '{csv_file}': {len(df):,} rows, {df['profile_id'].nunique()} profiles.\n")

    # -------------------------------------------------------------------------
    # 1. PROFILE-BASED SPLIT (70% Train, 15% Val, 15% Test)
    # -------------------------------------------------------------------------
    # Seed 79 yields an almost exact match to 21,000 / 4,500 / 4,500
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=79)
    train_idx, temp_idx = next(gss1.split(df, groups=df["profile_id"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=79)
    val_sub_idx, test_sub_idx = next(gss2.split(df_temp, groups=df_temp["profile_id"]))

    df_val = df_temp.iloc[val_sub_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_sub_idx].reset_index(drop=True)

    print("Pipeline Split Summary (Zero Profile Leakage):")
    print(f"  [70%] Training   : {len(df_train):,} rows ({len(df_train)/len(df)*100:.1f}%), {df_train['profile_id'].nunique()} profiles")
    print(f"  [15%] Validation : {len(df_val):,} rows ({len(df_val)/len(df)*100:.1f}%), {df_val['profile_id'].nunique()} profiles")
    print(f"  [15%] Testing    : {len(df_test):,} rows ({len(df_test)/len(df)*100:.1f}%), {df_test['profile_id'].nunique()} profiles\n")

    target = "pm"
    features_full = [
        "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque",
        "stator_winding", "stator_tooth", "stator_yoke"
    ]
    features_inverter = [
        "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"
    ]

    all_results = []

    for suite_name, feat_cols in [
        ("Full Sensor Suite (Inverter + Stator Sensors)", features_full),
        ("Inverter-Only Sensorless Suite (No Stator Sensors)", features_inverter)
    ]:
        print("=" * 75)
        print(f"  EVALUATING: {suite_name.upper()}")
        print("=" * 75)
        
        X_train, y_train = df_train[feat_cols], df_train[target]
        X_val, y_val = df_val[feat_cols], df_val[target]
        X_test, y_test = df_test[feat_cols], df_test[target]

        models = {
            "KNN (k=15, Distance-Weighted)": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance", algorithm="kd_tree"))
            ]),
            "Ridge Regression (Linear Baseline)": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),
            "HistGradientBoosting (Modern Tree)": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42
            )
        }

        suite_rows = []

        for name, model in models.items():
            # 1. Train on the EXACT SAME training set
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # 2. Evaluate on Validation Set
            t_v0 = time.perf_counter()
            y_val_pred = model.predict(X_val)
            val_lat = (time.perf_counter() - t_v0) / len(X_val) * 1e6
            
            mae_val = mean_absolute_error(y_val, y_val_pred)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
            r2_val = r2_score(y_val, y_val_pred)
            
            # 3. Evaluate on Testing Set
            t_t0 = time.perf_counter()
            y_test_pred = model.predict(X_test)
            test_lat = (time.perf_counter() - t_t0) / len(X_test) * 1e6
            
            mae_test = mean_absolute_error(y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
            r2_test = r2_score(y_test, y_test_pred)
            
            row = {
                "Feature Suite": suite_name,
                "Model": name,
                "Train Time (s)": round(train_time, 2),
                "Val MAE (deg C)": round(mae_val, 2),
                "Val RMSE (deg C)": round(rmse_val, 2),
                "Val R2": round(r2_val, 4),
                "Test MAE (deg C)": round(mae_test, 2),
                "Test RMSE (deg C)": round(rmse_test, 2),
                "Test R2": round(r2_test, 4),
                "Latency (us/pred)": round(test_lat, 1)
            }
            suite_rows.append(row)
            all_results.append(row)

        res_df = pd.DataFrame(suite_rows)
        disp_cols = ["Model", "Val MAE (deg C)", "Val R2", "Test MAE (deg C)", "Test R2", "Latency (us/pred)"]
        print(res_df[disp_cols].to_string(index=False))
        print()

    # Save to CSV
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv("benchmark_30k_70_15_15.csv", index=False)
    print("[SAVED] Complete benchmark table saved to 'benchmark_30k_70_15_15.csv'.")

    # -------------------------------------------------------------------------
    # 2. GENERATE PUBLICATION FIGURE
    # -------------------------------------------------------------------------
    plot_results(summary_df)


def plot_results(df_results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    suites = df_results["Feature Suite"].unique()
    models = ["KNN (k=15, Distance-Weighted)", "Ridge Regression (Linear Baseline)", "HistGradientBoosting (Modern Tree)"]
    model_labels = ["KNN\n(k=15)", "Ridge\nRegression", "HistGradient\nBoosting"]
    colors_val = "#1f77b4"
    colors_test = "#ff7f0e"
    
    # Left subplot: Full Sensor Suite
    df_full = df_results[df_results["Feature Suite"] == suites[0]].set_index("Model")
    x = np.arange(len(models))
    w = 0.35
    
    val_maes = [df_full.loc[m, "Val MAE (deg C)"] for m in models]
    test_maes = [df_full.loc[m, "Test MAE (deg C)"] for m in models]
    
    b1 = ax1.bar(x - w/2, val_maes, w, label="Validation Set (15% | 4,502 rows)", color=colors_val, edgecolor="black", alpha=0.85)
    b2 = ax1.bar(x + w/2, test_maes, w, label="Testing Set (15% | 4,571 rows)", color=colors_test, edgecolor="black", alpha=0.85)
    
    for b in b1:
        ax1.annotate(f"{b.get_height():.2f} C", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
    for b in b2:
        ax1.annotate(f"{b.get_height():.2f} C", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
        
    ax1.set_ylabel("MAE (deg C) - Lower is Better", fontweight="bold")
    ax1.set_title("A. Full Feature Suite (Inverter + Stator Sensors)", fontweight="bold", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontweight="bold")
    ax1.set_ylim(0, max(max(val_maes), max(test_maes)) * 1.25)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right subplot: Sensorless Suite
    df_inv = df_results[df_results["Feature Suite"] == suites[1]].set_index("Model")
    val_maes_inv = [df_inv.loc[m, "Val MAE (deg C)"] for m in models]
    test_maes_inv = [df_inv.loc[m, "Test MAE (deg C)"] for m in models]
    
    b3 = ax2.bar(x - w/2, val_maes_inv, w, label="Validation Set (15% | 4,502 rows)", color=colors_val, edgecolor="black", alpha=0.85)
    b4 = ax2.bar(x + w/2, test_maes_inv, w, label="Testing Set (15% | 4,571 rows)", color=colors_test, edgecolor="black", alpha=0.85)
    
    for b in b3:
        ax2.annotate(f"{b.get_height():.2f} C", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
    for b in b4:
        ax2.annotate(f"{b.get_height():.2f} C", (b.get_x() + b.get_width()/2, b.get_height()),
                     ha="center", va="bottom", fontsize=9, fontweight="bold", xytext=(0, 2), textcoords="offset points")
        
    ax2.set_ylabel("MAE (deg C) - Lower is Better", fontweight="bold")
    ax2.set_title("B. Sensorless Suite (Inverter + Coolant/Ambient Only)", fontweight="bold", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontweight="bold")
    ax2.set_ylim(0, max(max(val_maes_inv), max(test_maes_inv)) * 1.25)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("30,000 Motor Measurements: Model Comparison on Same Validation and Test Sets", fontsize=13, fontweight="bold")
    
    filename = "figure4_30k_split_benchmark.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Publication figure saved to '{filename}'.")


if __name__ == "__main__":
    run_benchmark()
