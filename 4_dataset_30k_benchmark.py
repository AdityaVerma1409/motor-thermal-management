"""
4_dataset_30k_benchmark.py
==========================
Evaluates the newly provided 30,000-sample dataset (motor_temperature_30000.csv):
  1. Inspects dataset properties (30,000 rows, all 69 profiles represented).
  2. Runs 5-Fold Leak-Free Cross-Validation (GroupKFold on profile_id) for:
       - Case A: Full Sensor Suite (including stator temperatures)
       - Case B: Inverter-Only Sensorless Rotor Estimation (coolant, ambient, voltages, currents, speed, torque)
  3. Evaluates Ridge Regression, K-Nearest Neighbors (KNN, k=15), and HistGradientBoosting.
"""

import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_30k_dataset(csv_path="motor_temperature_30000.csv"):
    print("=" * 75)
    print("  EVALUATING NEW 30,000-ROW DATASET: " + csv_path)
    print("=" * 75)
    
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    n_profiles = df["profile_id"].nunique()
    print(f"Loaded: {n_rows:,} rows x {n_cols} columns")
    print(f"Profile coverage: {n_profiles} unique profiles (uniform cross-profile sample)\n")

    target = "pm"
    features_full = [
        "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque",
        "stator_winding", "stator_tooth", "stator_yoke"
    ]
    features_no_stator = [
        "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"
    ]

    experiments = [
        ("Case A: With Stator Sensors (Full Thermal Suite)", features_full),
        ("Case B: Inverter Inputs Only (Without Stator Sensors)", features_no_stator)
    ]

    all_summaries = []

    for exp_name, feat_cols in experiments:
        print("=" * 75)
        print(f"  {exp_name.upper()}")
        print("=" * 75)
        
        gkf = GroupKFold(n_splits=5)
        
        models = {
            "Ridge Regression (Baseline)": Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0))
            ]),
            "K-Nearest Neighbors (KNN, k=15)": Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance", algorithm="kd_tree"))
            ]),
            "HistGradientBoosting (Modern Tree)": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42
            )
        }

        metrics = {name: {"mae": [], "rmse": [], "r2": [], "latency_us": []} for name in models}

        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=df["profile_id"])):
            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_test = df.iloc[test_idx].reset_index(drop=True)
            
            X_train, y_train = df_train[feat_cols], df_train[target]
            X_test, y_test = df_test[feat_cols], df_test[target]
            n_test = len(X_test)
            
            for m_name, model in models.items():
                model.fit(X_train, y_train)
                t0 = time.perf_counter()
                y_pred = model.predict(X_test)
                lat_us = (time.perf_counter() - t0) / n_test * 1e6
                
                metrics[m_name]["mae"].append(mean_absolute_error(y_test, y_pred))
                metrics[m_name]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                metrics[m_name]["r2"].append(r2_score(y_test, y_pred))
                metrics[m_name]["latency_us"].append(lat_us)

        # Print summary for this case
        case_rows = []
        for m_name in models:
            maes = metrics[m_name]["mae"]
            rmses = metrics[m_name]["rmse"]
            r2s = metrics[m_name]["r2"]
            lats = metrics[m_name]["latency_us"]
            row = {
                "Experiment": exp_name,
                "Model": m_name,
                "MAE (deg C)": f"{np.mean(maes):.2f} +/- {np.std(maes):.2f}",
                "RMSE (deg C)": f"{np.mean(rmses):.2f} +/- {np.std(rmses):.2f}",
                "R2 Score": f"{np.mean(r2s):.4f} +/- {np.std(r2s):.4f}",
                "Latency (us/pred)": f"{np.mean(lats):.1f} us"
            }
            case_rows.append(row)
            all_summaries.append(row)
            
        case_df = pd.DataFrame(case_rows)[["Model", "MAE (deg C)", "RMSE (deg C)", "R2 Score", "Latency (us/pred)"]]
        print(case_df.to_string(index=False))
        print()

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv("results_30k_benchmark.csv", index=False)
    print("=" * 75)
    print("[SUCCESS] Benchmark results on 30k dataset saved to 'results_30k_benchmark.csv'!")
    print("=" * 75)
    return summary_df


if __name__ == "__main__":
    evaluate_30k_dataset("motor_temperature_30000.csv")
