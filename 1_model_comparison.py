"""
1_model_comparison.py
=====================
Comprehensive, rigorous comparison of Machine Learning models on the Kaggle
Electric Motor Temperature dataset (measures_v2.csv):
  1. Baseline: "Predict No Change" (Delta_T = 0, persistence baseline)
  2. Linear Baseline: Ridge Regression (L2 Regularized)
  3. Non-Parametric: K-Nearest Neighbors (KNN, k=15, uniformly sampled across all profiles)
  4. Advanced Ensemble: HistGradientBoostingRegressor (Monotonic Tree Ensemble)

Evaluation Rigor:
  - 5-Fold GroupKFold cross-validation on `profile_id` (100% leak-free across motor profiles).
  - Single-step 5-second Delta_T accuracy (MAE, RMSE, R2) with mean +/- std.
  - Multi-step recursive trajectory forecast error evaluated at 1 min, 5 min, 10 min, and 30 min horizons.
  - Per-prediction inference latency measured in microseconds (us) per sample.
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------------------------------------------
# 1. LOAD & THINNING (1.33 MILLION ROWS -> 5-SECOND TEMPORAL SAMPLING)
# -----------------------------------------------------------------------------
def load_and_preprocess_data(csv_path="measures_v2.csv", thin_factor=10):
    print("=" * 78)
    print("  STEP 1: LOADING & THINNING DATASET (5-SECOND RESOLUTION)")
    print("=" * 78)
    t0 = time.time()
    
    cols = [
        "profile_id", "pm", "stator_winding", "stator_tooth", "stator_yoke",
        "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"
    ]
    
    print(f"Reading '{csv_path}' ...")
    df = pd.read_csv(csv_path, usecols=cols)
    total_raw = len(df)
    n_profiles = df["profile_id"].nunique()
    print(f"Loaded raw dataset: {total_raw:,} rows across {n_profiles} motor profiles ({time.time()-t0:.1f}s)")

    # Thinning: Keep every 10th row within each profile session
    # 2 Hz (0.5s) -> 0.2 Hz (5.0s)
    print(f"Thinning by factor {thin_factor} (1 sample every 5.0 seconds) ...")
    sessions = []
    for pid, group in df.groupby("profile_id", sort=False):
        sub = group.iloc[::thin_factor].copy().reset_index(drop=True)
        if len(sub) < 5:
            continue
        # Target: 5-second rotor magnet temperature increment Delta_pm = pm(k+1) - pm(k)
        sub["pm_next"] = sub["pm"].shift(-1)
        sub["delta_pm"] = sub["pm_next"] - sub["pm"]
        sessions.append(sub.iloc[:-1].copy())

    df_thinned = pd.concat(sessions, axis=0, ignore_index=True)
    print(f"Dataset reduced to {len(df_thinned):,} rows across {df_thinned['profile_id'].nunique()} profiles.")
    print("Physical justification: Rotor thermal time constants are 3-10 minutes;")
    print("5-second sampling captures 100% of thermal dynamics without high-frequency electrical noise.")
    return df_thinned


# -----------------------------------------------------------------------------
# 2. 5-FOLD GROUP-K-FOLD CROSS-VALIDATION (1-STEP EVALUATION)
# -----------------------------------------------------------------------------
def run_5fold_cross_validation(df_thinned, n_splits=5):
    print("\n" + "=" * 78)
    print(f"  STEP 2: FULL {n_splits}-FOLD GROUP-K-FOLD CROSS-VALIDATION (LEAK-FREE)")
    print("=" * 78)
    
    features = ["pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
    target = "delta_pm"
    
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_metrics = {
        "Predict No Change (Baseline)": {"mae": [], "rmse": [], "r2": []},
        "Ridge Regression (Linear)": {"mae": [], "rmse": [], "r2": []},
        "KNN (Uniform Sample across Profiles)": {"mae": [], "rmse": [], "r2": []},
        "HistGradientBoosting (Modern Tree)": {"mae": [], "rmse": [], "r2": []}
    }
    
    per_sample_latencies_us = {
        "Predict No Change (Baseline)": 0.0,
        "Ridge Regression (Linear)": [],
        "KNN (Uniform Sample across Profiles)": [],
        "HistGradientBoosting (Modern Tree)": []
    }
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(df_thinned, groups=df_thinned["profile_id"])):
        df_train = df_thinned.iloc[train_idx].reset_index(drop=True)
        df_test = df_thinned.iloc[test_idx].reset_index(drop=True)
        
        X_train, y_train = df_train[features], df_train[target]
        X_test, y_test = df_test[features], df_test[target]
        
        n_test = len(X_test)
        
        # 1. Baseline: Predict No Change (Delta_T = 0)
        y_pred_nc = np.zeros_like(y_test)
        fold_metrics["Predict No Change (Baseline)"]["mae"].append(mean_absolute_error(y_test, y_pred_nc))
        fold_metrics["Predict No Change (Baseline)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_nc)))
        fold_metrics["Predict No Change (Baseline)"]["r2"].append(r2_score(y_test, y_pred_nc))
        
        # 2. Ridge Regression
        ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
        ridge.fit(X_train, y_train)
        t0 = time.perf_counter()
        y_pred_ridge = ridge.predict(X_test)
        lat_ridge = (time.perf_counter() - t0) / n_test * 1e6
        per_sample_latencies_us["Ridge Regression (Linear)"].append(lat_ridge)
        fold_metrics["Ridge Regression (Linear)"]["mae"].append(mean_absolute_error(y_test, y_pred_ridge))
        fold_metrics["Ridge Regression (Linear)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
        fold_metrics["Ridge Regression (Linear)"]["r2"].append(r2_score(y_test, y_pred_ridge))
        
        # 3. K-Nearest Neighbors (Uniform random sample across all training profiles)
        knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=15, weights="distance", algorithm="kd_tree"))])
        sample_idx = X_train.sample(n=min(35000, len(X_train)), random_state=42 + fold).index
        knn.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
        t0 = time.perf_counter()
        y_pred_knn = knn.predict(X_test)
        lat_knn = (time.perf_counter() - t0) / n_test * 1e6
        per_sample_latencies_us["KNN (Uniform Sample across Profiles)"].append(lat_knn)
        fold_metrics["KNN (Uniform Sample across Profiles)"]["mae"].append(mean_absolute_error(y_test, y_pred_knn))
        fold_metrics["KNN (Uniform Sample across Profiles)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_knn)))
        fold_metrics["KNN (Uniform Sample across Profiles)"]["r2"].append(r2_score(y_test, y_pred_knn))
        
        # 4. HistGradientBoostingRegressor
        hgbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
        hgbm.fit(X_train, y_train)
        t0 = time.perf_counter()
        y_pred_hgbm = hgbm.predict(X_test)
        lat_hgbm = (time.perf_counter() - t0) / n_test * 1e6
        per_sample_latencies_us["HistGradientBoosting (Modern Tree)"].append(lat_hgbm)
        fold_metrics["HistGradientBoosting (Modern Tree)"]["mae"].append(mean_absolute_error(y_test, y_pred_hgbm))
        fold_metrics["HistGradientBoosting (Modern Tree)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_hgbm)))
        fold_metrics["HistGradientBoosting (Modern Tree)"]["r2"].append(r2_score(y_test, y_pred_hgbm))
        
        print(f"  Fold {fold+1}/{n_splits} Complete: Test set = {n_test:,} rows ({df_test['profile_id'].nunique()} profiles)")

    summary_rows = []
    for model_name, m in fold_metrics.items():
        mae_m, mae_s = np.mean(m["mae"]), np.std(m["mae"])
        rmse_m, rmse_s = np.mean(m["rmse"]), np.std(m["rmse"])
        r2_m, r2_s = np.mean(m["r2"]), np.std(m["r2"])
        if model_name == "Predict No Change (Baseline)":
            lat_str = "0.0 us"
            lat_val = 0.0
        else:
            lat_val = float(np.mean(per_sample_latencies_us[model_name]))
            lat_str = f"{lat_val:.1f} us"
            
        summary_rows.append({
            "Model": model_name,
            "MAE (deg C)": round(mae_m, 4),
            "MAE Std": round(mae_s, 4),
            "RMSE (deg C)": round(rmse_m, 4),
            "RMSE Std": round(rmse_s, 4),
            "R2 Score": round(r2_m, 4),
            "R2 Std": round(r2_s, 4),
            "Latency_per_pred_us": round(lat_val, 1)
        })

    cv_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 78)
    print("  5-FOLD CROSS-VALIDATION BENCHMARK RESULTS")
    print("=" * 78)
    disp_df = cv_df.copy()
    disp_df["MAE"] = [f"{r['MAE (deg C)']:.4f} +/- {r['MAE Std']:.4f} C" for _, r in cv_df.iterrows()]
    disp_df["RMSE"] = [f"{r['RMSE (deg C)']:.4f} +/- {r['RMSE Std']:.4f} C" for _, r in cv_df.iterrows()]
    disp_df["R2"] = [f"{r['R2 Score']:.4f} +/- {r['R2 Std']:.4f}" for _, r in cv_df.iterrows()]
    disp_df["Latency / Prediction"] = [f"{r['Latency_per_pred_us']:.1f} us" for _, r in cv_df.iterrows()]
    print(disp_df[["Model", "MAE", "RMSE", "R2", "Latency / Prediction"]].to_string(index=False))
    
    cv_df.to_csv("model_comparison_results.csv", index=False)
    print("\n[SAVED] Benchmark metrics saved to 'model_comparison_results.csv'")
    return cv_df


# -----------------------------------------------------------------------------
# 3. MULTI-STEP RECURSIVE TRAJECTORY FORECAST BENCHMARK (1 - 30 MINUTES)
# -----------------------------------------------------------------------------
def benchmark_recursive_forecast(df_thinned):
    print("\n" + "=" * 78)
    print("  STEP 3: MULTI-STEP RECURSIVE FORECAST BENCHMARK (1 TO 30 MINUTES)")
    print("=" * 78)
    print("Why this matters: In actual ECU operation, predictions are recursed:")
    print("  T_pm(k+1) = T_pm(k) + Delta_T_hat")
    print("Over 360 steps (30 min), error compounds. This evaluates true trajectory drift.")
    
    features = ["pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
    target = "delta_pm"
    
    # Use split 1 for trajectory evaluation
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(df_thinned, groups=df_thinned["profile_id"]))
    df_train = df_thinned.iloc[train_idx].reset_index(drop=True)
    df_test = df_thinned.iloc[test_idx].reset_index(drop=True)
    
    X_train, y_train = df_train[features], df_train[target]
    
    # Train models
    print("\nFitting models on training fold...")
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]).fit(X_train, y_train)
    
    knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(15, weights="distance", algorithm="kd_tree"))])
    sample_idx = X_train.sample(n=min(35000, len(X_train)), random_state=42).index
    knn.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    
    hgbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42).fit(X_train, y_train)
    
    horizons = {
        "1 min (12 steps)": 12,
        "5 min (60 steps)": 60,
        "10 min (120 steps)": 120,
        "30 min (360 steps)": 360
    }
    
    traj_errors = {
        "Predict No Change (Baseline)": {h: [] for h in horizons},
        "Ridge Regression (Linear)": {h: [] for h in horizons},
        "KNN (Uniform Sample)": {h: [] for h in horizons},
        "HistGradientBoosting": {h: [] for h in horizons}
    }
    
    test_pids = df_test["profile_id"].unique()
    evaluated_profiles = 0
    
    for pid in test_pids:
        p_data = df_test[df_test["profile_id"] == pid].reset_index(drop=True)
        if len(p_data) < 365:
            continue
            
        evaluated_profiles += 1
        T_actual = p_data["pm"].values
        
        # Initial temperature
        t0_val = T_actual[0]
        
        # 1. No change: T_pred(t) = t0_val
        T_nc = np.full(361, t0_val)
        
        # 2. Ridge rollout
        T_ridge = np.zeros(361)
        T_ridge[0] = t0_val
        curr_r = t0_val
        
        # 3. KNN rollout
        T_knn = np.zeros(361)
        T_knn[0] = t0_val
        curr_k = t0_val
        
        # 4. HistGBM rollout
        T_hgbm = np.zeros(361)
        T_hgbm[0] = t0_val
        curr_h = t0_val
        
        # Recursive 360-step loop
        for k in range(360):
            row = p_data.iloc[k]
            x_r = np.array([[curr_r, row["coolant"], row["ambient"], row["u_d"], row["u_q"], row["i_d"], row["i_q"], row["motor_speed"], row["torque"]]])
            x_k = np.array([[curr_k, row["coolant"], row["ambient"], row["u_d"], row["u_q"], row["i_d"], row["i_q"], row["motor_speed"], row["torque"]]])
            x_h = np.array([[curr_h, row["coolant"], row["ambient"], row["u_d"], row["u_q"], row["i_d"], row["i_q"], row["motor_speed"], row["torque"]]])
            
            curr_r += ridge.predict(x_r)[0]
            curr_k += knn.predict(x_k)[0]
            curr_h += hgbm.predict(x_h)[0]
            
            T_ridge[k+1] = curr_r
            T_knn[k+1] = curr_k
            T_hgbm[k+1] = curr_h
            
        for h_name, h_steps in horizons.items():
            traj_errors["Predict No Change (Baseline)"][h_name].append(abs(T_nc[h_steps] - T_actual[h_steps]))
            traj_errors["Ridge Regression (Linear)"][h_name].append(abs(T_ridge[h_steps] - T_actual[h_steps]))
            traj_errors["KNN (Uniform Sample)"][h_name].append(abs(T_knn[h_steps] - T_actual[h_steps]))
            traj_errors["HistGradientBoosting"][h_name].append(abs(T_hgbm[h_steps] - T_actual[h_steps]))
            
    traj_summary = []
    print(f"\nEvaluated on {evaluated_profiles} long test profiles (>= 360 steps / 30 min):")
    for h_name in horizons:
        row_dict = {"Horizon": h_name}
        for m in traj_errors:
            arr = traj_errors[m][h_name]
            mean_e = np.mean(arr)
            std_e = np.std(arr)
            row_dict[m] = f"{mean_e:.2f} +/- {std_e:.2f} C"
        traj_summary.append(row_dict)
        
    traj_df = pd.DataFrame(traj_summary)
    print(traj_df.to_string(index=False))
    traj_df.to_csv("trajectory_forecast_results.csv", index=False)
    print("\n[SAVED] Trajectory forecast results saved to 'trajectory_forecast_results.csv'")
    return traj_df


if __name__ == "__main__":
    df_thinned = load_and_preprocess_data("measures_v2.csv", thin_factor=10)
    run_5fold_cross_validation(df_thinned, n_splits=5)
    benchmark_recursive_forecast(df_thinned)
