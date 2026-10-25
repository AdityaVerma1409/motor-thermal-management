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

# 1. Load and thin
print("Loading measures_v2.csv...")
cols = [
    "profile_id", "pm", "stator_winding", "stator_tooth", "stator_yoke",
    "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"
]
df = pd.read_csv("measures_v2.csv", usecols=cols)
sessions = []
for pid, group in df.groupby("profile_id", sort=False):
    sub = group.iloc[::10].copy().reset_index(drop=True)
    if len(sub) < 5: continue
    sub["pm_next"] = sub["pm"].shift(-1)
    sub["delta_pm"] = sub["pm_next"] - sub["pm"]
    sessions.append(sub.iloc[:-1].copy())

df_thinned = pd.concat(sessions, axis=0, ignore_index=True)
print(f"Total thinned rows: {len(df_thinned):,} across {df_thinned['profile_id'].nunique()} profiles")

features = ["pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
target = "delta_pm"

# 2. 5-Fold GroupKFold Cross-Validation
gkf = GroupKFold(n_splits=5)
fold_results = {
    "No-Change (Baseline)": {"mae": [], "rmse": [], "r2": []},
    "Ridge Regression": {"mae": [], "rmse": [], "r2": []},
    "KNN (Uniform Sample 35k)": {"mae": [], "rmse": [], "r2": []},
    "HistGradientBoosting": {"mae": [], "rmse": [], "r2": []}
}

latencies = {"Ridge Regression": [], "KNN (Uniform Sample 35k)": [], "HistGradientBoosting": []}

for fold, (train_idx, test_idx) in enumerate(gkf.split(df_thinned, groups=df_thinned["profile_id"])):
    print(f"\n--- FOLD {fold+1}/5 ---")
    df_train = df_thinned.iloc[train_idx].reset_index(drop=True)
    df_test = df_thinned.iloc[test_idx].reset_index(drop=True)
    
    X_train, y_train = df_train[features], df_train[target]
    X_test, y_test = df_test[features], df_test[target]
    
    # Baseline: No-change
    y_pred_nc = np.zeros_like(y_test)
    fold_results["No-Change (Baseline)"]["mae"].append(mean_absolute_error(y_test, y_pred_nc))
    fold_results["No-Change (Baseline)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_nc)))
    fold_results["No-Change (Baseline)"]["r2"].append(r2_score(y_test, y_pred_nc))
    
    # Ridge
    m_ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    m_ridge.fit(X_train, y_train)
    t0 = time.time()
    y_pred_ridge = m_ridge.predict(X_test)
    t_ridge = (time.time() - t0) / len(X_test) * 1e6 # microsec
    latencies["Ridge Regression"].append(t_ridge)
    fold_results["Ridge Regression"]["mae"].append(mean_absolute_error(y_test, y_pred_ridge))
    fold_results["Ridge Regression"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_ridge)))
    fold_results["Ridge Regression"]["r2"].append(r2_score(y_test, y_pred_ridge))
    
    # KNN (Uniformly sampled across all profiles!)
    m_knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(15, weights="distance", algorithm="kd_tree"))])
    sample_idx = X_train.sample(n=min(35000, len(X_train)), random_state=42 + fold).index
    m_knn.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    t0 = time.time()
    y_pred_knn = m_knn.predict(X_test)
    t_knn = (time.time() - t0) / len(X_test) * 1e6 # microsec
    latencies["KNN (Uniform Sample 35k)"].append(t_knn)
    fold_results["KNN (Uniform Sample 35k)"]["mae"].append(mean_absolute_error(y_test, y_pred_knn))
    fold_results["KNN (Uniform Sample 35k)"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_knn)))
    fold_results["KNN (Uniform Sample 35k)"]["r2"].append(r2_score(y_test, y_pred_knn))
    
    # HistGBM
    m_hgbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42)
    m_hgbm.fit(X_train, y_train)
    t0 = time.time()
    y_pred_hgbm = m_hgbm.predict(X_test)
    t_hgbm = (time.time() - t0) / len(X_test) * 1e6 # microsec
    latencies["HistGradientBoosting"].append(t_hgbm)
    fold_results["HistGradientBoosting"]["mae"].append(mean_absolute_error(y_test, y_pred_hgbm))
    fold_results["HistGradientBoosting"]["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred_hgbm)))
    fold_results["HistGradientBoosting"]["r2"].append(r2_score(y_test, y_pred_hgbm))

print("\n" + "="*80)
print("5-FOLD CROSS-VALIDATION SUMMARY (MEAN +/- STD)")
print("="*80)
summary = []
for model_name in fold_results:
    mae_mean, mae_std = np.mean(fold_results[model_name]["mae"]), np.std(fold_results[model_name]["mae"])
    rmse_mean, rmse_std = np.mean(fold_results[model_name]["rmse"]), np.std(fold_results[model_name]["rmse"])
    r2_mean, r2_std = np.mean(fold_results[model_name]["r2"]), np.std(fold_results[model_name]["r2"])
    lat = f"{np.mean(latencies[model_name]):.1f} us" if model_name in latencies else "0.0 us"
    summary.append({
        "Model": model_name,
        "MAE (deg C)": f"{mae_mean:.4f} +/- {mae_std:.4f}",
        "RMSE (deg C)": f"{rmse_mean:.4f} +/- {rmse_std:.4f}",
        "R2 Score": f"{r2_mean:.4f} +/- {r2_std:.4f}",
        "Latency / Pred": lat
    })

sum_df = pd.DataFrame(summary)
print(sum_df.to_string(index=False))
sum_df.to_csv("scratch/cv_5fold_summary.csv", index=False)
