import time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

# Load thinned data
cols = ["profile_id", "pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
df = pd.read_csv("measures_v2.csv", usecols=cols)

sessions = []
for pid, group in df.groupby("profile_id", sort=False):
    sub = group.iloc[::10].copy().reset_index(drop=True)
    if len(sub) < 370: continue
    sub["pm_next"] = sub["pm"].shift(-1)
    sub["delta_pm"] = sub["pm_next"] - sub["pm"]
    sessions.append(sub.iloc[:-1].copy())

df_thinned = pd.concat(sessions, axis=0, ignore_index=True)
features = ["pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
target = "delta_pm"

gkf = GroupKFold(n_splits=5)
train_idx, test_idx = next(gkf.split(df_thinned, groups=df_thinned["profile_id"]))
df_train, df_test = df_thinned.iloc[train_idx], df_thinned.iloc[test_idx]

# Train models
print("Training models...")
ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(1.0))]).fit(df_train[features], df_train[target])

knn = Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(15, weights="distance", algorithm="kd_tree"))])
s_idx = df_train.sample(n=35000, random_state=42).index
knn.fit(df_train.loc[s_idx, features], df_train.loc[s_idx, target])

hgbm = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42).fit(df_train[features], df_train[target])

# Now perform multi-step recursive trajectory rollouts on held-out test profiles
# Evaluate error at 1 min (12 steps), 5 min (60 steps), 10 min (120 steps), 30 min (360 steps)
horizons = {"1 min (12 steps)": 12, "5 min (60 steps)": 60, "10 min (120 steps)": 120, "30 min (360 steps)": 360}

# Pick test profiles
test_pids = df_test["profile_id"].unique()
print(f"Evaluating recursive trajectory on {len(test_pids)} test profiles...")

traj_errors = {
    "No-Change": {h: [] for h in horizons},
    "Ridge": {h: [] for h in horizons},
    "KNN": {h: [] for h in horizons},
    "HistGBM": {h: [] for h in horizons}
}

for pid in test_pids:
    p_data = df_test[df_test["profile_id"] == pid].reset_index(drop=True)
    if len(p_data) < 365: continue
    
    # Evaluate starting from step 0 (or multiple windows)
    # Let's test a 360-step rollout
    T_actual = p_data["pm"].values
    
    # 1. No Change
    T_nc = np.full(361, T_actual[0])
    
    # 2. Ridge rollout
    T_ridge = np.zeros(361)
    T_ridge[0] = T_actual[0]
    
    # 3. KNN rollout
    T_knn = np.zeros(361)
    T_knn[0] = T_actual[0]
    
    # 4. HistGBM rollout
    T_hgbm = np.zeros(361)
    T_hgbm[0] = T_actual[0]
    
    curr_r = T_actual[0]
    curr_k = T_actual[0]
    curr_h = T_actual[0]
    
    for k in range(360):
        row = p_data.iloc[k]
        # features: ["pm", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
        # In recursive rollout, pm is the model's own predicted temperature, while operating inputs are the dynamic operating profile
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
        err_nc = abs(T_nc[h_steps] - T_actual[h_steps])
        err_r = abs(T_ridge[h_steps] - T_actual[h_steps])
        err_k = abs(T_knn[h_steps] - T_actual[h_steps])
        err_h = abs(T_hgbm[h_steps] - T_actual[h_steps])
        
        traj_errors["No-Change"][h_name].append(err_nc)
        traj_errors["Ridge"][h_name].append(err_r)
        traj_errors["KNN"][h_name].append(err_k)
        traj_errors["HistGBM"][h_name].append(err_h)

print("\n--- RECURSIVE TRAJECTORY FORECAST MAE (deg C) ---")
for h_name in horizons:
    print(f"\nHorizon: {h_name}")
    for m in ["No-Change", "Ridge", "KNN", "HistGBM"]:
        arr = traj_errors[m][h_name]
        print(f"  {m:10s} : MAE = {np.mean(arr):.2f} +/- {np.std(arr):.2f} deg C")
