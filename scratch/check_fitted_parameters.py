import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# Load data
cols = ["profile_id", "pm", "stator_winding", "stator_tooth", "stator_yoke", "coolant", "ambient", "u_d", "u_q", "i_d", "i_q", "motor_speed", "torque"]
df = pd.read_csv("measures_v2.csv", usecols=cols)

sessions = []
for pid, group in df.groupby("profile_id", sort=False):
    sub = group.iloc[::10].copy().reset_index(drop=True)
    if len(sub) < 5: continue
    sub["pm_next"] = sub["pm"].shift(-1)
    sub["stator_temp"] = sub["stator_tooth"] # tooth is typical proxy
    sub["stator_next"] = sub["stator_temp"].shift(-1)
    sessions.append(sub.iloc[:-1].copy())

df_train = pd.concat(sessions, axis=0, ignore_index=True)

# 1. Check v2 parameter fitting
from motor_thermal_advisory_v2 import TwoNodeHybridPMSMModel, TwoNodeConfig
model_v2 = TwoNodeHybridPMSMModel(TwoNodeConfig())

print("Fitting v2 thermal parameters on measures_v2.csv...")
# We can run the internal least squares and inspect opt.x vs bounds
t_pm = df_train["pm"].to_numpy(dtype=np.float64)
t_s = df_train["stator_temp"].to_numpy(dtype=np.float64)
t_pm_next = df_train["pm_next"].to_numpy(dtype=np.float64)
t_s_next = df_train["stator_next"].to_numpy(dtype=np.float64)
t_cool = df_train["coolant"].to_numpy(dtype=np.float64)
t_amb = df_train["ambient"].to_numpy(dtype=np.float64)
spd = df_train["motor_speed"].to_numpy(dtype=np.float64)
i_sq = (df_train["i_d"] ** 2 + df_train["i_q"] ** 2).to_numpy(dtype=np.float64)
u_sq = (df_train["u_d"] ** 2 + df_train["u_q"] ** 2).to_numpy(dtype=np.float64)
w_norm_sq = (np.abs(spd) / 6000.0) ** 2

is_cooldown = (i_sq < np.percentile(i_sq, 25)) & ((t_s - t_cool) > 4.0)
w_row = np.sqrt(np.where(is_cooldown, 5.0, 1.0))

x0 = np.array([5.5e-4, 8.5e-4, 0.65, 0.35, 2.2e-3, 1.0e-4, 1.5e-4, 5.0e-7, 6.0e-4, 1.8e-6, 3.0e-7])
lb = np.array([1.0e-4, 1.0e-4, 0.40, 0.15, 4.0e-4, 1.0e-5, 1.0e-5, 0.0, 0.0, 0.0, 0.0])
ub = np.array([3.0e-3, 5.0e-3, 0.90, 0.70, 1.0e-2, 8.0e-4, 1.0e-3, 1.0e-3, 5.0e-2, 1.0e-3, 1.0e-3])

param_names = [
    "alpha_ag0", "alpha_ag1", "gamma", "mu_cap_ratio", "alpha_cool", "beta_pm", "beta_s",
    "c_pm_i", "c_pm_w", "c_s_cu", "c_s_fe"
]

def residual_vec(theta: np.ndarray) -> np.ndarray:
    phys_p = tuple(theta[:7])
    c_pm_i, c_pm_w, c_s_cu, c_s_fe = theta[7:]
    phi11, phi12, phi21, phi22, gam11, gam12, gam21, gam22 = model_v2.compute_discrete_matrices_batch(spd, params=phys_p)
    b_pm, a_cool, b_s = theta[5], theta[4], theta[6]
    q_pm_proxy = c_pm_i * i_sq + c_pm_w * w_norm_sq
    cu_temp_scale = 1.0 + 0.00393 * (t_s - 20.0)
    q_s_proxy = c_s_cu * i_sq * cu_temp_scale + c_s_fe * u_sq
    in_pm = b_pm * t_amb + q_pm_proxy
    in_s = a_cool * t_cool + b_s * t_amb + q_s_proxy
    pred_pm_next = phi11 * t_pm + phi12 * t_s + gam11 * in_pm + gam12 * in_s
    pred_s_next = phi21 * t_pm + phi22 * t_s + gam21 * in_pm + gam22 * in_s
    res_pm = w_row * (pred_pm_next - t_pm_next)
    res_s = w_row * (pred_s_next - t_s_next)
    return np.concatenate([res_pm, res_s])

opt = least_squares(residual_vec, x0=x0, bounds=(lb, ub), method="trf", loss="soft_l1", max_nfev=150)

print("\n--- v2 FITTED PARAMETER BOUND CHECK ---")
for i, name in enumerate(param_names):
    val = opt.x[i]
    low = lb[i]
    high = ub[i]
    at_low = np.isclose(val, low, rtol=1e-3, atol=1e-6)
    at_high = np.isclose(val, high, rtol=1e-3, atol=1e-6)
    status = "AT LOWER BOUND!" if at_low else ("AT UPPER BOUND!" if at_high else "OK (Interior)")
    print(f"  {name:15s}: val={val:11.4e} | [min={low:11.4e}, max={high:11.4e}] | {status}")
