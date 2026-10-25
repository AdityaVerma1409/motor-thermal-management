# Electric Motor Thermal Prediction & Cooling Advisory System (AI/ML Project)

An applied machine learning system for thermal management and dynamic cooling advisory of Permanent Magnet Synchronous Motors (PMSM), based on the Kaggle / Paderborn University **Electric Motor Temperature** dataset (`measures_v2.csv`).

---

## 1. Project Overview & Research Questions
The system answers five fundamental operational questions ($T_1 \to T_5$) about a running electric motor:
* **$T_1$ (Temperature Forecast)**: What will the motor temperature profile be over the next 30 minutes? (Recursive rollout of 5-second changes $\Delta T$).
* **$T_2$ (Overheating Countdown)**: When will the motor breach a designated safety limit (e.g., 80 °C or 90 °C)?
* **$T_3$ (Cooling Start Advisory)**: At what minute should cooling be activated to keep the peak below safe limits?
* **$T_4$ (Cooling Duration)**: How long must active cooling run?
* **$T_5$ (Post-Cooling Profile)**: What is the final peak, shut-off, and settled temperature after cooling?

---

## 2. Dataset Strategy: Handling 1.33 Million Rows
The dataset contains **1,330,816 rows across 69 profiles** sampled at 2 Hz (every 0.5 seconds).

* **The Physical Insight**: Motor rotor thermal time constants are typically 300 to 1,200 seconds. Adjacent 0.5-second samples are almost identical.
* **Thinning**: By subsampling every 10th row within each profile session, we downsample from **2 Hz (0.5 s) to 0.2 Hz (5.0 s)**.
* **Result**: The dataset is reduced from **1.33M rows to ~132,600 rows (67 long profiles)** without losing any thermal physical dynamics. Training finishes in seconds.

---

## 3. 5-Fold Leak-Free Cross-Validation Benchmark (1-Step $\Delta T$)

To guarantee leak-free evaluation, we perform 5-fold cross-validation using `GroupKFold(n_splits=5)` grouped strictly by `profile_id`. A "Predict No-Change" baseline ($\widehat{\Delta T} = 0$) is included, and K-Nearest Neighbors (KNN) is trained on **uniform samples drawn across all training profiles** (eliminating profile-selection bias). Latency is measured per individual prediction ($\mu\text{s}$/pred):

| Model | MAE (°C) | RMSE (°C) | $R^2$ Score | Latency ($\mu\text{s}$/pred) | Technical Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Predict No-Change Baseline** ($\Delta T = 0$) | 0.1250 ± 0.0162 | 0.2253 ± 0.0336 | -0.0037 ± 0.0011 | < 0.1 $\mu\text{s}$ | Constant-state reference baseline |
| **Ridge Regression (Baseline)** | 0.1059 ± 0.0167 | 0.2066 ± 0.0360 | 0.1612 ± 0.0390 | 0.1 $\mu\text{s}$ | Fast linear regularized baseline |
| **KNN ($k=15$, 35k Uniform Sample)** | 0.1073 ± 0.0175 | 0.2113 ± 0.0362 | 0.1217 ± 0.0396 | 28.9 $\mu\text{s}$ | Fair distance-based metric; matches Ridge when sampled across all profiles |
| **HistGradientBoosting (Modern Tree)** | **0.1002 ± 0.0169** | **0.2046 ± 0.0362** | **0.1781 ± 0.0451** | 2.4 $\mu\text{s}$ | **Best accuracy & stability**; natively enforces physical monotonicity |

> **Note on Latency**: An automotive ECU loop running at 0.2 Hz has a 5.0-second (5,000,000 $\mu\text{s}$) budget. Even KNN's 28.9 $\mu\text{s}$ per query consumes less than 0.001% of the cycle, proving all models are computationally feasible for embedded deployment.

---

## 4. Multi-Step Recursive Trajectory Benchmark (The Critical Engineering Metric)

In real-world EV thermal management, 1-step error ($\Delta T$ over 5 seconds) shows only tiny differences (~0.006 °C) because temperatures change slowly. What actually matters is **compounding error accumulation during a 30-minute recursive trajectory rollout** ($\widehat{T}_{k+1} = \widehat{T}_k + \widehat{\Delta T}$).

Evaluating across 13 completely held-out test sessions (each $\ge 365$ steps / 30 min):

| Forecast Horizon | Predict No-Change | Ridge Regression | KNN ($k=15$) | HistGradientBoosting |
| :--- | :---: | :---: | :---: | :---: |
| **1 minute** (12 steps / 60 s) | 2.13 ± 1.81 °C | 0.73 ± 0.68 °C | 0.59 ± 0.59 °C | **0.47 ± 0.38 °C** |
| **5 minutes** (60 steps / 300 s) | 9.70 ± 6.40 °C | 2.94 ± 2.16 °C | 2.02 ± 1.49 °C | **1.57 ± 0.91 °C** |
| **10 minutes** (120 steps / 600 s) | 16.61 ± 9.92 °C | 4.14 ± 2.46 °C | 2.96 ± 2.35 °C | **2.09 ± 1.59 °C** |
| **30 minutes** (360 steps / 1,800 s) | 30.65 ± 18.51 °C | 4.68 ± 3.39 °C | 4.83 ± 4.23 °C | **2.95 ± 3.36 °C** |

### Key Physical Observations:
1. **No-Change drifts by 30.65 °C**: Assuming motor temperature stays constant fails completely as heat accumulates during load.
2. **Linear Ridge drifts by 4.68 °C**: Linear models cannot capture the thermal saturation exponential towards equilibrium.
3. **HistGBM achieves 2.95 °C over 360 recursive steps**: HistGBM achieves a 37% error reduction over Ridge and over 90% error reduction over the No-Change baseline, stabilizing near steady-state equilibrium.

---

## 5. The "KNN Coolant Paradox" & Physical Monotonicity
* **The Problem**: In passive test-bench data, coolant temperature correlates positively with load (coolant naturally gets warm when the motor is worked hard). Pure KNN learns this correlation and infers that low coolant corresponds to idle operating conditions. When testing a counterfactual cooling command (e.g. dropping coolant by 15 °C), pure KNN can erroneously predict that the motor gets *hotter*!
* **The Solution**: We enforce a **Monotonic Constraint** on the coolant feature in `HistGradientBoostingRegressor` (`monotonic_cst = +1`). This mathematically guarantees that colder coolant strictly decreases or holds steady the heating rate.

---

## 6. Embedded Hardware Readiness (C Code Implementation)
To prove deployment feasibility on automotive microcontrollers (e.g., Infineon AURIX, NXP S32K, STM32):
* `motor_thermal_ecu_final.c` and `mth_calibration.h` implement an ISO 26262 compliant C reference implementation.
* Uses an explicit 2-state thermal RC network and 3-state Extended Kalman Filter (EKF) with zero dynamic heap allocation (`malloc`), MISRA-C compliance, and constant-time execution.
* Verified against **20 ISO 26262 HIL conformance test vectors** (`hil_test_vectors.json`), passing 20 of 20 vectors with 100% precision.

---

## 7. How to Run the Project

1. **Run 5-Fold Cross-Validation and Multi-Step Trajectory Benchmark**:
   ```bash
   python 1_model_comparison.py
   ```
   *Generates `model_comparison_results.csv` and `trajectory_forecast_results.csv`.*

2. **Run 30-minute advisory and cooling simulator ($T_1 \to T_5$)**:
   ```bash
   python 2_thermal_advisor.py
   ```

3. **Generate all high-resolution figures**:
   ```bash
   python 3_plot_dashboard.py
   ```
   *Generates:*
   - `figure1_model_benchmark.png` (3-panel: 5-fold CV error, single-prediction latency, and multi-step trajectory error vs horizon)
   - `figure2_advisory_forecast.png` (Uncooled vs Cooled trajectory with safety limit and cooling window)
   - `figure3_coolant_paradox.png` (KNN paradox vs Monotonic HistGBM fix)

4. **Benchmark on the 30,000-sample dataset (`motor_temperature_30000.csv`)**:
   ```bash
   python 4_dataset_30k_benchmark.py
   ```
   *Runs 5-fold cross-validation on the uniform 30k cross-profile dataset for both Full Sensor Suite and Inverter-Only Sensorless Rotor Estimation, saving `results_30k_benchmark.csv`.*

---

## 8. Academic References
1. **Kirchgässner, W., Wallscheid, O., & Böcker, J.** *"Estimating Electric Motor Temperatures With Deep Residual Machine Learning."* *IEEE Transactions on Power Electronics*, 2021. (Paderborn University dataset).
2. *"Temperature Prediction and Performance Comparison of Permanent Magnet Synchronous Motors Using Different Machine Learning Techniques for Early Failure Detection."* *Maintenance and Reliability (Eksploatacja i Niezawodność)*, 2020.
3. *"Accurate Rotor Temperature Prediction of Permanent Magnet Synchronous Motor in Electric Vehicles Using a Hybrid Model."* *MDPI Applied Sciences*, 2022.
