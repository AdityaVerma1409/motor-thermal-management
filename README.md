# Electric Motor Thermal Prediction & Cooling Advisory System (AI/ML Project)

An applied machine learning system for thermal management of Permanent Magnet Synchronous Motors (PMSM), based on the Kaggle **Electric Motor Temperature** dataset.

---

## 1. Project Overview & Research Questions
The system answers five fundamental operational questions ($T_1 \to T_5$) about a running electric motor:
* **$T_1$ (Temperature Forecast)**: What will the motor temperature be over the next 30 minutes? (Recursive rollout of 5-second changes $\Delta T$).
* **$T_2$ (Overheating Countdown)**: When will the motor breach a designated safety limit (e.g., 80 °C or 90 °C)?
* **$T_3$ (Cooling Start Advisory)**: At what minute should cooling be activated to keep the peak safe?
* **$T_4$ (Cooling Duration)**: How long must active cooling run?
* **$T_5$ (Post-Cooling Profile)**: What is the final peak, shut-off, and settled temperature after cooling?

---

## 2. Dataset Strategy: Handling 1.33 Million Rows
The dataset (`measures_v2.csv`) contains **1,330,816 rows across 69 profiles** sampled at 2 Hz (every 0.5 seconds).

* **The Physical Insight**: Motor thermal dynamics are slow (time constants are on the order of minutes). Adjacent 0.5-second samples are almost identical.
* **Thinning**: By subsampling every 10th row within each profile session, we downsample from **2 Hz (0.5 s) to 0.2 Hz (5.0 s)**.
* **Result**: The dataset is reduced from **1.33M rows to ~133,000 rows** without losing any thermal physical dynamics. Training finishes in seconds.

---

## 3. Model Comparison Benchmark

Models are trained on 55 profiles and evaluated on **14 completely unseen held-out profiles** using `GroupKFold(n_splits=5)`:

| Model | MAE (°C) | RMSE (°C) | $R^2$ Score | Train Time | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ridge Regression (Baseline)** | 0.1359 | 0.2747 | 0.0923 | 0.03 s | Fast linear baseline |
| **K-Nearest Neighbors (KNN, $k=15$)** | 0.1449 | 0.2821 | 0.0431 | 0.03 s | Standard distance-based regression |
| **HistGradientBoosting (Modern Tree)** | **0.1285** | **0.2725** | **0.1067** | 1.77 s | **Best overall accuracy**; supports monotonic physical constraints |

---

## 4. The "KNN Coolant Paradox" (Key Project Insight)
* **The Problem**: In passive bench testing, coolant fluid naturally warms up as the motor operates at high load. As a result, pure KNN learns a spurious correlation where low coolant is associated with idle states. When simulating a cooling intervention (manually lowering coolant), KNN can predict that the motor gets *hotter*.
* **The Solution**: We enforce a **Monotonic Constraint** on coolant in `HistGradientBoostingRegressor` (`monotonic_cst`), guaranteeing that colder coolant strictly reduces or holds steady the heating rate.

---

## 5. How to Run the Project

1. **Train and compare models**:
   ```bash
   python 1_model_comparison.py
   ```
   *Generates `model_comparison_results.csv`.*

2. **Run 30-minute advisory and cooling simulator ($T_1 \to T_5$)**:
   ```bash
   python 2_thermal_advisor.py
   ```

3. **Generate graphs for your presentation/report**:
   ```bash
   python 3_plot_dashboard.py
   ```
   *Generates:*
   - `figure1_model_benchmark.png` (Accuracy comparison bar chart)
   - `figure2_advisory_forecast.png` (Uncooled vs Cooled trajectory with safety limit and cooling window)
   - `figure3_coolant_paradox.png` (KNN paradox vs Monotonic HistGBM fix)

---

## 6. Academic References
1. **Kirchgässner, W., Wallscheid, O., & Böcker, J.** *"Estimating Electric Motor Temperatures With Deep Residual Machine Learning."* *IEEE Transactions on Power Electronics*. (Original Paderborn University dataset).
2. *"Temperature Prediction and Performance Comparison of Permanent Magnet Synchronous Motors Using Different Machine Learning Techniques for Early Failure Detection."* *Maintenance and Reliability (Eksploatacja i Niezawodność)*.
3. *"Accurate Rotor Temperature Prediction of Permanent Magnet Synchronous Motor in Electric Vehicles Using a Hybrid Model."* *MDPI Applied Sciences*.
