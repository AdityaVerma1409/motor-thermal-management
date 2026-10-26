# PMSM Motor Thermal Prediction & Dynamic Cooling Advisory
## Complete Presentation Slide Deck (Canva / PowerPoint Ready)

Use this document to populate your Canva presentation slides:
👉 **Canva Deck Link**: `https://www.canva.com/design/DAHYhrbnf_4/_sYjYpRdl3EhrVn3bwa5yQ/edit`

---

## Slide 1: Title & Project Overview

### Visual Layout
* **Header / Title**: Large bold typography.
* **Hero Image / Graphic**: 3D Electric Vehicle Powertrain or Motor Stator/Rotor cutaway.
* **Sub-badges**: `AI/ML Dynamic Advisory` • `30-Minute Predictive Horizon` • `ISO 26262 Embedded C Ready`

### Slide Text
* **Title**: Predictive Thermal Management & Dynamic Cooling Advisory for Electric Vehicle PMSMs
* **Subtitle**: Machine Learning Virtual Sensing, 30-Minute Recursive Horizon Forecasting, and Embedded ECU Implementation
* **Team Members**: [Your Names]
* **Target Questions Answered**: What is the temperature now ($T_1$), when will it overheat ($T_2$), when should cooling start ($T_3$), how long should it run ($T_4$), and what is the post-cooling profile ($T_5$)?

### Speaker Notes
> "Welcome everyone. Today we present an applied machine learning and physical thermal management system for Permanent Magnet Synchronous Motors. Electric motor overheating is a major bottleneck in EV performance, battery drain, and warranty life. Today, we show how to forecast thermal behavior 30 minutes in advance, trigger optimal cooling before critical limits are reached, and verify everything on both machine learning benchmarks and ISO 26262 embedded C code."

---

## Slide 2: The Engineering Problem & Automotive Motivation

### Visual Layout
* **Left Column**: Diagram of a PMSM motor showing spinning rotor shaft and stationary stator jacket.
* **Right Column**: 3 Key Engineering Challenges with warning icons.

### Slide Text
* **1. The Rotor Demagnetization Threat**:
  * NdFeB permanent magnets lose magnetism irreversibly if exposed to temperatures above **100 °C – 150 °C**.
* **2. The Inaccessibility Bottleneck**:
  * Magnets are mounted on the spinning rotor shaft (up to 6,000–12,000 RPM).
  * Direct physical thermocouples cannot be wired to rotating shafts in production passenger vehicles.
* **3. Reactive vs. Predictive Cooling**:
  * Traditional thermal switches only trigger *after* the motor is already hot, forcing aggressive power de-rating (limp mode).
  * Our predictive system forecasts thermal trajectory **30 minutes ahead**, allowing energy-efficient preemptive cooling.

### Speaker Notes
> "In electric vehicles, you cannot stick a wire onto a rotor spinning at thousands of RPM. Carmakers must estimate rotor temperature virtually using inverter currents, voltages, speed, and coolant sensors. Furthermore, reactive cooling is inefficient—if you wait until the motor is already 85 °C, you have to throttle motor power. Our goal is predictive advisory."

---

## Slide 3: Dataset Strategy & Physics-Informed 10x Thinning

### Visual Layout
* **Graphic**: Flowchart showing:
  `Raw Dataset (1.33M rows @ 2 Hz)` $\longrightarrow$ `10x Subsampling` $\longrightarrow$ `Clean Time-Series (132k rows @ 0.2 Hz)`.
* **Side Table**: Comparison of raw vs. thinned properties.

### Slide Text
* **Dataset Source**: Kaggle / Paderborn University benchmark dataset (1,330,816 rows across 69 profiles).
* **The Physical Insight**:
  * Motor thermal time constants are **300 to 1,200 seconds** (minutes to heat up or cool down).
  * Adjacent 0.5-second samples (2 Hz) are 99.9% redundant and cause memory bottlenecks.
* **Physics-Informed Thinning**:
  * Subsampled every 10th row per session to **0.2 Hz (5.0-second intervals)**.
  * Shrank dataset from 1.33M rows to **132,611 rows** while preserving 100% of physical heating curves.
* **Leak-Free Cross-Validation**:
  * Partitioned strictly by `profile_id` via 5-Fold `GroupKFold`. Zero data leakage between test driving profiles and training.

### Speaker Notes
> "A key data science challenge was handling 1.3 million rows. Motor temperatures take minutes to change. Taking measurements every 0.5 seconds creates massive redundancy. By subsampling to 5-second intervals, we preserved 100% of the heating physics while reducing computational time by 90%."

---

## Slide 4: Target T1 — Candidate Model Benchmark (The 1-Step Illusion)

### Visual Layout
* **Top Half**: 5-Fold Cross-Validation accuracy table.
* **Bottom Half**: Bar chart comparing MAE, RMSE, and Per-Prediction Latency.

### Slide Text
* **T1 Definition**: Permanent Magnet Rotor Temperature prediction from operational features ($u_d, u_q, i_d, i_q, \omega, \tau, T_{\mathrm{cool}}, T_{\mathrm{amb}}$).
* **The 1-Step (5s) Error Illusion**:
  * Over a short 5-second step, all models look very close:
    * **Ridge Regression**: MAE = **0.1057 °C**, $R^2 = 0.1625$
    * **KNN ($k=15$, Uniform 35k)**: MAE = **0.1064 °C**, $R^2 = 0.1256$
    * **HistGradientBoosting**: MAE = **0.0989 °C**, $R^2 = 0.1843$
* **Fair Sampling Note**: When KNN is sampled uniformly across all 55 profiles, it matches Ridge accuracy (previous claims of KNN losing heavily were due to unfair sampling on only profile 4).
* **Single-Prediction Latency on ECU**:
  * Ridge: **0.1 µs** | HistGBM: **2.6 µs** | KNN: **30.6 µs**.
  * All models consume **< 0.001%** of the 5.0-second ECU cycle budget.

### Speaker Notes
> "In single-step predictions, the gap between Ridge and HistGBM is only 0.007 °C. An evaluator might ask: 'Why not just use Ridge?' The answer lies on the next slide: compounding error during 30-minute recursive forecasting."

---

## Slide 5: The Real Differentiator — 30-Minute Trajectory Rollout

### Visual Layout
* **Chart**: Multi-step recursive trajectory forecast curve from `figure1_model_benchmark.png` (Panel C).
* **Table**: Horizon vs. Model Error.

### Slide Text
* **The Compounding Error Test**:
  $$\widehat{T}_{k+1} = \widehat{T}_k + \widehat{\Delta T}(\widehat{T}_k, u_k) \quad \text{for } k = 1 \dots 360 \text{ steps (30 min)}$$
* **Empirical Trajectory MAE on Held-Out Test Sessions**:
  * **1 Minute**: No-Change: 2.32 °C | Ridge: 0.85 °C | **HistGBM: 0.54 °C**
  * **5 Minutes**: No-Change: 9.85 °C | Ridge: 3.05 °C | **HistGBM: 1.51 °C**
  * **10 Minutes**: No-Change: 16.59 °C | Ridge: 4.05 °C | **HistGBM: 1.90 °C**
  * **30 Minutes**:
    * **Predict No-Change Baseline**: Drifts by **30.32 °C** (total failure).
    * **Ridge Regression**: Drifts linearly by **4.54 °C**.
    * **KNN ($k=15$)**: Accumulates **3.93 °C** error.
    * **HistGradientBoosting**: **Stabilizes at 2.92 °C**.
* **Physical Reason**: HistGBM captures the non-linear exponential approach toward steady-state thermal equilibrium ($T \to T_\infty$), whereas linear models drift indefinitely.

### Speaker Notes
> "Here is our most compelling technical finding: over 30 minutes, assuming 'no change' fails by over 30 degrees. Linear Ridge drifts by 4.5 °C. But HistGradientBoosting stabilizes within 2.9 °C over 360 recursive steps because decision trees capture non-linear convective cooling saturation."

---

## Slide 6: The "KNN Coolant Paradox" & Physics-Informed Fix

### Visual Layout
* **Side-by-side comparison**:
  * Left: Unconstrained KNN showing counter-intuitive upward curve when cooling.
  * Right: Monotonic HistGBM showing physical downward slope.
* Reference graphic: `figure3_coolant_paradox.png`.

### Slide Text
* **The Observational Data Trap**:
  * In passive dynamometer bench runs, coolant fluid naturally warms up as the motor is driven hard.
  * Distance-based algorithms (KNN) learn a spurious correlation: *low coolant temperature = idle motor*.
* **The Failure**:
  * When simulating an active cooling intervention (dropping coolant by 15 °C), unconstrained KNN looks up idle neighbors and erroneously predicts that the motor gets **hotter**!
* **The Solution — Monotonic Gradient Boosting**:
  * We enforce a **Monotonic Constraint** on coolant in HistGBM (`monotonic_cst = +1`).
  * Mathematically guarantees:
    $$\frac{\partial \Delta T}{\partial T_{\mathrm{cool}}} \ge 0$$
  * Colder coolant strictly reduces or holds steady the heating rate.

### Speaker Notes
> "Machine learning models do not know physics unless you enforce it. Pure KNN suffered from a 'Coolant Paradox' where dropping coolant predicted hotter motor temperatures. By enforcing monotonicity constraints in HistGBM, we guarantee physically sound cooling simulations."

---

## Slide 7: Targets T2 & T3 — Overheating Onset & Predictive Control

### Visual Layout
* **Left**: Graphic illustrating time-to-limit crossing $T_2$ and safety trigger $T_3$.
* **Right**: HistGBM T2 performance metrics and control policy rule box.

### Slide Text
* **Target $T_2$: Time to Overheating (seconds)**:
  * A static row model cannot predict time-to-threshold.
  * We extracted **64,841 ground-truth crossing labels** from continuous time series ($T_{\mathrm{limit}} = 70\text{ }^\circ\text{C}$).
  * Trained a specialized HistGradientBoosting Regressor to predict time remaining directly from operating inputs:
    * **$R^2 = \mathbf{0.7508}$**
    * **$\text{MAE} = \mathbf{122.6\text{ s}}$** (2.04 minutes across a 30-minute horizon on unseen profiles).
* **Target $T_3$: Cooling Start Time (seconds from now)**:
  * Rule-Based Predictive Control Policy:
    $$T_3 = \max(0,\ T_2 - \tau_{\mathrm{margin}})$$
  * With a 60-second safety margin: if overheating is predicted in 640 s (10.7 min), cooling is commanded at **$t = 580\text{ s}$ ($9.7\text{ min}$)**.
  * **Academic Framing**: Labeled as a *control-policy demonstration*.

### Speaker Notes
> "For Target T2, we trained a dedicated gradient-boosted regressor on over 64,000 threshold crossings. It predicts how many seconds remain before overheating with an R² of 0.75. T3 uses this forecast to trigger active cooling 60 seconds before danger, preventing reactive de-rating."

---

## Slide 8: Targets T4 & T5 — First-Order Thermal Model & Sensitivity (Route B)

### Visual Layout
* **Left**: First-order thermal ODE and analytical formula for duration $T_4$.
* **Right**: Sensitivity Analysis Matrix ($\pm 20\%$ in $R_{\mathrm{th}}$ and $C_{\mathrm{th}}$).

### Slide Text
* **Governing First-Order Thermal ODE**:
  $$C_{\mathrm{th}} \frac{dT(t)}{dt} = P_{\mathrm{loss}}(t) - \frac{T(t) - T_{\mathrm{cool}}(t)}{R_{\mathrm{th}}}$$
  * Nominal parameters: $R_{\mathrm{th}} = 0.10\text{ K/W}$, $C_{\mathrm{th}} = 4,000\text{ J/K}$ ($\tau = 400.0\text{ s}$).
* **Target $T_4$ (Required Cooling Duration)**:
  * Analytical logarithmic decay duration to reach target $T_{\mathrm{target}} = 65.5\text{ }^\circ\text{C}$:
    $$T_4 = R_{\mathrm{th}} C_{\mathrm{th}} \ln\left( \frac{T_{\mathrm{start}} - T_{\infty,\mathrm{cool}}}{T_{\mathrm{target}} - T_{\infty,\mathrm{cool}}} \right) = \mathbf{778.4\text{ s}}\ (13.0\text{ min})$$
* **Target $T_5$ (Post-Cooling Profile)**:
  * Peak temperature held at **$68.50\text{ }^\circ\text{C}$** (below 70 °C limit).
  * Shut-off temp: **$65.50\text{ }^\circ\text{C}$** | 30-min settled temp: **$71.85\text{ }^\circ\text{C}$**.
* **Sensitivity Matrix Highlights**:
  * **$R_{\mathrm{th}} +20\%$ (Degraded Thermal Interface)**: Motor overheats **3.9 minutes faster** ($T_2 = 406.7\text{ s}$), requiring continuous maximum cooling.
  * **$C_{\mathrm{th}} +20\%$ (Higher Thermal Inertia)**: Delays overheating onset by **2.1 minutes**.
* **Honest Limitations**: Labeled as model-derived estimates awaiting dynamometer experimental validation.

### Speaker Notes
> "Because observational datasets lack active chiller modulation experiments, we adopted Route B: a first-principles lumped thermal model. Our sensitivity analysis shows that thermal resistance R_th is the primary driver of thermal runaway—a 20% degradation causes the motor to overheat nearly 4 minutes faster."

---

## Slide 9: Embedded ECU Hardware Readiness (C Code Implementation)

### Visual Layout
* **Graphic**: Automotive ECU block diagram (Infineon AURIX / STM32) showing sensor inputs, EKF state observer, and advisory output.
* **Badges**: `MISRA-C Compliant` • `Zero Dynamic Memory` • `100% Vector Pass (20/20)`

### Slide Text
* **Embedded Architecture**:
  * Implemented in [`motor_thermal_ecu_final.c`](file:///c:/Users/prave/OneDrive/Desktop/IDES%20AND%20CADS/vscodes/motor-thermal-management/motor_thermal_ecu_final.c).
  * **Zero Dynamic Heap Allocation**: No `malloc` calls; execution timing is 100% deterministic for real-time safety.
  * Combines an explicit 2-state thermal RC network with a 3-state Extended Kalman Filter (EKF) for rotor temperature and magnet flux tracking.
* **ISO 26262 HIL Conformance Testing**:
  * Evaluated across **20 Hardware-in-the-Loop test vectors** covering nominal cruise, deep field weakening, and chiller pump failures.
  * **Pass Rate: 20 of 20 vectors (100%)** with zero numerical deviation after fixing the $T_5$ end-temperature logic.
* **Framing**: Serves as a deployment-readiness proof-of-concept for automotive production ECUs.

### Speaker Notes
> "Finally, we proved that this is not just Python research code. We implemented an ISO 26262 compliant C reference engine with zero dynamic memory allocation and constant-time execution. It passes all 20 HIL conformance test vectors with 100% precision."

---

## Slide 10: Conclusion & Summary of Deliverables

### Visual Layout
* **Grid of 4 Key Takeaways** + Links to live dashboard and code.

### Slide Text
* **1. Rigorous Data Science**: 10x physical thinning, fair cross-profile KNN sampling, and leak-free 5-fold cross-validation.
* **2. 30-Minute Trajectory Superiority**: HistGradientBoosting captures non-linear thermal saturation, cutting 30-min forecast error to **2.92 °C** (vs. 30.32 °C baseline).
* **3. Physics-Guaranteed Advisory**: Monotonic constraints eliminate the Coolant Paradox; first-order sensitivity bounds duration ($T_4$) and final temperatures ($T_5$).
* **4. Production Hardware Ready**: Embedded C engine passes 20/20 ISO 26262 HIL vectors.
* **Interactive Artifact**: Live simulator and visual dashboard available in [`dashboard.html`](file:///c:/Users/prave/OneDrive/Desktop/IDES%20AND%20CADS/vscodes/motor-thermal-management/dashboard.html).

### Speaker Notes
> "In conclusion, we have built a complete, end-to-end thermal management solution: from raw 1.3M sensor data, to machine learning trajectory models, physics-informed control advisory, and automotive C deployment. Thank you, and we welcome your questions."

---

## Slide 11: Examiner Defense & Technical Q&A (Tough Questions Answered)

### Visual Layout
* **Header**: "Anticipated Questions & Defense Strategy"
* **Grid of 4 Q&A Cards**: High contrast cards answering the 4 most likely examiner questions.

### Slide Text
* **Q1: "Why not simply attach a thermocouple directly to the rotor magnets?"**
  * *Answer*: In production EV powertrains, the rotor spins at up to 12,000 RPM. Physical wires snap instantly; mechanical slip rings wear out within months and pose fire hazards. Wireless telemetry sensors are expensive and fragile in high EMI environments. Virtual sensing in software via inverter telemetry is the industry-standard automotive solution.
* **Q2: "Why choose HistGradientBoosting over a Deep Neural Network (LSTM / Transformer)?"**
  * *Answer*: Three reasons: (1) **ECU Latency**: HistGBM evaluates in $2.6\text{ }\mu\text{s}$, consuming $<0.001\%$ of the $5\text{s}$ ECU budget. (2) **Safety / Monotonicity**: We can enforce strict mathematical monotonicity constraints (`monotonic_cst = +1`) in tree models to guarantee physics compliance; LSTMs are prone to unconstrained counterfactual hallucinations. (3) **Embedded Portability**: Tree decision rules convert directly to fixed-time C if-else branching with zero dynamic memory allocation.
* **Q3: "Why did you use Route B (Lumped Thermal ODE) instead of pure ML for Targets T4 and T5?"**
  * *Answer*: The Paderborn benchmark dataset consists of passive observational driving cycles where coolant pump speeds were largely static; it contains zero commanded chiller intervention steps. Training an ML model on passive data to predict active cooling duration would cause severe out-of-distribution hallucinations. Route B uses a documented first-principles lumped thermal model ($C_{\mathrm{th}} \frac{dT}{dt} = P_{\mathrm{loss}} - \frac{T - T_{\mathrm{cool}}}{R_{\mathrm{th}}}$) with full $\pm 20\%$ sensitivity bounds, which is the mathematically rigorous engineering standard.
* **Q4: "Why did KNN appear to lose earlier, and what did you fix?"**
  * *Answer*: Originally, KNN was trained only on the first 30,000 rows, which all came from just the first 2 driving profiles (severe training bias). When we sampled 35,000 points uniformly across all 55 training profiles, KNN achieved $0.1064\text{ }^\circ\text{C}$ MAE—matching Ridge ($0.1057\text{ }^\circ\text{C}$). However, over 30-minute recursive rollouts, KNN drifts by $3.93\text{ }^\circ\text{C}$ and suffers from the Coolant Paradox, proving that HistGBM is the superior architecture for trajectory forecasting.

### Speaker Notes
> "If examiners question our choice of gradient boosting over deep learning, or why we used a lumped analytical ODE for T4/T5, these four answers demonstrate deep automotive domain mastery: safety constraints, embedded ECU execution budgets, and honest handling of observational vs. active intervention data."
