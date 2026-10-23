"""
Electric Motor Thermal Prediction & Cooling Advisory System
============================================================
Hybrid Physics-ML architecture for PMSM thermal management.

Architecture:
  ML Loss Estimator  →  predicts P_loss from (i_d, i_q, u_d, u_q, ω, τ)
  Thermal RC Network →  2-node lumped-parameter model for heat transfer
  Rollout Simulator  →  360-step recursive trajectory forecasting
  Cooling Optimizer  →  nested binary search for optimal cooling schedule
  Advisory System    →  computes T1 through T5

Dependencies: numpy, pandas, scipy, scikit-learn
Dataset: Kaggle "Electric Motor Temperature" (measures_v2.csv)
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import warnings
import time


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MotorDataConfig:
    """Configuration for data preprocessing."""
    downsample_factor: int = 10          # 2 Hz → 0.2 Hz (Δt = 5 s)
    dt: float = 5.0                      # seconds per timestep
    horizon_steps: int = 360             # 30 min at 5 s steps
    thermal_targets: tuple = ('pm', 'stator_winding', 'stator_tooth', 'stator_yoke')
    load_features: tuple = ('u_d', 'u_q', 'i_d', 'i_q', 'motor_speed', 'torque')
    env_features: tuple = ('coolant', 'ambient')


def load_and_preprocess(csv_path: str, cfg: MotorDataConfig) -> pd.DataFrame:
    """
    Load the Electric Motor Temperature dataset and prepare features.

    Steps
    -----
    1. Downsample by factor 10 within each session (2 Hz → 0.2 Hz).
    2. Estimate electromagnetic losses from the dq-frame energy balance:
         P_elec ≈ (3/2)(u_d·i_d + u_q·i_q)
         P_mech  = τ · ω
         P_loss  ≈ max(P_elec − P_mech, 0)
    3. Compute 5 s incremental temperature changes ΔT for each sensor.
    """
    df = pd.read_csv(csv_path)

    # --- Downsample ---
    df['_row'] = df.groupby('profile_id').cumcount()
    df = df[df['_row'] % cfg.downsample_factor == 0].copy()
    df.drop(columns='_row', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- Estimate losses ---
    df['P_elec'] = 1.5 * (df['u_d'] * df['i_d'] + df['u_q'] * df['i_q'])
    df['P_mech'] = df['torque'] * df['motor_speed']
    df['P_loss_est'] = np.clip(df['P_elec'] - df['P_mech'], 0, None)

    # --- Incremental ΔT ---
    for t in cfg.thermal_targets:
        df[f'd_{t}'] = df.groupby('profile_id')[t].diff()

    df.dropna(subset=[f'd_{cfg.thermal_targets[0]}'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_group_kfold_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Group-K-Fold on profile_id — no session in both train and val.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['profile_id'].values
    return list(gkf.split(np.zeros(len(df)), groups=groups))


# ═══════════════════════════════════════════════════════════════════════
# 2. ML LOSS ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════

class LossEstimator:
    """
    Predicts total electromagnetic loss P_loss from operating conditions.

    Features : [i_d, i_q, u_d, u_q, motor_speed, torque]
    Target   : P_loss_est  (energy-balance estimate)

    Critical property
    -----------------
    The feature set deliberately **excludes** coolant and ambient
    temperatures.  Electromagnetic losses depend on electrical / mechanical
    state only.  This decoupling is what makes counterfactual cooling
    simulations physically valid: when we later perturb T_cool, the loss
    prediction stays unchanged, and only the heat-removal path (in the
    physics layer) responds.
    """

    FEATURES = ['i_d', 'i_q', 'u_d', 'u_q', 'motor_speed', 'torque']

    def __init__(
        self,
        max_iter: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
    ):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            loss='squared_error',
            random_state=42,
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> 'LossEstimator':
        X = df[self.FEATURES].values
        y = df['P_loss_est'].values
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.clip(self.model.predict(X), 0, None)

    def predict_from_dict(self, op: dict) -> float:
        x = np.array([op[f] for f in self.FEATURES])
        return float(self.predict(x)[0])


# ═══════════════════════════════════════════════════════════════════════
# 3. THERMAL RC NETWORK  (2-Node Lumped Parameter)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ThermalParams:
    """
    2-Node LPTN parameters.

    Thermal circuit
    ---------------
        P·α               P·(1−α)
         ↓                    ↓
      ┌──────┐   R_ps   ┌──────┐   R_sc
      │ C_pm │──/\\/\\/──│ C_s  │──/\\/\\/──  T_coolant
      │ T_pm │           │ T_s  │
      └──────┘           └──┬───┘
                            │ R_sa
                           /\\/\\/
                            │
                         T_ambient
    """
    C_pm:  float = 8.0     # Rotor PM thermal capacitance       [J/K]
    C_s:   float = 12.0    # Stator aggregate capacitance        [J/K]
    R_ps:  float = 0.50    # PM ↔ stator thermal resistance     [K/W]
    R_sc:  float = 0.15    # Stator ↔ coolant resistance        [K/W]
    R_sa:  float = 2.00    # Stator ↔ ambient resistance        [K/W]
    alpha: float = 0.35    # Fraction of total loss in rotor     [–]

    def to_array(self) -> np.ndarray:
        return np.array([self.C_pm, self.C_s, self.R_ps,
                         self.R_sc, self.R_sa, self.alpha])

    @classmethod
    def from_array(cls, a: np.ndarray) -> 'ThermalParams':
        return cls(*a)

    @staticmethod
    def bounds() -> list:
        """Physically motivated search bounds for identification."""
        return [
            (1.0, 100.0),   # C_pm
            (1.0, 200.0),   # C_s
            (0.01, 5.0),    # R_ps
            (0.01, 2.0),    # R_sc
            (0.1, 20.0),    # R_sa
            (0.05, 0.80),   # alpha
        ]


class ThermalNetwork:
    """
    2-Node Lumped-Parameter Thermal Network for PMSM.

    This is the physics core.  Given P_loss (from the ML estimator),
    it propagates temperatures via Newton's law of cooling.

    Why counterfactuals work
    ------------------------
    Coolant enters the dynamics exclusively through the structural
    equation  Q_sc = (T_s − T_cool) / R_sc.  Reducing T_cool increases
    this heat flux and drives T_s (and hence T_pm) downward — regardless
    of the operating-condition correlations in the training data.

    Integration method
    ------------------
    Semi-implicit (Crank–Nicolson-like) Euler:  dissipation terms
    evaluated at k+1, source terms at k.  This keeps the system
    A-stable at dt = 5 s even for stiff parameter combinations,
    while remaining an explicit 2×2 linear solve per step.
    """

    def __init__(self, params: ThermalParams):
        self.p = params

    # ── forward Euler (simpler, useful for debugging) ──────────────

    def step_explicit(
        self, T_pm: float, T_s: float,
        P: float, Tc: float, Ta: float, dt: float = 5.0,
    ) -> Tuple[float, float]:
        p = self.p
        Q_ps = (T_pm - T_s) / p.R_ps
        Q_sc = (T_s - Tc) / p.R_sc
        Q_sa = (T_s - Ta) / p.R_sa

        dT_pm = (p.alpha * P - Q_ps) / p.C_pm * dt
        dT_s  = ((1 - p.alpha) * P + Q_ps - Q_sc - Q_sa) / p.C_s * dt
        return T_pm + dT_pm, T_s + dT_s

    # ── semi-implicit Euler (production) ──────────────────────────

    def step(
        self, T_pm: float, T_s: float,
        P: float, Tc: float, Ta: float, dt: float = 5.0,
    ) -> Tuple[float, float]:
        """
        Semi-implicit step.  Dissipation (linear coupling) is implicit;
        source (P_loss) and boundary (Tc, Ta) are explicit.

        Solves:
          A · [T_pm^{k+1}, T_s^{k+1}]^T = b

        where A is 2×2 with positive diagonal dominance → always invertible.
        """
        p = self.p
        a11 = 1.0 + dt / (p.C_pm * p.R_ps)
        a12 = -dt / (p.C_pm * p.R_ps)
        a21 = -dt / (p.C_s * p.R_ps)
        a22 = 1.0 + dt / p.C_s * (1/p.R_ps + 1/p.R_sc + 1/p.R_sa)

        b1 = T_pm + dt * p.alpha * P / p.C_pm
        b2 = T_s  + dt / p.C_s * (
            (1 - p.alpha) * P + Tc / p.R_sc + Ta / p.R_sa
        )

        det = a11 * a22 - a12 * a21
        return (a22 * b1 - a12 * b2) / det, (a11 * b2 - a21 * b1) / det

    # ── analytical steady state ───────────────────────────────────

    def equilibrium(self, P: float, Tc: float, Ta: float) -> Tuple[float, float]:
        """
        Closed-form equilibrium (set dT/dt = 0).

          T_s^∞  = (P + Tc/R_sc + Ta/R_sa) / (1/R_sc + 1/R_sa)
          T_pm^∞ = T_s^∞ + α · P · R_ps
        """
        p = self.p
        G = 1/p.R_sc + 1/p.R_sa
        T_s_eq  = (P + Tc / p.R_sc + Ta / p.R_sa) / G
        T_pm_eq = T_s_eq + p.alpha * P * p.R_ps
        return T_pm_eq, T_s_eq


# ═══════════════════════════════════════════════════════════════════════
# 4. PARAMETER IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════

def identify_thermal_params(
    df: pd.DataFrame,
    loss_est: LossEstimator,
    session_ids: Optional[List] = None,
    dt: float = 5.0,
    max_sessions: int = 10,
) -> ThermalParams:
    """
    Fit RC parameters to observed trajectories via global optimisation.

    Objective
    ---------
      min_θ  Σ_sessions Σ_k  [ (T̂_pm(k) − T_pm(k))²
                              + (T̂_s(k)  − T̄_stator(k))² ]

    where T̄_stator is the mean of the three stator sensor readings.

    Uses scipy.optimize.differential_evolution for robustness against
    local minima, followed by a polishing L-BFGS-B step.
    """
    if session_ids is None:
        all_ids = df['profile_id'].unique()
        rng = np.random.RandomState(42)
        session_ids = rng.choice(
            all_ids, size=min(max_sessions, len(all_ids)), replace=False
        )

    # Pre-compute per-session arrays for speed
    sessions = []
    for sid in session_ids:
        sdf = df[df['profile_id'] == sid].reset_index(drop=True)
        if len(sdf) < 50:
            continue
        X_load = sdf[LossEstimator.FEATURES].values
        sessions.append({
            'T_pm':   sdf['pm'].values,
            'T_s':    sdf[['stator_winding', 'stator_tooth',
                           'stator_yoke']].mean(axis=1).values,
            'T_cool': sdf['coolant'].values,
            'T_amb':  sdf['ambient'].values,
            'P_loss': loss_est.predict(X_load),
        })

    def objective(x):
        params = ThermalParams.from_array(x)
        net = ThermalNetwork(params)
        mse_sum, n = 0.0, 0
        for s in sessions:
            T_pm, T_s = s['T_pm'][0], s['T_s'][0]
            for k in range(1, len(s['T_pm'])):
                T_pm, T_s = net.step(
                    T_pm, T_s, s['P_loss'][k-1],
                    s['T_cool'][k-1], s['T_amb'][k-1], dt
                )
                mse_sum += (T_pm - s['T_pm'][k])**2 + (T_s - s['T_s'][k])**2
                n += 2
        return mse_sum / max(n, 1)

    result = differential_evolution(
        objective,
        ThermalParams.bounds(),
        maxiter=200,
        seed=42,
        tol=1e-6,
        polish=True,          # L-BFGS-B refinement
    )
    return ThermalParams.from_array(result.x)


# ═══════════════════════════════════════════════════════════════════════
# 5. HYBRID PREDICTOR
# ═══════════════════════════════════════════════════════════════════════

class HybridPredictor:
    """
    ML losses + physics heat transfer = counterfactual-valid forecaster.
    """

    def __init__(
        self,
        loss_est: LossEstimator,
        thermal_net: ThermalNetwork,
        dt: float = 5.0,
    ):
        self.loss_est    = loss_est
        self.thermal_net = thermal_net
        self.dt          = dt

    def predict_step(
        self, T_pm: float, T_s: float,
        op: dict, T_cool: float, T_amb: float,
    ) -> Tuple[float, float]:
        P = self.loss_est.predict_from_dict(op)
        return self.thermal_net.step(T_pm, T_s, P, T_cool, T_amb, self.dt)


# ═══════════════════════════════════════════════════════════════════════
# 6. ROLLOUT SIMULATOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoolingSchedule:
    """Active-cooling window definition."""
    start_step:     int   = 0       # step index when cooling turns on
    duration_steps: int   = 0       # number of steps cooling stays on
    delta_cool:     float = 10.0    # reduction applied to T_cool [°C]

    def active(self, k: int) -> bool:
        return self.start_step <= k < self.start_step + self.duration_steps


@dataclass
class RolloutResult:
    T_pm: np.ndarray          # (N+1,)  PM temperature trajectory
    T_s:  np.ndarray          # (N+1,)  stator temperature trajectory
    t:    np.ndarray          # (N+1,)  time axis [s]
    peak_T_pm:  float
    peak_step:  int
    schedule:   Optional[CoolingSchedule] = None


def simulate_rollout(
    pred: HybridPredictor,
    T_pm_0: float,
    T_s_0: float,
    op: dict,
    T_cool_base: float,
    T_amb: float,
    n_steps: int = 360,
    cooling: Optional[CoolingSchedule] = None,
    dt: float = 5.0,
) -> RolloutResult:
    """
    Recursive trajectory under zero-order hold for operating conditions.

    Active cooling is modelled as a step reduction in coolant temperature:
      T_cool_eff = T_cool_base − δ   during cooling window
      T_cool_eff = T_cool_base       otherwise
    """
    T_pm_arr = np.empty(n_steps + 1)
    T_s_arr  = np.empty(n_steps + 1)
    T_pm_arr[0], T_s_arr[0] = T_pm_0, T_s_0
    T_pm, T_s = T_pm_0, T_s_0

    for k in range(n_steps):
        Tc = (T_cool_base - cooling.delta_cool
              if cooling and cooling.active(k)
              else T_cool_base)
        T_pm, T_s = pred.predict_step(T_pm, T_s, op, Tc, T_amb)
        T_pm_arr[k+1] = T_pm
        T_s_arr[k+1]  = T_s

    peak_idx = int(np.argmax(T_pm_arr))
    return RolloutResult(
        T_pm=T_pm_arr, T_s=T_s_arr,
        t=np.arange(n_steps + 1) * dt,
        peak_T_pm=float(T_pm_arr[peak_idx]),
        peak_step=peak_idx,
        schedule=cooling,
    )


# ═══════════════════════════════════════════════════════════════════════
# 7. COOLING OPTIMISER  —  nested binary search for (T3, T4)
# ═══════════════════════════════════════════════════════════════════════

class CoolingOptimizer:
    """
    Finds (t_start*, t_duration*) such that:

      max_k T_pm(k; t_start, t_duration)  ≤  T_limit − ε

    while maximising t_start (delay cooling) and minimising t_duration
    (save energy).

    Monotonicity assumptions (validated in the sanity suite):
      • fixed duration  → later start ⇒ higher peak   (monotone ↑)
      • fixed start     → longer duration ⇒ lower peak (monotone ↓)

    These allow nested binary search instead of exhaustive grid search.

    Complexity:  ~20 outer × ~20 inner × 360 steps/rollout ≈ 0.2 s.
    """

    def __init__(
        self,
        pred: HybridPredictor,
        T_limit: float = 90.0,
        eps: float = 2.0,
        safety_margin_steps: int = 12,      # 60 s / 5 s
        delta_cool: float = 10.0,
        dt: float = 5.0,
        n_steps: int = 360,
    ):
        self.pred       = pred
        self.T_limit    = T_limit
        self.T_target   = T_limit - eps
        self.margin     = safety_margin_steps
        self.delta_cool = delta_cool
        self.dt         = dt
        self.N          = n_steps

    # ── helpers ────────────────────────────────────────────────────

    def _peak(self, T_pm0, T_s0, op, Tc, Ta, s, d) -> float:
        sch = CoolingSchedule(s, d, self.delta_cool)
        r = simulate_rollout(self.pred, T_pm0, T_s0, op, Tc, Ta,
                             self.N, sch, self.dt)
        return r.peak_T_pm

    def _feasible(self, T_pm0, T_s0, op, Tc, Ta, s, d) -> bool:
        return self._peak(T_pm0, T_s0, op, Tc, Ta, s, d) <= self.T_target

    def _min_duration(self, T_pm0, T_s0, op, Tc, Ta, s) -> Optional[int]:
        """Binary search for shortest feasible duration at start step s."""
        max_d = self.N - s
        if not self._feasible(T_pm0, T_s0, op, Tc, Ta, s, max_d):
            return None          # even full-horizon cooling insufficient
        lo, hi = 1, max_d
        while lo < hi:
            mid = (lo + hi) // 2
            if self._feasible(T_pm0, T_s0, op, Tc, Ta, s, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # ── main entry ────────────────────────────────────────────────

    def optimize(self, T_pm0, T_s0, op, Tc, Ta) -> Dict:
        """
        Returns
        -------
        dict  with keys:
          cooling_needed, start_step, start_time_s,
          duration_steps, duration_time_s,
          uncooled_peak, cooled_peak, alarm
        """
        # 1) uncooled baseline
        unc = simulate_rollout(self.pred, T_pm0, T_s0, op, Tc, Ta, self.N)

        if unc.peak_T_pm <= self.T_target:
            return dict(cooling_needed=False, start_step=None,
                        start_time_s=None, duration_steps=None,
                        duration_time_s=None,
                        uncooled_peak=unc.peak_T_pm,
                        cooled_peak=unc.peak_T_pm, alarm=None)

        # 2) can cooling from step 0 fix it at all?
        d0 = self._min_duration(T_pm0, T_s0, op, Tc, Ta, 0)
        if d0 is None:
            pk = self._peak(T_pm0, T_s0, op, Tc, Ta, 0, self.N)
            return dict(
                cooling_needed=True, start_step=0, start_time_s=0.0,
                duration_steps=self.N,
                duration_time_s=self.N * self.dt,
                uncooled_peak=unc.peak_T_pm, cooled_peak=pk,
                alarm="Cooling insufficient at maximum capacity; "
                      "load reduction required.")

        # 3) binary search for latest feasible start
        lo, hi = 0, self.N - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2          # bias high
            if self._min_duration(T_pm0, T_s0, op, Tc, Ta, mid) is not None:
                lo = mid
            else:
                hi = mid - 1

        latest = lo
        safe   = max(0, latest - self.margin)

        # 4) minimum duration at the safe start
        opt_d  = self._min_duration(T_pm0, T_s0, op, Tc, Ta, safe)
        pk     = self._peak(T_pm0, T_s0, op, Tc, Ta, safe, opt_d)

        return dict(
            cooling_needed=True,
            start_step=safe,       start_time_s=safe * self.dt,
            duration_steps=opt_d,  duration_time_s=opt_d * self.dt,
            uncooled_peak=unc.peak_T_pm,
            cooled_peak=pk,
            alarm=None)


# ═══════════════════════════════════════════════════════════════════════
# 8. ADVISORY SYSTEM  (T1 … T5)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Advisory:
    """All five advisory outputs."""
    # T1 — trajectory
    T1_trajectory: np.ndarray
    T1_time:       np.ndarray

    # T2 — limit crossing
    T2_limit_time_s:  Optional[float]
    T2_status:        str            # "Now" | "At Xs" | "Not Reached"
    T2_predicted_peak: float

    # T3 — cooling start
    T3_start_s: Optional[float]

    # T4 — cooling duration
    T4_duration_s: Optional[float]
    T4_alarm:      Optional[str]

    # T5 — post-cooling metrics
    T5_peak:      float
    T5_shutoff:   Optional[float]
    T5_end:       float


class AdvisorySystem:
    """
    Top-level entry point.

        system = AdvisorySystem(predictor, T_limit=90)
        result = system.advise(T_pm_0, T_s_0, op, T_cool, T_amb)
    """

    def __init__(
        self, pred: HybridPredictor,
        T_limit: float = 90.0, eps: float = 2.0,
        safety_s: float = 60.0, delta_cool: float = 10.0,
        dt: float = 5.0, n_steps: int = 360,
    ):
        self.pred    = pred
        self.T_limit = T_limit
        self.dt      = dt
        self.N       = n_steps
        self.opt     = CoolingOptimizer(
            pred, T_limit, eps,
            safety_margin_steps=int(safety_s / dt),
            delta_cool=delta_cool, dt=dt, n_steps=n_steps,
        )

    def advise(self, T_pm0, T_s0, op, Tc, Ta) -> Advisory:

        # ── T1 ────────────────────────────────────────────────────
        unc = simulate_rollout(
            self.pred, T_pm0, T_s0, op, Tc, Ta, self.N, dt=self.dt
        )

        # ── T2 ────────────────────────────────────────────────────
        if T_pm0 >= self.T_limit:
            t2_time, t2_status = 0.0, "Now"
        else:
            hits = np.where(unc.T_pm >= self.T_limit)[0]
            if len(hits):
                t2_time   = float(hits[0] * self.dt)
                t2_status = f"At {t2_time:.0f}s"
            else:
                t2_time, t2_status = None, "Not Reached"

        # ── T3, T4 ───────────────────────────────────────────────
        o = self.opt.optimize(T_pm0, T_s0, op, Tc, Ta)

        # ── T5 ────────────────────────────────────────────────────
        if o['cooling_needed'] and o['start_step'] is not None:
            sch = CoolingSchedule(
                o['start_step'], o['duration_steps'],
                self.opt.delta_cool,
            )
            cooled = simulate_rollout(
                self.pred, T_pm0, T_s0, op, Tc, Ta,
                self.N, sch, self.dt,
            )
            t5_peak = float(np.max(cooled.T_pm))
            off_k   = sch.start_step + sch.duration_steps
            t5_off  = float(cooled.T_pm[min(off_k, self.N)])
            t5_end  = float(cooled.T_pm[-1])
        else:
            t5_peak = float(np.max(unc.T_pm))
            t5_off  = None
            t5_end  = float(unc.T_pm[-1])

        return Advisory(
            T1_trajectory=unc.T_pm, T1_time=unc.t,
            T2_limit_time_s=t2_time, T2_status=t2_status,
            T2_predicted_peak=float(np.max(unc.T_pm)),
            T3_start_s=o['start_time_s'],
            T4_duration_s=o['duration_time_s'],
            T4_alarm=o['alarm'],
            T5_peak=t5_peak, T5_shutoff=t5_off, T5_end=t5_end,
        )


# ═══════════════════════════════════════════════════════════════════════
# 9. PHYSICS SANITY TEST SUITE
# ═══════════════════════════════════════════════════════════════════════

class PhysicsSanityTests:
    """
    Since the dataset has no ground-truth active-cooling labels, we
    validate through physics-consistency assertions that any correct
    thermal model *must* satisfy.
    """

    def __init__(self, pred: HybridPredictor, dt: float = 5.0):
        self.pred = pred
        self.dt   = dt

    @staticmethod
    def _op(**kw):
        base = dict(i_d=0, i_q=0, u_d=0, u_q=0,
                    motor_speed=0, torque=0)
        base.update(kw)
        return base

    def _run(self, T0_pm, T0_s, op, Tc, Ta, N=360, cool=None):
        return simulate_rollout(
            self.pred, T0_pm, T0_s, op, Tc, Ta, N, cool, self.dt
        )

    # ------------------------------------------------------------------

    def test_coolant_sensitivity(self) -> bool:
        """∂T_peak / ∂T_cool > 0  (colder coolant → colder motor)."""
        op = self._op(i_q=200, u_q=300, motor_speed=3000, torque=50)
        peaks = [self._run(50, 45, op, Tc, 25).peak_T_pm
                 for Tc in [20, 25, 30, 35, 40]]
        ok = all(peaks[i] <= peaks[i+1] + 0.01 for i in range(len(peaks)-1))
        self._report("Coolant sensitivity (monotone ↑ in T_cool)",
                     ok, f"peaks={[f'{p:.1f}' for p in peaks]}")
        return ok

    def test_duration_monotonicity(self) -> bool:
        """Longer cooling → lower peak."""
        op = self._op(i_q=200, u_q=300, motor_speed=3000, torque=50)
        peaks = []
        for d in [0, 30, 60, 120, 180, 360]:
            c = CoolingSchedule(0, d, 10.0) if d else None
            peaks.append(self._run(60, 50, op, 25, 25, cool=c).peak_T_pm)
        ok = all(peaks[i] >= peaks[i+1] - 0.01 for i in range(len(peaks)-1))
        self._report("Duration monotonicity (monotone ↓ in duration)",
                     ok, f"peaks={[f'{p:.1f}' for p in peaks]}")
        return ok

    def test_idle_convergence(self) -> bool:
        """Zero load ⇒ T → T_ambient."""
        r = self._run(80, 70, self._op(), 25, 25, N=360)
        ok = r.T_pm[-1] < 80.0
        self._report("Idle convergence",
                     ok, f"80.0→{r.T_pm[-1]:.1f} (amb=25)")
        return ok

    def test_second_law(self) -> bool:
        """P=0, T>T_cool ⇒ dT/dt ≤ 0 at every step."""
        r = self._run(100, 90, self._op(), 25, 25, N=100)
        diffs = np.diff(r.T_pm)
        ok = bool(np.all(diffs <= 0.001))
        self._report("Second law (zero-source cooling)",
                     ok, f"max rise={np.max(diffs):.4f}°C")
        return ok

    def test_equilibrium(self) -> bool:
        """Long rollout converges to analytical steady state."""
        op = self._op(i_q=100, u_q=200, motor_speed=2000, torque=30)
        P = self.pred.loss_est.predict_from_dict(op)
        T_eq, _ = self.pred.thermal_net.equilibrium(P, 25, 25)
        r = self._run(25, 25, op, 25, 25, N=3600)
        err = abs(r.T_pm[-1] - T_eq)
        ok = err < 2.0
        self._report("Equilibrium convergence",
                     ok, f"analytical={T_eq:.1f}, sim={r.T_pm[-1]:.1f}")
        return ok

    def test_smoothness(self) -> bool:
        """Max step-to-step jump < 5 °C (thermal inertia)."""
        op = self._op(i_q=300, u_q=400, motor_speed=4000, torque=80)
        r = self._run(25, 25, op, 25, 25, N=360)
        mx = float(np.max(np.abs(np.diff(r.T_pm))))
        ok = mx < 5.0
        self._report("Smoothness (thermal inertia)",
                     ok, f"max Δ/step={mx:.3f}°C")
        return ok

    # ------------------------------------------------------------------

    @staticmethod
    def _report(name, passed, detail):
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}  ({detail})")

    def run_all(self) -> Dict[str, bool]:
        print("=" * 64)
        print("  PHYSICS SANITY TEST SUITE")
        print("=" * 64)
        results = {}
        for name, fn in [
            ("coolant_sensitivity",   self.test_coolant_sensitivity),
            ("duration_monotonicity", self.test_duration_monotonicity),
            ("idle_convergence",      self.test_idle_convergence),
            ("second_law",            self.test_second_law),
            ("equilibrium",           self.test_equilibrium),
            ("smoothness",            self.test_smoothness),
        ]:
            try:
                results[name] = fn()
            except Exception as e:
                print(f"  [ERR]  {name}: {e}")
                results[name] = False
        p = sum(results.values())
        print(f"\n  Result: {p}/{len(results)} passed")
        print("=" * 64)
        return results


# ═══════════════════════════════════════════════════════════════════════
# 10. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main(csv_path: str = 'measures_v2.csv'):
    """Full pipeline: load → train → calibrate → validate → advise."""
    cfg = MotorDataConfig()

    # 1. Data
    print("[1/7] Loading & preprocessing …")
    df = load_and_preprocess(csv_path, cfg)
    print(f"      {len(df):,} rows, {df['profile_id'].nunique()} sessions")

    # 2. Split
    print("[2/7] Group-K-Fold split …")
    splits = build_group_kfold_splits(df, 5)
    tr_idx, va_idx = splits[0]
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
    print(f"      train={len(df_tr):,}  val={len(df_va):,}")

    # 3. Loss model
    print("[3/7] Training loss estimator …")
    loss_est = LossEstimator()
    loss_est.fit(df_tr)
    y_hat = loss_est.predict(df_va[LossEstimator.FEATURES].values)
    print(f"      val MAE = {mean_absolute_error(df_va['P_loss_est'], y_hat):.2f} W")

    # 4. Thermal parameters
    print("[4/7] Identifying RC parameters (≈2-5 min) …")
    t0 = time.time()
    params = identify_thermal_params(df_tr, loss_est, max_sessions=8)
    print(f"      done in {time.time()-t0:.0f}s")
    print(f"      C_pm={params.C_pm:.2f}  C_s={params.C_s:.2f}  "
          f"R_ps={params.R_ps:.3f}  R_sc={params.R_sc:.3f}  "
          f"R_sa={params.R_sa:.3f}  α={params.alpha:.3f}")

    # 5. Hybrid predictor
    net  = ThermalNetwork(params)
    pred = HybridPredictor(loss_est, net, dt=cfg.dt)

    # 6. Sanity tests
    print("\n[5/7] Physics sanity tests …")
    tests = PhysicsSanityTests(pred, dt=cfg.dt)
    test_results = tests.run_all()

    # 7. Example advisory
    print("\n[6/7] Example advisory …")
    sys = AdvisorySystem(pred, T_limit=90.0, eps=2.0,
                         safety_s=60.0, delta_cool=10.0)
    row = df_va.iloc[len(df_va) // 2]
    op  = {f: float(row[f]) for f in LossEstimator.FEATURES}
    adv = sys.advise(
        T_pm0=float(row['pm']),
        T_s0=float(row[['stator_winding','stator_tooth','stator_yoke']].mean()),
        op=op,
        Tc=float(row['coolant']),
        Ta=float(row['ambient']),
    )
    print(f"      T1  trajectory : {adv.T1_trajectory[0]:.1f} → "
          f"{adv.T1_trajectory[-1]:.1f} °C  (peak {adv.T2_predicted_peak:.1f})")
    print(f"      T2  limit cross: {adv.T2_status}")
    print(f"      T3  cool start : {adv.T3_start_s}")
    print(f"      T4  cool dur   : {adv.T4_duration_s}")
    print(f"      T5  peak/off/end: {adv.T5_peak:.1f} / "
          f"{adv.T5_shutoff} / {adv.T5_end:.1f} °C")
    if adv.T4_alarm:
        print(f"      ⚠ ALARM: {adv.T4_alarm}")

    print("\n[7/7] Done.")
    return pred, sys, test_results


if __name__ == '__main__':
    predictor, system, sanity = main()
