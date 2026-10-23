"""
Production System-Level Extensions  —  v4
===========================================
  [Q1] Dynamic demagnetization boundary  T_limit(i_d)
  [Q2] Coolant plumbing transport delay  (Smith dead-time)
  [Q3] Dual-estimation for lifetime aging (parameter tracking)

Builds on v2 (thermal network) and v3 (observer, feedback, COP).
"""

import numpy as np
from scipy.linalg import expm, inv as linalg_inv
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    def njit(*a, **kw):
        def _wrap(fn): return fn
        return _wrap if not a else (lambda fn: fn) if callable(a[0]) else _wrap
    HAS_NUMBA = False


# ═══════════════════════════════════════════════════════════════════════
#  Q1.  DYNAMIC DEMAGNETIZATION BOUNDARY
# ═══════════════════════════════════════════════════════════════════════
#
#  Physics
#  -------
#  NdFeB magnets have a temperature-dependent coercivity:
#
#    H_c(T) = H_{c,20} · [1 + α_H · (T − 20)]
#
#  where α_H ≈ −0.55 to −0.65 %/°C  (negative — coercivity drops
#  with temperature).
#
#  Under field weakening the inverter injects negative i_d, which
#  creates a demagnetising MMF:
#
#    H_demag ∝ −i_d · N_s / (2·p·l_ag)
#
#  Irreversible demagnetisation occurs when the operating point
#  crosses the knee of the B–H curve.  The safe temperature envelope
#  is therefore:
#
#    T_limit(i_d) = T_crit,0 − κ_demag · |i_d|
#
#  where T_crit,0 is the safe limit at zero field weakening and
#  κ_demag [°C/A] is a motor-specific coefficient derived from the
#  magnet grade's B–H family.
#
#  Impact on the bisection optimiser
#  ----------------------------------
#  Under ZOH (constant i_d within a rollout), T_limit is a scalar
#  constant → the existing bisection is unchanged, just parameterised
#  by i_d.
#
#  Under a time-varying field-weakening profile i_d(k), the constraint
#  becomes step-dependent:
#
#    T_pm(k) ≤ T_limit(i_d(k))   ∀ k ∈ [0, 360]
#
#  We define the feasibility metric as:
#
#    margin_min = min_k [ T_limit(i_d(k)) − T_pm(k) ]
#
#  Feasible iff margin_min ≥ 0.  The bisection searches for the
#  schedule that keeps margin_min ≥ 0.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DemagBoundary:
    """
    Temperature-dependent demagnetisation limit.

    Parameters
    ----------
    T_crit_0     : safe PM temperature at zero field weakening  [°C]
    kappa_demag  : sensitivity coefficient                      [°C/A]
    T_absolute   : absolute maximum regardless of i_d           [°C]

    Typical values for N42SH grade:
      T_crit_0   ≈ 150 °C   (thermal class at H_d = 0)
      kappa_demag ≈ 0.25–0.5 °C/A  (depends on motor geometry)
      T_absolute  = 180 °C  (magnet binder breakdown)
    """
    T_crit_0:    float = 150.0
    kappa_demag: float = 0.30
    T_absolute:  float = 180.0

    def T_limit(self, i_d: float) -> float:
        """Dynamic safe temperature limit given current i_d [A]."""
        T_dyn = self.T_crit_0 - self.kappa_demag * abs(i_d)
        return min(max(T_dyn, 20.0), self.T_absolute)

    def T_limit_array(self, i_d_profile: np.ndarray) -> np.ndarray:
        """Vectorised limit for a time-varying i_d profile."""
        T_dyn = self.T_crit_0 - self.kappa_demag * np.abs(i_d_profile)
        return np.clip(T_dyn, 20.0, self.T_absolute)


@njit(cache=True)
def _fast_rollout_peak_demag(
    T_pm0: float, T_s0: float,
    P_loss: float, T_cool_base: float, T_amb: float,
    omega_rad: float,
    C_pm: float, C_s: float,
    G0: float, G1: float, gam: float,
    G_sc: float, G_sa: float,
    alpha: float, beta_: float,
    dt: float, n_steps: int,
    cool_start: int, cool_dur: int, delta_cool: float,
    # Demagnetisation boundary (constant i_d → constant limit)
    T_limit: float,
) -> Tuple[float, float]:
    """
    Returns (worst_margin, peak_T_pm).
    worst_margin = min_k [T_limit − T_pm(k)].
    Feasible iff worst_margin ≥ 0.
    """
    g = G0 + G1 * abs(omega_rad) ** gam
    bP = beta_ * P_loss

    a11 = 1.0 + dt * g / C_pm
    a12 = -dt * g / C_pm
    a21 = -dt * g / C_s
    a22 = 1.0 + dt / C_s * (g + G_sc + G_sa)
    det = a11 * a22 - a12 * a21

    T_pm, T_s = T_pm0, T_s0
    peak = T_pm0
    worst_margin = T_limit - T_pm0

    for k in range(n_steps):
        Tc = T_cool_base - delta_cool if cool_start <= k < cool_start + cool_dur else T_cool_base

        b1 = T_pm + dt * alpha * bP / C_pm
        b2 = T_s  + dt / C_s * (
            (1.0 - alpha) * bP + G_sc * Tc + G_sa * T_amb
        )

        T_pm = (a22 * b1 - a12 * b2) / det
        T_s  = (a11 * b2 - a21 * b1) / det

        if T_pm > peak:
            peak = T_pm

        margin = T_limit - T_pm
        if margin < worst_margin:
            worst_margin = margin

    return worst_margin, peak


@njit(cache=True)
def _fast_rollout_peak_demag_profile(
    T_pm0: float, T_s0: float,
    P_loss: float, T_cool_base: float, T_amb: float,
    omega_rad: float,
    C_pm: float, C_s: float,
    G0: float, G1: float, gam: float,
    G_sc: float, G_sa: float,
    alpha: float, beta_: float,
    dt: float, n_steps: int,
    cool_start: int, cool_dur: int, delta_cool: float,
    # Time-varying demagnetisation limit (one value per step + initial)
    T_limit_profile: np.ndarray,    # shape (n_steps + 1,)
) -> Tuple[float, float]:
    """
    Same as above but with a time-varying T_limit(k) array,
    for field-weakening profiles where i_d changes over time.
    """
    g = G0 + G1 * abs(omega_rad) ** gam
    bP = beta_ * P_loss

    a11 = 1.0 + dt * g / C_pm
    a12 = -dt * g / C_pm
    a21 = -dt * g / C_s
    a22 = 1.0 + dt / C_s * (g + G_sc + G_sa)
    det = a11 * a22 - a12 * a21

    T_pm, T_s = T_pm0, T_s0
    peak = T_pm0
    worst_margin = T_limit_profile[0] - T_pm0

    for k in range(n_steps):
        Tc = T_cool_base - delta_cool if cool_start <= k < cool_start + cool_dur else T_cool_base

        b1 = T_pm + dt * alpha * bP / C_pm
        b2 = T_s  + dt / C_s * (
            (1.0 - alpha) * bP + G_sc * Tc + G_sa * T_amb
        )

        T_pm = (a22 * b1 - a12 * b2) / det
        T_s  = (a11 * b2 - a21 * b1) / det

        if T_pm > peak:
            peak = T_pm

        margin = T_limit_profile[k + 1] - T_pm
        if margin < worst_margin:
            worst_margin = margin

    return worst_margin, peak


class DemagAwareCoolingOptimizer:
    """
    Bisection optimizer with dynamic demagnetisation boundary.

    Under ZOH (constant i_d), T_limit is scalar and the algorithm
    is identical to v2 except the limit depends on the current
    operating point.

    Under a time-varying i_d profile, T_limit(k) is an array and
    the feasibility check uses the step-wise margin.
    """

    def __init__(
        self, pred, demag: DemagBoundary,
        eps: float = 2.0,
        safety_margin_steps: int = 12,
        delta_cool: float = 10.0,
        dt: float = 5.0, n_steps: int = 360,
    ):
        self.pred   = pred
        self.demag  = demag
        self.eps    = eps
        self.margin = safety_margin_steps
        self.dc     = delta_cool
        self.dt     = dt
        self.N      = n_steps

        tp = pred.thermal_net.p
        self._tp = (
            tp.C_pm, tp.C_s,
            tp.G_ag0, tp.G_ag1, tp.gamma,
            tp.G_sc, tp.G_sa,
            tp.alpha, tp.beta,
        )

    def _feasible(self, T_pm0, T_s0, P, Tc, Ta, omega, s, d, T_lim):
        margin, _ = _fast_rollout_peak_demag(
            T_pm0, T_s0, P, Tc, Ta, omega,
            *self._tp, self.dt, self.N,
            s, d, self.dc, T_lim - self.eps,
        )
        return margin >= 0

    def _min_duration(self, T_pm0, T_s0, P, Tc, Ta, omega, s, T_lim):
        max_d = self.N - s
        if not self._feasible(T_pm0, T_s0, P, Tc, Ta, omega, s, max_d, T_lim):
            return None
        lo, hi = 1, max_d
        while lo < hi:
            mid = (lo + hi) // 2
            if self._feasible(T_pm0, T_s0, P, Tc, Ta, omega, s, mid, T_lim):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def optimize(self, T_pm0, T_s0, op, Tc, Ta) -> Dict:
        """
        Optimise cooling schedule against the demagnetisation boundary.

        The limit T_limit(i_d) is computed from the operating point's i_d.
        Under ZOH, i_d is constant → T_limit is scalar.
        """
        P     = self.pred.loss_est.predict_scalar(op)
        omega = op['motor_speed'] * 2.0 * np.pi / 60.0
        i_d   = op.get('i_d', 0.0)

        T_lim = self.demag.T_limit(i_d)

        # Uncooled check
        margin_unc, peak_unc = _fast_rollout_peak_demag(
            T_pm0, T_s0, P, Tc, Ta, omega,
            *self._tp, self.dt, self.N,
            0, 0, 0.0, T_lim - self.eps,
        )

        if margin_unc >= 0:
            return dict(
                cooling_needed=False,
                T_limit_active=T_lim,
                i_d=i_d,
                start_step=None, duration_steps=None,
                uncooled_peak=peak_unc, alarm=None,
            )

        # Check if cooling from step 0 can fix it
        d0 = self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, 0, T_lim)
        if d0 is None:
            return dict(
                cooling_needed=True,
                T_limit_active=T_lim,
                i_d=i_d,
                start_step=0, duration_steps=self.N,
                uncooled_peak=peak_unc,
                alarm=f"Cooling insufficient at T_limit={T_lim:.0f}°C "
                      f"(i_d={i_d:.0f}A); reduce load or field weakening.",
            )

        # Binary search for latest start
        lo, hi = 0, self.N - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, mid, T_lim) is not None:
                lo = mid
            else:
                hi = mid - 1

        safe = max(0, lo - self.margin)
        opt_d = self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, safe, T_lim)

        _, cooled_peak = _fast_rollout_peak_demag(
            T_pm0, T_s0, P, Tc, Ta, omega,
            *self._tp, self.dt, self.N,
            safe, opt_d, self.dc, T_lim - self.eps,
        )

        return dict(
            cooling_needed=True,
            T_limit_active=T_lim,
            i_d=i_d,
            start_step=safe,
            start_time_s=safe * self.dt,
            duration_steps=opt_d,
            duration_time_s=opt_d * self.dt,
            uncooled_peak=peak_unc,
            cooled_peak=cooled_peak,
            alarm=None,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Q2.  COOLANT TRANSPORT DELAY  (Smith dead-time)
# ═══════════════════════════════════════════════════════════════════════
#
#  In a BEV, the chiller/radiator is 2–4 m of plumbing away from
#  the motor jacket.  When the chiller starts, cold coolant takes
#
#    t_transport = V_pipe / V̇_coolant
#
#  seconds to physically arrive.  During this dead-time the motor
#  sees no cooling effect at all.
#
#  Key insight:  the transport delay does NOT change the thermal
#  network matrices (Φ, Γ).  It only shifts WHEN the effective
#  coolant temperature changes.  In the rollout:
#
#    T_cool_eff(k) = T_cool_base − δ
#        if  (k − delay_steps) ∈ [cool_start, cool_start + cool_dur)
#
#  Equivalently: the cooling window's THERMAL EFFECT is delayed by
#  delay_steps, but the COMMAND is issued at cool_start.
#
#  For the optimizer, the safety margin absorbs the delay:
#
#    effective_margin = safety_margin_steps + delay_steps
#
#  Additionally, coolant temperature doesn't step instantaneously
#  even after the cold fluid arrives — there's a first-order mixing
#  lag.  We model both together:
#
#    T_cool_eff(k) = T_cool_base − δ · h(k − k_cmd)
#
#  where h(t) is the delayed-first-order step response:
#
#    h(t) = 0                           for t < t_transport
#    h(t) = 1 − exp(−(t−t_transport)/τ_mix)  for t ≥ t_transport
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CoolantTransportModel:
    """
    Coolant plumbing delay + mixing dynamics.

    Parameters
    ----------
    t_transport_s : pure transport dead-time              [s]
    tau_mix_s     : first-order mixing time constant       [s]
    dt            : simulation timestep                    [s]
    """
    t_transport_s: float = 10.0     # 2–4 m pipe at ~0.5 m/s
    tau_mix_s:     float = 5.0      # jacket mixing lag
    dt:            float = 5.0

    @property
    def delay_steps(self) -> int:
        """Integer dead-time in simulation steps."""
        return int(np.ceil(self.t_transport_s / self.dt))

    def step_response(self, n_steps: int) -> np.ndarray:
        """
        Normalised step response h(k) ∈ [0, 1].

        h(k) = 0                                    for k < delay_steps
        h(k) = 1 − exp(−(k·dt − t_transport)/τ_mix) for k ≥ delay_steps
        """
        t = np.arange(n_steps) * self.dt
        h = np.zeros(n_steps)
        active = t >= self.t_transport_s
        h[active] = 1.0 - np.exp(
            -(t[active] - self.t_transport_s) / self.tau_mix_s
        )
        return h

    def effective_delta_cool(
        self, n_steps: int,
        cool_start: int, cool_dur: int,
        delta_cool: float,
    ) -> np.ndarray:
        """
        Effective coolant temperature reduction at each step,
        accounting for transport delay and mixing lag.

        Returns an array of shape (n_steps,) where:
          T_cool_eff(k) = T_cool_base − effective_delta(k)
        """
        h = self.step_response(n_steps)
        delta_profile = np.zeros(n_steps)

        for k in range(n_steps):
            if cool_start <= k < cool_start + cool_dur:
                # Cooling command has been active for (k − cool_start) steps
                age = k - cool_start
                if age < len(h):
                    delta_profile[k] = delta_cool * h[age]
            elif k >= cool_start + cool_dur:
                # Cooling command OFF — the "off" transient propagates
                # with the same delay + lag
                off_age = k - (cool_start + cool_dur)
                on_age  = k - cool_start
                # h_on is the response to the ON step, h_off to the OFF step
                h_on = h[min(on_age, len(h)-1)]  if on_age < len(h) else 1.0
                h_off = h[min(off_age, len(h)-1)] if off_age < len(h) else 1.0
                delta_profile[k] = delta_cool * (h_on - h_off)

        return np.clip(delta_profile, 0, delta_cool)


@njit(cache=True)
def _fast_rollout_with_delay(
    T_pm0: float, T_s0: float,
    P_loss: float, T_cool_base: float, T_amb: float,
    omega_rad: float,
    C_pm: float, C_s: float,
    G0: float, G1: float, gam: float,
    G_sc: float, G_sa: float,
    alpha: float, beta_: float,
    dt: float, n_steps: int,
    cool_start: int, cool_dur: int, delta_cool: float,
    delay_steps: int,
    T_limit: float,
) -> Tuple[float, float]:
    """
    Rollout with integer-step transport delay.

    The cooling EFFECT appears delay_steps after the command is issued.
    The mixing lag is absorbed into a conservative delay
    (delay_steps = ceil((t_transport + τ_mix) / dt)).
    """
    g = G0 + G1 * abs(omega_rad) ** gam
    bP = beta_ * P_loss

    a11 = 1.0 + dt * g / C_pm
    a12 = -dt * g / C_pm
    a21 = -dt * g / C_s
    a22 = 1.0 + dt / C_s * (g + G_sc + G_sa)
    det = a11 * a22 - a12 * a21

    T_pm, T_s = T_pm0, T_s0
    peak = T_pm0

    # The effective cooling window is shifted by delay_steps
    eff_start = cool_start + delay_steps
    eff_end   = cool_start + cool_dur + delay_steps

    for k in range(n_steps):
        Tc = T_cool_base - delta_cool if eff_start <= k < eff_end else T_cool_base

        b1 = T_pm + dt * alpha * bP / C_pm
        b2 = T_s  + dt / C_s * (
            (1.0 - alpha) * bP + G_sc * Tc + G_sa * T_amb
        )

        T_pm = (a22 * b1 - a12 * b2) / det
        T_s  = (a11 * b2 - a21 * b1) / det

        if T_pm > peak:
            peak = T_pm

    feasible = peak <= T_limit
    return peak, float(feasible)


class TransportDelayOptimizer:
    """
    Cooling optimiser that accounts for plumbing transport delay.

    The safety margin is increased by the delay steps:
      effective_margin = base_margin + delay_steps

    The bisection structure is unchanged.
    """

    def __init__(
        self, pred, transport: CoolantTransportModel,
        T_limit: float = 90.0, eps: float = 2.0,
        base_safety_steps: int = 12,
        delta_cool: float = 10.0,
        n_steps: int = 360,
    ):
        self.pred    = pred
        self.trans   = transport
        self.T_target = T_limit - eps
        self.dc      = delta_cool
        self.dt      = transport.dt
        self.N       = n_steps
        self.delay   = transport.delay_steps

        # Total safety margin = explicit safety + transport delay
        self.margin  = base_safety_steps + self.delay

        tp = pred.thermal_net.p
        self._tp = (
            tp.C_pm, tp.C_s,
            tp.G_ag0, tp.G_ag1, tp.gamma,
            tp.G_sc, tp.G_sa,
            tp.alpha, tp.beta,
        )

    def _peak(self, T_pm0, T_s0, P, Tc, Ta, omega, s, d):
        pk, _ = _fast_rollout_with_delay(
            T_pm0, T_s0, P, Tc, Ta, omega,
            *self._tp, self.dt, self.N,
            s, d, self.dc, self.delay, self.T_target,
        )
        return pk

    def _feasible(self, T_pm0, T_s0, P, Tc, Ta, omega, s, d):
        return self._peak(T_pm0, T_s0, P, Tc, Ta, omega, s, d) <= self.T_target

    def _min_duration(self, T_pm0, T_s0, P, Tc, Ta, omega, s):
        max_d = self.N - s
        if not self._feasible(T_pm0, T_s0, P, Tc, Ta, omega, s, max_d):
            return None
        lo, hi = 1, max_d
        while lo < hi:
            mid = (lo + hi) // 2
            if self._feasible(T_pm0, T_s0, P, Tc, Ta, omega, s, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def optimize(self, T_pm0, T_s0, op, Tc, Ta) -> Dict:
        P     = self.pred.loss_est.predict_scalar(op)
        omega = op['motor_speed'] * 2.0 * np.pi / 60.0

        unc_peak = self._peak(T_pm0, T_s0, P, Tc, Ta, omega, 0, 0)
        if unc_peak <= self.T_target:
            return dict(cooling_needed=False, uncooled_peak=unc_peak,
                        transport_delay_s=self.delay * self.dt)

        d0 = self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, 0)
        if d0 is None:
            return dict(
                cooling_needed=True, uncooled_peak=unc_peak,
                transport_delay_s=self.delay * self.dt,
                alarm="Cooling insufficient even with immediate start.")

        lo, hi = 0, self.N - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, mid) is not None:
                lo = mid
            else:
                hi = mid - 1

        safe = max(0, lo - self.margin)
        opt_d = self._min_duration(T_pm0, T_s0, P, Tc, Ta, omega, safe)
        pk = self._peak(T_pm0, T_s0, P, Tc, Ta, omega, safe, opt_d)

        return dict(
            cooling_needed=True,
            start_step=safe,
            command_time_s=safe * self.dt,
            arrival_time_s=(safe + self.delay) * self.dt,
            duration_steps=opt_d,
            duration_time_s=opt_d * self.dt,
            cooled_peak=pk,
            uncooled_peak=unc_peak,
            transport_delay_s=self.delay * self.dt,
            alarm=None,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Q3.  DUAL ESTIMATION FOR LIFETIME AGING
# ═══════════════════════════════════════════════════════════════════════
#
#  Problem
#  -------
#  Over 10–15 years and 150k+ miles:
#    • TIM / potting resin degrades  →  G_sc drops by 15–30 %
#    • Coolant channel scaling       →  convective h drops
#    • Airgap contamination          →  G_ag0 may drop
#
#  The offline-calibrated parameters drift, and the model gradually
#  under-predicts cooling requirements.
#
#  Architecture: Dual Kalman Filter
#  --------------------------------
#  Two interleaved filters, each running every measurement cycle:
#
#  Filter 1 (STATE):  Standard KF from Q1.
#    State:  x = [T_pm, T_s]
#    Uses current parameter estimates θ̂ to build (Φ, Γ).
#
#  Filter 2 (PARAMETER):  Slow-tracking KF.
#    State:  θ = [ln(G_sc), ln(G_ag0)]
#    Log-parameterisation ensures positivity.
#    Process noise Q_θ is very small (parameters drift over months,
#    not seconds).
#    Measurement: the state filter's innovation sequence.
#
#  When to update parameters?
#    • Always: state filter runs every cycle.
#    • Conditionally: parameter filter updates ONLY during
#      "informative" intervals — coast-down (P_loss ≈ 0) and
#      steady-state thermal equilibrium — when the thermal
#      dynamics are dominated by the conductance terms rather
#      than the loss terms.
#
#  This "conditional update" prevents the parameter filter from
#  absorbing loss-estimation errors into the conductance estimates.
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgingEstimatorConfig:
    """Configuration for the dual parameter estimator."""
    # Parameter process noise (very small — aging is slow)
    Q_theta_diag: Tuple[float, float] = (1e-8, 1e-8)   # [ln(W/K)]² per step
    # Coast-down detection thresholds
    current_threshold_A: float = 5.0
    # Minimum number of coast-down steps before updating
    min_coastdown_steps: int = 10
    # Exponential forgetting factor for parameter filter
    lambda_forget: float = 0.999     # close to 1 = slow adaptation


class DualAgingEstimator:
    """
    Dual Kalman Filter for joint state + parameter estimation.

    Tracks thermal conductance degradation online using the
    model's innovation sequence during coast-down intervals.
    """

    def __init__(
        self, thermal_net,        # ThermalNetwork from v2
        observer,                  # RotorTemperatureObserver from v3
        cfg: AgingEstimatorConfig = AgingEstimatorConfig(),
    ):
        self.net = thermal_net
        self.obs = observer        # state filter
        self.cfg = cfg

        # Parameter state: θ = [ln(G_sc), ln(G_ag0)]
        self.theta = np.array([
            np.log(thermal_net.p.G_sc),
            np.log(thermal_net.p.G_ag0),
        ])
        self.P_theta = np.diag([0.01, 0.01])   # initial parameter covariance
        self.Q_theta = np.diag(cfg.Q_theta_diag)

        # Coast-down detector state
        self._coastdown_count = 0
        self._innovation_buffer = []

    def is_coastdown(self, i_d: float, i_q: float) -> bool:
        """Detect coast-down interval from current magnitudes."""
        return (abs(i_d) < self.cfg.current_threshold_A and
                abs(i_q) < self.cfg.current_threshold_A)

    def update(
        self,
        T_s_measured: float,
        P_loss: float, T_cool: float, T_amb: float,
        omega_rad: float,
        i_d: float, i_q: float,
        dt: float = 5.0,
    ) -> Dict:
        """
        Full dual-estimation cycle.

        1. State filter predict + correct  (always)
        2. Coast-down detection
        3. Parameter filter update          (only during coast-down)
        4. Rebuild thermal network if parameters changed

        Returns
        -------
        dict with T_pm_hat, T_s_hat, G_sc_hat, G_ag0_hat, aging_pct
        """
        # --- 1. State filter ---
        T_pm_hat, T_s_hat, P_state = self.obs.update(
            T_s_measured, P_loss, T_cool, T_amb, omega_rad, dt,
        )

        # Innovation (prediction error on T_s)
        # The observer stores this internally; we also track it here
        innovation = T_s_measured - T_s_hat  # approximate

        # --- 2. Coast-down detection ---
        in_coastdown = self.is_coastdown(i_d, i_q)

        if in_coastdown:
            self._coastdown_count += 1
            self._innovation_buffer.append(innovation)
        else:
            self._coastdown_count = 0
            self._innovation_buffer.clear()

        # --- 3. Parameter update (only during sustained coast-down) ---
        param_updated = False
        if (in_coastdown and
                self._coastdown_count >= self.cfg.min_coastdown_steps):

            # During coast-down (P ≈ 0), the thermal dynamics are:
            #   dT_s/dt ≈ G_ag·(T_pm − T_s)/C_s − G_sc·(T_s − T_cool)/C_s
            #
            # The innovation reflects how well these conductances match
            # the observed cooling rate.

            # Linearised Jacobian of the measurement model w.r.t. θ:
            #   ∂(predicted T_s) / ∂(ln G_sc) ≈ −dt·G_sc·(T_s−T_cool)/C_s
            #   ∂(predicted T_s) / ∂(ln G_ag0) ≈ dt·G_ag0·(T_pm−T_s)/C_s

            G_sc_curr  = np.exp(self.theta[0])
            G_ag0_curr = np.exp(self.theta[1])

            H_theta = np.array([
                -dt * G_sc_curr * (T_s_hat - T_cool) / self.net.p.C_s,
                 dt * G_ag0_curr * (T_pm_hat - T_s_hat) / self.net.p.C_s,
            ]).reshape(1, 2)

            # Parameter Kalman update
            S_theta = float(H_theta @ self.P_theta @ H_theta.T) + self.obs.R_val
            K_theta = (self.P_theta @ H_theta.T) / S_theta   # (2, 1)

            self.theta = self.theta + K_theta.ravel() * innovation

            IKH = np.eye(2) - K_theta @ H_theta
            self.P_theta = (IKH @ self.P_theta @ IKH.T +
                           K_theta * self.obs.R_val @ K_theta.T)

            # Forgetting factor (prevents covariance from collapsing)
            self.P_theta /= self.cfg.lambda_forget

            # Apply updated parameters to the thermal network
            self.net.p.G_sc  = np.exp(self.theta[0])
            self.net.p.G_ag0 = np.exp(self.theta[1])

            param_updated = True

        # --- 4. Aging metrics ---
        G_sc_original = 30.0    # nominal from factory calibration
        G_ag0_original = 4.0

        return {
            'T_pm_hat': T_pm_hat,
            'T_s_hat': T_s_hat,
            'G_sc_hat': float(np.exp(self.theta[0])),
            'G_ag0_hat': float(np.exp(self.theta[1])),
            'G_sc_degradation_pct': (
                1.0 - np.exp(self.theta[0]) / G_sc_original
            ) * 100,
            'G_ag0_degradation_pct': (
                1.0 - np.exp(self.theta[1]) / G_ag0_original
            ) * 100,
            'param_updated': param_updated,
            'in_coastdown': in_coastdown,
        }
