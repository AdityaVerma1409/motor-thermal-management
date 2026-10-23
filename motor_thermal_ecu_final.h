/**
 * @file  motor_thermal_ecu_final.h
 * @brief PMSM Thermal Advisory — Production MISRA-C:2012 / AUTOSAR.
 *
 * Features
 * --------
 *   [1] Singularity-free float32 Sylvester expm via sinhc(x).
 *   [2] Dual 1-D speed LUT for ZOH propagators (Φ, Γ).
 *   [3] 3-state EKF: [T_pm, T_s, λ_pm] with back-EMF flux fusion.
 *   [4] Gated coast-down RLS aging estimator for G_sc, G_ag0.
 *   [5] Fast Tier-1 demagnetisation clamp (inline, ISR-safe).
 *   [6] Double-bisection advisory with pipe dead-time shift.
 *
 * Constraints
 * -----------
 *   • Zero heap allocation (no malloc/free).  < 2 KB SRAM.
 *   • All loops bounded at compile time.
 *   • float32 throughout.  No VLAs, no recursion.
 *   • Single point of return per function body.
 *
 * Target MCUs: Infineon AURIX TC387 / NXP S32K3 / dSPACE MicroAutoBox.
 */

#ifndef MOTOR_THERMAL_ECU_FINAL_H
#define MOTOR_THERMAL_ECU_FINAL_H

#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════ compile-time knobs ══════════════════════ */

#define MTH_LUT_N           32U     /**< Speed LUT grid points.  */
#define MTH_HORIZON         360U    /**< Rollout steps (30 min).  */
#define MTH_BISECT_ITER     20U     /**< Max bisection iters.     */
#define MTH_DT              5.0f    /**< Timestep  [s].           */
#define MTH_PI_F            3.14159265f

/* ══════════════════════ data types ══════════════════════════════ */

/** Thermal-network parameters (NVM-resident, field-updatable). */
typedef struct {
    float C_pm;     float C_s;
    float G_ag0;    float G_ag1;    float gamma;
    float G_sc;     float G_sa;
    float alpha;    float beta;
} Mth_Params_t;

/** Pre-computed ZOH entry: Φ (2×2, flat) + Γ (2×3, flat). */
typedef struct {
    float omega;
    float Phi[4];
    float Gam[6];
} Mth_LutEntry_t;

/** Speed LUT (const flash). */
typedef struct {
    Mth_LutEntry_t e[MTH_LUT_N];
    uint32_t n;
    float w_min, w_max, w_step;
} Mth_Lut_t;

/** Semi-implicit Euler pre-computed coefficients (stack). */
typedef struct {
    float a11, a12, a21, a22, inv_det;
    float c_alpha_Cpm;   /* α·β / C_pm                */
    float c_1malpha_Cs;  /* (1−α)·β / C_s             */
    float c_Gsc_Cs;      /* G_sc / C_s                 */
    float c_Gsa_Cs;      /* G_sa / C_s                 */
} Mth_Coeffs_t;

/** Demagnetisation boundary. */
typedef struct {
    float T_crit0;       /**< Safe limit at i_d=0 [°C]  */
    float kappa;         /**< Sensitivity [°C/A]         */
    float T_abs;         /**< Absolute ceiling [°C]      */
} Mth_Demag_t;

/** Magnet flux parameters for EKF. */
typedef struct {
    float lambda_ref;    /**< Flux linkage at T_ref [Wb]  */
    float alpha_lambda;  /**< Temp coefficient [1/°C]     */
    float T_ref;         /**< Reference temperature [°C]  */
} Mth_FluxParams_t;

/**
 * @brief 3-state EKF: x = [T_pm, T_s, λ_pm].
 *
 * Observation y = [T_s_meas, λ_pm_obs].
 *   T_s_meas from stator thermistor.
 *   λ_pm_obs from back-EMF: λ = (u_q − R_s·i_q) / ω  (at |ω|>ω_min).
 */
typedef struct {
    float x[3];      /**< State estimate.                       */
    float P[9];      /**< 3×3 covariance (row-major).           */
    float Q[9];      /**< 3×3 process noise cov.                */
    float R[4];      /**< 2×2 measurement noise cov.            */
    float lambda_ref;
    float alpha_lam;
    float T_ref_lam;
    float omega_min; /**< Minimum speed for flux observation [rad/s] */
} Mth_EKF_t;

/** RLS aging estimator state. */
typedef struct {
    float theta[2];     /**< [ln G_sc, ln G_ag0]                  */
    float P_rls[4];     /**< 2×2 inverse covariance (row-major)   */
    float lambda;       /**< Forgetting factor (0.995–0.9999)     */
    float G_sc_nom;     /**< Nominal factory G_sc [W/K]           */
    float G_ag0_nom;    /**< Nominal factory G_ag0 [W/K]          */
    uint16_t cd_count;  /**< Coast-down sample counter            */
    float i_thresh;     /**< Coast-down current threshold [A]     */
    uint16_t cd_min;    /**< Minimum samples before update        */
} Mth_RLS_t;

/** Operating point (model inputs). */
typedef struct {
    float P_loss;     float T_cool;    float T_amb;
    float omega;      float i_d;       float i_q;
    float u_q;        /**< q-axis voltage for flux obs [V] */
    float R_s;        /**< Stator resistance [Ω]           */
} Mth_Op_t;

/** Advisory output. */
typedef struct {
    float T2_peak;          float T2_cross_s;
    float T3_cmd_s;         float T4_dur_s;
    float T5_peak;          float T5_off;       float T5_end;
    float T_limit_active;   float i_d_limit_A;
    uint8_t alarm;          /**< 0=OK 1=insufficient 2=runaway */
} Mth_Advisory_t;

/* ══════════════════════ public API ══════════════════════════════ */

/* ── LUT build (startup) ─────────────────────────────────────── */
void Mth_BuildLut(Mth_Lut_t *lut, const Mth_Params_t *p);

/* ── Coefficient pre-compute (once per advisory cycle) ────────── */
void Mth_PreCoeffs(Mth_Coeffs_t *c, const Mth_Params_t *p, float omega);

/* ── Demagnetisation clamp (inline, ISR-safe) ─────────────────── */
static inline float Mth_DemagClamp(const Mth_Demag_t *d, float id)
{
    float t = d->T_crit0 - d->kappa * fabsf(id);
    if (t < 20.0f)  { t = 20.0f; }
    if (t > d->T_abs) { t = d->T_abs; }
    return t;
}

static inline float Mth_DemagIdLimit(const Mth_Demag_t *d,
                                     float T_pm, float eps)
{
    /* Max |i_d| before demag at current T_pm. */
    float headroom = d->T_crit0 - eps - T_pm;
    float result;
    if (headroom <= 0.0f) { result = 0.0f; }
    else                  { result = headroom / d->kappa; }
    return result;
}

/* ── Single rollout (inner kernel) ────────────────────────────── */
float Mth_RolloutPeak(const Mth_Coeffs_t *c,
                      float Tp0, float Ts0,
                      float P, float Tc, float Ta,
                      uint16_t cs, uint16_t cd,
                      float dc, uint16_t delay);

/* ── Full advisory (T1–T5 + i_d limit) ───────────────────────── */
void Mth_Advisory(Mth_Advisory_t *out,
                  const Mth_Params_t *p,
                  const Mth_Op_t *op,
                  const Mth_Demag_t *demag,
                  float Tp0, float Ts0,
                  float eps, uint16_t safety,
                  float dc, uint16_t delay);

/* ── 3-state EKF ─────────────────────────────────────────────── */
void Mth_EKF_Init(Mth_EKF_t *kf,
                  float Ts_meas,
                  const Mth_FluxParams_t *fp,
                  const float Q9[9], const float R4[4],
                  float omega_min);

float Mth_EKF_Step(Mth_EKF_t *kf,
                   const Mth_Lut_t *lut,
                   float Ts_meas, float lambda_obs,
                   uint8_t flux_valid,
                   float P_loss, float Tc, float Ta,
                   float omega);

/* ── RLS aging estimator ─────────────────────────────────────── */
void Mth_RLS_Init(Mth_RLS_t *rls,
                  float G_sc_nom, float G_ag0_nom,
                  float lambda_forget,
                  float i_thresh, uint16_t cd_min);

uint8_t Mth_RLS_Step(Mth_RLS_t *rls,
                     Mth_Params_t *p,
                     float Tp_hat, float Ts_hat,
                     float Ts_meas, float Tc,
                     float omega, float id, float iq,
                     float C_s, float dt);

#ifdef __cplusplus
}
#endif
#endif /* MOTOR_THERMAL_ECU_FINAL_H */
