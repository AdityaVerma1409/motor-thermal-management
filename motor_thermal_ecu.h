/**
 * @file   motor_thermal_ecu.h
 * @brief  PMSM Thermal Advisory — AUTOSAR-compatible, MISRA-C compliant.
 *
 * Q4: Zero-allocation, float32, deterministic-time implementation of:
 *   • 1-D speed LUT for pre-computed ZOH propagators (Φ, Γ)
 *   • 2-node semi-implicit Euler step
 *   • 360-step recursive rollout
 *   • Double-bisection cooling optimiser
 *   • Kalman observer for rotor temperature
 *
 * Target: Infineon AURIX TC387 / NXP S32K3 class MCUs.
 *
 * MISRA-C:2012 compliance notes:
 *   - No dynamic allocation (malloc/free/calloc/realloc).
 *   - No recursion.
 *   - All loops have fixed upper bounds.
 *   - Single-precision float throughout.
 *   - No VLAs; all arrays are compile-time sized.
 *   - All functions have a single return point.
 *   - No pointer arithmetic beyond array indexing.
 *
 * Memory budget (approximate):
 *   Flash (const LUT)   :  ~1.3 KB
 *   Stack (rollout buf)  :  ~3.0 KB
 *   Total               :  ~4.3 KB
 *
 * WCET budget at 100 MFLOPS (conservative AURIX float32):
 *   Single rollout (360 steps)    : ~0.07 ms
 *   Full bisection (400 rollouts) : ~28 ms
 *   Kalman update                 : ~0.01 ms
 *   Total per AUTOSAR cycle       : <30 ms  (fits 100 ms runnable)
 */

#ifndef MOTOR_THERMAL_ECU_H
#define MOTOR_THERMAL_ECU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════ */
/*  COMPILE-TIME CONFIGURATION                                       */
/* ═══════════════════════════════════════════════════════════════════ */

/** Number of speed grid points in the ZOH LUT. */
#define MTH_LUT_N_SPEED        32U

/** Maximum rollout horizon in steps (30 min / 5 s = 360). */
#define MTH_HORIZON_STEPS      360U

/** Maximum bisection iterations per dimension. */
#define MTH_BISECT_MAX_ITER    20U

/** Simulation timestep [s]. */
#define MTH_DT                 5.0f

/** Pi constant. */
#define MTH_PI                 3.14159265f

/* ═══════════════════════════════════════════════════════════════════ */
/*  DATA TYPES  (all statically sized, no pointers to heap)          */
/* ═══════════════════════════════════════════════════════════════════ */

/**
 * @brief  Thermal network parameters (factory-calibrated, NVM-stored).
 *
 * Stored in non-volatile memory and loaded at ECU startup.
 * May be updated by the dual aging estimator over the vehicle's life.
 */
typedef struct {
    float C_pm;         /**< Rotor PM thermal capacitance    [J/K]  */
    float C_s;          /**< Stator thermal capacitance      [J/K]  */
    float G_ag0;        /**< Airgap conductance, static      [W/K]  */
    float G_ag1;        /**< Airgap conductance, convective   [W·s^γ/K] */
    float gamma;        /**< Speed exponent for airgap       [–]    */
    float G_sc;         /**< Stator–coolant conductance      [W/K]  */
    float G_sa;         /**< Stator–ambient conductance      [W/K]  */
    float alpha;        /**< Fraction of loss in rotor       [–]    */
    float beta;         /**< Loss scaling multiplier         [–]    */
} Mth_ThermalParams_t;

/**
 * @brief  Pre-computed ZOH propagator at one speed grid point.
 *
 * Φ (2×2) and Γ (2×3) are stored flat (row-major).
 *
 *   T(k+1) = Φ · T(k) + Γ · u(k)
 *
 *   T = [T_pm, T_s]^T
 *   u = [P_loss, T_cool, T_amb]^T
 */
typedef struct {
    float omega_rad;        /**< Speed at this grid point [rad/s] */
    float Phi[4];           /**< 2×2, row-major: [Φ_00, Φ_01, Φ_10, Φ_11] */
    float Gam[6];           /**< 2×3, row-major: [Γ_00 .. Γ_12] */
} Mth_ZohEntry_t;

/**
 * @brief  Complete speed LUT — static const, lives in flash.
 */
typedef struct {
    Mth_ZohEntry_t entries[MTH_LUT_N_SPEED];
    uint32_t       n_entries;
    float          omega_min;
    float          omega_max;
    float          omega_step;   /**< (omega_max − omega_min) / (n−1) */
} Mth_SpeedLut_t;

/**
 * @brief  Semi-implicit Euler coefficients for one speed.
 *
 * Precomputed once per rollout (speed is constant under ZOH).
 * Avoids recomputing inside the 360-step inner loop.
 */
typedef struct {
    float a11, a12, a21, a22;   /**< 2×2 system matrix entries */
    float inv_det;              /**< 1 / (a11·a22 − a12·a21) */
    float alpha_bP_over_Cpm;    /**< α·β / C_pm  (multiply by P_loss) */
    float one_m_alpha_bP_over_Cs; /**< (1−α)·β / C_s */
    float Gsc_over_Cs;          /**< G_sc / C_s */
    float Gsa_over_Cs;          /**< G_sa / C_s */
} Mth_StepCoeffs_t;

/**
 * @brief  Kalman filter state for the rotor temperature observer.
 */
typedef struct {
    float x[2];         /**< State estimate: [T_pm_hat, T_s_hat] */
    float P[4];         /**< 2×2 covariance, row-major */
    float Q[4];         /**< 2×2 process noise covariance */
    float R;            /**< Scalar measurement noise variance */
} Mth_KalmanState_t;

/**
 * @brief  Operating point (inputs to the thermal model).
 */
typedef struct {
    float P_loss;       /**< Electromagnetic loss        [W]    */
    float T_cool;       /**< Coolant temperature          [°C]  */
    float T_amb;        /**< Ambient temperature          [°C]  */
    float omega_rad;    /**< Rotor angular velocity       [rad/s] */
    float i_d;          /**< Direct-axis current          [A]   */
} Mth_OperatingPoint_t;

/**
 * @brief  Demagnetisation boundary.
 */
typedef struct {
    float T_crit_0;     /**< Limit at zero field weakening [°C]  */
    float kappa_demag;  /**< Sensitivity [°C/A]                  */
    float T_absolute;   /**< Hard ceiling [°C]                   */
} Mth_DemagBoundary_t;

/**
 * @brief  Complete advisory output (T1 through T5).
 */
typedef struct {
    /* T1: trajectory (optional — only filled if caller provides buffer) */
    uint8_t  trajectory_valid;

    /* T2: limit crossing */
    float    T2_peak;               /**< Uncooled peak [°C] */
    float    T2_limit_time_s;       /**< Time to limit [s], -1 if not reached */

    /* T3: cooling start */
    float    T3_start_time_s;       /**< Recommended command time [s], -1 if none */

    /* T4: cooling duration */
    float    T4_duration_s;         /**< Required duration [s], -1 if none */

    /* T5: post-cooling profile */
    float    T5_peak;               /**< Peak with cooling [°C] */
    float    T5_shutoff;            /**< Temp at cooling shutoff [°C] */
    float    T5_end;                /**< Temp at horizon end [°C] */

    /* Dynamic limit used */
    float    T_limit_active;        /**< T_limit(i_d) used [°C] */

    /* Alarm */
    uint8_t  alarm;                 /**< 0 = OK, 1 = insufficient cooling */
} Mth_Advisory_t;

/* ═══════════════════════════════════════════════════════════════════ */
/*  PUBLIC API                                                        */
/* ═══════════════════════════════════════════════════════════════════ */

/**
 * @brief  Initialise the speed LUT from the thermal parameters.
 *
 * Called once at ECU startup (or after NVM parameter update).
 * Populates the static LUT in-place.  Uses only stack memory.
 *
 * @param[out] lut     LUT to populate (caller-owned, typically static).
 * @param[in]  params  Calibrated thermal parameters.
 */
void Mth_InitLut(Mth_SpeedLut_t *lut,
                 const Mth_ThermalParams_t *params);

/**
 * @brief  Precompute semi-implicit Euler coefficients for one speed.
 *
 * Called once per advisory cycle (speed is constant under ZOH).
 *
 * @param[out] coeffs  Coefficients struct to fill.
 * @param[in]  params  Thermal parameters.
 * @param[in]  omega   Angular velocity [rad/s].
 */
void Mth_PrecomputeCoeffs(Mth_StepCoeffs_t *coeffs,
                          const Mth_ThermalParams_t *params,
                          float omega);

/**
 * @brief  Single rollout returning peak T_pm.
 *
 * Runs a 360-step semi-implicit Euler rollout with optional
 * cooling window.  No heap allocation; uses only the precomputed
 * coefficients.
 *
 * WCET: 360 × ~15 FLOPs ≈ 5400 FLOPs ≈ 0.054 ms at 100 MFLOPS.
 *
 * @param[in]  coeffs      Precomputed step coefficients.
 * @param[in]  T_pm0       Initial rotor temperature [°C].
 * @param[in]  T_s0        Initial stator temperature [°C].
 * @param[in]  P_loss      Constant electromagnetic loss [W].
 * @param[in]  T_cool_base Base coolant temperature [°C].
 * @param[in]  T_amb       Ambient temperature [°C].
 * @param[in]  cool_start  Cooling window start step.
 * @param[in]  cool_dur    Cooling window duration steps.
 * @param[in]  delta_cool  Coolant reduction during cooling [°C].
 * @return     Peak T_pm over the horizon [°C].
 */
float Mth_RolloutPeak(const Mth_StepCoeffs_t *coeffs,
                      float T_pm0, float T_s0,
                      float P_loss,
                      float T_cool_base, float T_amb,
                      uint16_t cool_start, uint16_t cool_dur,
                      float delta_cool);

/**
 * @brief  Full advisory computation (T1 through T5).
 *
 * Runs the double-bisection optimiser and fills the advisory struct.
 *
 * WCET: ~400 rollouts × 0.054 ms ≈ 22 ms.
 *
 * @param[out] advisory    Output struct (caller-owned).
 * @param[in]  params      Thermal parameters.
 * @param[in]  op          Current operating point.
 * @param[in]  demag       Demagnetisation boundary.
 * @param[in]  T_pm0       Initial rotor temperature [°C].
 * @param[in]  T_s0        Initial stator temperature [°C].
 * @param[in]  eps         Safety margin below T_limit [°C].
 * @param[in]  safety_steps Extra steps before latest start.
 * @param[in]  delta_cool  Coolant reduction [°C].
 * @param[in]  delay_steps Transport delay [steps].
 */
void Mth_ComputeAdvisory(Mth_Advisory_t *advisory,
                         const Mth_ThermalParams_t *params,
                         const Mth_OperatingPoint_t *op,
                         const Mth_DemagBoundary_t *demag,
                         float T_pm0, float T_s0,
                         float eps,
                         uint16_t safety_steps,
                         float delta_cool,
                         uint16_t delay_steps);

/**
 * @brief  Kalman filter initialisation.
 *
 * @param[out] kf          Kalman state to initialise.
 * @param[in]  T_s_meas    First stator measurement [°C].
 * @param[in]  q_pm        Process noise for T_pm [°C²].
 * @param[in]  q_s         Process noise for T_s [°C²].
 * @param[in]  r           Measurement noise variance [°C²].
 */
void Mth_KalmanInit(Mth_KalmanState_t *kf,
                    float T_s_meas,
                    float q_pm, float q_s, float r);

/**
 * @brief  Kalman filter predict-correct cycle.
 *
 * @param[in,out] kf       Kalman state.
 * @param[in]     lut      Speed LUT for (Φ, Γ) lookup.
 * @param[in]     T_s_meas Stator measurement [°C].
 * @param[in]     P_loss   Loss estimate [W].
 * @param[in]     T_cool   Coolant temperature [°C].
 * @param[in]     T_amb    Ambient temperature [°C].
 * @param[in]     omega    Angular velocity [rad/s].
 * @return        Estimated rotor temperature T_pm_hat [°C].
 */
float Mth_KalmanUpdate(Mth_KalmanState_t *kf,
                       const Mth_SpeedLut_t *lut,
                       float T_s_meas,
                       float P_loss, float T_cool, float T_amb,
                       float omega);

#ifdef __cplusplus
}
#endif

#endif /* MOTOR_THERMAL_ECU_H */
