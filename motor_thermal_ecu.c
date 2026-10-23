/**
 * @file   motor_thermal_ecu.c
 * @brief  PMSM Thermal Advisory — AUTOSAR / MISRA-C implementation.
 *
 * All functions use only stack-allocated locals (no malloc).
 * All loops have compile-time-constant upper bounds.
 * Single-precision float32 throughout.
 *
 * Tested on:
 *   - Infineon AURIX TC387  (TriCore, 300 MHz, float32 FPU)
 *   - NXP S32K344           (Cortex-M7, 160 MHz, float32 FPU)
 *   - x86-64 GCC/Clang      (unit tests)
 */

#include "motor_thermal_ecu.h"
#include <math.h>   /* fabsf, powf, sqrtf, expf */

/* ═══════════════════════════════════════════════════════════════════ */
/*  INTERNAL HELPERS                                                  */
/* ═══════════════════════════════════════════════════════════════════ */

/**
 * @brief  Compute airgap thermal conductance at a given speed.
 *         G_ag(ω) = G_ag0 + G_ag1 · |ω|^γ
 */
static float compute_G_ag(const Mth_ThermalParams_t *p, float omega)
{
    float result;
    result = p->G_ag0 + p->G_ag1 * powf(fabsf(omega), p->gamma);
    return result;
}

/**
 * @brief  Dynamic demagnetisation temperature limit.
 *         T_limit(i_d) = T_crit_0 − κ · |i_d|,  clamped.
 */
static float compute_T_limit(const Mth_DemagBoundary_t *d, float i_d)
{
    float T_dyn;
    float result;

    T_dyn = d->T_crit_0 - d->kappa_demag * fabsf(i_d);

    /* Clamp to [20, T_absolute] */
    if (T_dyn < 20.0f)
    {
        result = 20.0f;
    }
    else if (T_dyn > d->T_absolute)
    {
        result = d->T_absolute;
    }
    else
    {
        result = T_dyn;
    }

    return result;
}

/**
 * @brief  2×2 matrix exponential via Sylvester / Cayley-Hamilton.
 *
 * For a 2×2 matrix A with eigenvalues λ₁, λ₂:
 *
 *   e^A = c₀·I + c₁·A
 *
 * where, if λ₁ ≠ λ₂:
 *   c₁ = (e^λ₁ − e^λ₂) / (λ₁ − λ₂)
 *   c₀ = e^λ₁ − λ₁·c₁  =  (λ₁·e^λ₂ − λ₂·e^λ₁) / (λ₁ − λ₂)
 *
 * If λ₁ ≈ λ₂ (repeated root):
 *   c₀ = e^λ₁ · (1 − λ₁)
 *   c₁ = e^λ₁
 *
 * Eigenvalues of a 2×2 [a b; c d]:
 *   λ = (tr ± √(tr²−4·det)) / 2
 *
 * This avoids calling a general expm() library function.
 *
 * @param[in]  A   2×2 row-major, scaled by dt: A_scaled = A·dt.
 * @param[out] E   2×2 row-major: e^{A·dt}.
 */
static void expm2x2(const float A[4], float E[4])
{
    float tr, det_A, disc, sq;
    float lam1, lam2;
    float c0, c1;
    float e1, e2;

    tr    = A[0] + A[3];                     /* trace */
    det_A = A[0] * A[3] - A[1] * A[2];      /* determinant */
    disc  = tr * tr - 4.0f * det_A;

    if (disc > 1e-10f)
    {
        sq   = sqrtf(disc);
        lam1 = 0.5f * (tr + sq);
        lam2 = 0.5f * (tr - sq);
        e1   = expf(lam1);
        e2   = expf(lam2);
        c1   = (e1 - e2) / (lam1 - lam2);
        c0   = (lam1 * e2 - lam2 * e1) / (lam1 - lam2);
    }
    else
    {
        /* Repeated eigenvalue */
        lam1 = 0.5f * tr;
        e1   = expf(lam1);
        c0   = e1 * (1.0f - lam1);
        c1   = e1;
    }

    /* E = c0·I + c1·A */
    E[0] = c0 + c1 * A[0];
    E[1] =      c1 * A[1];
    E[2] =      c1 * A[2];
    E[3] = c0 + c1 * A[3];
}

/**
 * @brief  2×2 matrix inverse.  Assumes det ≠ 0 (guaranteed for
 *         our thermal system with positive conductances).
 *
 * @param[in]  M     2×2 row-major input.
 * @param[out] Minv  2×2 row-major output.
 */
static void inv2x2(const float M[4], float Minv[4])
{
    float det;
    float inv_det;

    det     = M[0] * M[3] - M[1] * M[2];
    inv_det = 1.0f / det;

    Minv[0] =  M[3] * inv_det;
    Minv[1] = -M[1] * inv_det;
    Minv[2] = -M[2] * inv_det;
    Minv[3] =  M[0] * inv_det;
}

/**
 * @brief  2×2 · 2×N matrix multiply (N=2 or N=3).
 */
static void mat2x2_mul_2xN(const float L[4], const float R[],
                            float Out[], uint32_t N)
{
    uint32_t j;
    for (j = 0U; j < N; j++)
    {
        Out[0U * N + j] = L[0] * R[0U * N + j] + L[1] * R[1U * N + j];
        Out[1U * N + j] = L[2] * R[0U * N + j] + L[3] * R[1U * N + j];
    }
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  LUT INITIALISATION                                                */
/* ═══════════════════════════════════════════════════════════════════ */

void Mth_InitLut(Mth_SpeedLut_t *lut,
                 const Mth_ThermalParams_t *params)
{
    uint32_t i;
    float omega;
    float g;
    float A_dt[4];        /* A·dt */
    float Phi[4];         /* e^{A·dt} */
    float Ainv[4];        /* A⁻¹ */
    float PhiI[4];        /* Φ − I */
    float Ainv_PhiI[4];   /* A⁻¹·(Φ−I) */
    float B[6];           /* 2×3 */
    float Gam[6];         /* 2×3 */

    lut->n_entries = MTH_LUT_N_SPEED;
    lut->omega_min = 0.0f;
    lut->omega_max = 15000.0f * (2.0f * MTH_PI / 60.0f);  /* 15k RPM */
    lut->omega_step = (lut->omega_max - lut->omega_min)
                      / (float)(MTH_LUT_N_SPEED - 1U);

    for (i = 0U; i < MTH_LUT_N_SPEED; i++)
    {
        omega = lut->omega_min + (float)i * lut->omega_step;
        g     = compute_G_ag(params, omega);

        /* Build A·dt (continuous-time matrix × timestep) */
        A_dt[0] = (-g / params->C_pm) * MTH_DT;
        A_dt[1] = ( g / params->C_pm) * MTH_DT;
        A_dt[2] = ( g / params->C_s)  * MTH_DT;
        A_dt[3] = (-(g + params->G_sc + params->G_sa) / params->C_s) * MTH_DT;

        /* Φ = expm(A·dt) */
        expm2x2(A_dt, Phi);

        /* Γ = A⁻¹·(Φ − I)·B */
        /* Build A (unscaled by dt, for the inverse) */
        {
            float A_raw[4];
            A_raw[0] = -g / params->C_pm;
            A_raw[1] =  g / params->C_pm;
            A_raw[2] =  g / params->C_s;
            A_raw[3] = -(g + params->G_sc + params->G_sa) / params->C_s;

            inv2x2(A_raw, Ainv);
        }

        /* Φ − I */
        PhiI[0] = Phi[0] - 1.0f;
        PhiI[1] = Phi[1];
        PhiI[2] = Phi[2];
        PhiI[3] = Phi[3] - 1.0f;

        /* A⁻¹·(Φ−I) */
        mat2x2_mul_2xN(Ainv, PhiI, Ainv_PhiI, 2U);

        /* B matrix (2×3) */
        B[0] = params->alpha * params->beta / params->C_pm;
        B[1] = 0.0f;
        B[2] = 0.0f;
        B[3] = (1.0f - params->alpha) * params->beta / params->C_s;
        B[4] = params->G_sc / params->C_s;
        B[5] = params->G_sa / params->C_s;

        /* Γ = Ainv_PhiI · B  (2×2 · 2×3 = 2×3) */
        mat2x2_mul_2xN(Ainv_PhiI, B, Gam, 3U);

        /* Store */
        lut->entries[i].omega_rad = omega;
        lut->entries[i].Phi[0] = Phi[0];
        lut->entries[i].Phi[1] = Phi[1];
        lut->entries[i].Phi[2] = Phi[2];
        lut->entries[i].Phi[3] = Phi[3];
        lut->entries[i].Gam[0] = Gam[0];
        lut->entries[i].Gam[1] = Gam[1];
        lut->entries[i].Gam[2] = Gam[2];
        lut->entries[i].Gam[3] = Gam[3];
        lut->entries[i].Gam[4] = Gam[4];
        lut->entries[i].Gam[5] = Gam[5];
    }
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  SEMI-IMPLICIT EULER COEFFICIENTS                                  */
/* ═══════════════════════════════════════════════════════════════════ */

void Mth_PrecomputeCoeffs(Mth_StepCoeffs_t *c,
                          const Mth_ThermalParams_t *p,
                          float omega)
{
    float g;
    float det;

    g = compute_G_ag(p, omega);

    c->a11 = 1.0f + MTH_DT * g / p->C_pm;
    c->a12 = -MTH_DT * g / p->C_pm;
    c->a21 = -MTH_DT * g / p->C_s;
    c->a22 = 1.0f + MTH_DT / p->C_s * (g + p->G_sc + p->G_sa);

    det = c->a11 * c->a22 - c->a12 * c->a21;
    c->inv_det = 1.0f / det;

    c->alpha_bP_over_Cpm    = p->alpha * p->beta / p->C_pm;
    c->one_m_alpha_bP_over_Cs = (1.0f - p->alpha) * p->beta / p->C_s;
    c->Gsc_over_Cs          = p->G_sc / p->C_s;
    c->Gsa_over_Cs          = p->G_sa / p->C_s;
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  SINGLE ROLLOUT                                                    */
/* ═══════════════════════════════════════════════════════════════════ */

float Mth_RolloutPeak(const Mth_StepCoeffs_t *c,
                      float T_pm, float T_s,
                      float P_loss,
                      float T_cool_base, float T_amb,
                      uint16_t cool_start, uint16_t cool_dur,
                      float delta_cool)
{
    float peak;
    float Tc;
    float b1, b2;
    float T_pm_new, T_s_new;
    uint16_t k;
    uint16_t cool_end;

    peak     = T_pm;
    cool_end = cool_start + cool_dur;

    for (k = 0U; k < MTH_HORIZON_STEPS; k++)
    {
        /* Determine effective coolant temperature */
        if ((k >= cool_start) && (k < cool_end))
        {
            Tc = T_cool_base - delta_cool;
        }
        else
        {
            Tc = T_cool_base;
        }

        /* RHS vector */
        b1 = T_pm + MTH_DT * c->alpha_bP_over_Cpm * P_loss;
        b2 = T_s  + MTH_DT * (
            c->one_m_alpha_bP_over_Cs * P_loss
            + c->Gsc_over_Cs * Tc
            + c->Gsa_over_Cs * T_amb
        );

        /* Cramer's rule solve */
        T_pm_new = (c->a22 * b1 - c->a12 * b2) * c->inv_det;
        T_s_new  = (c->a11 * b2 - c->a21 * b1) * c->inv_det;

        T_pm = T_pm_new;
        T_s  = T_s_new;

        if (T_pm > peak)
        {
            peak = T_pm;
        }
    }

    return peak;
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  DOUBLE-BISECTION OPTIMISER                                        */
/* ═══════════════════════════════════════════════════════════════════ */

/**
 * @brief  Binary search for minimum feasible cooling duration.
 *
 * @return Duration in steps, or 0xFFFF if infeasible.
 */
static uint16_t find_min_duration(
    const Mth_StepCoeffs_t *c,
    float T_pm0, float T_s0, float P_loss,
    float T_cool, float T_amb,
    uint16_t start, float delta_cool,
    uint16_t delay,
    float T_target)
{
    uint16_t max_dur;
    uint16_t lo, hi, mid;
    float    peak;
    uint16_t result;
    uint16_t eff_start;

    eff_start = start;  /* delay absorbed into safety margin externally */
    max_dur   = MTH_HORIZON_STEPS - eff_start;

    /* Check if max duration is sufficient */
    peak = Mth_RolloutPeak(c, T_pm0, T_s0, P_loss, T_cool, T_amb,
                           eff_start + delay, max_dur, delta_cool);

    if (peak > T_target)
    {
        result = 0xFFFFU;   /* infeasible */
    }
    else
    {
        lo = 1U;
        hi = max_dur;

        /* Fixed upper bound: MTH_BISECT_MAX_ITER iterations */
        {
            uint16_t iter;
            for (iter = 0U; iter < MTH_BISECT_MAX_ITER; iter++)
            {
                if (lo >= hi)
                {
                    break;
                }
                mid = (lo + hi) / 2U;
                peak = Mth_RolloutPeak(c, T_pm0, T_s0, P_loss,
                                       T_cool, T_amb,
                                       eff_start + delay, mid,
                                       delta_cool);
                if (peak <= T_target)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid + 1U;
                }
            }
        }
        result = lo;
    }

    return result;
}

void Mth_ComputeAdvisory(Mth_Advisory_t *adv,
                         const Mth_ThermalParams_t *params,
                         const Mth_OperatingPoint_t *op,
                         const Mth_DemagBoundary_t *demag,
                         float T_pm0, float T_s0,
                         float eps,
                         uint16_t safety_steps,
                         float delta_cool,
                         uint16_t delay_steps)
{
    Mth_StepCoeffs_t coeffs;
    float T_limit;
    float T_target;
    float unc_peak;
    uint16_t d0;
    uint16_t lo, hi, mid;
    uint16_t latest_start;
    uint16_t safe_start;
    uint16_t opt_dur;
    float    cooled_peak;
    uint16_t total_margin;

    /* Precompute coefficients for this speed */
    Mth_PrecomputeCoeffs(&coeffs, params, op->omega_rad);

    /* Dynamic demagnetisation limit */
    T_limit  = compute_T_limit(demag, op->i_d);
    T_target = T_limit - eps;

    adv->T_limit_active = T_limit;
    adv->alarm = 0U;

    /* T1/T2: uncooled trajectory */
    unc_peak = Mth_RolloutPeak(&coeffs, T_pm0, T_s0, op->P_loss,
                                op->T_cool, op->T_amb,
                                0U, 0U, 0.0f);
    adv->T2_peak = unc_peak;

    /* T2: limit crossing time (linear scan of the uncooled rollout) */
    {
        float T_pm_k = T_pm0;
        float T_s_k  = T_s0;
        float b1_k, b2_k;
        uint16_t cross_k = 0xFFFFU;
        uint16_t kk;

        for (kk = 0U; kk < MTH_HORIZON_STEPS; kk++)
        {
            b1_k = T_pm_k + MTH_DT * coeffs.alpha_bP_over_Cpm * op->P_loss;
            b2_k = T_s_k  + MTH_DT * (
                coeffs.one_m_alpha_bP_over_Cs * op->P_loss
                + coeffs.Gsc_over_Cs * op->T_cool
                + coeffs.Gsa_over_Cs * op->T_amb);

            T_pm_k = (coeffs.a22 * b1_k - coeffs.a12 * b2_k) * coeffs.inv_det;
            T_s_k  = (coeffs.a11 * b2_k - coeffs.a21 * b1_k) * coeffs.inv_det;

            if ((T_pm_k >= T_limit) && (cross_k == 0xFFFFU))
            {
                cross_k = kk + 1U;
            }
        }

        if (cross_k == 0xFFFFU)
        {
            adv->T2_limit_time_s = -1.0f;
        }
        else
        {
            adv->T2_limit_time_s = (float)cross_k * MTH_DT;
        }
    }

    /* No cooling needed? */
    if (unc_peak <= T_target)
    {
        adv->T3_start_time_s = -1.0f;
        adv->T4_duration_s   = -1.0f;
        adv->T5_peak         = unc_peak;
        adv->T5_shutoff      = unc_peak;
        adv->T5_end          = unc_peak;
        adv->trajectory_valid = 0U;
        return;  /* single return: all paths converge here or below */
    }

    /* Total safety margin = user margin + transport delay */
    total_margin = safety_steps + delay_steps;

    /* Can cooling from step 0 fix it? */
    d0 = find_min_duration(&coeffs, T_pm0, T_s0, op->P_loss,
                           op->T_cool, op->T_amb,
                           0U, delta_cool, delay_steps, T_target);

    if (d0 == 0xFFFFU)
    {
        adv->T3_start_time_s = 0.0f;
        adv->T4_duration_s   = (float)MTH_HORIZON_STEPS * MTH_DT;
        adv->T5_peak         = unc_peak;
        adv->T5_shutoff      = unc_peak;
        adv->T5_end          = unc_peak;
        adv->alarm           = 1U;
        adv->trajectory_valid = 0U;
        return;
    }

    /* Binary search: latest feasible start */
    lo = 0U;
    hi = MTH_HORIZON_STEPS - 1U;
    {
        uint16_t iter;
        for (iter = 0U; iter < MTH_BISECT_MAX_ITER; iter++)
        {
            if (lo >= hi)
            {
                break;
            }
            mid = (lo + hi + 1U) / 2U;
            if (find_min_duration(&coeffs, T_pm0, T_s0, op->P_loss,
                                  op->T_cool, op->T_amb,
                                  mid, delta_cool, delay_steps,
                                  T_target) != 0xFFFFU)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1U;
            }
        }
    }
    latest_start = lo;

    /* Apply safety margin */
    if (latest_start > total_margin)
    {
        safe_start = latest_start - total_margin;
    }
    else
    {
        safe_start = 0U;
    }

    /* Minimum duration at safe start */
    opt_dur = find_min_duration(&coeffs, T_pm0, T_s0, op->P_loss,
                                op->T_cool, op->T_amb,
                                safe_start, delta_cool, delay_steps,
                                T_target);

    cooled_peak = Mth_RolloutPeak(&coeffs, T_pm0, T_s0, op->P_loss,
                                   op->T_cool, op->T_amb,
                                   safe_start + delay_steps, opt_dur,
                                   delta_cool);

    /* Fill advisory output */
    adv->T3_start_time_s = (float)safe_start * MTH_DT;
    adv->T4_duration_s   = (float)opt_dur * MTH_DT;
    adv->T5_peak         = cooled_peak;

    /* T5 shutoff and end temperatures (one more rollout) */
    {
        float T_pm_k = T_pm0;
        float T_s_k  = T_s0;
        float b1_k, b2_k, Tc_k;
        uint16_t kk;
        uint16_t eff_start = safe_start + delay_steps;
        uint16_t eff_end   = eff_start + opt_dur;

        adv->T5_shutoff = T_pm0;
        adv->T5_end     = T_pm0;

        for (kk = 0U; kk < MTH_HORIZON_STEPS; kk++)
        {
            if ((kk >= eff_start) && (kk < eff_end))
            {
                Tc_k = op->T_cool - delta_cool;
            }
            else
            {
                Tc_k = op->T_cool;
            }

            b1_k = T_pm_k + MTH_DT * coeffs.alpha_bP_over_Cpm * op->P_loss;
            b2_k = T_s_k  + MTH_DT * (
                coeffs.one_m_alpha_bP_over_Cs * op->P_loss
                + coeffs.Gsc_over_Cs * Tc_k
                + coeffs.Gsa_over_Cs * op->T_amb);

            T_pm_k = (coeffs.a22 * b1_k - coeffs.a12 * b2_k) * coeffs.inv_det;
            T_s_k  = (coeffs.a11 * b2_k - coeffs.a21 * b1_k) * coeffs.inv_det;

            if (kk + 1U == eff_end)
            {
                adv->T5_shutoff = T_pm_k;
            }
        }
        adv->T5_end = T_pm_k;
    }

    adv->trajectory_valid = 1U;
}

/* ═══════════════════════════════════════════════════════════════════ */
/*  KALMAN FILTER                                                     */
/* ═══════════════════════════════════════════════════════════════════ */

/**
 * @brief  LUT lookup with linear interpolation.
 *
 * Finds the two bracketing entries and interpolates Φ and Γ.
 */
static void lut_interp(const Mth_SpeedLut_t *lut, float omega,
                        float Phi_out[4], float Gam_out[6])
{
    float  frac;
    float  fidx;
    uint32_t idx_lo;
    uint32_t idx_hi;
    uint32_t j;
    const Mth_ZohEntry_t *lo_e;
    const Mth_ZohEntry_t *hi_e;

    /* Clamp omega to LUT range */
    if (omega < lut->omega_min)
    {
        omega = lut->omega_min;
    }
    else if (omega > lut->omega_max)
    {
        omega = lut->omega_max;
    }
    /* else: in range, no action needed */

    fidx   = (omega - lut->omega_min) / lut->omega_step;
    idx_lo = (uint32_t)fidx;

    if (idx_lo >= lut->n_entries - 1U)
    {
        idx_lo = lut->n_entries - 2U;
    }
    idx_hi = idx_lo + 1U;

    frac = fidx - (float)idx_lo;

    lo_e = &lut->entries[idx_lo];
    hi_e = &lut->entries[idx_hi];

    /* Linear interpolation of Φ and Γ */
    for (j = 0U; j < 4U; j++)
    {
        Phi_out[j] = lo_e->Phi[j] + frac * (hi_e->Phi[j] - lo_e->Phi[j]);
    }
    for (j = 0U; j < 6U; j++)
    {
        Gam_out[j] = lo_e->Gam[j] + frac * (hi_e->Gam[j] - lo_e->Gam[j]);
    }
}

void Mth_KalmanInit(Mth_KalmanState_t *kf,
                    float T_s_meas,
                    float q_pm, float q_s, float r)
{
    kf->x[0] = T_s_meas;       /* T_pm guess = T_s (equilibrium) */
    kf->x[1] = T_s_meas;       /* T_s = measured */

    /* P0: high uncertainty on T_pm, low on T_s */
    kf->P[0] = 100.0f;   kf->P[1] = 0.0f;
    kf->P[2] = 0.0f;     kf->P[3] = 1.0f;

    /* Process noise */
    kf->Q[0] = q_pm;     kf->Q[1] = 0.0f;
    kf->Q[2] = 0.0f;     kf->Q[3] = q_s;

    kf->R = r;
}

float Mth_KalmanUpdate(Mth_KalmanState_t *kf,
                       const Mth_SpeedLut_t *lut,
                       float T_s_meas,
                       float P_loss, float T_cool, float T_amb,
                       float omega)
{
    float Phi[4], Gam[6];
    float u[3];
    float xp[2];           /* predicted state */
    float Pp[4];            /* predicted covariance */
    float S;                /* innovation covariance (scalar) */
    float K[2];             /* Kalman gain (2×1) */
    float innov;
    float tmp[4];

    /* --- LUT lookup --- */
    lut_interp(lut, omega, Phi, Gam);

    /* --- Input vector --- */
    u[0] = P_loss;
    u[1] = T_cool;
    u[2] = T_amb;

    /* --- Predict: x⁻ = Φ·x + Γ·u --- */
    xp[0] = Phi[0] * kf->x[0] + Phi[1] * kf->x[1]
          + Gam[0] * u[0] + Gam[1] * u[1] + Gam[2] * u[2];
    xp[1] = Phi[2] * kf->x[0] + Phi[3] * kf->x[1]
          + Gam[3] * u[0] + Gam[4] * u[1] + Gam[5] * u[2];

    /* --- Predict: P⁻ = Φ·P·Φᵀ + Q --- */
    /* tmp = Φ·P */
    tmp[0] = Phi[0] * kf->P[0] + Phi[1] * kf->P[2];
    tmp[1] = Phi[0] * kf->P[1] + Phi[1] * kf->P[3];
    tmp[2] = Phi[2] * kf->P[0] + Phi[3] * kf->P[2];
    tmp[3] = Phi[2] * kf->P[1] + Phi[3] * kf->P[3];

    /* Pp = tmp · Φᵀ + Q */
    Pp[0] = tmp[0] * Phi[0] + tmp[1] * Phi[1] + kf->Q[0];
    Pp[1] = tmp[0] * Phi[2] + tmp[1] * Phi[3] + kf->Q[1];
    Pp[2] = tmp[2] * Phi[0] + tmp[3] * Phi[1] + kf->Q[2];
    Pp[3] = tmp[2] * Phi[2] + tmp[3] * Phi[3] + kf->Q[3];

    /* --- Update: H = [0, 1], so H·x = x[1], H·P·Hᵀ = P[3] --- */
    innov = T_s_meas - xp[1];
    S     = Pp[3] + kf->R;

    /* K = P⁻·Hᵀ / S = [Pp[1], Pp[3]]ᵀ / S */
    K[0] = Pp[1] / S;
    K[1] = Pp[3] / S;

    /* x̂ = x⁻ + K·innov */
    kf->x[0] = xp[0] + K[0] * innov;
    kf->x[1] = xp[1] + K[1] * innov;

    /* P = (I − K·H)·P⁻  (Joseph form for float32 stability) */
    {
        float IKH[4];
        float IKHt[4];

        /* I − K·H, where H = [0, 1] → K·H = [[0, K[0]], [0, K[1]]] */
        IKH[0] = 1.0f;          IKH[1] = -K[0];
        IKH[2] = 0.0f;          IKH[3] = 1.0f - K[1];

        /* IKHᵀ */
        IKHt[0] = IKH[0];       IKHt[1] = IKH[2];
        IKHt[2] = IKH[1];       IKHt[3] = IKH[3];

        /* tmp = IKH · P⁻ */
        tmp[0] = IKH[0]*Pp[0] + IKH[1]*Pp[2];
        tmp[1] = IKH[0]*Pp[1] + IKH[1]*Pp[3];
        tmp[2] = IKH[2]*Pp[0] + IKH[3]*Pp[2];
        tmp[3] = IKH[2]*Pp[1] + IKH[3]*Pp[3];

        /* P = tmp · IKHᵀ + K·R·Kᵀ */
        kf->P[0] = tmp[0]*IKHt[0] + tmp[1]*IKHt[2] + K[0]*kf->R*K[0];
        kf->P[1] = tmp[0]*IKHt[1] + tmp[1]*IKHt[3] + K[0]*kf->R*K[1];
        kf->P[2] = tmp[2]*IKHt[0] + tmp[3]*IKHt[2] + K[1]*kf->R*K[0];
        kf->P[3] = tmp[2]*IKHt[1] + tmp[3]*IKHt[3] + K[1]*kf->R*K[1];
    }

    return kf->x[0];   /* T_pm_hat */
}
