/**
 * @file  motor_thermal_ecu_final.c
 * @brief Production MISRA-C:2012 implementation.
 */

#include "motor_thermal_ecu_final.h"
#include <math.h>  /* fabsf sqrtf expf coshf sinhf cosf sinf */

/* ═══════════════════════════════════════════════════════════════ */
/*  INTERNAL — sinhc-based 2×2 matrix exponential                 */
/* ═══════════════════════════════════════════════════════════════ */

/**
 * @brief  sinhc(x) = sinh(x)/x,  singularity-free at x = 0.
 *
 * For |x²| < 1e-4  use Taylor:  1 + x²/6 + x⁴/120.
 * This avoids the 0/0 indeterminate form.
 */
static float sinhc_f32(float x)
{
    float x2 = x * x;
    float result;

    if (x2 < 1e-4f)
    {
        result = 1.0f + x2 * (0.166666667f + x2 * 0.008333333f);
    }
    else
    {
        result = sinhf(x) / x;
    }
    return result;
}

/**
 * @brief  sinc(x) = sin(x)/x,  for the complex-eigenvalue branch.
 */
static float sinc_f32(float x)
{
    float x2 = x * x;
    float result;

    if (x2 < 1e-4f)
    {
        result = 1.0f - x2 * (0.166666667f - x2 * 0.008333333f);
    }
    else
    {
        result = sinf(x) / x;
    }
    return result;
}

/**
 * @brief  Singularity-free 2×2 matrix exponential.
 *
 * Given A·dt (2×2 row-major), computes E = e^{A·dt}.
 *
 * Method (Sylvester / Cayley-Hamilton via sinhc):
 *   s  = tr(A·dt) / 2
 *   B  = A·dt − s·I            (traceless)
 *   q² = B₀₀² + B₀₁·B₁₀      (= −det B)
 *
 *   q² > 0  (real distinct):  c₀ = cosh(q),  c₁ = sinhc(q)
 *   q² < 0  (complex conj.):  p = √(−q²), c₀ = cos(p), c₁ = sinc(p)
 *   q² ≈ 0  (repeated):       c₀ = 1, c₁ = 1
 *
 *   E = eˢ · (c₀·I + c₁·B)
 *
 * No division by (λ₁−λ₂) anywhere — singularity-free.
 */
static void expm2x2(const float Adt[4], float E[4])
{
    float s, es;
    float B0, B1, B2, B3;
    float q2, c0, c1;

    s = 0.5f * (Adt[0] + Adt[3]);
    B0 = Adt[0] - s;   B1 = Adt[1];
    B2 = Adt[2];        B3 = Adt[3] - s;   /* B3 = -B0 */

    q2 = B0 * B0 + B1 * B2;
    es = expf(s);

    if (q2 > 1e-8f)
    {
        float q = sqrtf(q2);
        c0 = coshf(q);
        c1 = sinhc_f32(q);
    }
    else if (q2 < -1e-8f)
    {
        float p = sqrtf(-q2);
        c0 = cosf(p);
        c1 = sinc_f32(p);
    }
    else
    {
        c0 = 1.0f;
        c1 = 1.0f;
    }

    E[0] = es * (c0 + c1 * B0);
    E[1] = es * (c1 * B1);
    E[2] = es * (c1 * B2);
    E[3] = es * (c0 + c1 * B3);
}

/* ── 2×2 helpers ──────────────────────────────────────────────── */

static void inv2x2(const float M[4], float I[4])
{
    float d = 1.0f / (M[0]*M[3] - M[1]*M[2]);
    I[0] =  M[3]*d;  I[1] = -M[1]*d;
    I[2] = -M[2]*d;  I[3] =  M[0]*d;
}

static float G_ag(const Mth_Params_t *p, float w)
{
    return p->G_ag0 + p->G_ag1 * powf(fabsf(w), p->gamma);
}

/* ═══════════════════════════════════════════════════════════════ */
/*  LUT BUILD  (called once at startup)                           */
/* ═══════════════════════════════════════════════════════════════ */

void Mth_BuildLut(Mth_Lut_t *lut, const Mth_Params_t *p)
{
    uint32_t i;

    lut->n     = MTH_LUT_N;
    lut->w_min = 0.0f;
    lut->w_max = 15000.0f * (2.0f * MTH_PI_F / 60.0f);
    lut->w_step = (lut->w_max - lut->w_min) / (float)(MTH_LUT_N - 1U);

    for (i = 0U; i < MTH_LUT_N; i++)
    {
        float w = lut->w_min + (float)i * lut->w_step;
        float g = G_ag(p, w);

        float Adt[4];
        Adt[0] = (-g / p->C_pm) * MTH_DT;
        Adt[1] = ( g / p->C_pm) * MTH_DT;
        Adt[2] = ( g / p->C_s)  * MTH_DT;
        Adt[3] = (-(g + p->G_sc + p->G_sa) / p->C_s) * MTH_DT;

        float Phi[4];
        expm2x2(Adt, Phi);

        /* A (un-scaled) and its inverse */
        float A[4], Ainv[4];
        A[0] = Adt[0]/MTH_DT; A[1] = Adt[1]/MTH_DT;
        A[2] = Adt[2]/MTH_DT; A[3] = Adt[3]/MTH_DT;
        inv2x2(A, Ainv);

        /* Φ−I */
        float PhiI[4];
        PhiI[0] = Phi[0]-1.0f; PhiI[1] = Phi[1];
        PhiI[2] = Phi[2];      PhiI[3] = Phi[3]-1.0f;

        /* Ainv · (Φ−I) */
        float AP[4];
        AP[0] = Ainv[0]*PhiI[0] + Ainv[1]*PhiI[2];
        AP[1] = Ainv[0]*PhiI[1] + Ainv[1]*PhiI[3];
        AP[2] = Ainv[2]*PhiI[0] + Ainv[3]*PhiI[2];
        AP[3] = Ainv[2]*PhiI[1] + Ainv[3]*PhiI[3];

        /* B matrix (2×3) */
        float B[6];
        B[0] = p->alpha * p->beta / p->C_pm;
        B[1] = 0.0f;
        B[2] = 0.0f;
        B[3] = (1.0f - p->alpha) * p->beta / p->C_s;
        B[4] = p->G_sc / p->C_s;
        B[5] = p->G_sa / p->C_s;

        /* Γ = AP · B  (2×2 · 2×3) */
        float Gam[6];
        uint32_t j;
        for (j = 0U; j < 3U; j++)
        {
            Gam[0U*3U+j] = AP[0]*B[0U*3U+j] + AP[1]*B[1U*3U+j];
            Gam[1U*3U+j] = AP[2]*B[0U*3U+j] + AP[3]*B[1U*3U+j];
        }

        lut->e[i].omega = w;
        {
            uint32_t k;
            for (k = 0U; k < 4U; k++) { lut->e[i].Phi[k] = Phi[k]; }
            for (k = 0U; k < 6U; k++) { lut->e[i].Gam[k] = Gam[k]; }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════ */
/*  SEMI-IMPLICIT COEFFICIENTS                                    */
/* ═══════════════════════════════════════════════════════════════ */

void Mth_PreCoeffs(Mth_Coeffs_t *c, const Mth_Params_t *p, float w)
{
    float g = G_ag(p, w);

    c->a11 = 1.0f + MTH_DT * g / p->C_pm;
    c->a12 = -MTH_DT * g / p->C_pm;
    c->a21 = -MTH_DT * g / p->C_s;
    c->a22 = 1.0f + MTH_DT / p->C_s * (g + p->G_sc + p->G_sa);
    c->inv_det = 1.0f / (c->a11 * c->a22 - c->a12 * c->a21);

    c->c_alpha_Cpm  = p->alpha * p->beta / p->C_pm;
    c->c_1malpha_Cs = (1.0f - p->alpha) * p->beta / p->C_s;
    c->c_Gsc_Cs     = p->G_sc / p->C_s;
    c->c_Gsa_Cs     = p->G_sa / p->C_s;
}

/* ═══════════════════════════════════════════════════════════════ */
/*  SINGLE ROLLOUT  (with pipe dead-time)                         */
/* ═══════════════════════════════════════════════════════════════ */

float Mth_RolloutPeak(const Mth_Coeffs_t *c,
                      float Tp, float Ts,
                      float P, float Tc_base, float Ta,
                      uint16_t cs, uint16_t cd,
                      float dc, uint16_t delay)
{
    float peak = Tp;
    uint16_t es = cs + delay;     /* effective cooling start */
    uint16_t ee = es + cd;        /* effective cooling end   */
    uint16_t k;

    for (k = 0U; k < MTH_HORIZON; k++)
    {
        float Tc = ((k >= es) && (k < ee)) ? (Tc_base - dc) : Tc_base;

        float b1 = Tp + MTH_DT * c->c_alpha_Cpm * P;
        float b2 = Ts + MTH_DT * (c->c_1malpha_Cs * P
                    + c->c_Gsc_Cs * Tc + c->c_Gsa_Cs * Ta);

        Tp = (c->a22 * b1 - c->a12 * b2) * c->inv_det;
        Ts = (c->a11 * b2 - c->a21 * b1) * c->inv_det;

        if (Tp > peak) { peak = Tp; }
    }
    return peak;
}

/* ═══════════════════════════════════════════════════════════════ */
/*  DOUBLE-BISECTION ADVISORY                                     */
/* ═══════════════════════════════════════════════════════════════ */

static uint16_t min_dur(const Mth_Coeffs_t *c,
                        float Tp0, float Ts0,
                        float P, float Tc, float Ta,
                        uint16_t s, float dc, uint16_t dly,
                        float Tgt)
{
    uint16_t mx = MTH_HORIZON - s;
    uint16_t result;

    if (Mth_RolloutPeak(c, Tp0, Ts0, P, Tc, Ta, s, mx, dc, dly) > Tgt)
    {
        result = 0xFFFFU;
    }
    else
    {
        uint16_t lo = 1U, hi = mx, mid;
        uint16_t it;
        for (it = 0U; it < MTH_BISECT_ITER; it++)
        {
            if (lo >= hi) { break; }
            mid = (lo + hi) / 2U;
            if (Mth_RolloutPeak(c, Tp0, Ts0, P, Tc, Ta,
                                s, mid, dc, dly) <= Tgt)
            { hi = mid; } else { lo = mid + 1U; }
        }
        result = lo;
    }
    return result;
}

void Mth_Advisory(Mth_Advisory_t *out,
                  const Mth_Params_t *p,
                  const Mth_Op_t *op,
                  const Mth_Demag_t *demag,
                  float Tp0, float Ts0,
                  float eps, uint16_t safety,
                  float dc, uint16_t delay)
{
    Mth_Coeffs_t c;
    float Tlim, Tgt, unc_pk;
    uint16_t d0, lo, hi, mid, it, latest, safe_s, opt_d;
    uint16_t tot_margin;

    Mth_PreCoeffs(&c, p, op->omega);

    Tlim = Mth_DemagClamp(demag, op->i_d);
    Tgt  = Tlim - eps;
    out->T_limit_active = Tlim;
    out->i_d_limit_A    = Mth_DemagIdLimit(demag, Tp0, eps);
    out->alarm = 0U;

    /* T1/T2 — uncooled */
    unc_pk = Mth_RolloutPeak(&c, Tp0, Ts0, op->P_loss,
                              op->T_cool, op->T_amb,
                              0U, 0U, 0.0f, 0U);
    out->T2_peak = unc_pk;

    /* T2 — limit crossing time (inline scan) */
    {
        float tp = Tp0, ts = Ts0;
        uint16_t cross = 0xFFFFU;
        uint16_t k;
        for (k = 0U; k < MTH_HORIZON; k++)
        {
            float b1 = tp + MTH_DT * c.c_alpha_Cpm * op->P_loss;
            float b2 = ts + MTH_DT * (c.c_1malpha_Cs * op->P_loss
                        + c.c_Gsc_Cs * op->T_cool
                        + c.c_Gsa_Cs * op->T_amb);
            tp = (c.a22*b1 - c.a12*b2) * c.inv_det;
            ts = (c.a11*b2 - c.a21*b1) * c.inv_det;
            if ((tp >= Tlim) && (cross == 0xFFFFU)) { cross = k+1U; }
        }
        out->T2_cross_s = (cross == 0xFFFFU) ? -1.0f
                          : (float)cross * MTH_DT;
    }

    if (unc_pk <= Tgt)
    {
        out->T3_cmd_s = -1.0f;   out->T4_dur_s = -1.0f;
        out->T5_peak = unc_pk;   out->T5_off = unc_pk;
        out->T5_end = unc_pk;
        return;
    }

    tot_margin = safety + delay;

    d0 = min_dur(&c, Tp0, Ts0, op->P_loss, op->T_cool, op->T_amb,
                 0U, dc, delay, Tgt);

    if (d0 == 0xFFFFU)
    {
        out->T3_cmd_s = 0.0f;
        out->T4_dur_s = (float)MTH_HORIZON * MTH_DT;
        out->T5_peak = unc_pk; out->T5_off = unc_pk; out->T5_end = unc_pk;
        out->alarm = 1U;
        return;
    }

    /* Latest feasible start */
    lo = 0U;  hi = MTH_HORIZON - 1U;
    for (it = 0U; it < MTH_BISECT_ITER; it++)
    {
        if (lo >= hi) { break; }
        mid = (lo + hi + 1U) / 2U;
        if (min_dur(&c, Tp0, Ts0, op->P_loss, op->T_cool, op->T_amb,
                    mid, dc, delay, Tgt) != 0xFFFFU)
        { lo = mid; } else { hi = mid - 1U; }
    }
    latest = lo;
    safe_s = (latest > (uint16_t)tot_margin) ? (uint16_t)(latest - tot_margin) : 0U;

    opt_d = min_dur(&c, Tp0, Ts0, op->P_loss, op->T_cool, op->T_amb,
                    safe_s, dc, delay, Tgt);

    out->T3_cmd_s = (float)safe_s * MTH_DT;
    out->T4_dur_s = (float)opt_d * MTH_DT;

    /* T5 — post-cooling trajectory */
    {
        float tp = Tp0, ts = Ts0, pk5 = Tp0;
        uint16_t es5 = safe_s + delay;
        uint16_t ee5 = es5 + opt_d;
        uint16_t k;
        out->T5_off = Tp0;
        for (k = 0U; k < MTH_HORIZON; k++)
        {
            float Tc = ((k >= es5) && (k < ee5))
                       ? (op->T_cool - dc) : op->T_cool;
            float b1 = tp + MTH_DT * c.c_alpha_Cpm * op->P_loss;
            float b2 = ts + MTH_DT * (c.c_1malpha_Cs * op->P_loss
                        + c.c_Gsc_Cs * Tc + c.c_Gsa_Cs * op->T_amb);
            tp = (c.a22*b1 - c.a12*b2) * c.inv_det;
            ts = (c.a11*b2 - c.a21*b1) * c.inv_det;
            if (tp > pk5) { pk5 = tp; }
            if (k + 1U == ee5) { out->T5_off = tp; }
        }
        out->T5_peak = pk5;
        out->T5_end  = tp;
    }
}

/* ═══════════════════════════════════════════════════════════════ */
/*  3-STATE EKF  [T_pm, T_s, λ_pm]                               */
/* ═══════════════════════════════════════════════════════════════ */

/** LUT interpolation — finds bracketing entries and lerps Φ, Γ. */
static void lut_lerp(const Mth_Lut_t *l, float w,
                     float Phi[4], float Gam[6])
{
    float f;
    uint32_t lo, hi, j;

    if (w < l->w_min) { w = l->w_min; }
    if (w > l->w_max) { w = l->w_max; }
    f  = (w - l->w_min) / l->w_step;
    lo = (uint32_t)f;
    if (lo >= l->n - 1U) { lo = l->n - 2U; }
    hi = lo + 1U;
    f -= (float)lo;

    for (j = 0U; j < 4U; j++)
    { Phi[j] = l->e[lo].Phi[j] + f*(l->e[hi].Phi[j] - l->e[lo].Phi[j]); }
    for (j = 0U; j < 6U; j++)
    { Gam[j] = l->e[lo].Gam[j] + f*(l->e[hi].Gam[j] - l->e[lo].Gam[j]); }
}

void Mth_EKF_Init(Mth_EKF_t *kf, float Ts_meas,
                  const Mth_FluxParams_t *fp,
                  const float Q9[9], const float R4[4],
                  float omega_min)
{
    uint32_t i;

    kf->x[0] = Ts_meas;            /* T_pm guess = T_s  */
    kf->x[1] = Ts_meas;            /* T_s = measured     */
    kf->x[2] = fp->lambda_ref;     /* λ_pm = nominal     */

    /* P0 = diag(100, 1, 0.001) */
    for (i = 0U; i < 9U; i++) { kf->P[i] = 0.0f; }
    kf->P[0] = 100.0f;  kf->P[4] = 1.0f;  kf->P[8] = 0.001f;

    for (i = 0U; i < 9U; i++) { kf->Q[i] = Q9[i]; }
    for (i = 0U; i < 4U; i++) { kf->R[i] = R4[i]; }

    kf->lambda_ref = fp->lambda_ref;
    kf->alpha_lam  = fp->alpha_lambda;
    kf->T_ref_lam  = fp->T_ref;
    kf->omega_min  = omega_min;
}

float Mth_EKF_Step(Mth_EKF_t *kf,
                   const Mth_Lut_t *lut,
                   float Ts_meas, float lambda_obs,
                   uint8_t flux_valid,
                   float P_loss, float Tc, float Ta,
                   float omega)
{
    float Phi2[4], Gam2[6];
    float u[3];
    float xp[3];      /* predicted state */
    float F[9];        /* 3×3 Jacobian */
    float Pp[9];       /* predicted cov */
    uint8_t n_obs;     /* 1 or 2 measurements */
    float result;

    /* ── 1. LUT lookup ─────────────────────────────────────────── */
    lut_lerp(lut, omega, Phi2, Gam2);
    u[0] = P_loss;  u[1] = Tc;  u[2] = Ta;

    /* ── 2. Predict state ──────────────────────────────────────── */
    /* Thermal: [T_pm', T_s'] = Φ·[T_pm, T_s] + Γ·u */
    xp[0] = Phi2[0]*kf->x[0] + Phi2[1]*kf->x[1]
          + Gam2[0]*u[0] + Gam2[1]*u[1] + Gam2[2]*u[2];
    xp[1] = Phi2[2]*kf->x[0] + Phi2[3]*kf->x[1]
          + Gam2[3]*u[0] + Gam2[4]*u[1] + Gam2[5]*u[2];

    /* Flux: λ_pm' = λ_pm + α_λ·λ_ref·(T_pm' − T_pm) */
    {
        float dTp = xp[0] - kf->x[0];
        xp[2] = kf->x[2] + kf->alpha_lam * kf->lambda_ref * dTp;
    }

    /* ── 3. Jacobian F = ∂f/∂x ─────────────────────────────────── */
    F[0] = Phi2[0];  F[1] = Phi2[1];  F[2] = 0.0f;
    F[3] = Phi2[2];  F[4] = Phi2[3];  F[5] = 0.0f;
    F[6] = kf->alpha_lam * kf->lambda_ref * (Phi2[0] - 1.0f);
    F[7] = kf->alpha_lam * kf->lambda_ref *  Phi2[1];
    F[8] = 1.0f;

    /* ── 4. Predicted covariance Pp = F·P·Fᵀ + Q  (3×3) ─────── */
    {
        float FP[9];
        uint32_t r, c, k;
        /* FP = F · P */
        for (r = 0U; r < 3U; r++) {
            for (c = 0U; c < 3U; c++) {
                float s = 0.0f;
                for (k = 0U; k < 3U; k++) {
                    s += F[r*3U+k] * kf->P[k*3U+c];
                }
                FP[r*3U+c] = s;
            }
        }
        /* Pp = FP · Fᵀ + Q */
        for (r = 0U; r < 3U; r++) {
            for (c = 0U; c < 3U; c++) {
                float s = 0.0f;
                for (k = 0U; k < 3U; k++) {
                    s += FP[r*3U+k] * F[c*3U+k]; /* Fᵀ col=row */
                }
                Pp[r*3U+c] = s + kf->Q[r*3U+c];
            }
        }
    }

    /* ── 5. Measurement update ─────────────────────────────────── */
    /*
     * y = [T_s_meas]            always present
     * y = [T_s_meas, λ_obs]     when flux_valid && |ω|>ω_min
     *
     * H₁ = [0 1 0]
     * H₂ = [0 0 1]
     *
     * We perform sequential scalar updates (one per measurement)
     * for simplicity and numerical robustness in float32.
     */
    n_obs = (flux_valid != 0U && fabsf(omega) > kf->omega_min)
            ? 2U : 1U;

    /* Copy predicted state into x, cov into P */
    kf->x[0] = xp[0];  kf->x[1] = xp[1];  kf->x[2] = xp[2];
    {
        uint32_t i;
        for (i = 0U; i < 9U; i++) { kf->P[i] = Pp[i]; }
    }

    /* ── Update 1: T_s measurement  (H = [0 1 0]) ─────────────── */
    {
        float innov = Ts_meas - kf->x[1];
        float S = kf->P[4] + kf->R[0];  /* P[1,1] + R_Ts */
        float K[3];
        float IKH[9];
        float Pnew[9];
        uint32_t r, c, k;

        K[0] = kf->P[1] / S;   /* P[0,1] / S */
        K[1] = kf->P[4] / S;   /* P[1,1] / S */
        K[2] = kf->P[7] / S;   /* P[2,1] / S */

        kf->x[0] += K[0] * innov;
        kf->x[1] += K[1] * innov;
        kf->x[2] += K[2] * innov;

        /* Joseph: P = (I−K·H)·P·(I−K·H)ᵀ + K·R·Kᵀ */
        /* I − K·H, H=[0 1 0] → KH row r = [0, K[r], 0] */
        for (r = 0U; r < 3U; r++) {
            for (c = 0U; c < 3U; c++) {
                IKH[r*3U+c] = ((r == c) ? 1.0f : 0.0f)
                             - ((c == 1U) ? K[r] : 0.0f);
            }
        }
        /* Pnew = IKH · P · IKHᵀ + K·R_Ts·Kᵀ */
        {
            float tmp[9];
            for (r = 0U; r < 3U; r++) {
                for (c = 0U; c < 3U; c++) {
                    float s = 0.0f;
                    for (k = 0U; k < 3U; k++) {
                        s += IKH[r*3U+k] * kf->P[k*3U+c];
                    }
                    tmp[r*3U+c] = s;
                }
            }
            for (r = 0U; r < 3U; r++) {
                for (c = 0U; c < 3U; c++) {
                    float s = 0.0f;
                    for (k = 0U; k < 3U; k++) {
                        s += tmp[r*3U+k] * IKH[c*3U+k];
                    }
                    Pnew[r*3U+c] = s + K[r] * kf->R[0] * K[c];
                }
            }
        }
        for (r = 0U; r < 9U; r++) { kf->P[r] = Pnew[r]; }
    }

    /* ── Update 2: flux measurement  (H = [0 0 1]) ────────────── */
    if (n_obs == 2U)
    {
        float innov = lambda_obs - kf->x[2];
        float S = kf->P[8] + kf->R[3];  /* P[2,2] + R_lam */
        float K[3];
        float IKH[9];
        float Pnew[9];
        uint32_t r, c, k;

        K[0] = kf->P[2] / S;
        K[1] = kf->P[5] / S;
        K[2] = kf->P[8] / S;

        kf->x[0] += K[0] * innov;
        kf->x[1] += K[1] * innov;
        kf->x[2] += K[2] * innov;

        for (r = 0U; r < 3U; r++) {
            for (c = 0U; c < 3U; c++) {
                IKH[r*3U+c] = ((r == c) ? 1.0f : 0.0f)
                             - ((c == 2U) ? K[r] : 0.0f);
            }
        }
        {
            float tmp[9];
            for (r = 0U; r < 3U; r++) {
                for (c = 0U; c < 3U; c++) {
                    float s = 0.0f;
                    for (k = 0U; k < 3U; k++) {
                        s += IKH[r*3U+k] * kf->P[k*3U+c];
                    }
                    tmp[r*3U+c] = s;
                }
            }
            for (r = 0U; r < 3U; r++) {
                for (c = 0U; c < 3U; c++) {
                    float s = 0.0f;
                    for (k = 0U; k < 3U; k++) {
                        s += tmp[r*3U+k] * IKH[c*3U+k];
                    }
                    Pnew[r*3U+c] = s + K[r] * kf->R[3] * K[c];
                }
            }
        }
        for (r = 0U; r < 9U; r++) { kf->P[r] = Pnew[r]; }
    }

    result = kf->x[0];  /* T_pm_hat */
    return result;
}

/* ═══════════════════════════════════════════════════════════════ */
/*  GATED COAST-DOWN RLS AGING ESTIMATOR                          */
/* ═══════════════════════════════════════════════════════════════ */

void Mth_RLS_Init(Mth_RLS_t *rls,
                  float G_sc_nom, float G_ag0_nom,
                  float lambda_forget,
                  float i_thresh, uint16_t cd_min)
{
    rls->theta[0] = logf(G_sc_nom);
    rls->theta[1] = logf(G_ag0_nom);
    rls->P_rls[0] = 0.01f; rls->P_rls[1] = 0.0f;
    rls->P_rls[2] = 0.0f;  rls->P_rls[3] = 0.01f;
    rls->lambda    = lambda_forget;
    rls->G_sc_nom  = G_sc_nom;
    rls->G_ag0_nom = G_ag0_nom;
    rls->cd_count  = 0U;
    rls->i_thresh  = i_thresh;
    rls->cd_min    = cd_min;
}

uint8_t Mth_RLS_Step(Mth_RLS_t *rls,
                     Mth_Params_t *p,
                     float Tp_hat, float Ts_hat,
                     float Ts_meas, float Tc,
                     float omega, float id, float iq,
                     float C_s, float dt)
{
    uint8_t updated = 0U;

    (void)omega;  /* Reserved for speed-dependent aging; silences -Wunused */

    /* Coast-down detection */
    if ((fabsf(id) < rls->i_thresh) && (fabsf(iq) < rls->i_thresh))
    {
        rls->cd_count++;
    }
    else
    {
        rls->cd_count = 0U;
    }

    if (rls->cd_count >= rls->cd_min)
    {
        /*
         * During coast-down (P≈0), the stator prediction error
         * depends primarily on G_sc and G_ag:
         *
         *   ε = Ts_meas − Ts_hat
         *
         *   ∂ε/∂(ln G_sc)  ≈ −dt · G_sc · (Ts − Tc) / C_s
         *   ∂ε/∂(ln G_ag0) ≈  dt · G_ag0 · (Tp − Ts) / C_s
         */
        float G_sc_cur  = expf(rls->theta[0]);
        float G_ag0_cur = expf(rls->theta[1]);

        float phi[2];
        phi[0] = -dt * G_sc_cur  * (Ts_hat - Tc) / C_s;
        phi[1] =  dt * G_ag0_cur * (Tp_hat - Ts_hat) / C_s;

        float eps = Ts_meas - Ts_hat;

        /* RLS gain: K = P·φ / (λ + φᵀ·P·φ) */
        float Pphi[2];
        Pphi[0] = rls->P_rls[0]*phi[0] + rls->P_rls[1]*phi[1];
        Pphi[1] = rls->P_rls[2]*phi[0] + rls->P_rls[3]*phi[1];

        float denom = rls->lambda
                    + phi[0]*Pphi[0] + phi[1]*Pphi[1];

        if (fabsf(denom) > 1e-10f)
        {
            float K[2];
            K[0] = Pphi[0] / denom;
            K[1] = Pphi[1] / denom;

            /* Update θ */
            rls->theta[0] += K[0] * eps;
            rls->theta[1] += K[1] * eps;

            /* Update P = (P − K·φᵀ·P) / λ */
            float Pnew[4];
            Pnew[0] = (rls->P_rls[0] - K[0]*Pphi[0]) / rls->lambda;
            Pnew[1] = (rls->P_rls[1] - K[0]*Pphi[1]) / rls->lambda;
            Pnew[2] = (rls->P_rls[2] - K[1]*Pphi[0]) / rls->lambda;
            Pnew[3] = (rls->P_rls[3] - K[1]*Pphi[1]) / rls->lambda;

            rls->P_rls[0] = Pnew[0]; rls->P_rls[1] = Pnew[1];
            rls->P_rls[2] = Pnew[2]; rls->P_rls[3] = Pnew[3];

            /* Apply to thermal params */
            p->G_sc  = expf(rls->theta[0]);
            p->G_ag0 = expf(rls->theta[1]);

            updated = 1U;
        }
    }

    return updated;
}
