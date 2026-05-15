// ============================================================
// TH-LIF v1 — Tunneling-Hazard LIF CUDA kernels (forward + backward)
// ============================================================
// Source: Dolphin notebook Cell 65 (PP-SNN v4, 2026-04-29)
// Patent: TURKPATENT 2026/007632 (neuron family) + 2026/004809 (hybrid system)
// Author: Can Dikmen
//
// Math:
//   lambda(v) = lam0 * exp(-kappa * (phi0 - eta * v))     // tunneling hazard
//   p         = 1 - exp(-lambda)                          // spike probability
//   spike     = 1 if p >= 0.5 else 0                      // deterministic threshold
//   v_next    = beta*v + I_in - spike*V_th                // leaky integration + soft reset
//
// Backward (natural hazard gradient, no surrogate):
//   dp/dv     = exp(-lambda) * lambda * kappa * eta
//   gs[t]     = (g_out[t] + gv) * dp_dv[t]   (clamped to +/- 1)
//   gi[t]     = gs[t]
//   gv        = gs * beta                    (zeroed if spike, reset breaks BPTT)
//
// Tensor layout: (B, T, N) row-major, float32 only (AMP forbidden).
// Constraints:
//   - exponent clamped to [-10, 10] to prevent overflow in expf
//   - dp/dv clamped to [0, 2]     to prevent gradient explosion
//   - gs    clamped to [-1, 1]    to prevent BPTT divergence
// ============================================================

extern "C" __global__
void thlif_fwd(const float* in, float* spk, float* dp_dv,
               int B, int T, int N,
               float beta, float threshold,
               float lam0, float kappa, float eta, float phi0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / N, n = idx % N;
    if (b >= B || n >= N) return;
    float v = 0;
    for (int t = 0; t < T; t++) {
        int o = b * T * N + t * N + n;
        v = beta * v + in[o];
        float exponent = -kappa * (phi0 - eta * v);
        if (exponent > 10.f) exponent = 10.f;
        if (exponent < -10.f) exponent = -10.f;
        float lam = lam0 * expf(exponent);
        float exp_neg_lam = expf(-lam);
        float p = 1.f - exp_neg_lam;
        float grad = exp_neg_lam * lam * kappa * eta;
        if (grad > 2.f) grad = 2.f;
        if (grad < 0.f) grad = 0.f;
        dp_dv[o] = grad;
        float s = (p >= 0.5f) ? 1.f : 0.f;
        spk[o] = s;
        v -= s * threshold;
    }
}

extern "C" __global__
void thlif_bwd(const float* go, const float* spk, const float* dp_dv,
               float* gi, int B, int T, int N,
               float beta, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / N, n = idx % N;
    if (b >= B || n >= N) return;
    float gv = 0;
    for (int t = T - 1; t >= 0; t--) {
        int o = b * T * N + t * N + n;
        float s = spk[o];
        float surr = dp_dv[o];
        float gs = (go[o] + gv) * surr;
        if (gs > 1.f) gs = 1.f;
        if (gs < -1.f) gs = -1.f;
        gi[o] = gs;
        gv = gs * beta;
        if (s > 0.5f) gv = 0.f;       // reset breaks gradient through time
    }
}
