"""
th_lif_v1.py — TH-LIF v1 CUDA kernel Python wrapper.

Importable as a module from cuda/wrappers/. The CUDA kernel source is
embedded inline in this file for self-containment (no path dependencies in
Colab). The canonical .cu version lives at cuda/kernels/th_lif_v1.cu;
this file must stay in sync with that source.

Patent: TÜRKPATENT 2026/007632 (neuron family) + 2026/004809 (hybrid system).
Author: Can Dikmen.

Usage:
    from th_lif_v1 import CUDATHLIF, _THLIFFunc

    layer = CUDATHLIF(n_in=64, n_out=128, beta=0.95, threshold=1.0,
                      lam0=0.1, kappa=2.0, eta=1.5, phi0=1.0)
    x = torch.randn(B, T, 64, device="cuda")   # float32 mandatory
    spk = layer(x)                              # (B, T, 128)

Constraints:
    - float32 only. AMP/half-precision forbidden (expf overflow).
    - Tensor shape (B, T, N), CUDA device.
    - Hazard parameters (lam0, kappa, eta, phi0) are FIXED per layer instance.
      The PP-SNN v4 / battery SoH / NeuroSentinel pipelines all use fixed
      values; learnable hazard parameters are reserved for L-TH-LIF (separate
      module), not this kernel.
"""

import numpy as np
import torch
import torch.nn as nn
import cupy as cp


# ============================================================
# 1. Zero-copy torch <-> cupy bridge
# ============================================================
def _t2c(t):
    """Zero-copy torch tensor to cupy ndarray. Autograd-safe."""
    tc = t.contiguous()
    mem = cp.cuda.UnownedMemory(tc.data_ptr(), tc.numel() * tc.element_size(), tc)
    return cp.ndarray(tc.shape, cp.float32, cp.cuda.MemoryPointer(mem, 0),
                      strides=tuple(s * 4 for s in tc.stride()))


# ============================================================
# 2. CUDA kernels (inline copy of kernels/th_lif_v1.cu)
# ============================================================
_THLIF_FWD_SRC = r"""extern "C" __global__
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
}"""

_THLIF_BWD_SRC = r"""extern "C" __global__
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
        if (s > 0.5f) gv = 0.f;
    }
}"""

_thlif_fwd = cp.RawKernel(_THLIF_FWD_SRC, "thlif_fwd")
_thlif_bwd = cp.RawKernel(_THLIF_BWD_SRC, "thlif_bwd")


# ============================================================
# 3. Autograd Function bridge
# ============================================================
class _THLIFFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, beta, threshold, lam0, kappa, eta, phi0):
        assert inp.is_cuda and inp.dtype == torch.float32, \
            "TH-LIF kernel requires CUDA float32 input"
        B, T, N = inp.shape
        ic = inp.contiguous().float()
        spk   = torch.zeros(B, T, N, device=inp.device, dtype=torch.float32)
        dp_dv = torch.zeros(B, T, N, device=inp.device, dtype=torch.float32)
        grid  = ((B * N + 255) // 256,)
        _thlif_fwd(grid, (256,),
                   (_t2c(ic), _t2c(spk), _t2c(dp_dv),
                    np.int32(B), np.int32(T), np.int32(N),
                    np.float32(beta), np.float32(threshold),
                    np.float32(lam0), np.float32(kappa),
                    np.float32(eta), np.float32(phi0)))
        cp.cuda.Device().synchronize()
        ctx.save_for_backward(spk, dp_dv)
        ctx.p = (B, T, N, beta, threshold)
        return spk

    @staticmethod
    def backward(ctx, go):
        spk, dp_dv = ctx.saved_tensors
        B, T, N, beta, threshold = ctx.p
        gi = torch.zeros(B, T, N, device=go.device, dtype=torch.float32)
        _thlif_bwd(((B * N + 255) // 256,), (256,),
                   (_t2c(go.contiguous().float()), _t2c(spk), _t2c(dp_dv),
                    _t2c(gi),
                    np.int32(B), np.int32(T), np.int32(N),
                    np.float32(beta), np.float32(threshold)))
        cp.cuda.Device().synchronize()
        return gi, None, None, None, None, None, None


# ============================================================
# 4. nn.Module wrapper
# ============================================================
class CUDATHLIF(nn.Module):
    """
    TH-LIF layer with custom CUDA kernel.

    Hazard parameters (lam0, kappa, eta, phi0) are FIXED at init time. Only
    the input projection nn.Linear is learned. For learnable hazard
    parameters, see L-TH-LIF (separate module).

    Patent: TÜRKPATENT 2026/007632 (neuron family) + 2026/004809 (hybrid system).
    """
    def __init__(self, n_in, n_out, beta=0.95, threshold=1.0,
                 lam0=0.1, kappa=2.0, eta=1.5, phi0=1.0, gain=1.0,
                 bias=False):
        super().__init__()
        self.proj = nn.Linear(n_in, n_out, bias=bias)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.lam0 = float(lam0)
        self.kappa = float(kappa)
        self.eta = float(eta)
        self.phi0 = float(phi0)
        self.gain = float(gain)
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, x):
        """
        x:   (B, T, n_in)   float32 CUDA
        out: (B, T, n_out)  float32 CUDA, binary spikes
        """
        ic = self.proj(x) * self.gain
        return _THLIFFunc.apply(ic, self.beta, self.threshold,
                                self.lam0, self.kappa, self.eta, self.phi0)

    def extra_repr(self):
        return (f"n_in={self.n_in}, n_out={self.n_out}, beta={self.beta}, "
                f"threshold={self.threshold}, lam0={self.lam0}, "
                f"kappa={self.kappa}, eta={self.eta}, phi0={self.phi0}")


# ============================================================
# 5. Sanity check (run only when invoked as script)
# ============================================================
def _sanity_check():
    """Quick smoke test: forward, backward, deterministic output, speed."""
    import time
    torch.manual_seed(42)
    device = "cuda"
    B, T, N_in, N_out = 32, 64, 16, 16
    layer = CUDATHLIF(N_in, N_out).to(device)

    # Forward
    x = torch.randn(B, T, N_in, device=device, requires_grad=True)
    spk = layer(x)
    assert spk.shape == (B, T, N_out), f"shape mismatch: {spk.shape}"
    assert ((spk == 0) | (spk == 1)).all(), "spikes must be binary"
    print(f"[OK] Forward: shape={tuple(spk.shape)}, mean_rate={spk.mean().item():.4f}")

    # Backward
    loss = spk.sum()
    loss.backward()
    assert x.grad is not None, "no input gradient"
    assert torch.isfinite(x.grad).all(), "non-finite input gradient"
    print(f"[OK] Backward: input_grad_norm={x.grad.norm().item():.4f}")

    # Determinism (same input -> same output)
    spk2 = layer(x.detach())
    assert (spk == spk2).all(), "non-deterministic forward!"
    print(f"[OK] Deterministic forward verified")

    # Speed
    x = torch.randn(64, 256, 64, device=device)
    layer = CUDATHLIF(64, 64).to(device)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = layer(x)
    torch.cuda.synchronize()
    print(f"[OK] Speed: 100 forwards (B=64, T=256, N=64): {(time.time()-t0)*1000:.1f} ms")


if __name__ == "__main__":
    _sanity_check()
