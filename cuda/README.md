# Custom CUDA Kernel — TH-LIF v1

This directory contains the custom CUDA kernel for TH-LIF (claim 2,
exponential variant) and the Python autograd wrapper that exposes it as a
drop-in `nn.Module`.

## Layout

```
cuda/
├── README.md                 ← this file
├── registry.json             ← metadata for each kernel (math, hyperparams, provenance)
├── kernels/
│   └── th_lif_v1.cu          ← raw CUDA C source (74 lines, fwd + bwd)
└── wrappers/
    └── th_lif_v1.py          ← Python autograd Function + nn.Module wrapper
```

## Why a custom kernel

A pure-PyTorch TH-LIF loops over T timesteps in Python, launches multiple
small CUDA ops per step (multiply, exp, sigmoid, surrogate, ...), and
materializes intermediate membrane states for the surrogate-gradient
backward pass. On a recent A100 with B=64, T=256, N=64, this takes about
95 ms per forward call.

The fused kernel exploits three properties of the hazard formulation:

1. **Analytic natural gradient.** The hazard `λ = λ₀·exp(−κ(φ₀ − η·V))`
   has an exact derivative `dp/dV = exp(−λ)·λ·κ·η`. No surrogate function
   is needed and no surrogate-related memory traffic is generated.
2. **Single launch per direction.** The whole T-loop runs inside one CUDA
   thread per `(batch, neuron)` pair. No kernel launch overhead per step.
3. **Minimal saved tensors.** Only `spk[t]` and `dp_dv[t]` are saved for
   the backward pass — no membrane states, no intermediate exponentials.

On the same A100 / B=64 / T=256 / N=64 benchmark, the fused kernel runs at
about 0.30 ms per forward call. That is roughly 312× faster than the
PyTorch baseline.

## Usage

```python
import torch
from cuda.wrappers.th_lif_v1 import CUDATHLIF

# Layer with fixed hazard parameters
layer = CUDATHLIF(
    n_in=64, n_out=128,
    beta=0.95, threshold=1.0,
    lam0=0.1, kappa=2.0, eta=1.5, phi0=1.0,
).to("cuda")

# Input must be float32 on CUDA, shape (B, T, n_in)
x = torch.randn(32, 256, 64, device="cuda")
spk = layer(x)        # (32, 256, 128), binary

# Standard PyTorch autograd backward
spk.sum().backward()
```

## Constraints (also enforced by assertions in the wrapper)

- **float32 only.** AMP / half-precision is forbidden because `expf`
  overflows past `|exponent| > ~87`. The forward clamps the exponent to
  `[−10, 10]`, but on FP16 even that range overflows the representable
  domain.
- **Shape `(B, T, N)` row-major.** No transposes, no NCHW.
- **CUDA device.** No CPU fallback. The wrapper asserts `inp.is_cuda`.
- **Fixed hazard parameters per layer instance.** Learnable hazard
  parameters require a different kernel; that case is covered by
  `L-TH-LIF` (see `hlif_neurons/liquid.py`).

## Compilation

The kernel is JIT-compiled at first use through `cupy.RawKernel`. There is
no separate build step. Requirements:

- CUDA Toolkit 11.0 or newer (the wrapper was developed against CUDA 12.x)
- cupy matching the installed CUDA version (e.g. `cupy-cuda12x`)
- PyTorch ≥ 1.10 with CUDA support

Install via:

```bash
pip install -e ".[cuda]"
```

## Sanity check

Run the wrapper module as a script for a quick smoke test:

```bash
python -m cuda.wrappers.th_lif_v1
```

Expected output:

```
[OK] Forward: shape=(32, 64, 16), mean_rate=0.NNNN
[OK] Backward: input_grad_norm=0.NNNN
[OK] Deterministic forward verified
[OK] Speed: 100 forwards (B=64, T=256, N=64): NN.N ms
```

## Numerical stability

The forward kernel applies three clamps to prevent overflow and gradient
explosion:

| Clamp | Range | Purpose |
|-------|-------|---------|
| `exponent` | `[−10, 10]` | Prevents `expf` overflow on supra-threshold inputs |
| `dp_dv` | `[0, 2]` | Bounds the natural gradient |
| `gs` (backward) | `[−1, 1]` | Prevents BPTT divergence over long sequences |

The reset step zeros the gradient through time (`gv = 0` after spike), which
breaks back-propagation through the spike event. This is intentional: it
matches the patent's specification and prevents the unboundedly large
gradients that would otherwise flow back through hard reset.

## Cross-reference

The TH-LIF v1 entry in `registry.json` carries a `cross_reference_fpga`
field that points to the matching FPGA neuron id `th_lif_exp_2026_04` in
`../fpga/measurements.json`. Both implement the same mathematical
formulation; the .cu file is the software fast path, the .v file is the
edge deployment path.
