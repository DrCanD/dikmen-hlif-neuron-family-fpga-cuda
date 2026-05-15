# H-LIF Neuron Family — FPGA & CUDA Reference Implementation
[![DOI](https://zenodo.org/badge/1239588631.svg)](https://doi.org/10.5281/zenodo.20201251)


> **Patent reference repository.** This repo contains the patent-scoped subset of the
> Dikmen spiking neuron library, the corresponding custom CUDA kernel for TH-LIF,
> and the FPGA energy/resource measurements from Arty A7-35T (post-route SAIF).

**Patent:** TÜRKPATENT application **2026/007632**, filed 14.05.2026
*"Hazard Function-Based Spiking Neuron Family, Liquid Extension, and N-Island
Heterogeneous Neuromorphic Processing System"* — owner: İstinye University.

---

## What is here

```
.
├── hlif_neurons/      ← 17 patent-covered neuron implementations (PyTorch)
├── cuda/              ← custom CUDA kernel for TH-LIF (forward + backward)
├── fpga/              ← FPGA validation database + RTL (Arty A7-35T)
├── docs/              ← claim mapping, validation summary
└── examples/          ← minimal usage scripts
```

| Layer | Content | 
|-------|---------|
| `hlif_neurons/` | 17 PyTorch neuron classes |
| `cuda/` | 74-line CUDA C kernel + Python autograd wrapper |
| `fpga/` | post-route resource, power, energy on Arty A7-35T |

---

## Quick start

```bash
pip install -e .
```

```python
import torch
from hlif_neurons import HLIFRegistry

# Pick any of 17 patent-covered models
neuron = HLIFRegistry.create("TH-LIF", size=128)

# Standard forward pass
x = torch.randn(32, 100, 128)   # (batch, timesteps, features)
spikes, state = neuron(x)        # spikes: (32, 100, 128), state: dict

# List by patent claim
print(HLIFRegistry.list_by_claim("claim_5_physics"))
# ['DB-LIF', 'RT-LIF', 'Thermal-LIF', 'Spin-LIF', 'Piezo-LIF']
```

---

## Custom CUDA kernel

`cuda/kernels/th_lif_v1.cu` is a fused forward+backward kernel for TH-LIF.
Compared to a pure-PyTorch implementation, the kernel exploits the analytic
natural-gradient property of the hazard formulation (no surrogate function
needed) and a single launch per direction.

```python
from cuda.wrappers.th_lif_v1 import CUDATHLIF
layer = CUDATHLIF(n_in=64, n_out=128, lam0=0.1, kappa=2.0, eta=1.5, phi0=1.0)
x = torch.randn(B, T, 64, device="cuda")   # float32 mandatory
spk = layer(x)                              # (B, T, 128)
```

Constraints: float32 only (AMP/half-precision overflows `expf`), shape `(B, T, N)`,
CUDA device. The reset step zeros the gradient through time, so BPTT does not
cross spike events. See `cuda/wrappers/th_lif_v1.py` for a sanity-check script.

---

## FPGA validation (Arty A7-35T)

Single-neuron TH-LIF on Digilent Arty A7-35T (Xilinx Artix-7 `xc7a35ticsg324-1L`):

| Metric | Value | Notes |
|--------|-------|-------|
| LUT (production) | **171** | 0.82% of 20,800 available |
| FF (production) | **81** | 0.19% of 41,600 available |
| DSP48E1 | 1 | β·V multiply, V_state_reg merged into DSP AREG |
| BRAM18 | 2 | softplus + tunneling LUTs (1024×16 each) |
| Clock | 50 MHz | WNS 8.703 ns, WHS 0.044 ns, 0 failing endpoints |
| Cycles per step | 6 | 7-state FSM, ~8.33 M steps/sec |
| Dynamic power | ≤2 mW | upper bound (production, no ILA debug) |
| Energy per step | **≤240 pJ** | conservative upper bound |
| Energy per spike | ≤2.56 nJ | 12 spikes per 128-step cycle |
| Functional check | bit-exact | 5 references (float, fixed-point, Icarus, xsim, hardware) |
| Confidence | High | SAIF-annotated post-route, 32% nets matched |

Arithmetic: Q4.12 signed fixed-point with saturating multiply/add/subtract.
Tested on a 4-phase periodic square-wave current stimulus, 128-step cycle,
4 hardware cycles captured via ILA. Zero numerical drift, zero cycle-to-cycle
variance.

See `fpga/measurements.json` for the full citable record. The Verilog
RTL (`th_lif_top.v`, `tb_th_lif_top.v`) and LUT initialization files
(`softplus_1024x16.mem`, `tunneling_1024x16.mem`) are in `fpga/rtl/` and
`fpga/mem_inits/` respectively.

---

## Citation

If you use this software or reference the patent in academic work, please cite:

```bibtex
@software{dikmen2026hlif,
  author    = {Dikmen, İsmail Can},
  title     = {H-LIF Neuron Family — FPGA \& CUDA Reference Implementation},
  year      = {2026},
  institution = {İstinye University},
  url       = {https://github.com/DrCanD/dikmen-hlif-neuron-family-fpga-cuda},
  license   = {Apache-2.0},
  note      = {Reference implementation for TÜRKPATENT 2026/007632}
}

@misc{dikmen2026patent_hlif,
  author    = {Dikmen, İsmail Can},
  title     = {Hazard Function-Based Spiking Neuron Family, Liquid Extension,
               and N-Island Heterogeneous Neuromorphic Processing System},
  year      = {2026},
  note      = {TÜRKPATENT application 2026/007632, filed 14.05.2026},
  institution = {İstinye University}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

The Apache-2.0 license includes an explicit patent grant from contributors to
users of the software. Independent of that grant, the TÜRKPATENT application
2026/007632 is the property of İstinye University; commercial use of the
patented inventions in jurisdictions where the application proceeds to grant
should be coordinated through the İstinye University Technology Transfer
Office (İstinye TTO).

---

## Author

**İsmail Can Dikmen** — Assistant Professor, Electrical and Electronics
Engineering, İstinye University, Istanbul, Türkiye.

- ORCID: [0000-0002-7747-7777](https://orcid.org/0000-0002-7747-7777)
- Web: [Google Scholar](https://scholar.google.com/citations?user=c4OrnOQAAAAJ)
- IEEE Senior Member
