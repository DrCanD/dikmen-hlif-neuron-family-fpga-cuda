# Changelog

All notable changes to this repository are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] — 2026-05-15

Initial release. Patent-scoped reference implementation aligned with
TÜRKPATENT application **2026/007632** (filed 14.05.2026).

### Added

**Neuron implementations (`hlif_neurons/`)**

- `base.py` — `BaseNeuron`, `FastSigmoidSurrogate`, `StraightThroughBernoulli`,
  `spike_hard`, `spike_stochastic` (carried over unchanged from the parent
  library `dikmen-spiking-neurons` v1.2.0).
- `h_lif.py` — 11 base H-LIF implementations: `DetLIF`, `StochLIF`, `BHLIF`,
  `THLIF`, `DBLIF`, `RTLIF`, `ThermalLIF`, `CHLIF`, `MoELIF`, `AttLIF`,
  `HyperNetLIF`.
- `h_lif_extensions.py` — Physics extensions: `SpinLIF`, `PiezoLIF`.
- `h_lif_variants.py` — **New in this repo.** TH-LIF realization variants
  specified in patent claim 3: `THLIFLinear` (piecewise-linear hazard ramp,
  hardware-friendly), `THLIFSigmoid` (saturating sigmoid hazard,
  gradient-stable). Together with `DetLIF` (λ₀ = 0 form, already present in
  `h_lif.py`) and `THLIF` (exponential canonical embodiment), these provide
  the four claim-3 forms.
- `liquid.py` — `LTHLIF`, `LBHLIF` (claim 4, state-dependent hazard
  parameters via single-layer linear transformation).
- `__init__.py` — `HLIFRegistry` central registry with 17 entries, plus
  a `list_by_claim()` method that maps patent claim keys to model names.

**CUDA kernel (`cuda/`)**

- `kernels/th_lif_v1.cu` — 74-line fused CUDA C kernel pair (forward +
  backward) for TH-LIF exponential. Uses the analytic natural gradient
  `dp/dV = exp(−λ)·λ·κ·η` (no surrogate function). float32 only.
- `wrappers/th_lif_v1.py` — Python autograd Function and `nn.Module`
  wrapper (`CUDATHLIF`) with zero-copy torch ↔ cupy bridge.
- `registry.json` — Renamed from `On_CUDA.json` in the source notebooks.
  Per-kernel metadata: math, hyperparameters, design constraints, project
  usage, cross-reference to FPGA database, patent provenance.

**FPGA validation (`fpga/`)**

- `measurements.json` — Renamed from `On_FPGA.json`. Citable database of
  Arty A7-35T validation: LUT 171, FF 81, DSP 1, BRAM18 2, WNS 8.703 ns at
  50 MHz, ≤2 mW dynamic, ≤240 pJ/step, bit-exact across 5 references
  (Python float, Q4.12 fixed-point, Icarus, xsim, hardware ILA), High
  confidence SAIF-annotated.
- `rtl/`, `mem_inits/` — Verilog RTL (`th_lif_top.v` 109 lines,
  `tb_th_lif_top.v` 81 lines) and Q4.12 LUT initialization files
  (`softplus_1024x16.mem`, `tunneling_1024x16.mem`, 1024 entries each).
- `reports/power_summary.md` — Human-readable power report summary.

**Documentation (`docs/`)**

- `claim_mapping.md` — One-to-one mapping from each patent claim
  (1, 2, 3, 4, 5, 6, system-level 7) to the specific class and file that
  implements it.
- `validation_summary.md` — Consolidated FPGA results, software-side
  benchmarks (SHD ablation 17.4 % vs 2.7 %, RDRD 99.3 %, DIAT-µSAT 93.8 %),
  CUDA kernel speedup (≈312× vs. PyTorch eager).

**Examples and tests**

- `examples/quickstart.py` — Minimal usage, walks through one model per
  claim.
- `examples/compare_th_variants.py` — Side-by-side comparison of the four
  claim-3 forms on identical input.
- `examples/cuda_kernel_demo.py` — End-to-end demo of the CUDA kernel
  (forward, backward, determinism, speed).
- `tests/test_registry.py` — pytest covering all 17 models for
  instantiation, forward shape, binary spike property, gradient flow,
  registry consistency.

**Project files**

- `README.md`, `LICENSE` (Apache 2.0), `NOTICE` (patent notice, parent
  repo provenance), `CITATION.cff`, `setup.py`, `requirements.txt`.

### Naming change from source notebooks

In the source notebooks the FPGA validation database and CUDA kernel
metadata were named `On_FPGA.json` and `On_CUDA.json`. In this repository
they are renamed to:

- `On_FPGA.json` → `fpga/measurements.json`
- `On_CUDA.json` → `cuda/registry.json`

The folder context now disambiguates the filenames and makes the
repository navigable without prior knowledge of the source convention.

### Pending (for a future minor release)

- Add a separate kernel for L-TH-LIF (learnable hazard parameters; currently
  PyTorch-only).
- Add KV260 measurements once the board is fully deployed.
