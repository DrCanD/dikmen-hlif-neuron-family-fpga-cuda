# Validation Summary

Patent: **TÜRKPATENT 2026/007632**, filed 14.05.2026
Repository: dikmen-hlif-neuron-family-fpga-cuda v1.0.0

This document consolidates the validation evidence supporting the patent
claims, drawn from FPGA synthesis, functional cross-verification, and
software-side benchmarks. For the citable raw record see
`fpga/measurements.json` and `cuda/registry.json`.

---

## 1. FPGA validation (Arty A7-35T, Xilinx Artix-7)

**Design under test:** Single TH-LIF neuron, exponential variant, Q4.12
signed fixed-point arithmetic, 7-state FSM, 6 cycles per neuron step.

### Resource utilization (production build)

| Resource | Used | Available | Utilization |
|----------|-----:|----------:|------------:|
| LUT      |  171 |    20,800 |  0.82 %     |
| FF       |   81 |    41,600 |  0.19 %     |
| DSP48E1  |    1 |        90 |  1.11 %     |
| BRAM18   |    2 |       100 |  4.00 %     |
| IOB      |    6 |       210 |  2.86 %     |

The two BRAM18 instances hold the softplus (1024×16) and tunneling (1024×16)
hazard lookup tables. The single DSP48E1 absorbs the β·V multiply, with
`V_state_reg` merged into the DSP AREG to save a flip-flop.

### Timing (post-route)

- Clock: 50 MHz (period 20 ns)
- WNS: 8.703 ns (positive margin)
- WHS: 0.044 ns (positive margin)
- Failing endpoints: 0
- Neuron step rate: 8.33 M steps/sec (6 cycles/step at 50 MHz)

### Power (Vivado XPower Analyzer, SAIF-annotated)

| Component | Production | Validation (ILA on) |
|-----------|-----------:|--------------------:|
| Device static | 62.0 mW | 62.0 mW |
| Core dynamic | 1.0 mW | 10.0 mW (incl. 8 mW ILA debug) |
| Total on-chip (upper bound) | ≤2.0 mW dynamic | 72.0 mW |

The 62 mW device static term is the baseline leakage for the
`xc7a35ticsg324-1L` part and is **not** attributable to the neuron itself.
ASIC or low-leakage FPGA platforms (e.g. Lattice iCE40 family) would reduce
this contribution by 10–100×.

### Energy

| Metric | Value | Notes |
|--------|------:|-------|
| Energy per step | ≤240 pJ | conservative upper bound (production) |
| Energy per spike | ≤2.56 nJ | 12 spikes per 128-step cycle |
| Energy per 128-step cycle | 15.4 nJ | |

Calculation basis:
`E_step = P_dyn / f_step` with P_dyn ≤ 2 mW and f_step = 8.33 MHz.

### Functional cross-validation (bit-exact across 5 references)

| Reference | Type | Spikes/cycle | Step match |
|-----------|------|:------------:|:----------:|
| Python float reference | software | 12 | — |
| Python fixed-point Q4.12 (saturating) | software | 12 | — |
| Icarus Verilog behavioral | RTL simulation | 12 | 128/128 |
| Vivado xsim behavioral | RTL simulation | 12 | 128/128 |
| Arty A7-35T hardware (ILA capture) | hardware | 12 | 4 cycles observed |

Stimulus: 4-phase periodic square-wave current, 128-step cycle.
Spike positions per cycle: `[15, 26, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93]`.
Membrane state at cycle boundary: `0xF113` (Q4.12) = −0.9329.
Zero numerical drift, zero cycle-to-cycle variance.

### Confidence assessment

- **Overall confidence:** High
- **Estimation method:** SAIF-annotated post-route, post-implementation
  timing simulation, 100 μs window
- **SAIF design nets matched:** 1430 / 4499 = 32 %
- **Unmatched signals note:** ~3069 signals are predominantly ILA trace
  memory and BSCAN debug hub internals; these are fed by Vivado's vectorless
  probabilistic estimator, not measured activity.

---

## 2. Software-side benchmarks (selected)

### SHD ablation — confirms TNS metric (claim 7 supporting evidence)

| Model | Acc (full) | Acc (temporal-destroyed) | Δ |
|-------|-----------:|-------------------------:|---:|
| Standard LIF | 70.3 % | 67.6 % | 2.7 % |
| TH-LIF | 78.9 % | 61.5 % | **17.4 %** |

Higher Δ means stronger reliance on the temporal dimension. TH-LIF's 6.4×
larger drop confirms that the dual-pathway hazard exploits temporal
structure substantially more than the threshold-only baseline.

### RDRD micro-Doppler (radar, 8-class human activity)

- TH-LIF CNN-equivalent F1: 99.3 % (matches CNN, lower energy)
- Energy advantage vs. CNN baseline on RDRD: 27 %

### DIAT-µSAT micro-Doppler (radar, drone classification)

- TH-LIF accuracy: 93.8 % (vs. CNN 94.1 %, within noise)
- 8-bit quantization-friendly, suitable for AKD1500 deployment

---

## 3. CUDA kernel performance (TH-LIF v1)

Implementation: single fused kernel per direction (forward, backward),
B·N grid threads, T sequential timesteps per thread.

### Throughput (a recent A100, FP32, B=64, T=256, N=64)

| Implementation | Time per forward | Speedup |
|----------------|-----------------:|--------:|
| PyTorch eager (baseline) | ~95 ms | 1.0× |
| CUDA kernel (this repo) | ~0.30 ms | **≈312×** |

The speedup stems from three factors: (1) fusing the entire timestep loop
into one kernel launch, (2) using the analytic natural-gradient
`dp/dv = exp(-λ)·λ·κ·η` instead of a surrogate function call, and
(3) saving only the surrogate value `dp_dv[t]` and the spike `spk[t]` for the
backward pass (no intermediate membrane states).

### Numerical stability

- Exponent clamped to `[−10, 10]` (forward)
- `dp/dv` clamped to `[0, 2]` (forward)
- `gs` clamped to `[−1, 1]` (backward)
- Reset gradient breaking active (`gv = 0` after spike)

The kernel is bit-exact on identical inputs and matches the PyTorch
implementation within `1e−6` FP32 tolerance.

---

## 4. Patent evidence summary

| Claim | Evidence | Where |
|-------|----------|-------|
| Claim 1 | 17 implementations, all runnable | `hlif_neurons/`, `tests/test_registry.py` |
| Claim 2 | TH-LIF kernel + Arty A7-35T validation | `cuda/`, `fpga/measurements.json` |
| Claim 3 | Three alternative forms implemented | `h_lif_variants.py`, `DetLIF` |
| Claim 4 | L-TH-LIF / L-BH-LIF state-dependent | `liquid.py` |
| Claim 5 | Five physics variants, all runnable | `h_lif.py`, `h_lif_extensions.py` |
| Claim 6 | Four architecture variants, all runnable | `h_lif.py` |
| Claim 7 system | Two-island instance: SoC patent 2026/004809 | companion patent |
| TNS metric | SHD ablation 17.4 % vs 2.7 % | software benchmark above |

---

## 5. Caveats

- The FPGA validation covers TH-LIF only (exponential variant). The other 16
  models in the registry are validated at the software level only (PyTorch
  forward / backward correctness). Hardware feasibility for L-TH-LIF,
  HyperNet-LIF, and MoE-LIF is the subject of ongoing work.
- The 62 mW device static term is the FPGA part baseline and is not a
  property of the TH-LIF design. Energy comparisons that include this term
  are FPGA-platform-specific, not neuron-specific.
- The 32 % SAIF design-nets-matched rate is normal for designs with ILA
  debug instrumentation present; the unmatched 68 % falls inside the
  vectorless probabilistic estimator's scope and contributes to the
  reported confidence levels rather than being silently ignored.
