# FPGA Validation — Arty A7-35T

This directory contains the FPGA validation evidence for the TH-LIF neuron
(patent claim 2), implemented on a Digilent Arty A7-35T board with a Xilinx
Artix-7 `xc7a35ticsg324-1L` part.

## Layout

```
fpga/
├── README.md              ← this file
├── measurements.json      ← citable database of resource, timing, power, energy
├── rtl/
│   ├── README.md
│   └── (Verilog RTL — pending upload from lab machine)
├── mem_inits/
│   ├── README.md
│   └── (LUT initialization files — pending upload)
└── reports/
    └── power_summary.md   ← human-readable summary of the power report
```

## Headline numbers

| Metric | Value | Confidence |
|--------|-------|------------|
| LUT (production) | 171 | High |
| FF (production) | 81 | High |
| DSP48E1 | 1 | High |
| BRAM18 | 2 | High |
| Clock | 50 MHz, WNS 8.703 ns | High (post-route) |
| Dynamic power | ≤ 2 mW (production) | High (SAIF-annotated) |
| Energy per step | ≤ 240 pJ | High (conservative upper bound) |

## Functional cross-validation

Bit-exact agreement across five references (Python float, Python Q4.12,
Icarus, xsim, hardware ILA) on a standard 128-step stimulus cycle.
Twelve spikes per cycle at positions
`[15, 26, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93]`.

## Tool flow

| Step | Tool | Version |
|------|------|---------|
| Synthesis | Vivado | 2025.2 |
| Implementation | Vivado | 2025.2 |
| Power analysis | Vivado XPower Analyzer (SAIF-annotated post-route) | 2025.2 |
| RTL simulation | Icarus Verilog, Vivado xsim | — |
| Hardware capture | Vivado ILA (4096-sample window) | 2025.2 |

## Reading order

1. `measurements.json` — the citable record. Every number reported in
   manuscripts and the patent specification traces back to a field here.
2. `docs/validation_summary.md` (one level up) — human-readable narrative
   of the same data.
3. `rtl/` — the actual Verilog (once uploaded).
