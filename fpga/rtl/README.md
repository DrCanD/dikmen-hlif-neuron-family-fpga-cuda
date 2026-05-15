# FPGA RTL Sources — Arty A7-35T

This directory contains the Verilog RTL sources for the TH-LIF single-neuron
implementation validated on a Digilent Arty A7-35T board (Xilinx Artix-7
`xc7a35ticsg324-1L`).

## Files

| File | Lines | Description |
|------|------:|-------------|
| `th_lif_top.v` | 109 | Top-level RTL module, 7-state FSM, 6 cycles per neuron step |
| `tb_th_lif_top.v` | 81 | Testbench with 4-phase periodic square-wave stimulus |

The lookup table initialization files (`softplus_1024x16.mem`,
`tunneling_1024x16.mem`) live in `../mem_inits/`.

## Provenance

These files were produced as part of the TH-LIF FPGA Implementation effort
(April 2026) and validated through:

- Icarus Verilog behavioral simulation
- Vivado xsim behavioral simulation
- Arty A7-35T hardware (ILA capture)

All five reference implementations (two software, two RTL-sim, one hardware)
agree bit-exactly on 12 spikes per 128-step cycle for the standard stimulus
pattern. See `../measurements.json` for the full citable record.

## Reproduction notes

The Vivado project that built these results is recoverable from the RTL +
Vivado 2025.2:

```
File → Project → New → RTL Project
Add Sources:         th_lif_top.v
Add Sim Sources:     tb_th_lif_top.v
Add Constraints:     (none required — default for xc7a35ticsg324-1L)
Add Memory Init:     ../mem_inits/softplus_1024x16.mem
                     ../mem_inits/tunneling_1024x16.mem
Target board:        xc7a35ticsg324-1L (Arty A7-35T)
Synthesis strategy:  Vivado Synthesis Defaults
Implementation:      Vivado Implementation Defaults
```

Run synthesis and implementation. The expected post-route utilization is
`LUT=171, FF=81, DSP=1, BRAM18=2, IOB=6` with WNS=8.703 ns @ 50 MHz.

For power analysis: open the implemented design, run `Report Power` with
`Activity` set to the SAIF file produced by post-implementation timing
simulation. The expected core dynamic power is 1.0 mW (production, no ILA)
or 10.0 mW (validation, ILA active).
