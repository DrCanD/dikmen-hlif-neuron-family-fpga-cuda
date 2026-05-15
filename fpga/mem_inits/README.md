# FPGA Memory Initialization Files

This directory contains the `.mem` files used to initialize the hazard
lookup tables in the TH-LIF FPGA implementation.

## Files

| File | Entries | Width | Format |
|------|--------:|------:|--------|
| `softplus_1024x16.mem` | 1024 | 16-bit Q4.12 | hex, one entry per line |
| `tunneling_1024x16.mem` | 1024 | 16-bit Q4.12 | hex, one entry per line |

Both files are read at synthesis time via Verilog `$readmemh` directives in
`../rtl/th_lif_top.v`, and each ends up mapped to one BRAM18 instance after
implementation.

## Contents

- **`softplus_1024x16.mem`** — Discretized softplus function `softplus(x) = log(1 + exp(x))`,
  used in the classical (threshold-crossing) hazard pathway of stage A.
- **`tunneling_1024x16.mem`** — Discretized tunneling hazard
  `exp(−κ·(φ₀ − η·V))`, used in the sub-threshold barrier-penetration
  pathway of stage B.

Both functions are sampled at 1024 evenly spaced points in the membrane
potential range and rounded to Q4.12 fixed-point. The first line of each
file is a comment; the remaining 1024 lines are the entries in row order.
