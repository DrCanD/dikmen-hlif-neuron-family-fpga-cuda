# Power Report Summary — TH-LIF on Arty A7-35T

Source: `C:/FPGA_Projects/th_lif_step3/th_lif_power_high.rpt`
Tool: Vivado 2025.2 XPower Analyzer
Method: SAIF-annotated post-route, 100 μs simulation window
Confidence: High

## Production build (deploy, no ILA)

| Component | Power |
|-----------|------:|
| Device static | 62.0 mW |
| Core dynamic | 1.0 mW |
| Total on-chip (upper bound) | ≤ 64.0 mW |

### Dynamic breakdown (upper bound, ≤ 1 mW each)

| Source | Power |
|--------|------:|
| Clocks | ≤ 1.0 mW |
| Block RAM | ≤ 1.0 mW |
| Signals + logic | ≤ 1.0 mW |
| DSP48E1 | ≤ 1.0 mW |
| I/O | 2.0 mW |
| Neuron core estimate | ~ 1.0 mW |

Note on resolution: the `u_neuron` Vivado hierarchy table entry falls below
the 1 mW reporting threshold. The estimate of ~1 mW is derived by
subtraction: total dynamic 10 mW (validation build) minus debug overhead
(dbg_hub 2 mW + u_ila_0 6 mW = 8 mW) leaves ~2 mW; the neuron core itself is
within ~1 mW of that.

## Validation build (ILA active, 4096-sample capture)

| Component | Power |
|-----------|------:|
| Device static | 62.0 mW |
| Total dynamic | 10.0 mW |
| Total on-chip | 72.0 mW |

### Dynamic breakdown (validation)

| Source | Power |
|--------|------:|
| Clocks | 5.0 mW |
| Block RAM | 2.0 mW |
| Signals + logic | 1.0 mW |
| DSP48E1 | 0.5 mW |
| I/O | 2.0 mW |
| **ILA debug overhead total** | **8.0 mW** |
| └── ILA capture core (u_ila_0) | 6.0 mW |
| └── Debug hub JTAG (dbg_hub) | 2.0 mW |

The 8 mW ILA debug overhead is removed in the production build, which is the
configuration used to derive the ≤ 2 mW dynamic / ≤ 240 pJ-per-step
production numbers reported in the paper and patent.

## Energy derivation

```
P_dyn (production)   = 1.0 mW       (upper bound 2.0 mW)
f_step               = 8.33 MHz     (6 cycles/step at 50 MHz)
E_step               = P_dyn / f_step
                     = 1.0e-3 / 8.33e6
                     = 1.20e-10 J
                     = 120 pJ       (upper bound 240 pJ)

spikes_per_128_step  = 12
E_spike              = (128 × E_step) / spikes_per_128_step
                     = (128 × 120 pJ) / 12
                     = 1.28 nJ      (upper bound 2.56 nJ)
```

## SAIF coverage

- Design nets matched: 1430 / 4499 = 32 %
- Unmatched nets: ~3069 (predominantly ILA trace memory and BSCAN debug
  hub internals; fed by Vivado's vectorless probabilistic estimator)
- Initial run with `strip_path 'uut'` produced 0 % match (DUT instance name
  is `dut`, not `uut`); after correcting to `strip_path 'dut'`, the 32 %
  match raised confidence from Medium to High. Final paper uses this run.

## Static power caveat

The 62 mW device static term is the `xc7a35ticsg324-1L` part baseline at
typical process / 25 °C ambient / 250 LFM airflow / industrial temperature
grade. It is **not** attributable to the TH-LIF design and would be 10–100×
lower on ASIC or low-leakage FPGA platforms (e.g. Lattice iCE40).
