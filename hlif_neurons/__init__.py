"""
H-LIF Neuron Family (TÜRKPATENT 2026/007632)
============================================
Patent-scoped subset of the Dikmen spiking neuron library.

This package contains the 15 spiking neuron models covered by TÜRKPATENT
application 2026/007632 ("Hazard Function-Based Spiking Neuron Family, Liquid
Extension, and N-Island Heterogeneous Neuromorphic Processing System"), plus
two TH-LIF realization variants (linearized, sigmoid) specified in claim 3.

Patent-covered models (17 entries in registry):

  Baseline (claim 3 limit cases, deterministic / single-pathway):
    - Det-LIF      λ0 = 0 deterministic Heaviside
    - Stoch-LIF    sigmoid hazard with additive noise

  Core hazard (claim 2, dual-pathway TH-LIF):
    - BH-LIF       barrier-only hazard (claim 1 with single hazard path)
    - TH-LIF       exponential dual-pathway (canonical embodiment)
    - TH-LIF-Linear   linearized form (claim 3)
    - TH-LIF-Sigmoid  sigmoid form (claim 3)

  Physics-inspired (claim 5):
    - DB-LIF       double-barrier bistable
    - RT-LIF       resonant tunneling band-pass
    - Thermal-LIF  Arrhenius kinetics
    - Spin-LIF     magnetic spin precession
    - Piezo-LIF    piezoelectric stress

  Architecture-inspired (claim 6):
    - CH-LIF       cross-coupled barrier modulation
    - MoE-LIF      mixture-of-experts hazard
    - Att-LIF      attention-modulated barrier
    - HyperNet-LIF meta-network barrier parameters

  Liquid extension (claim 4):
    - L-TH-LIF     state-dependent TH-LIF
    - L-BH-LIF     state-dependent BH-LIF

For the full 39-model library (including MS-IF, I-LIF, F-LIF, K-LIF, W-LIF,
C-LIF families) see github.com/DrCanD/dikmen-spiking-neurons.

Author:   İsmail Can Dikmen
Affil:    İstinye University, Department of Electrical and Electronics Engineering
License:  Apache-2.0 (with explicit patent grant - see LICENSE and NOTICE)
"""

__version__ = "1.0.0"
__author__ = "İsmail Can Dikmen"
__patent__ = "TÜRKPATENT 2026/007632"

from .base import BaseNeuron, spike_hard, spike_stochastic
from .h_lif import REGISTRY as _HLIF_REG
from .h_lif_extensions import REGISTRY as _HLIF_EXT_REG
from .h_lif_variants import REGISTRY as _HLIF_VAR_REG
from .liquid import REGISTRY as _LIQUID_REG


class HLIFRegistry:
    """Central registry for all 17 patent-covered neuron models."""

    _all = {}
    _all.update(_HLIF_REG)        # 11 base H-LIF
    _all.update(_HLIF_EXT_REG)    # 2 physics extensions (Spin, Piezo)
    _all.update(_HLIF_VAR_REG)    # 2 TH-LIF variants (Linear, Sigmoid)
    _all.update(_LIQUID_REG)      # 2 liquid (L-TH-LIF, L-BH-LIF)

    _claims = {
        "claim_1_general": ["BH-LIF", "TH-LIF"],
        "claim_2_th_lif": ["TH-LIF"],
        "claim_3_variants": ["Det-LIF", "TH-LIF-Linear", "TH-LIF-Sigmoid"],
        "claim_4_liquid": ["L-TH-LIF", "L-BH-LIF"],
        "claim_5_physics": ["DB-LIF", "RT-LIF", "Thermal-LIF", "Spin-LIF", "Piezo-LIF"],
        "claim_6_architecture": ["CH-LIF", "MoE-LIF", "Att-LIF", "HyperNet-LIF"],
    }

    @classmethod
    def create(cls, name: str, size: int = 128, **kwargs) -> BaseNeuron:
        if name not in cls._all:
            raise KeyError(
                f"Unknown neuron '{name}'. Available: {list(cls._all.keys())}"
            )
        return cls._all[name](size=size, **kwargs)

    @classmethod
    def list_all(cls) -> list:
        return sorted(cls._all.keys())

    @classmethod
    def list_by_claim(cls, claim_key: str) -> list:
        if claim_key not in cls._claims:
            raise KeyError(
                f"Unknown claim key '{claim_key}'. Available: {list(cls._claims.keys())}"
            )
        return cls._claims[claim_key]

    @classmethod
    def summary(cls) -> str:
        lines = [
            f"H-LIF Neuron Family v{__version__}",
            f"Patent: {__patent__}",
            f"Total: {len(cls._all)} patent-covered models",
            "",
        ]
        for claim, models in cls._claims.items():
            lines.append(f"  {claim}: {', '.join(models)}")
        return "\n".join(lines)


__all__ = ["BaseNeuron", "HLIFRegistry", "spike_hard", "spike_stochastic"]
