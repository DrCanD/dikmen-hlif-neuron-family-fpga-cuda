"""
H-LIF Variant Forms: Alternative TH-LIF hazard realizations.

Claim 3 of TÜRKPATENT 2026/007632 states that the threshold-crossing hazard
circuit (12) and barrier-penetration hazard circuit (13) of the TH-LIF neuron
unit (10) may alternatively be realized in linearized, sigmoid, or
λ0 = 0 deterministic forms. The exponential form is the canonical embodiment
and lives in h_lif.py as THLIF; the deterministic limit lives there as DetLIF.

This module supplies the remaining two forms.

Models:
- TH-LIF-Linear:  piecewise-linear hazard, slope κ·η around the operating point
- TH-LIF-Sigmoid: sigmoid hazard, saturates at λ_max
"""

import torch
import torch.nn as nn

from .base import BaseNeuron, spike_stochastic


class THLIFLinear(BaseNeuron):
    """#4a TH-LIF Linearized form (patent claim 3).

    Hazard ramp instead of exponential:
        h(V) = λ0 * relu(1 + κ·(η·V − φ0))

    Equivalent to a first-order Taylor expansion of the exponential form
    around the threshold. Cheaper on hardware (no exp lookup table), at the
    cost of saturation behavior in the supra-threshold regime.
    """
    _family = "h_lif"
    _description = "TH-LIF with linearized hazard ramp (claim 3, hardware-friendly)."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lambda_0 = nn.Parameter(torch.full((size,), -2.0))
        self.log_kappa = nn.Parameter(torch.full((size,), 0.5))
        self.eta = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.5))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        lam_0 = self.log_lambda_0.exp()
        kappa = self.log_kappa.exp()
        # Linearized barrier: ramp instead of exp
        ramp = torch.relu(1.0 + kappa * (self.eta * mem - self.phi_0))
        h = lam_0 * ramp
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class THLIFSigmoid(BaseNeuron):
    """#4b TH-LIF Sigmoid form (patent claim 3).

    Saturating hazard:
        h(V) = λ_max * sigmoid(κ·(η·V − φ0))

    Equivalent in spirit to the classical (threshold-crossing) pathway of
    BH-LIF but without the barrier-penetration term. Useful when only the
    supra-threshold regime matters and gradient stability is preferred over
    sub-threshold sensitivity.
    """
    _family = "h_lif"
    _description = "TH-LIF with sigmoid hazard saturation (claim 3, gradient-stable)."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lambda_max = nn.Parameter(torch.full((size,), 1.0))
        self.log_kappa = nn.Parameter(torch.full((size,), 0.5))
        self.eta = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.5))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        lam_max = self.log_lambda_max.exp()
        kappa = self.log_kappa.exp()
        h = lam_max * torch.sigmoid(kappa * (self.eta * mem - self.phi_0))
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


REGISTRY = {
    "TH-LIF-Linear": THLIFLinear,
    "TH-LIF-Sigmoid": THLIFSigmoid,
}
