"""
Liquid Branch: Input-dependent hazard dynamics.
Axis: How fast to adapt? (ODE-governed parameters)
Models: L-TH-LIF (#17), L-BH-LIF (#32)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseNeuron, spike_stochastic


class LTHLIF(BaseNeuron):
    """#17 Liquid TH-LIF. Hazard params (kappa, lambda) are input-dependent."""
    _family = "liquid"
    _description = "Liquid tunneling-hazard: spike mechanism itself is input-dependent."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.eta = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.5))
        self.w_kappa = nn.Linear(size, size, bias=True)
        self.w_lambda = nn.Linear(size, size, bias=True)
        self.w_tau = nn.Linear(size, size, bias=True)

    def single_step(self, x_t, state):
        tau_factor = torch.sigmoid(self.w_tau(x_t))
        beta_dyn = self.beta * tau_factor + (1 - tau_factor) * 0.5
        mem = beta_dyn * state["mem"] + x_t

        kappa = F.softplus(self.w_kappa(x_t))
        lam_0 = F.softplus(self.w_lambda(x_t))
        barrier = self.phi_0 - self.eta * mem
        h = lam_0 * torch.exp((-kappa * barrier).clamp(-10, 10))
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class LBHLIF(BaseNeuron):
    """#32 Liquid BH-LIF. Both dual-pathway params are input-dependent."""
    _family = "liquid"
    _description = "Liquid dual-pathway: both classical and tunneling hazard are input-dependent."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.eta = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.5))
        self.w_kappa = nn.Linear(size, size, bias=True)
        self.w_lambda_q = nn.Linear(size, size, bias=True)
        self.w_lambda_cl = nn.Linear(size, size, bias=True)
        self.w_delta = nn.Linear(size, size, bias=True)
        self.w_tau = nn.Linear(size, size, bias=True)

    def single_step(self, x_t, state):
        tau_factor = torch.sigmoid(self.w_tau(x_t))
        beta_dyn = self.beta * tau_factor + (1 - tau_factor) * 0.5
        mem = beta_dyn * state["mem"] + x_t

        kappa = F.softplus(self.w_kappa(x_t))
        lam_0 = F.softplus(self.w_lambda_q(x_t))
        lam_max = F.softplus(self.w_lambda_cl(x_t))
        delta = F.softplus(self.w_delta(x_t)) + 0.01

        lam_cl = lam_max * torch.sigmoid((mem - self.threshold) / delta)
        lam_q = lam_0 * torch.exp((-kappa * (self.phi_0 - self.eta * mem)).clamp(-10, 10))
        prob = (1.0 - torch.exp(-(lam_cl + lam_q))).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


REGISTRY = {
    "L-TH-LIF": LTHLIF,
    "L-BH-LIF": LBHLIF,
}
