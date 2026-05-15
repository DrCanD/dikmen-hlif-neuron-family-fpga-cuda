"""
H-LIF Family: Hazard-Based Leaky Integrate-and-Fire
Axis: How to make the spike decision? (threshold mechanism variations)
11 models: Det, Stoch, BH, TH, DB, RT, Thermal, CH, MoE, Att, HyperNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import BaseNeuron, spike_hard, spike_stochastic


class DetLIF(BaseNeuron):
    """#1 Deterministic LIF. V >= theta -> spike. Zero learnable neuron params."""
    _family = "h_lif"
    _description = "Fixed threshold, deterministic. V >= theta -> spike."

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        spk = spike_hard(mem, self.threshold)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class StochLIF(BaseNeuron):
    """#2 Stochastic LIF. Fixed threshold + Gaussian noise injection."""
    _family = "h_lif"
    _description = "Fixed threshold + Gaussian noise injection."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_sigma = nn.Parameter(torch.full((size,), -1.0))
        self.log_delta = nn.Parameter(torch.full((size,), 0.0))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        sigma = self.log_sigma.exp()
        delta = self.log_delta.exp()
        if self.training:
            noise = torch.randn_like(mem) * sigma
            prob = torch.sigmoid((mem + noise - self.threshold) / delta)
        else:
            prob = torch.sigmoid((mem - self.threshold) / delta)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class BHLIF(BaseNeuron):
    """#3 Barrier-Hazard LIF. Dual-pathway: classical sigmoid + barrier-crossing."""
    _family = "h_lif"
    _description = "Dual-pathway hazard: classical sigmoid + barrier-crossing."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lambda_max = nn.Parameter(torch.full((size,), 1.0))
        self.log_delta = nn.Parameter(torch.full((size,), 0.0))
        self.log_lambda_0 = nn.Parameter(torch.full((size,), -2.0))
        self.log_kappa = nn.Parameter(torch.full((size,), 0.5))
        self.eta = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.5))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        lam_max = self.log_lambda_max.exp()
        delta = self.log_delta.exp()
        lam_0 = self.log_lambda_0.exp()
        kappa = self.log_kappa.exp()
        lam_cl = lam_max * torch.sigmoid((mem - self.threshold) / delta)
        lam_q = lam_0 * torch.exp((-kappa * (self.phi_0 - self.eta * mem)).clamp(-10, 10))
        prob = (1.0 - torch.exp(-(lam_cl + lam_q))).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class THLIF(BaseNeuron):
    """#4 Tunneling-Hazard LIF. Sub-threshold firing via quantum tunneling analogy."""
    _family = "h_lif"
    _description = "Sub-threshold firing via quantum tunneling analogy. Flagship model."

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
        lam_q = lam_0 * torch.exp((-kappa * (self.phi_0 - self.eta * mem)).clamp(-10, 10))
        prob = (1.0 - torch.exp(-lam_q)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class DBLIF(BaseNeuron):
    """#5 Double-Barrier LIF. Bistable: two sequential barriers, double-well potential."""
    _family = "h_lif"
    _description = "Bistable: two sequential barriers, double-well potential."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.phi_1 = nn.Parameter(torch.full((size,), 0.6))
        self.phi_2 = nn.Parameter(torch.full((size,), 1.2))
        self.alpha_up = nn.Parameter(torch.full((size,), 0.8))
        self.alpha_down = nn.Parameter(torch.full((size,), 0.3))
        self.log_beta_rest = nn.Parameter(torch.full((size,), math.log(0.95)))
        self.log_beta_exc = nn.Parameter(torch.full((size,), math.log(0.98)))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "excited": torch.zeros(batch_size, self.size, device=device),
        }

    def single_step(self, x_t, state):
        mem, exc = state["mem"], state["excited"]
        beta_eff = exc * self.log_beta_exc.exp() + (1 - exc) * self.log_beta_rest.exp()
        mem = beta_eff * mem + x_t

        go_up = (mem > self.phi_1).float() * (1 - exc)
        exc = exc + go_up * torch.sigmoid(self.alpha_up)
        exc = exc.clamp(0, 1)

        spk = spike_hard(mem, self.phi_2)
        go_down = spk
        exc = exc * (1 - go_down)
        mem = mem - spk * self.phi_2
        return spk, {"mem": mem, "excited": exc}


class RTLIF(BaseNeuron):
    """#6 Resonant-Tunneling LIF. Band-pass: Gaussian hazard peaks at resonant voltage."""
    _family = "h_lif"
    _description = "Band-pass: Gaussian hazard peaks at resonant voltage."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.v_peak = nn.Parameter(torch.full((size,), 0.8))
        self.log_sigma = nn.Parameter(torch.full((size,), -0.5))
        self.log_lambda_0 = nn.Parameter(torch.full((size,), 0.0))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        sigma = self.log_sigma.exp()
        lam = self.log_lambda_0.exp() * torch.exp(-0.5 * ((mem - self.v_peak) / sigma) ** 2)
        prob = (1.0 - torch.exp(-lam)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class ThermalLIF(BaseNeuron):
    """#7 Thermal-Hazard LIF. Arrhenius kinetics: learnable temperature controls barrier crossing rate."""
    _family = "h_lif"
    _description = "Arrhenius kinetics: learnable temperature controls barrier crossing rate."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lambda_0 = nn.Parameter(torch.full((size,), 0.0))
        self.E_a = nn.Parameter(torch.full((size,), 1.0))
        self.log_T = nn.Parameter(torch.full((size,), 0.0))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        T = self.log_T.exp().clamp(min=0.01)
        lam = self.log_lambda_0.exp() * torch.exp((-self.E_a / T).clamp(-10, 10))
        eff_lam = lam * torch.sigmoid(mem - self.threshold * 0.5)
        prob = (1.0 - torch.exp(-eff_lam)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class CHLIF(BaseNeuron):
    """#8 Coupled-Hazard LIF. Lateral coupling: neighbors modulate each other's barrier height."""
    _family = "h_lif"
    _description = "Lateral coupling: neighbors modulate each other's barrier height."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lam = nn.Parameter(torch.full((size,), -1.0))
        self.coupling = nn.Parameter(torch.randn(size, size) * 0.01)
        self.log_kappa = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.2))
        self.eta = nn.Parameter(torch.full((size,), 0.7))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        lam = self.log_lam.exp()
        kappa = self.log_kappa.exp()
        lateral = torch.matmul(mem, torch.tanh(self.coupling))
        eff_phi = self.phi_0 + 0.1 * lateral
        barrier = eff_phi - self.eta * mem
        h = lam * torch.exp((-kappa * barrier).clamp(-10, 10))
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class MoELIF(BaseNeuron):
    """#9 Mixture-of-Experts Hazard LIF. Multiple expert hazard functions + learned gating."""
    _family = "h_lif"
    _description = "Multiple expert hazard functions + learned gating network."

    def __init__(self, size, beta=0.95, threshold=1.0, n_experts=3):
        super().__init__(size, beta, threshold)
        self.n_experts = n_experts
        self.log_lam = nn.Parameter(torch.randn(n_experts, size) * 0.5 - 1.5)
        self.log_kappa = nn.Parameter(torch.randn(n_experts, size) * 0.3 + 0.5)
        self.phi = nn.Parameter(torch.randn(n_experts, size) * 0.3 + 1.0)
        self.eta = nn.Parameter(torch.randn(n_experts, size) * 0.2 + 0.7)
        self.gate = nn.Linear(size, n_experts)

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        gate_w = F.softmax(self.gate(mem), dim=-1)
        h_total = torch.zeros_like(mem)
        for e in range(self.n_experts):
            lam = self.log_lam[e].exp()
            kap = self.log_kappa[e].exp()
            barrier = self.phi[e] - self.eta[e] * mem
            h_e = lam * torch.exp((-kap * barrier).clamp(-10, 10))
            h_total = h_total + gate_w[:, e:e + 1] * h_e
        prob = (1.0 - torch.exp(-h_total)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class AttLIF(BaseNeuron):
    """#10 Attention-Hazard LIF. Temporal attention spotlight lowers barrier at salient moments."""
    _family = "h_lif"
    _description = "Temporal attention spotlight lowers barrier at salient moments."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.log_lambda_0 = nn.Parameter(torch.full((size,), -1.5))
        self.log_kappa = nn.Parameter(torch.full((size,), 0.5))
        self.phi_0 = nn.Parameter(torch.full((size,), 1.2))
        self.eta = nn.Parameter(torch.full((size,), 0.7))
        self.attn_key = nn.Linear(size, size // 4)
        self.attn_query = nn.Linear(size, size // 4)
        self.attn_scale = nn.Parameter(torch.tensor(0.1))

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        lambda_0 = self.log_lambda_0.exp()
        kappa = self.log_kappa.exp()
        q = self.attn_query(mem)
        k = self.attn_key(mem)
        attn_score = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        effective_phi = self.phi_0 - self.attn_scale * attn_score
        barrier = effective_phi - self.eta * mem
        h = lambda_0 * torch.exp((-kappa * barrier).clamp(-10, 10))
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


class HyperNetLIF(BaseNeuron):
    """#11 HyperNetwork-Modulated Hazard LIF. Meta-network generates barrier params per timestep."""
    _family = "h_lif"
    _description = "Meta-network dynamically generates barrier parameters per timestep."

    def __init__(self, size, beta=0.95, threshold=1.0, hidden=32):
        super().__init__(size, beta, threshold)
        self.hyper = nn.Sequential(
            nn.Linear(size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, size * 4),  # generates lambda_0, kappa, eta, phi_0
        )

    def single_step(self, x_t, state):
        mem = self.beta * state["mem"] + x_t
        params = self.hyper(mem)
        lam_0 = F.softplus(params[:, : self.size])
        kappa = F.softplus(params[:, self.size : 2 * self.size])
        eta = torch.sigmoid(params[:, 2 * self.size : 3 * self.size])
        phi_0 = params[:, 3 * self.size :]
        barrier = phi_0 - eta * mem
        h = lam_0 * torch.exp((-kappa * barrier).clamp(-10, 10))
        prob = (1.0 - torch.exp(-h)).clamp(0, 1)
        spk = spike_stochastic(prob, self.training)
        mem = mem - spk * self.threshold
        return spk, {"mem": mem}


REGISTRY = {
    "Det-LIF": DetLIF,
    "Stoch-LIF": StochLIF,
    "BH-LIF": BHLIF,
    "TH-LIF": THLIF,
    "DB-LIF": DBLIF,
    "RT-LIF": RTLIF,
    "Thermal-LIF": ThermalLIF,
    "CH-LIF": CHLIF,
    "MoE-LIF": MoELIF,
    "Att-LIF": AttLIF,
    "HyperNet-LIF": HyperNetLIF,
}
