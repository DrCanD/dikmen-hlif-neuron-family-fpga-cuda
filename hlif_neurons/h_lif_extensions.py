"""
H-LIF Physics Branch Extensions: Spin-LIF (#30), Piezo-LIF (#31)
"""

import torch
import torch.nn as nn
import math

from .base import BaseNeuron, spike_hard


class SpinLIF(BaseNeuron):
    """#30 Magnetic Spin-Precession LIF. 2D spin vector precesses under input 'field'."""
    _family = "h_lif"
    _description = "Magnetic spin precession: 2D spin driven by input field, spike on spin-up."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.gamma = nn.Parameter(torch.full((size,), 1.0))
        self.log_T2 = nn.Parameter(torch.full((size,), math.log(10.0)))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "spin_x": torch.zeros(batch_size, self.size, device=device),
            "spin_z": torch.ones(batch_size, self.size, device=device) * -0.5,
        }

    def single_step(self, x_t, state):
        sx, sz = state["spin_x"], state["spin_z"]
        T2 = self.log_T2.exp()
        B = x_t
        dsx = self.gamma * (sz * B) - sx / T2
        dsz = -self.gamma * (sx * B) - (sz + 0.5) / T2
        sx = sx + dsx * 0.1
        sz = sz + dsz * 0.1
        mem = sz + 0.5
        spk = spike_hard(mem, self.threshold * 0.8)
        sx = sx * (1 - spk)
        sz = sz - spk * 0.5
        return spk, {"mem": mem, "spin_x": sx, "spin_z": sz}


class PiezoLIF(BaseNeuron):
    """#31 Piezoelectric-Stress LIF. Stress-dependent threshold: accumulated input shifts barrier."""
    _family = "h_lif"
    _description = "Piezoelectric analogy: accumulated stress shifts the firing threshold."

    def __init__(self, size, beta=0.95, threshold=1.0):
        super().__init__(size, beta, threshold)
        self.d33 = nn.Parameter(torch.full((size,), 1.0))
        self.k_s = nn.Parameter(torch.full((size,), 0.1))
        self.stress_decay = nn.Parameter(torch.full((size,), 0.99))

    def init_state(self, batch_size, device=None):
        device = device or next(self.parameters()).device
        return {
            "mem": torch.zeros(batch_size, self.size, device=device),
            "stress": torch.zeros(batch_size, self.size, device=device),
        }

    def single_step(self, x_t, state):
        stress = torch.sigmoid(self.stress_decay) * state["stress"] + x_t.abs()
        V = self.d33 * x_t
        dynamic_thresh = self.threshold + self.k_s * stress
        mem = self.beta * state["mem"] + V
        spk = spike_hard(mem, dynamic_thresh)
        mem = mem - spk * dynamic_thresh
        stress = stress * (1 - spk * 0.5)
        return spk, {"mem": mem, "stress": stress}


REGISTRY = {
    "Spin-LIF": SpinLIF,
    "Piezo-LIF": PiezoLIF,
}
