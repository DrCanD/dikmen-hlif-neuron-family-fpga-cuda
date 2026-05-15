"""
Base neuron interface and surrogate gradient utilities.
All 34 models inherit from BaseNeuron.
"""

import torch
import torch.nn as nn
import math
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════
#  Surrogate Gradients
# ═══════════════════════════════════════════════════════════

class FastSigmoidSurrogate(torch.autograd.Function):
    """Surrogate gradient: forward = Heaviside, backward = fast sigmoid."""
    scale = 25.0

    @staticmethod
    def forward(ctx, membrane, threshold):
        if not torch.is_tensor(threshold):
            threshold = membrane.new_tensor(threshold)
        ctx.save_for_backward(membrane, threshold)
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        grad = grad_output / (FastSigmoidSurrogate.scale * (membrane - threshold).abs() + 1.0) ** 2
        return grad, None


class StraightThroughBernoulli(torch.autograd.Function):
    """Straight-through estimator for Bernoulli sampling."""
    @staticmethod
    def forward(ctx, prob):
        return torch.bernoulli(prob)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def spike_hard(mem, threshold=1.0):
    """Deterministic spike with surrogate gradient."""
    return FastSigmoidSurrogate.apply(mem, threshold)


def spike_stochastic(prob, training=True):
    """Stochastic spike with straight-through gradient."""
    if training:
        spk = StraightThroughBernoulli.apply(prob)
        return spk - prob.detach() + prob
    else:
        return (prob > 0.5).float()


# ═══════════════════════════════════════════════════════════
#  Base Neuron
# ═══════════════════════════════════════════════════════════

class BaseNeuron(nn.Module, ABC):
    """
    Common interface for all Dikmen spiking neuron models.

    Forward signature:
        spikes, state = neuron(x, state=None)
        x:      [batch, time, features] or [batch, features] (single step)
        spikes: same shape as x
        state:  dict with model-specific state tensors

    All temporal unrolling happens inside forward().
    """

    _family = "base"
    _description = "Abstract base neuron"

    def __init__(self, size: int, beta: float = 0.95, threshold: float = 1.0):
        super().__init__()
        self.size = size
        self.beta = beta
        self.threshold = threshold

    @abstractmethod
    def single_step(self, x_t, state):
        """Process one timestep. Returns (spike, new_state)."""
        ...

    def init_state(self, batch_size, device=None):
        """Default state: single membrane tensor of zeros."""
        device = device or next(self.parameters(), torch.tensor(0.0)).device
        return {"mem": torch.zeros(batch_size, self.size, device=device)}

    def forward(self, x, state=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        if state is None:
            state = self.init_state(B, x.device)

        spikes = []
        for t in range(T):
            spk, state = self.single_step(x[:, t, :], state)
            spikes.append(spk)

        return torch.stack(spikes, dim=1), state

    @property
    def learnable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def family(self):
        return self._family

    @property
    def description(self):
        return self._description

    def extra_repr(self):
        return (
            f"size={self.size}, beta={self.beta}, threshold={self.threshold}, "
            f"family={self.family}, params={self.learnable_param_count}"
        )
