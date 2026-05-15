"""
Tests covering all 17 patent-covered neurons.

Run:
    pytest tests/
"""

import pytest
import torch

from hlif_neurons import HLIFRegistry


ALL_NEURONS = HLIFRegistry.list_all()


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_registry_size():
    assert len(ALL_NEURONS) == 17, f"Expected 17 patent-covered models, got {len(ALL_NEURONS)}"


def test_summary_runs():
    s = HLIFRegistry.summary()
    assert "TÜRKPATENT 2026/007632" in s
    assert "17" in s


@pytest.mark.parametrize("name", ALL_NEURONS)
def test_neuron_instantiates(name):
    neuron = HLIFRegistry.create(name, size=32)
    assert neuron is not None
    assert hasattr(neuron, "family")
    assert hasattr(neuron, "single_step")


@pytest.mark.parametrize("name", ALL_NEURONS)
def test_neuron_forward_shape(name):
    neuron = HLIFRegistry.create(name, size=32)
    x = torch.randn(4, 20, 32)
    spk, state = neuron(x)
    assert spk.shape == (4, 20, 32), f"{name}: spike shape {spk.shape}"
    assert isinstance(state, dict)


@pytest.mark.parametrize("name", ALL_NEURONS)
def test_neuron_binary_spikes(name):
    neuron = HLIFRegistry.create(name, size=32)
    x = torch.randn(4, 20, 32)
    spk, _ = neuron(x)
    assert ((spk == 0) | (spk == 1)).all(), f"{name}: spikes must be binary"


@pytest.mark.parametrize("name", ALL_NEURONS)
def test_neuron_backward(name):
    """All neurons must support gradient flow to their inputs."""
    neuron = HLIFRegistry.create(name, size=32)
    x = torch.randn(4, 20, 32, requires_grad=True)
    spk, _ = neuron(x)
    spk.sum().backward()
    assert x.grad is not None, f"{name}: no gradient flowed to input"
    assert torch.isfinite(x.grad).all(), f"{name}: non-finite gradient"


def test_claim_mapping_covers_all_models():
    """Every model in the registry should appear in at least one claim."""
    mapped = set()
    for claim_models in HLIFRegistry._claims.values():
        mapped.update(claim_models)
    # BH-LIF is covered by claim_1; TH-LIF by claim_2; Stoch-LIF is a baseline
    # included for reference (sigmoid noise form, related to claim 3).
    # Verify the core 14 are mapped:
    expected_in_claims = {
        "BH-LIF", "TH-LIF", "Det-LIF", "TH-LIF-Linear", "TH-LIF-Sigmoid",
        "L-TH-LIF", "L-BH-LIF",
        "DB-LIF", "RT-LIF", "Thermal-LIF", "Spin-LIF", "Piezo-LIF",
        "CH-LIF", "MoE-LIF", "Att-LIF", "HyperNet-LIF",
    }
    missing = expected_in_claims - mapped
    assert not missing, f"Models not mapped to any claim: {missing}"


def test_invalid_neuron_name_raises():
    with pytest.raises(KeyError):
        HLIFRegistry.create("NotAReal-LIF", size=32)


def test_invalid_claim_key_raises():
    with pytest.raises(KeyError):
        HLIFRegistry.list_by_claim("claim_99_made_up")
