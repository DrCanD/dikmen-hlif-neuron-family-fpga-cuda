"""
Quickstart: instantiate any of the 17 patent-covered models and run forward.

Run:
    python examples/quickstart.py
"""

import torch

from hlif_neurons import HLIFRegistry


def main():
    torch.manual_seed(0)

    print(HLIFRegistry.summary())
    print()

    # Instantiate any of the 17 neurons by name
    neuron = HLIFRegistry.create("TH-LIF", size=128)
    print(f"Created: {neuron}")
    print(f"  family: {neuron.family}")
    print(f"  params: {neuron.learnable_param_count}")
    print()

    # Standard forward pass
    B, T, N = 16, 50, 128
    x = torch.randn(B, T, N)
    spikes, state = neuron(x)

    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Spike  shape: {tuple(spikes.shape)}")
    print(f"Spike  mean rate: {spikes.mean().item():.4f}")
    print(f"State  keys: {list(state.keys())}")
    print()

    # Walk through claims and instantiate one model from each
    print("One model per patent claim:")
    for claim_key in ["claim_2_th_lif", "claim_3_variants", "claim_4_liquid",
                      "claim_5_physics", "claim_6_architecture"]:
        names = HLIFRegistry.list_by_claim(claim_key)
        first = names[0]
        n = HLIFRegistry.create(first, size=64)
        spk, _ = n(torch.randn(4, 20, 64))
        print(f"  {claim_key:25s} -> {first:15s} rate={spk.mean().item():.3f}")


if __name__ == "__main__":
    main()
