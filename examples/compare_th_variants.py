"""
Compare the four realization forms of the TH-LIF hazard from patent claim 3:

    1. Exponential (canonical):  λ = λ₀ · exp(−κ(φ₀ − η·V))
    2. Linearized:               λ = λ₀ · relu(1 + κ(η·V − φ₀))
    3. Sigmoid:                  λ = λ_max · sigmoid(κ(η·V − φ₀))
    4. λ₀ = 0 deterministic:     V ≥ θ → spike (Heaviside)

All four are run on the same noisy current input and the resulting spike
trains are compared on rate, jitter, and average membrane.

Run:
    python examples/compare_th_variants.py
"""

import torch

from hlif_neurons import HLIFRegistry


def main():
    torch.manual_seed(0)
    B, T, N = 8, 200, 64

    # Same input drives all four neurons
    x = torch.randn(B, T, N) * 0.4 + 0.6   # mildly positive drive

    variants = {
        "TH-LIF (exponential)": "TH-LIF",
        "TH-LIF-Linear       ": "TH-LIF-Linear",
        "TH-LIF-Sigmoid      ": "TH-LIF-Sigmoid",
        "Det-LIF (λ₀=0)      ": "Det-LIF",
    }

    print(f"{'Variant':24s} | {'mean_rate':>9s} | {'spk_var':>8s} | {'params':>7s}")
    print("-" * 60)

    for label, model_key in variants.items():
        neuron = HLIFRegistry.create(model_key, size=N)
        with torch.no_grad():
            spk, _ = neuron(x)
        rate = spk.mean().item()
        var = spk.var().item()
        params = neuron.learnable_param_count
        print(f"{label} | {rate:9.4f} | {var:8.4f} | {params:7d}")

    print()
    print("Interpretation:")
    print("  - Exponential canonical: rate sensitive to (κ, η, φ₀) jointly")
    print("  - Linearized: cheaper on hardware (no exp LUT), saturates supra-θ")
    print("  - Sigmoid: gradient-stable, saturates at λ_max in supra-θ regime")
    print("  - Det-LIF (λ₀=0): no stochasticity, baseline LIF behavior")


if __name__ == "__main__":
    main()
