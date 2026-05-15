"""
Demo: use the custom TH-LIF CUDA kernel.

Requires a CUDA-capable GPU and cupy matching the installed CUDA version
(e.g. `pip install -e ".[cuda]"`).

Run:
    python examples/cuda_kernel_demo.py
"""

import sys
import time

import torch


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; this demo needs a GPU.")
        sys.exit(0)

    try:
        # Path setup so the cuda/ subpackage imports cleanly
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from cuda.wrappers.th_lif_v1 import CUDATHLIF
    except ImportError as e:
        print(f"Could not import the CUDA wrapper: {e}")
        print("Install cupy with: pip install -e \".[cuda]\"")
        sys.exit(0)

    torch.manual_seed(42)
    device = "cuda"

    # Build a TH-LIF layer with the canonical hyperparameters
    layer = CUDATHLIF(
        n_in=64, n_out=128,
        beta=0.95, threshold=1.0,
        lam0=0.1, kappa=2.0, eta=1.5, phi0=1.0,
    ).to(device)

    # Forward pass
    B, T, N_in = 32, 256, 64
    x = torch.randn(B, T, N_in, device=device, requires_grad=True)
    spk = layer(x)

    assert spk.shape == (B, T, 128), f"shape mismatch: {spk.shape}"
    assert ((spk == 0) | (spk == 1)).all(), "spikes must be binary"
    print(f"[OK] Forward: shape={tuple(spk.shape)}, mean_rate={spk.mean().item():.4f}")

    # Backward pass (autograd through the custom kernel)
    loss = spk.sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print(f"[OK] Backward: input_grad_norm={x.grad.norm().item():.4f}")

    # Determinism
    with torch.no_grad():
        spk2 = layer(x.detach())
    assert (spk == spk2).all(), "non-deterministic forward!"
    print("[OK] Deterministic forward verified")

    # Speed: compare against a naive PyTorch implementation
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        _ = layer(x.detach())
    torch.cuda.synchronize()
    t_cuda = (time.time() - t0) * 1000
    print(f"[OK] Speed (CUDA kernel): 100 forwards (B={B}, T={T}, N=64): {t_cuda:.1f} ms")


if __name__ == "__main__":
    main()
