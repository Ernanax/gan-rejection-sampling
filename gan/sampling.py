"""
Sampling methods. Each returns a tensor of shape (n_samples, 784) with values in [-1, 1].
"""
import torch

from gan.calibration import apply_calibrator
from gan.model import LATENT_DIM
from gan.obrs import acceptance, compute_gamma


@torch.no_grad()
def baseline(G, n_samples, device, batch_size=256):
    """Plain generator samples."""
    out = [G(torch.randn(batch_size, LATENT_DIM, device=device)).cpu()
           for _ in range(-(-n_samples // batch_size))]
    return torch.cat(out)[:n_samples]


@torch.no_grad()
def obrs(G, D, n_samples, device, K=10, batch_size=500):
    """Optimal Budgeted Rejection Sampling with budget K."""
    accepted, n = [], 0
    while n < n_samples:
        x = G(torch.randn(batch_size, LATENT_DIM, device=device))
        d = D(x).clamp(1e-6, 1 - 1e-6)
        r_x = d / (1 - d + 1e-6)
        keep = (torch.rand_like(d) <= acceptance(r_x, compute_gamma(r_x, K))).squeeze(1)
        accepted.append(x[keep].cpu())
        n += int(keep.sum())
    return torch.cat(accepted)[:n_samples]


@torch.no_grad()
def mh_gan(G, D, n_samples, device, K=10, calibrator=None, chain_batch_size=128):
    """
    Metropolis-Hastings GAN: one Markov chain of length K per sample, run in parallel batches.
    Chains start from a generated sample so no real image can be returned.
    """
    def score(x):
        d = D(x)
        if calibrator is not None:
            d = apply_calibrator(d, calibrator).to(device)
        return d.clamp(1e-6, 1 - 1e-6)

    out, n_accepted, n_proposals = [], 0, 0
    while sum(len(o) for o in out) < n_samples:
        b = min(chain_batch_size, n_samples - sum(len(o) for o in out))
        x = G(torch.randn(b, LATENT_DIM, device=device))
        for _ in range(K):
            x_new = G(torch.randn(b, LATENT_DIM, device=device))
            ratio = ((1.0 / score(x) - 1.0) / (1.0 / score(x_new) - 1.0)).clamp(max=1.0)
            accept = (torch.rand(b, 1, device=device) <= ratio).squeeze(1)
            x[accept] = x_new[accept]
            n_accepted += int(accept.sum())
            n_proposals += b
        out.append(x.cpu())
    print(f"MH-GAN acceptance rate: {n_accepted / n_proposals:.4f}")
    return torch.cat(out)


@torch.no_grad()
def hard_truncation(G, n_samples, device, psi=0.7, batch_size=256):
    """Reject latent vectors whose norm exceeds psi·sqrt(latent_dim)."""
    out, n = [], 0
    while n < n_samples:
        z = torch.randn(batch_size * 4, LATENT_DIM, device=device)
        z = z[z.norm(dim=1) <= psi * LATENT_DIM ** 0.5]
        if len(z):
            out.append(G(z).cpu())
            n += len(z)
    return torch.cat(out)[:n_samples]


@torch.no_grad()
def soft_truncation(G, n_samples, device, psi=0.7, batch_size=256):
    """Scale latent vectors by psi (pulls samples toward the mode)."""
    out = [G(psi * torch.randn(batch_size, LATENT_DIM, device=device)).cpu()
           for _ in range(-(-n_samples // batch_size))]
    return torch.cat(out)[:n_samples]


def sample(method, G, D, n_samples, device, K=10, psi=0.7, calibrator=None):
    if method == "baseline":
        return baseline(G, n_samples, device)
    if method == "obrs":
        return obrs(G, D, n_samples, device, K=K)
    if method == "mh":
        return mh_gan(G, D, n_samples, device, K=K)
    if method == "mh-calibrated":
        return mh_gan(G, D, n_samples, device, K=K, calibrator=calibrator)
    if method == "hard-trunc":
        return hard_truncation(G, n_samples, device, psi=psi)
    if method == "soft-trunc":
        return soft_truncation(G, n_samples, device, psi=psi)
    raise ValueError(f"Unknown method: {method}")


METHODS = ["baseline", "obrs", "mh", "mh-calibrated", "hard-trunc", "soft-trunc"]
