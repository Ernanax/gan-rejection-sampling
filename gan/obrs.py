"""
Optimal Budgeted Rejection Sampling (OBRS) and Training with OBRS (Tw/OBRS).

Budget K = expected number of generated samples needed per accepted sample.
"""
import torch

from gan.model import LATENT_DIM


def compute_gamma(r_x, K, max_iter=100, eps=1e-6):
    """Binary search for gamma such that mean(min(gamma·r(x), 1)) = 1/K."""
    gamma_min, gamma_max = 0.0, 1000.0
    for _ in range(max_iter):
        gamma = (gamma_min + gamma_max) / 2
        budget = torch.mean(torch.clamp(gamma * r_x, max=1.0))
        if budget > 1 / K:
            gamma_max = gamma
        else:
            gamma_min = gamma
        if torch.abs(budget - 1 / K) < eps:
            break
    return gamma


def acceptance(r_x, gamma):
    """Optimal acceptance probability a_O(x)."""
    return torch.clamp(r_x * gamma, max=1.0)


def f_div(u):
    """Generator of the GAN f-divergence."""
    return u * torch.log(u + 1e-9) - (u + 1) * torch.log(u + 1 + 1e-9)


def D_train(x, G, D, D_optimizer, criterion, device):
    """Standard BCE discriminator step."""
    D.zero_grad()
    y_real = torch.ones(x.shape[0], 1, device=device)
    y_fake = torch.zeros(x.shape[0], 1, device=device)
    x_fake = G(torch.randn(x.shape[0], LATENT_DIM, device=device)).detach()

    D_loss = criterion(torch.sigmoid(D(x)), y_real) + criterion(torch.sigmoid(D(x_fake)), y_fake)
    D_loss.backward()
    D_optimizer.step()
    return D_loss.item()


def G_obrs_train(x, G, D, G_optimizer, device, K=2.0):
    """Generator step with the Tw/OBRS loss."""
    G.zero_grad()
    x_fake = G(torch.randn(x.shape[0], LATENT_DIM, device=device))
    prob = torch.clamp(torch.sigmoid(D(x_fake)), 1e-6, 1 - 1e-6)

    r_x = prob / (1.0 - prob)

    weights = acceptance(r_x.detach(), compute_gamma(r_x.detach(), K))

    G_loss = torch.mean(K * weights * f_div(r_x / (K * weights)))
    G_loss.backward()
    G_optimizer.step()
    return G_loss.item()
