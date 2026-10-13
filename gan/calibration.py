"""Discriminator calibration (needed by MH-GAN, whose acceptance ratio relies on D(x))."""
import pickle

import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from gan.model import LATENT_DIM


def collect_scores(D, G, real_loader, n_samples, device):
    """Raw D scores on n_samples real (label 1) and n_samples fake (label 0) images."""
    scores, labels = [], []
    with torch.no_grad():
        for imgs, _ in real_loader:
            d = D(imgs.view(imgs.size(0), -1).to(device)).squeeze().clamp(1e-6, 1 - 1e-6)
            scores.append(d.cpu())
            labels.append(torch.ones(d.shape[0]))
            if sum(s.shape[0] for s in scores) >= n_samples:
                break
        collected = 0
        while collected < n_samples:
            d = D(G(torch.randn(128, LATENT_DIM, device=device))).squeeze().clamp(1e-6, 1 - 1e-6)
            scores.append(d.cpu())
            labels.append(torch.zeros(d.shape[0]))
            collected += 128
    return torch.cat(scores)[:2 * n_samples].numpy(), torch.cat(labels)[:2 * n_samples].numpy()


def dawid_z(scores, labels):
    """Dawid's Z statistic: |Z| < 2 means D is well calibrated."""
    Z = np.sum(labels - scores) / np.sqrt(np.sum(scores * (1 - scores)))
    print(f"Z-statistic: {Z:.4f} ({'well calibrated' if abs(Z) < 2 else 'miscalibrated'})")
    return Z


def fit_calibrator(scores, labels, method="logistic"):
    if method == "logistic":
        calibrator = LogisticRegression().fit(scores.reshape(-1, 1), labels)
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'logistic' or 'isotonic'.")
    dawid_z(apply_calibrator(torch.tensor(scores), calibrator).numpy(), labels)
    return calibrator


def apply_calibrator(raw_scores, calibrator):
    """Map raw D scores (any shape) to calibrated probabilities, same shape."""
    s = raw_scores.detach().cpu().numpy().reshape(-1, 1)
    if isinstance(calibrator, LogisticRegression):
        calibrated = calibrator.predict_proba(s)[:, 1]
    else:
        calibrated = calibrator.predict(s.reshape(-1))
    return torch.tensor(calibrated, dtype=torch.float32).reshape(raw_scores.shape)


def save_calibrator(calibrator, path):
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator(path):
    with open(path, "rb") as f:
        return pickle.load(f)
