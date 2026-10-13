"""FID and precision/recall (k-NN manifolds) on Inception-v3 features."""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors
from torchvision.models import Inception_V3_Weights, inception_v3


def load_inception(device):
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


@torch.no_grad()
def inception_features(imgs, inception, device, batch_size=128):
    """imgs: (N, 784) or (N, 1, 28, 28) to (N, 2048) features."""
    imgs = imgs.view(-1, 1, 28, 28)
    feats = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i + batch_size].repeat(1, 3, 1, 1)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False).to(device)
        feats.append(inception(batch).cpu().numpy())
        if device.type == "mps":
            torch.mps.empty_cache()
    return np.concatenate(feats)


def real_features(loader, inception, device, n_samples):
    imgs, n = [], 0
    for x, _ in loader:
        imgs.append(x)
        n += len(x)
        if n >= n_samples:
            break
    return inception_features(torch.cat(imgs)[:n_samples], inception, device)


def fid(real, fake):
    mu_r, sigma_r = real.mean(0), np.cov(real, rowvar=False)
    mu_f, sigma_f = fake.mean(0), np.cov(fake, rowvar=False)
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_f
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def precision_recall(real, fake, k=5):
    """
    Precision: share of fake samples inside the real manifold.
    Recall: share of real samples inside the fake manifold.
    A manifold is the union of balls reaching each point's k-th nearest neighbour.
    """
    nn_real = NearestNeighbors(n_neighbors=k + 1).fit(real)
    nn_fake = NearestNeighbors(n_neighbors=k + 1).fit(fake)
    real_radii = nn_real.kneighbors(real)[0][:, k]
    fake_radii = nn_fake.kneighbors(fake)[0][:, k]

    d, idx = nn_real.kneighbors(fake, n_neighbors=1)
    precision = np.mean(d[:, 0] <= real_radii[idx[:, 0]])
    d, idx = nn_fake.kneighbors(real, n_neighbors=1)
    recall = np.mean(d[:, 0] <= fake_radii[idx[:, 0]])
    return float(precision), float(recall)
