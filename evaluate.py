"""Evaluate one or several sampling methods with FID, precision, recall and generation time."""
import argparse
import os
import time

from gan.calibration import load_calibrator
from gan.metrics import fid, inception_features, load_inception, precision_recall, real_features
from gan.sampling import METHODS, sample
from gan.utils import CHECKPOINT_DIR, get_device, load_models, mnist_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampling methods.")
    parser.add_argument("--methods", nargs="+", default=["baseline", "obrs"], choices=METHODS)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--K", type=int, nargs="+", default=[10], help="One or several values to sweep.")
    parser.add_argument("--psi", type=float, nargs="+", default=[0.7], help="One or several values to sweep.")
    args = parser.parse_args()

    device = get_device()
    G, D = load_models(device)
    calibrator = load_calibrator(os.path.join(CHECKPOINT_DIR, "calibrator.pkl")) \
        if "mh-calibrated" in args.methods else None

    print("Extracting real features (MNIST test set)...")
    inception = load_inception(device)
    real = real_features(mnist_loader(train=False, batch_size=256), inception, device, args.n_samples)

    # Each method is swept over its own parameter: K for OBRS/MH-GAN, psi for truncations
    runs = []
    for m in args.methods:
        if m in ("obrs", "mh", "mh-calibrated"):
            runs += [(m, {"K": k}) for k in args.K]
        elif m in ("hard-trunc", "soft-trunc"):
            runs += [(m, {"psi": p}) for p in args.psi]
        else:
            runs.append((m, {}))

    print(f"\n{'Method':<15}{'Params':<12}{'FID':>8}{'Precision':>11}{'Recall':>8}{'Time (s)':>10}")
    for method, params in runs:
        t0 = time.time()
        fake = sample(method, G, D, args.n_samples, device, calibrator=calibrator, **params)
        elapsed = time.time() - t0
        feats = inception_features(fake, inception, device)
        p, r = precision_recall(real, feats)
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"{method:<15}{param_str:<12}{fid(real, feats):>8.2f}{p:>11.3f}{r:>8.3f}{elapsed:>10.1f}")
