"""Fit a calibrator on the discriminator scores (used by --method mh-calibrated)."""
import argparse
import os

from gan.calibration import collect_scores, dawid_z, fit_calibrator, save_calibrator
from gan.utils import CHECKPOINT_DIR, get_device, load_models, mnist_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the discriminator.")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of real and of fake samples.")
    parser.add_argument("--method", default="logistic", choices=["logistic", "isotonic"])
    args = parser.parse_args()

    device = get_device()
    G, D = load_models(device)

    scores, labels = collect_scores(D, G, mnist_loader(train=False, batch_size=256), args.n_samples, device)

    print("Before calibration:")
    dawid_z(scores, labels)
    print(f"After {args.method} calibration:")
    calibrator = fit_calibrator(scores, labels, args.method)

    path = os.path.join(CHECKPOINT_DIR, "calibrator.pkl")
    save_calibrator(calibrator, path)
    print(f"Calibrator saved to {path}")
