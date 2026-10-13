"""Generate MNIST samples with a given sampling method and save them as PNG files."""
import argparse
import os

import torchvision

from gan.calibration import load_calibrator
from gan.sampling import METHODS, sample
from gan.utils import CHECKPOINT_DIR, ROOT, get_device, load_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MNIST samples.")
    parser.add_argument("--method", default="obrs", choices=METHODS)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--K", type=int, default=10, help="OBRS budget / MH-GAN chain length.")
    parser.add_argument("--psi", type=float, default=0.7, help="Truncation parameter.")
    parser.add_argument("--out_dir", default=os.path.join(ROOT, "samples"))
    args = parser.parse_args()

    device = get_device()
    G, D = load_models(device)
    calibrator = load_calibrator(os.path.join(CHECKPOINT_DIR, "calibrator.pkl")) \
        if args.method == "mh-calibrated" else None

    print(f"Generating {args.n_samples} samples with '{args.method}'...")
    imgs = sample(args.method, G, D, args.n_samples, device, K=args.K, psi=args.psi, calibrator=calibrator)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, x in enumerate(imgs):
        torchvision.utils.save_image(x.view(1, 28, 28), os.path.join(args.out_dir, f"{i}.png"))
    print(f"Samples saved to {args.out_dir}")
