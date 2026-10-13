# Rejection Sampling for GANs

Improving GAN samples on MNIST by using the discriminator to filter the generator's output. Two methods are implemented and compared with truncation baselines:

- **MH-GAN** ([Turner et al., 2019](https://arxiv.org/abs/1811.11357)): a Metropolis-Hastings chain of length K where the discriminator decides whether to accept each new proposal. Optional calibration of the discriminator (logistic or isotonic).
- **OBRS / Tw/OBRS** ([Verine et al., 2024](https://arxiv.org/abs/2311.00460)): Optimal Budgeted Rejection Sampling, which accepts each sample with an optimal probability under a budget K. Here OBRS is also used in the **training loss** of the generator.

Generator and discriminator are 4-layer MLPs. The details are in [report.pdf](report.pdf).

## Results

Averaged over 5 runs of 10,000 samples each.

| Method | Params | FID ↓ | Precision ↑ | Recall ↑ | Gen. time (s) |
|---|---|---|---|---|---|
| Baseline | — | 32.40 | 0.511 | 0.210 | 0.1 |
| MH-GAN | K = 10 | 26.85 | 0.525 | 0.241 | 2.8 |
| MH-GAN | K = 200 | 25.91 | 0.521 | 0.292 | 44.0 |
| Hard truncation | ψ = 0.7 | 36.84 | 0.405 | 0.312 | 0.1 |
| Soft truncation | ψ = 0.9 | 36.38 | 0.421 | 0.286 | 0.1 |
| Tw/OBRS | K = 2 | 42.6 | **0.56** | 0.31 | 0.5 |
| **Tw/OBRS** | **K = 10** | **23.7** | 0.48 | 0.30 | 5.1 |

Tw/OBRS with K = 10 gets the best FID and is about 9× faster than MH-GAN with K = 200.

## Project structure

```
gan/
├─ model.py         Generator and discriminator
├─ obrs.py          OBRS acceptance function and Tw/OBRS training losses
├─ sampling.py      Baseline, OBRS, MH-GAN, hard and soft truncation
├─ calibration.py   Discriminator calibration (Dawid's Z statistic, logistic and isotonic)
├─ metrics.py       FID, precision and recall on Inception-v3 features
└─ utils.py         Device, MNIST loader, checkpoint save and load
train.py            Train the GAN with Tw/OBRS
calibrate.py        Fit the discriminator calibrator
generate.py         Save samples as PNG images
evaluate.py         Compare sampling methods
checkpoints/        Trained G, D and calibrator
```

## Usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The trained models are included in `checkpoints/`, so you can generate samples or run evaluations directly. MNIST is downloaded to `data/` the first time it is needed. Each script detects CUDA, Apple Silicon (MPS) or CPU on its own.

**Generate samples**. The PNG images are written to `samples/`:

```bash
python generate.py --method obrs --K 10 --n_samples 10000
python generate.py --method mh-calibrated --K 50
python generate.py --method soft-trunc --psi 0.9
```

The available methods are `baseline`, `obrs`, `mh`, `mh-calibrated`, `hard-trunc` and `soft-trunc`.

**Evaluate methods** (FID, precision, recall and generation time). Each method is run once for each value of `--K` or `--psi`:

```bash
python evaluate.py --methods baseline obrs mh-calibrated --K 2 10 50
python evaluate.py --methods hard-trunc soft-trunc --psi 0.5 0.7 0.9
```

**Retrain** (this overwrites `checkpoints/G.pth` and `checkpoints/D.pth`):

```bash
python train.py --epochs 300
python calibrate.py --method logistic
```

## Authors

Emile Descroix, Anael Ernadotte. M2 IASD project (2025–2026).
