"""Train a GAN on MNIST with the Tw/OBRS generator loss."""
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from gan.model import MNIST_DIM, Discriminator, Generator
from gan.obrs import D_train, G_obrs_train
from gan.utils import get_device, mnist_loader, save_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN on MNIST with Tw/OBRS.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--K", type=float, default=2.0, help="OBRS budget used in the generator loss.")
    args = parser.parse_args()

    device = get_device()
    loader = mnist_loader(train=True, batch_size=args.batch_size, shuffle=True)

    G, D = Generator().to(device), Discriminator().to(device)
    if torch.cuda.device_count() > 1:
        G, D = nn.DataParallel(G), nn.DataParallel(D)

    criterion = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    print("Start training")
    for epoch in range(1, args.epochs + 1):
        for x, _ in loader:
            x = x.view(-1, MNIST_DIM).to(device)
            # D is updated once for every two G updates to keep it from dominating
            D_train(x, G, D, D_optimizer, criterion, device)
            G_obrs_train(x, G, D, G_optimizer, device, K=args.K)
            G_obrs_train(x, G, D, G_optimizer, device, K=args.K)

        if epoch % 10 == 0:
            save_models(G, D)
            print(f"Epoch {epoch}/{args.epochs} — checkpoints saved")

    print("Training done")
