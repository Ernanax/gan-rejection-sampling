import os

import torch
from torchvision import datasets, transforms

from gan.model import Discriminator, Generator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
DATA_DIR = os.path.join(ROOT, "data")


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


def mnist_loader(train, batch_size, data_dir=DATA_DIR, shuffle=False):
    """MNIST scaled to [-1, 1] (matches the generator's tanh output). Downloaded on first use."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=4, pin_memory=torch.cuda.is_available())


def save_models(G, D, folder=CHECKPOINT_DIR):
    os.makedirs(folder, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(folder, "G.pth"))
    torch.save(D.state_dict(), os.path.join(folder, "D.pth"))


def _load(model, path, device):
    ckpt = torch.load(path, map_location=device)
    # Strip the "module." prefix added by DataParallel
    model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()})
    return model.to(device).eval()


def load_models(device, folder=CHECKPOINT_DIR):
    G = _load(Generator(), os.path.join(folder, "G.pth"), device)
    D = _load(Discriminator(), os.path.join(folder, "D.pth"), device)
    return G, D
