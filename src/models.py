"""Baseline model: pretrained CIFAR-10 ResNet wrapped with input normalization."""
import torch
import torch.nn as nn

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class Normalized(nn.Module):
    def __init__(self, model, mean=CIFAR_MEAN, std=CIFAR_STD):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def load_baseline(arch="cifar10_resnet20", device="cuda"):
    net = torch.hub.load("chenyaofo/pytorch-cifar-models", arch, pretrained=True)
    return Normalized(net).to(device).eval()


def load_checkpoint(path, arch="cifar10_resnet20", device="cuda"):
    """Load a defended model saved by defenses/adv_train.py."""
    model = load_baseline(arch, device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()
