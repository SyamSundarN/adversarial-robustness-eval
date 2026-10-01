"""CIFAR-10 loaders. Images stay in [0, 1]; normalization lives inside the model
so attacks operate in pixel space and eps values mean what they say."""
import torch
from torchvision import datasets, transforms


def get_loaders(batch_size=128, root="./data", num_workers=2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_tf = transforms.ToTensor()
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
