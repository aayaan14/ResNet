import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def build_transforms(strategy: str):
    """
    Returns (train_transform, test_transform) based on augmentation strategy.

    Strategies:
        baseline    - RandomCrop + HFlip + Normalize
        cutout      - baseline + Cutout (erasing a random patch)
        autoaugment - baseline + AutoAugment (CIFAR-10 policy)
    """
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    )

    base_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if strategy == "baseline":
        train_transforms = base_train + [transforms.ToTensor(), normalize]

    elif strategy == "cutout":
        # RandomErasing is PyTorch's built-in Cutout equivalent
        # Applied after ToTensor since it operates on tensors
        train_transforms = base_train + [
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(1.0, 1.0), value=0),
        ]

    elif strategy == "autoaugment":
        train_transforms = base_train + [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            normalize,
        ]

    else:
        raise ValueError(f"Unknown augmentation strategy: '{strategy}'")

    test_transforms = [transforms.ToTensor(), normalize]

    return transforms.Compose(train_transforms), transforms.Compose(test_transforms)


def build_dataloaders(cfg: dict):
    """
    Builds train and test DataLoaders from config.

    Expected cfg keys: root, batch_size, num_workers, augmentation.strategy
    """
    train_transform, test_transform = build_transforms(cfg["augmentation"]["strategy"])

    dataset = cfg["data"]["dataset"]
    root = cfg["data"]["root"]

    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=train_transform)
        testset  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'")

    trainloader = DataLoader(
        trainset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    return trainloader, testloader