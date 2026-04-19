import argparse
import yaml
import wandb
import torch
import torch.optim as optim
from dotenv import load_dotenv  

load_dotenv() 

from src.models import build_model
from src.data import build_dataloaders
from src.engine import fit
from src.utils import set_seed, get_device


def build_optimizer(model, cfg: dict):
    name = cfg["training"]["optimizer"]
    lr   = cfg["training"]["lr"]
    wd   = cfg["training"]["weight_decay"]

    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=cfg["training"]["momentum"], weight_decay=wd)
    elif name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'")


def build_scheduler(optimizer, cfg: dict):
    name   = cfg["training"]["scheduler"]
    epochs = cfg["training"]["epochs"]

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "steplr":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg["training"]["step_size"], gamma=cfg["training"]["gamma"])
    elif name == "onecycle":
        # OneCycleLR needs steps_per_epoch — we patch this in main() after loaders are built
        raise RuntimeError("Use build_onecycle_scheduler() after building dataloaders.")
    else:
        raise ValueError(f"Unknown scheduler: '{name}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    set_seed(cfg["experiment"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # Data
    trainloader, testloader = build_dataloaders(cfg)

    # Model
    model = build_model(
        arch=cfg["model"]["arch"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    # Optimizer
    optimizer = build_optimizer(model, cfg)

    # Scheduler
    if cfg["training"]["scheduler"] == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["training"]["lr"],
            epochs=cfg["training"]["epochs"],
            steps_per_epoch=len(trainloader),
        )
    else:
        scheduler = build_scheduler(optimizer, cfg)

    # WandB
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["experiment"]["name"],
        group=cfg["experiment"]["group"],
        config=cfg,         # logs the full config — great for comparing runs
    )
    wandb.watch(model, log="gradients", log_freq=100)

    # Train
    fit(model, trainloader, testloader, optimizer, scheduler, cfg, device)

    wandb.finish()


if __name__ == "__main__":
    main()