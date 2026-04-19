import argparse
import yaml
import torch

from src.models import build_model
from src.data import build_dataloaders
from src.engine import evaluate
from src.utils import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (optional, inferred from config if not set)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["experiment"]["seed"])
    device = get_device()

    # Infer checkpoint path from experiment name if not explicitly passed
    checkpoint_path = args.checkpoint or f"checkpoints/{cfg['experiment']['name']}_best.pth"

    print(f"Loading model  : {cfg['model']['arch']}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Device         : {device}")

    # Build model and load weights
    model = build_model(
        arch=cfg["model"]["arch"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.\n")

    # Only need the test loader
    _, testloader = build_dataloaders(cfg)

    criterion = torch.nn.CrossEntropyLoss()
    test_metrics = evaluate(model, testloader, criterion, device)

    print(f"Test Loss : {test_metrics['loss']:.4f}")
    print(f"Test Acc  : {test_metrics['acc']:.2f}%")


if __name__ == "__main__":
    main()