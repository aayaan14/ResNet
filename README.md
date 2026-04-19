# ResNet on CIFAR-10

ResNet-18 and ResNet-50 trained from scratch on CIFAR-10, with structured ablations tracked via WandB.

## Project Structure

```
.
├── configs/
│   ├── resnet18_baseline.yaml
│   └── resnet50_baseline.yaml
├── checkpoints/             # saved per-experiment (auto-created)
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── blocks.py        # BasicBlock, Bottleneck
│   │   └── resnet.py        # ResNet, resnet18(), resnet50()
│   ├── data.py              # Dataloaders + augmentation strategies
│   ├── engine.py            # train_one_epoch, evaluate, fit
│   └── utils.py             # set_seed, get_device
├── train.py                 # Training entry point
├── inference.py             # Evaluate a saved checkpoint
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt

# Run a single experiment
python train.py --config configs/resnet18_baseline.yaml

# Run an ablation (e.g. swap optimizer)
python train.py --config configs/resnet18_adamw.yaml
```

Checkpoints are saved to `checkpoints/<experiment_name>_best.pth` whenever test accuracy improves.

## Inference

Evaluate a saved checkpoint on the test set:

```bash
# Infer checkpoint path from config name (looks for checkpoints/<experiment_name>_best.pth)
python inference.py --config configs/resnet18_baseline.yaml

# Or point to a checkpoint explicitly
python inference.py --config configs/resnet18_baseline.yaml --checkpoint checkpoints/some_other.pth
```

## Ablations

Each ablation is a new config file. The only fields you need to change:

| Ablation         | Key to change                | Values                              |
|------------------|------------------------------|-------------------------------------|
| Architecture     | `model.arch`                 | `resnet18`, `resnet50`              |
| Optimizer        | `training.optimizer`         | `sgd`, `adamw`                      |
| Scheduler        | `training.scheduler`         | `cosine`, `steplr`, `onecycle`      |
| Weight decay     | `training.weight_decay`      | `1e-4`, `5e-4`, `1e-3`              |
| Augmentation     | `augmentation.strategy`      | `baseline`, `cutout`, `autoaugment` |

Set `experiment.group` to the same value across related runs so WandB groups them correctly.