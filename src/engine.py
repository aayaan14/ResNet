import os
import time
import torch
import torch.nn as nn
import wandb


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "loss": total_loss / len(loader),
        "acc": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "loss": total_loss / len(loader),
        "acc": 100.0 * correct / total,
    }


def fit(model, trainloader, testloader, optimizer, scheduler, cfg, device):
    """Full training loop with wandb logging."""
    criterion = nn.CrossEntropyLoss()
    epochs = cfg["training"]["epochs"]
    best_acc = 0.0

    for epoch in range(epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(model, trainloader, criterion, optimizer, device)
        test_metrics  = evaluate(model, testloader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # Console
        print(
            f"Epoch {epoch+1:03d}/{epochs} | {elapsed:.1f}s | LR: {current_lr:.5f} | "
            f"Train Loss: {train_metrics['loss']:.3f} | Train Acc: {train_metrics['acc']:.2f}% | "
            f"Test Loss: {test_metrics['loss']:.3f} | Test Acc: {test_metrics['acc']:.2f}%"
        )

        # WandB
        wandb.log({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train/loss": train_metrics["loss"],
            "train/acc":  train_metrics["acc"],
            "test/loss":  test_metrics["loss"],
            "test/acc":   test_metrics["acc"],
        })

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{cfg['experiment']['name']}_best.pth")

    print(f"\nBest Test Acc: {best_acc:.2f}%")
    wandb.summary["best_test_acc"] = best_acc