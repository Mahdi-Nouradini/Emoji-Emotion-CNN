import random
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from data.generator import get_dataloaders, CLASS_NAMES, NUM_CLASSES
from models.cnn import SimpleCNN, BetterCNN, count_parameters
from utils.evaluation import compute_metrics, evaluate

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    return metrics


def fit_model(model, train_loader, val_loader, device, epochs=18, lr=1e-3, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1_macro"])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_state = deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.3f} | "
            f"val_f1={val_metrics['f1_macro']:.3f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Target classes: {CLASS_NAMES}\n")

    train_loader, val_loader, test_loader, _, _ = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    # 1. Train & Evaluate Baseline CNN
    print("--- Training Baseline CNN (SimpleCNN) ---")
    baseline_model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Trainable parameters: {count_parameters(baseline_model):,}")
    baseline_model, _ = fit_model(
        baseline_model, train_loader, val_loader, device, epochs=16, lr=1.2e-3
    )

    test_metrics_base, y_true_base, y_pred_base = evaluate(
        baseline_model, test_loader, criterion, device
    )
    print("\nBaseline Test Classification Report:")
    print(classification_report(y_true_base, y_pred_base, target_names=CLASS_NAMES, digits=3))

    # 2. Train & Evaluate Optimized CNN
    print("\n--- Training Optimized CNN (BetterCNN) ---")
    better_model = BetterCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Trainable parameters: {count_parameters(better_model):,}")
    better_model, _ = fit_model(
        better_model, train_loader, val_loader, device, epochs=18, lr=1.2e-3
    )

    test_metrics_better, y_true_better, y_pred_better = evaluate(
        better_model, test_loader, criterion, device
    )
    print("\nOptimized CNN Test Classification Report:")
    print(classification_report(y_true_better, y_pred_better, target_names=CLASS_NAMES, digits=3))

    # 3. Final Comparison Summary
    comparison = pd.DataFrame([
        {
            "model": "SimpleCNN (Baseline)",
            "params": count_parameters(baseline_model),
            **test_metrics_base,
        },
        {
            "model": "BetterCNN (BatchNorm + AdaptivePool)",
            "params": count_parameters(better_model),
            **test_metrics_better,
        },
    ])
    print("\n--- Final Benchmark Comparison ---")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()