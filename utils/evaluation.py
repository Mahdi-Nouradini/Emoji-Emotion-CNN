import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    return metrics, np.array(all_targets), np.array(all_preds)


def plot_history(history, title_prefix="Model"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{title_prefix} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].plot(history["val_f1"], label="Val Macro-F1")
    axes[1].set_title(f"{title_prefix} - Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def denormalize(img_tensor, mean, std):
    x = img_tensor.cpu().clone()
    x = x * std[:, None, None] + mean[:, None, None]
    x = x.clamp(0, 1)
    return np.transpose(x.numpy(), (1, 2, 0))


@torch.no_grad()
def show_predictions(model, dataset, class_names, mean, std, device, n=12, mistakes_only=False):
    model.eval()
    chosen = []

    for idx in range(len(dataset)):
        x, y = dataset[idx]
        logits = model(x.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].cpu()
        pred = int(probs.argmax().item())
        conf = float(probs[pred].item())

        if mistakes_only and pred == int(y.item()):
            continue

        chosen.append((x, int(y.item()), pred, conf))
        if len(chosen) >= n:
            break

    if len(chosen) == 0:
        print("No examples matched the filter.")
        return

    cols = 4
    rows = math.ceil(len(chosen) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (x, y_true, pred, conf) in zip(axes, chosen):
        ax.imshow(denormalize(x, mean, std))
        ok = (y_true == pred)
        ax.set_title(
            f"T: {class_names[y_true]}\nP: {class_names[pred]} ({conf:.2f})",
            color=("green" if ok else "crimson"),
            fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_feature_maps(model, sample_tensor, mean, std, device, max_channels=8):
    model.eval()
    x = sample_tensor.unsqueeze(0).to(device)

    first_conv = model.features[0]
    relu_after = model.features[1]
    fmap = relu_after(first_conv(x)).cpu()[0]

    fig, axes = plt.subplots(1, max_channels + 1, figsize=(2.2 * (max_channels + 1), 2.5))
    axes[0].imshow(denormalize(sample_tensor, mean, std))
    axes[0].set_title("Input")
    axes[0].axis("off")

    for i in range(max_channels):
        axes[i + 1].imshow(fmap[i].numpy(), cmap="viridis")
        axes[i + 1].set_title(f"Map {i}")
        axes[i + 1].axis("off")

    plt.suptitle("First Conv Layer Feature Maps", y=1.03)
    plt.tight_layout()
    plt.show()