import copy
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def build_avocado_classifier(
    num_classes: int = 4,
    freeze_backbone: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )

    return model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None,
    unfreeze_at_epoch: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    if device is None:
        device = get_device()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc": [],  "val_acc": [],
    }

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    no_improve_count = 0

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        if unfreeze_at_epoch and epoch == unfreeze_at_epoch:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)

        model.train()
        running_loss, running_correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == labels).sum().item()
            total += inputs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
            f"| {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_count = 0
            if save_path:
                save_dl_model(model, save_path)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                break

    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_dl_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
    device: Optional[torch.device] = None,
    model_name: str = "MobileNetV2",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}\n")
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig_path = Path(save_dir) / f"{model_name.replace(' ', '_').lower()}_cm.png"
        plt.savefig(fig_path, dpi=150)

    plt.show()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "cm": cm,
        "y_pred": y_pred,
        "y_true": y_true,
    }


def plot_training_history(history: Dict[str, List[float]], save_dir: Optional[str] = None) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], "s--", label="Val Loss", color="coral")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="steelblue")
    axes[1].plot(epochs, history["val_acc"], "s--", label="Val Acc", color="coral")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig_path = Path(save_dir) / "training_history.png"
        plt.savefig(fig_path, dpi=150)

    plt.show()


INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_single_image_dl(
    model: nn.Module,
    image_path: str,
    class_names: List[str],
    device: Optional[torch.device] = None,
) -> Tuple[str, float, List[float]]:
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), probs.tolist()


def save_dl_model(model: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_dl_model(path: str, num_classes: int = 4, device: Optional[torch.device] = None) -> nn.Module:
    if device is None:
        device = get_device()
    model = build_avocado_classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
