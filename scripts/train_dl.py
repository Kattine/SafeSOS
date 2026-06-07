"""
Train MobileNetV3-Small on the 9+1-class SafeSOS gesture problem.

Pipeline:
    1. Build datasets from data/processed/{train,val,test}/<class>/*.jpg
    2. Fine-tune MobileNetV3-Small (ImageNet pretrained) for N epochs
    3. Run temperature scaling on the val set (post-hoc calibration)
    4. Save best checkpoint + temperature to models/checkpoints/

Device selection is automatic:
    - CUDA (Colab T4 / L4)  -> use it
    - MPS  (Mac M-series)   -> use it
    - else                  -> CPU (slow but works)

Run:
    python scripts/train_dl.py --epochs 10 --batch-size 64 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    checkpoint_dir: str = "models/checkpoints"
    num_classes: int = 10                # 9 gestures + no_gesture
    image_size: int = 224
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    unfreeze_last_n_blocks: int = 3      # rest is frozen for speed
    early_stop_patience: int = 3


# ============================================================
# Device selection
# ============================================================

def pick_device() -> torch.device:
    """CUDA -> MPS -> CPU, in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Data loading
# ============================================================

def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """Train uses augmentation, val/test use deterministic preprocessing."""
    # ImageNet statistics (MobileNetV3 pretrained on ImageNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # cutout-like
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_tf, eval_tf


def _is_real_image_file(path: str) -> bool:
    """Filter for ImageFolder: keeps real images, rejects junk.

    Drops macOS AppleDouble sidecar files (`._*` — created automatically when
    tarring on macOS) and other dotfiles like `.DS_Store`. These contain no
    pixels and crash PIL with `UnidentifiedImageError`.
    """
    import os
    name = os.path.basename(path)
    if name.startswith("."):
        return False  # covers ._*, .DS_Store, ._.DS_Store, etc.
    return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Returns train/val/test DataLoaders plus the class name list."""
    train_tf, eval_tf = build_transforms(cfg.image_size)
    root = Path(cfg.data_dir)

    train_ds = datasets.ImageFolder(
        root / "train", transform=train_tf, is_valid_file=_is_real_image_file,
    )
    val_ds = datasets.ImageFolder(
        root / "val", transform=eval_tf, is_valid_file=_is_real_image_file,
    )
    test_ds = datasets.ImageFolder(
        root / "test", transform=eval_tf, is_valid_file=_is_real_image_file,
    )

    # Sanity: same class order across splits
    assert train_ds.classes == val_ds.classes == test_ds.classes, \
        "Class folders differ across splits — check data/processed/"

    common_kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, **common_kwargs)

    return train_loader, val_loader, test_loader, train_ds.classes


# ============================================================
# Model
# ============================================================

def build_model(num_classes: int, unfreeze_last_n_blocks: int) -> nn.Module:
    """MobileNetV3-Small with a new classification head.

    Freezing strategy: only the last N feature blocks plus the classifier
    head are trainable. Faster training, less overfitting on small data.
    """
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    net = models.mobilenet_v3_small(weights=weights)

    # Replace head for our num_classes
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)

    # Freeze early features
    feature_blocks = list(net.features)
    n_total = len(feature_blocks)
    n_freeze = max(0, n_total - unfreeze_last_n_blocks)
    for i, block in enumerate(feature_blocks):
        requires_grad = i >= n_freeze
        for p in block.parameters():
            p.requires_grad = requires_grad
    # Classifier is always trainable
    for p in net.classifier.parameters():
        p.requires_grad = True

    return net


# ============================================================
# Training & evaluation loops
# ============================================================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (loss, top-1 accuracy)."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """One training epoch. Returns average loss."""
    model.train()
    loss_sum, total = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * y.size(0)
        total += y.size(0)
    return loss_sum / total


# ============================================================
# Temperature scaling (post-hoc calibration)
# ============================================================

def fit_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 100,
) -> float:
    """Fit a single scalar temperature T on the validation set.

    Following Guo et al. 2017 — solves
        T* = argmin_T  NLL(softmax(logits / T), y)
    via L-BFGS. Improves selective-prediction reliability for free.
    """
    model.eval()
    # Collect all val logits & labels (val set is small, fits in RAM)
    all_logits, all_y = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            all_logits.append(model(x))
            all_y.append(y)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_y, dim=0)

    # Single trainable scalar, kept on the same device as logits
    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


# ============================================================
# Main
# ============================================================

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-last-n-blocks", type=int, default=3)
    p.add_argument("--early-stop-patience", type=int, default=3)
    args = p.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    print(f"[train] device = {device}")
    print(f"[train] config = {asdict(cfg)}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader, class_names = build_dataloaders(cfg)
    print(f"[train] classes = {class_names}")

    # Model
    model = build_model(cfg.num_classes, cfg.unfreeze_last_n_blocks).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] params: {trainable:,} trainable / {total:,} total")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs,
    )

    # Training loop with early stopping
    best_val_acc, patience = 0.0, 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(log)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
              f"({elapsed:.1f}s)")

        # Early stopping on val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(),
                       Path(cfg.checkpoint_dir) / "mobilenet_best.pth")
            print(f"  ↳ new best val_acc={val_acc:.4f}, checkpoint saved")
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"  ↳ early stop at epoch {epoch}")
                break

    # Reload best checkpoint
    model.load_state_dict(
        torch.load(Path(cfg.checkpoint_dir) / "mobilenet_best.pth",
                   map_location=device)
    )

    # Calibration
    print("[train] fitting temperature on val set...")
    T = fit_temperature(model, val_loader, device)
    print(f"[train] temperature T = {T:.4f}")

    # Final test eval (uncalibrated logits — calibration affects probabilities,
    # not argmax accuracy)
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[train] test_acc = {test_acc:.4f}")

    # Save artefacts
    artefacts = {
        "config": asdict(cfg),
        "class_names": class_names,
        "temperature": T,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "history": history,
    }
    with open(Path(cfg.checkpoint_dir) / "mobilenet_meta.json", "w") as f:
        json.dump(artefacts, f, indent=2)

    print(f"[done] artefacts saved to {cfg.checkpoint_dir}/")


if __name__ == "__main__":
    main()
