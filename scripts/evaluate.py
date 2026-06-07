"""
Unified evaluation of all three SafeSOS models on the held-out test set.

Reads:
    models/checkpoints/keypoint_svm.joblib
    models/checkpoints/keypoint_rf.joblib
    models/checkpoints/keypoint_best.joblib  (SVM by val macro-F1)
    models/checkpoints/mobilenet_best.pth
    models/checkpoints/mobilenet_meta.json
    data/processed/keypoints/test.npz
    data/processed/test/<class>/*.jpg

Writes:
    data/outputs/metrics.json                      (single table summary)
    reports/figures/confusion_<model>.png          (one per model)
    reports/figures/per_class_accuracy.png         (grouped bar chart)
    data/outputs/predictions_<model>.npz           (raw probs + labels, for later analysis)

The naive baseline is computed at eval time (majority class from train labels)
since there's nothing to load.

Run:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)


# ============================================================
# Result container
# ============================================================

@dataclass
class ModelResult:
    """Standardised per-model evaluation result."""
    name: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion: np.ndarray            # (K, K)
    per_class_f1: np.ndarray         # (K,)
    per_class_support: np.ndarray    # (K,)
    probs: np.ndarray | None         # (N, K) calibrated probabilities, or None for naive
    y_true: np.ndarray               # (N,)
    y_pred: np.ndarray               # (N,)


# ============================================================
# Data loading
# ============================================================

def load_keypoint_split(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load (X, y, class_names) from saved npz."""
    data = np.load(npz_path, allow_pickle=True)
    return data["X"].astype(np.float32), data["y"].astype(np.int64), list(data["class_names"])


def make_image_test_loader(
    data_dir: Path,
    image_size: int = 224,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """torchvision-style test loader; only used for the DL model."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Apply the dotfile filter from train_dl.py to skip macOS junk
    def is_valid(path: str) -> bool:
        name = Path(path).name
        if name.startswith("."):
            return False
        return path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(data_dir / "test", transform=tf, is_valid_file=is_valid)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    return ds, loader


# ============================================================
# Per-model evaluation
# ============================================================

def evaluate_naive(y_train: np.ndarray, y_test: np.ndarray, K: int) -> ModelResult:
    """Majority-class predictor: always predicts argmax(np.bincount(y_train))."""
    from models.naive_baseline import MajorityClassifier
    clf = MajorityClassifier().fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = clf.predict(np.zeros((len(y_test), 1)))
    probs = clf.predict_proba(np.zeros((len(y_test), 1)))

    return ModelResult(
        name="naive",
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        confusion=confusion_matrix(y_test, y_pred, labels=list(range(K))),
        per_class_f1=f1_score(y_test, y_pred, average=None, labels=list(range(K)), zero_division=0),
        per_class_support=np.bincount(y_test, minlength=K),
        probs=probs,
        y_true=y_test,
        y_pred=y_pred,
    )


def evaluate_classical(
    name: str,
    ckpt_path: Path,
    X_test: np.ndarray,
    y_test: np.ndarray,
    K: int,
) -> ModelResult:
    """Load a saved KeypointClassifier and evaluate."""
    from models.classical_ml import KeypointClassifier
    clf = KeypointClassifier.load(ckpt_path)
    y_pred = clf.predict(X_test)
    probs = clf.predict_proba(X_test)

    return ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        confusion=confusion_matrix(y_test, y_pred, labels=list(range(K))),
        per_class_f1=f1_score(y_test, y_pred, average=None, labels=list(range(K)), zero_division=0),
        per_class_support=np.bincount(y_test, minlength=K),
        probs=probs,
        y_true=y_test,
        y_pred=y_pred,
    )


def evaluate_dl(
    ckpt_path: Path,
    meta_path: Path,
    data_dir: Path,
    K: int,
) -> ModelResult:
    """Run MobileNetV3 over the test set, returning calibrated probabilities."""
    import torch
    import torch.nn.functional as F
    from models.deep_learning import SelectiveMobileNet

    with open(meta_path) as f:
        meta = json.load(f)
    temperature = float(meta["temperature"])

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"[evaluate_dl] device = {device}")

    sm = SelectiveMobileNet(
        num_classes=K,
        confidence_threshold=0.0,   # we don't abstain here; that's a separate experiment
        temperature=temperature,
        device=device,
    )
    sm.load(str(ckpt_path))
    sm.model.eval()

    _, loader = make_image_test_loader(data_dir)

    all_probs, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = sm.model(x.to(device))
            scaled = logits / temperature
            probs = F.softmax(scaled, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_y.append(y.numpy())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0)
    y_pred = probs.argmax(axis=1)

    return ModelResult(
        name="mobilenet",
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(K))),
        per_class_f1=f1_score(y_true, y_pred, average=None, labels=list(range(K)), zero_division=0),
        per_class_support=np.bincount(y_true, minlength=K),
        probs=probs,
        y_true=y_true,
        y_pred=y_pred,
    )


# ============================================================
# Plotting
# ============================================================

def plot_confusion(
    result: ModelResult,
    class_names: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    cm = result.confusion
    # Row-normalise to make per-class recall visible
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{result.name}  (acc {result.accuracy:.3f}, F1 {result.macro_f1:.3f})")

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val > 0.5 else "black",
                    fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recall")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_per_class_accuracy(
    results: list[ModelResult],
    class_names: list[str],
    out_path: Path,
) -> None:
    """Grouped bar: per-class F1 across all models."""
    import matplotlib.pyplot as plt
    K = len(class_names)
    width = 0.8 / len(results)
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, r in enumerate(results):
        offset = (i - (len(results) - 1) / 2) * width
        ax.bar(x + offset, r.per_class_f1, width, label=r.name)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 across all models")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ============================================================
# Persistence
# ============================================================

def save_metrics_summary(results: list[ModelResult], out_path: Path) -> None:
    summary = {
        "models": [
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "macro_f1": r.macro_f1,
                "weighted_f1": r.weighted_f1,
                "per_class_f1": r.per_class_f1.tolist(),
                "per_class_support": r.per_class_support.tolist(),
            }
            for r in results
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] {out_path}")


def save_predictions(result: ModelResult, out_path: Path) -> None:
    """Save raw probabilities for downstream selective-prediction analysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        y_true=result.y_true,
        y_pred=result.y_pred,
        probs=result.probs if result.probs is not None else np.array([]),
    )
    print(f"[save] {out_path}")


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keypoint-dir", default="data/processed/keypoints")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--outputs-dir", default="data/outputs")
    p.add_argument("--figures-dir", default="reports/figures")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kp_dir = Path(args.keypoint_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)

    # ---- Data ----
    X_train, y_train, names_train = load_keypoint_split(kp_dir / "train.npz")
    X_test,  y_test,  names_test  = load_keypoint_split(kp_dir / "test.npz")
    assert names_train == names_test
    class_names = names_train
    K = len(class_names)
    print(f"[data] classes ({K}): {class_names}")
    print(f"[data] test set: {len(y_test)} samples")

    # ---- Evaluate every model on the same test split ----
    results: list[ModelResult] = []

    print("\n[1/4] naive baseline")
    results.append(evaluate_naive(y_train, y_test, K))

    print("\n[2/4] classical: keypoint_svm")
    results.append(evaluate_classical(
        "svm", ckpt_dir / "keypoint_svm.joblib", X_test, y_test, K,
    ))

    print("\n[3/4] classical: keypoint_rf")
    results.append(evaluate_classical(
        "rf", ckpt_dir / "keypoint_rf.joblib", X_test, y_test, K,
    ))

    print("\n[4/4] deep learning: mobilenet")
    results.append(evaluate_dl(
        ckpt_dir / "mobilenet_best.pth",
        ckpt_dir / "mobilenet_meta.json",
        data_dir,
        K,
    ))

    # ---- Print headline table ----
    print("\n" + "=" * 60)
    print(f"  {'model':<15s} {'acc':>10s} {'macro-F1':>12s} {'weighted-F1':>15s}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r.name:<15s} {r.accuracy:>10.4f} {r.macro_f1:>12.4f} {r.weighted_f1:>15.4f}")
    print("=" * 60)

    # ---- Save artefacts ----
    save_metrics_summary(results, outputs_dir / "metrics.json")
    for r in results:
        plot_confusion(r, class_names, figures_dir / f"confusion_{r.name}.png")
        save_predictions(r, outputs_dir / f"predictions_{r.name}.npz")

    plot_per_class_accuracy(results, class_names, figures_dir / "per_class_f1.png")

    print("\n[done] artefacts in:")
    print(f"  {outputs_dir}/metrics.json")
    print(f"  {outputs_dir}/predictions_*.npz")
    print(f"  {figures_dir}/confusion_*.png")
    print(f"  {figures_dir}/per_class_f1.png")


if __name__ == "__main__":
    main()
