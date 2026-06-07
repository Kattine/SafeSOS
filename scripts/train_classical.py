"""Train classical models (SVM, RF) on MediaPipe keypoints.

Reads: data/processed/keypoints/{train,val,test}.npz
Writes: models/checkpoints/keypoint_{svm,rf,best}.joblib and classical_metrics.json

Examples:
    python scripts/train_classical.py
    python scripts/train_classical.py --backends svm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow imports when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)


# Data loading

def load_split(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load X, y, class_names from a saved .npz."""
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    class_names = list(data["class_names"])
    return X, y, class_names


def assert_same_classes(*name_lists: list[str]) -> list[str]:
    reference = name_lists[0]
    for names in name_lists[1:]:
        if names != reference:
            raise RuntimeError(
                f"Class lists differ between splits.\n"
                f"  reference: {reference}\n"
                f"  other:     {names}"
            )
    return reference


# Train one backend

def train_backend(
    backend: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: list[str],
    checkpoint_dir: Path,
) -> dict:
    """Fit one backend (svm | rf), evaluate, save."""
    # Import here so the script can still import without optional deps loaded.
    from models.classical_ml import KeypointClassifier

    print(f"\n[train] backend = {backend}")
    clf = KeypointClassifier(backend=backend)
    clf.fit(X_train, y_train, X_val, y_val)

    # Evaluate on val and test.
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    metrics = {
        "backend": backend,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro")),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "test_macro_f1": float(f1_score(y_test, test_pred, average="macro")),
        "test_confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }

    print(f"  val  accuracy={metrics['val_accuracy']:.4f}  macro-F1={metrics['val_macro_f1']:.4f}")
    print(f"  test accuracy={metrics['test_accuracy']:.4f}  macro-F1={metrics['test_macro_f1']:.4f}")

    # Save model.
    ckpt_path = checkpoint_dir / f"keypoint_{backend}.joblib"
    clf.save(ckpt_path)
    print(f"  saved -> {ckpt_path}")

    # Save text report.
    report_path = checkpoint_dir / f"keypoint_{backend}_report.txt"
    report = classification_report(
        y_test, test_pred, target_names=class_names, digits=4,
    )
    report_path.write_text(report)

    return metrics


# Entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keypoint-dir", default="data/processed/keypoints")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument(
        "--backends", nargs="+", default=["svm", "rf"],
        choices=["svm", "rf"],
        help="which backends to train; default both",
    )
    return p.parse_args()


def select_best(all_metrics: list[dict]) -> dict:
    """Pick the backend with highest validation macro-F1."""
    return max(all_metrics, key=lambda m: m["val_macro_f1"])


def main() -> None:
    args = parse_args()
    kp_dir = Path(args.keypoint_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("[load] keypoints from", kp_dir)
    X_train, y_train, names_train = load_split(kp_dir / "train.npz")
    X_val,   y_val,   names_val   = load_split(kp_dir / "val.npz")
    X_test,  y_test,  names_test  = load_split(kp_dir / "test.npz")
    class_names = assert_same_classes(names_train, names_val, names_test)
    print(f"[load] train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    print(f"[load] classes={class_names}")

    all_metrics: list[dict] = []
    for backend in args.backends:
        m = train_backend(
            backend=backend,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   y_test=y_test,
            class_names=class_names,
            checkpoint_dir=ckpt_dir,
        )
        all_metrics.append(m)

    best = select_best(all_metrics)
    print(f"\n[best] {best['backend']} "
          f"(val macro-F1 = {best['val_macro_f1']:.4f})")

    # Copy best model to a stable filename.
    import shutil
    best_path = ckpt_dir / "keypoint_best.joblib"
    shutil.copy2(ckpt_dir / f"keypoint_{best['backend']}.joblib", best_path)
    print(f"[best] copied -> {best_path}")

    # Save summary metrics.
    summary_path = ckpt_dir / "classical_metrics.json"
    summary_path.write_text(json.dumps({
        "class_names": class_names,
        "best_backend": best["backend"],
        "all_results": all_metrics,
    }, indent=2))
    print(f"[done] metrics summary -> {summary_path}")


if __name__ == "__main__":
    main()
