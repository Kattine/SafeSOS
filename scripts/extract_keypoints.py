"""Extract MediaPipe hand keypoints from processed images to .npz files.

Reads: data/processed/{train,val,test}/<class>/*.jpg
Writes: data/processed/keypoints/{train,val,test}.npz and meta.json

Run:
    python scripts/extract_keypoints.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as script or module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm.auto import tqdm


VALID_EXTS = {".jpg", ".jpeg", ".png"}


# Path collection

def collect_split(split_dir: Path) -> tuple[list[Path], np.ndarray, list[str]]:
    """Return image paths, integer labels, and class names for one split."""
    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    paths: list[Path] = []
    labels: list[int] = []
    for label_idx, class_dir in enumerate(class_dirs):
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in VALID_EXTS:
                paths.append(p)
                labels.append(label_idx)

    return paths, np.array(labels, dtype=np.int64), class_names


def assert_consistent_classes(split_classes: dict[str, list[str]]) -> list[str]:
    """All splits must have the same class list in the same order."""
    reference = None
    for split, classes in split_classes.items():
        if reference is None:
            reference = classes
        elif classes != reference:
            raise RuntimeError(
                f"Class lists differ across splits.\n"
                f"  reference: {reference}\n"
                f"  {split}: {classes}"
            )
    assert reference is not None
    return reference


# Keypoint extraction

def extract_for_split(
    paths: list[Path],
    labels: np.ndarray,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MediaPipe on paths and return X, y, and detection mask."""
    # Lazy import keeps this file importable without MediaPipe.
    from models.classical_ml import HandKeypointExtractor

    extractor = HandKeypointExtractor()
    try:
        X, valid_mask = extractor.extract_batch(paths, verbose=verbose)
    finally:
        extractor.close()

    y = labels[valid_mask]
    return X, y, valid_mask


# Save

def save_split(
    out_dir: Path,
    split: str,
    X: np.ndarray,
    y: np.ndarray,
    valid_mask: np.ndarray,
    class_names: list[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"
    np.savez_compressed(
        path,
        X=X,
        y=y,
        valid_mask=valid_mask,
        class_names=np.array(class_names, dtype=object),
        detection_rate=np.float32(valid_mask.mean()),
    )
    return path


def per_class_detection_rate(
    valid_mask: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict[str, float]:
    """Return per-class hand detection rates."""
    rates: dict[str, float] = {}
    for label_idx, name in enumerate(class_names):
        idx = labels == label_idx
        if idx.any():
            rates[name] = float(valid_mask[idx].mean())
        else:
            rates[name] = float("nan")
    return rates


# Entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument("--out-dir", default="data/processed/keypoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_dir)
    out_dir = Path(args.out_dir)

    splits = ["train", "val", "test"]
    split_classes: dict[str, list[str]] = {}
    split_data: dict[str, tuple[list[Path], np.ndarray]] = {}

    # First pass: collect paths and class lists.
    for split in splits:
        split_dir = processed_root / split
        if not split_dir.is_dir():
            print(f"[error] missing split directory: {split_dir}")
            sys.exit(1)
        paths, labels, class_names = collect_split(split_dir)
        if len(paths) == 0:
            print(f"[error] no images found in {split_dir}")
            sys.exit(1)
        split_classes[split] = class_names
        split_data[split] = (paths, labels)
        print(f"[collect] {split}: {len(paths)} images, {len(class_names)} classes")

    class_names = assert_consistent_classes(split_classes)

    # Second pass: extract and save.
    summary: dict[str, dict] = {}
    for split in splits:
        paths, labels = split_data[split]
        print(f"\n[extract] running MediaPipe on {split} ({len(paths)} images)...")
        X, y, valid_mask = extract_for_split(paths, labels, verbose=True)

        out_path = save_split(out_dir, split, X, y, valid_mask, class_names)
        rates = per_class_detection_rate(valid_mask, labels, class_names)
        overall = float(valid_mask.mean())

        summary[split] = {
            "n_images": int(valid_mask.size),
            "n_with_hand_detected": int(valid_mask.sum()),
            "overall_detection_rate": overall,
            "per_class_detection_rate": rates,
            "output_file": str(out_path),
        }

        print(f"[extract] {split}: detection rate = {overall:.3f}")
        for cls, r in rates.items():
            print(f"  {cls:<14s} {r:.3f}")

    # Save summary.
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "class_names": class_names,
        "feature_dim": 63,
        "splits": summary,
    }, indent=2))
    print(f"\n[done] keypoints saved to {out_dir}/")
    print(f"[done] summary written to {meta_path}")


if __name__ == "__main__":
    main()
