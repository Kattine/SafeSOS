"""Split raw images into train/val/test with stratified sampling.

Reads: data/raw/<class>/*.jpg
Writes: data/processed/{train,val,test}/<class>/*.jpg and split_manifest.json

Examples:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-per-class 3000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from tqdm.auto import tqdm


# Constants

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
VALID_EXTS = {".jpg", ".jpeg", ".png"}


# Core logic

def list_class_files(class_dir: Path) -> list[Path]:
    """Return sorted list of image files under class_dir (deterministic)."""
    files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    return sorted(files)


def stratified_split(
    files: list[Path],
    ratios: dict[str, float],
    rng: random.Random,
) -> dict[str, list[Path]]:
    """Shuffle files and split by train/val/test ratios."""
    shuffled = files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    # Assign remainder to test to keep totals exact.
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train:n_train + n_val],
        "test":  shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def copy_split(
    files: list[Path],
    dest_class_dir: Path,
) -> list[str]:
    """Copy files to dest_class_dir; return the file names actually written."""
    dest_class_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for src in files:
        dst = dest_class_dir / src.name
        shutil.copy2(src, dst)
        written.append(src.name)
    return written


def process_class(
    cls: str,
    raw_root: Path,
    processed_root: Path,
    max_per_class: int | None,
    rng: random.Random,
) -> dict[str, list[str]]:
    """Process a single class directory: cap, split, copy, return manifest."""
    class_dir = raw_root / cls
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")

    files = list_class_files(class_dir)
    if not files:
        raise RuntimeError(f"No images found in {class_dir}")

    # Optional cap per class.
    if max_per_class is not None and len(files) > max_per_class:
        files = rng.sample(files, max_per_class)

    splits = stratified_split(files, SPLIT_RATIOS, rng)

    manifest_for_class: dict[str, list[str]] = {}
    for split_name, split_files in splits.items():
        dest = processed_root / split_name / cls
        written = copy_split(split_files, dest)
        manifest_for_class[split_name] = written

    return manifest_for_class


# Entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    p.add_argument(
        "--max-per-class", type=int, default=3000,
        help="cap per-class images before splitting; default 3000",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_dir)
    processed_root = Path(args.processed_dir)
    rng = random.Random(args.seed)

    classes = sorted([d.name for d in raw_root.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class directories in {raw_root}")
    print(f"[prepare] found classes: {classes}")

    full_manifest: dict[str, dict[str, list[str]]] = {}
    counts = {split: {cls: 0 for cls in classes} for split in SPLIT_RATIOS}

    for cls in tqdm(classes, desc="classes"):
        per_class = process_class(
            cls=cls,
            raw_root=raw_root,
            processed_root=processed_root,
            max_per_class=args.max_per_class,
            rng=rng,
        )
        full_manifest[cls] = per_class
        for split_name, files in per_class.items():
            counts[split_name][cls] = len(files)

    # Save split manifest.
    manifest_path = processed_root / "split_manifest.json"
    processed_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "seed": args.seed,
        "split_ratios": SPLIT_RATIOS,
        "max_per_class": args.max_per_class,
        "files": full_manifest,
    }, indent=2))

    # Print split summary.
    print("\n[prepare] split counts:")
    header = f"  {'class':<14s} " + " ".join(f"{s:>7s}" for s in SPLIT_RATIOS)
    print(header)
    for cls in classes:
        row = f"  {cls:<14s} " + " ".join(
            f"{counts[s][cls]:>7d}" for s in SPLIT_RATIOS
        )
        print(row)
    totals_row = f"  {'TOTAL':<14s} " + " ".join(
        f"{sum(counts[s].values()):>7d}" for s in SPLIT_RATIOS
    )
    print(totals_row)
    print(f"\n[done] manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
