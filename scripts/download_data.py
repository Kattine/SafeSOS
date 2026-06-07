"""Download HaGRID and keep only SafeSOS classes.

Writes images in ImageFolder layout:
    data/raw/<class_name>/<image_id>.jpg

Examples:
    python scripts/download_data.py --max-per-class 4500
    python scripts/download_data.py --zip-path /path/to/local/hagrid.zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

from tqdm.auto import tqdm


# Constants

TARGET_CLASSES: list[str] = [
    "call", "stop", "palm", "fist", "ok",
    "peace", "one", "two_up", "three", "no_gesture",
]

REPO_ID = "cj-mills/hagrid-classification-512p-no-gesture-150k-zip"
ZIP_FILENAME = "hagrid-classification-512p-no-gesture-150k.zip"

IMG_EXTS = (".jpg", ".jpeg", ".png")


# Zip input

def acquire_zip(zip_path: Path | None) -> Path:
    """Return a local zip path, downloading from HF when needed."""
    if zip_path is not None:
        zip_path = Path(zip_path).expanduser().resolve()
        if not zip_path.is_file():
            print(f"[error] --zip-path does not point to a file: {zip_path}")
            sys.exit(1)
        print(f"[download] using local zip: {zip_path}")
        return zip_path

    print(f"[download] fetching {REPO_ID}/{ZIP_FILENAME} via huggingface_hub...")
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=ZIP_FILENAME,
        repo_type="dataset",
    )
    print(f"[download] cached at: {local}")
    return Path(local)


# Zip layout

def detect_zip_layout(zf: zipfile.ZipFile) -> tuple[str, set[str]]:
    """Infer common path prefix and available class folders."""
    sample_paths: list[str] = []
    for info in zf.infolist():
        name = info.filename
        if name.lower().endswith(IMG_EXTS):
            sample_paths.append(name)
            if len(sample_paths) >= 200:
                break

    if not sample_paths:
        raise RuntimeError(
            "No image entries found inside the zip. "
            f"First 5 entries seen: {[i.filename for i in zf.infolist()[:5]]}"
        )

    # Normalize separators.
    sample_paths = [p.replace("\\", "/") for p in sample_paths]

    parts0 = sample_paths[0].split("/")
    candidate_prefix = "/".join(parts0[:-2]) if len(parts0) > 2 else ""

    if candidate_prefix:
        if not all(p.startswith(candidate_prefix + "/") for p in sample_paths):
            candidate_prefix = ""

    # Class name is the parent folder.
    classes: set[str] = set()
    for p in sample_paths:
        parts = p.split("/")
        if len(parts) < 2:
            continue
        classes.add(parts[-2])

    return candidate_prefix, classes


def class_from_entry(filename: str, prefix: str) -> str | None:
    """Extract the class name from a zip entry path, or None if unparseable."""
    name = filename.replace("\\", "/")
    if prefix and not name.startswith(prefix + "/"):
        return None
    parts = name.split("/")
    if len(parts) < 2:
        return None
    return parts[-2]


# Extraction

def ensure_class_dirs(out_dir: Path) -> None:
    for cls in TARGET_CLASSES:
        (out_dir / cls).mkdir(parents=True, exist_ok=True)


def extract_filtered(
    zip_path: Path,
    out_dir: Path,
    max_per_class: int | None,
) -> dict[str, int]:
    """Open the zip and stream out only the entries we want."""
    counters: dict[str, int] = {c: 0 for c in TARGET_CLASSES}
    target_set = set(TARGET_CLASSES)

    with zipfile.ZipFile(zip_path, "r") as zf:
        prefix, available = detect_zip_layout(zf)
        print(f"[zip] detected prefix: {repr(prefix)}")
        print(f"[zip] found {len(available)} class folders: {sorted(available)}")

        missing = target_set - available
        if missing:
            print(f"[warn] target classes not present in zip: {sorted(missing)}")
            # Continue even if some targets are missing.

        ensure_class_dirs(out_dir)

        # Pre-filter for progress bar total.
        all_entries = [
            info for info in zf.infolist()
            if info.filename.lower().endswith(IMG_EXTS)
        ]
        print(f"[zip] total image entries: {len(all_entries)}")

        for info in tqdm(all_entries, desc="extracting"):
            cls = class_from_entry(info.filename, prefix)
            if cls is None or cls not in target_set:
                continue
            if max_per_class is not None and counters[cls] >= max_per_class:
                continue

            idx = counters[cls]
            dst = out_dir / cls / f"{idx:06d}.jpg"
            with zf.open(info, "r") as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out)
            counters[cls] += 1

            # Stop early if all classes hit cap.
            if max_per_class is not None and all(
                counters[c] >= max_per_class for c in TARGET_CLASSES
            ):
                print("[extract] reached max-per-class for all targets, stopping")
                break

    return counters


# Entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument(
        "--max-per-class", type=int, default=None,
        help="cap samples per class; default: keep all available",
    )
    p.add_argument(
        "--zip-path", default=None,
        help="path to a local copy of the HaGRID zip "
             "(skips huggingface_hub download)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    zip_path = acquire_zip(Path(args.zip_path) if args.zip_path else None)
    counters = extract_filtered(zip_path, out_dir, max_per_class=args.max_per_class)

    print("\n[done] per-class counts written to disk:")
    total = 0
    for cls in TARGET_CLASSES:
        n = counters[cls]
        total += n
        print(f"  {cls:<14s} {n:>6d}")
    print(f"  {'TOTAL':<14s} {total:>6d}")
    print(f"\n[done] images saved under {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
