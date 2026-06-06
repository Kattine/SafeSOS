"""
Download the HaGRID classification subset (cj-mills/hagrid-classification-512p-no-gesture-150k-zip)
and extract ONLY the 9 emergency gesture classes we need.

The full subset is ~10 GB; after filtering we keep ~1-2 GB.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

TARGET_CLASSES = [
    "call", "stop", "palm", "fist", "ok",
    "peace", "one", "two_up", "three", "no_gesture",
]

REPO_ID = "cj-mills/hagrid-classification-512p-no-gesture-150k-zip"
RAW_DIR = Path("data/raw")


def download_subset() -> Path:
    """Download the zipped dataset from Hugging Face Hub."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] Pulling {REPO_ID} into {RAW_DIR}")
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(RAW_DIR),
    )
    return Path(local_dir)


def filter_to_target_classes(extracted_root: Path) -> None:
    """Remove subdirectories that are not in TARGET_CLASSES."""
    # TODO: implement after inspecting the actual archive layout
    print(f"[filter] Keeping only: {TARGET_CLASSES}")


def main() -> None:
    extracted = download_subset()
    filter_to_target_classes(extracted)
    print("[done] Data ready at data/raw/")


if __name__ == "__main__":
    main()
