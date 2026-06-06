"""
Project setup: download data, build features, train all three models.

Usage:
    python setup.py --all          # do everything
    python setup.py --download     # only download data
    python setup.py --train        # only train (assumes data is ready)
"""

import argparse
from scripts import download_data, train_pipeline


def run(download: bool, train: bool) -> None:
    """Orchestrate setup steps."""
    if download:
        print("[setup] Downloading HaGRID subset...")
        download_data.main()
    if train:
        print("[setup] Training all three models...")
        train_pipeline.main()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--train", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.all:
        run(download=True, train=True)
    else:
        run(download=args.download, train=args.train)
