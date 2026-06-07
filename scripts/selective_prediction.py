"""Selective prediction analysis for risk-coverage trade-offs.

Reads predictions from evaluate.py outputs and writes figures + JSON metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root or module mode.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


# Core metrics

def confidence_from_probs(probs: np.ndarray) -> np.ndarray:
    """Confidence = max calibrated softmax / proba per sample."""
    return probs.max(axis=1)


def risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, np.ndarray]:
    """For each tau in thresholds, compute coverage and selective-risk."""
    y_pred = probs.argmax(axis=1)
    conf = confidence_from_probs(probs)
    correct = (y_pred == y_true).astype(np.float32)

    coverages = np.empty_like(thresholds, dtype=np.float64)
    selective_risks = np.empty_like(thresholds, dtype=np.float64)
    selective_accs = np.empty_like(thresholds, dtype=np.float64)

    n = len(y_true)
    for i, tau in enumerate(thresholds):
        accepted = conf >= tau
        n_acc = int(accepted.sum())
        coverages[i] = n_acc / n
        if n_acc == 0:
            selective_risks[i] = 0.0
            selective_accs[i] = 1.0
        else:
            selective_accs[i] = float(correct[accepted].mean())
            selective_risks[i] = 1.0 - selective_accs[i]

    return {
        "thresholds": thresholds,
        "coverage": coverages,
        "selective_risk": selective_risks,
        "selective_accuracy": selective_accs,
    }


def aurc(coverages: np.ndarray, selective_risks: np.ndarray) -> float:
    """Area under risk-coverage curve (lower is better)."""
    # Sort by coverage so integration is stable.
    order = np.argsort(coverages)
    c = coverages[order]
    r = selective_risks[order]
    # NumPy 2.x renamed trapz -> trapezoid.
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(r, c))


def per_class_abstention(
    probs: np.ndarray,
    y_true: np.ndarray,
    tau: float,
    K: int,
) -> dict[str, np.ndarray]:
    """Per-class accepted / abstained / correct-when-accepted counts at tau."""
    conf = confidence_from_probs(probs)
    y_pred = probs.argmax(axis=1)
    accepted = conf >= tau

    abst_rate = np.zeros(K)
    acc_when_accepted = np.zeros(K)
    n_accepted = np.zeros(K)

    for k in range(K):
        mask = y_true == k
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        abst_rate[k] = 1.0 - float(accepted[mask].mean())

        acc_k_mask = mask & accepted
        n_accepted[k] = int(acc_k_mask.sum())
        if n_accepted[k] > 0:
            acc_when_accepted[k] = float((y_pred[acc_k_mask] == k).mean())

    return {
        "abstention_rate": abst_rate,
        "accuracy_when_accepted": acc_when_accepted,
        "n_accepted": n_accepted.astype(int),
    }


# I/O

def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Load (probs, y_true) from a predictions_*.npz file.

    Returns None if probs is empty (e.g. naive baseline doesn't get a curve).
    """
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    probs = data["probs"]
    if probs.size == 0:
        return None
    y_true = data["y_true"]
    return probs, y_true


# Plotting

def plot_risk_coverage(
    curves: dict[str, dict[str, np.ndarray]],
    out_path: Path,
    operating_tau: float = 0.75,
) -> None:
    """Plot risk-coverage curves for all models."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colours = {"svm": "#1f77b4", "rf": "#ff7f0e", "mobilenet": "#2ca02c"}
    markers = {"svm": "o", "rf": "s", "mobilenet": "D"}

    op_coverages: dict[str, float] = {}
    raw_risks: dict[str, float] = {}

    for name, curve in curves.items():
        ax.plot(
            curve["coverage"], curve["selective_risk"],
            marker=markers.get(name, "o"), markersize=4,
            color=colours.get(name),
            label=f"{name}", linewidth=1.6,
        )
        # operating point at requested tau
        idx_op = int(np.argmin(np.abs(curve["thresholds"] - operating_tau)))
        op_cov = curve["coverage"][idx_op]
        op_risk = curve["selective_risk"][idx_op]
        op_coverages[name] = float(op_cov)
        # mark operating point with a larger marker outline
        ax.scatter([op_cov], [op_risk], s=160,
                   facecolors="none", edgecolors=colours.get(name, "k"),
                   linewidths=2, zorder=10)
        # raw risk = at full coverage (tau=0)
        raw_risks[name] = float(curve["selective_risk"][0])

    # Reference line: SVM raw error at full coverage.
    if "svm" in raw_risks:
        ax.axhline(raw_risks["svm"], linestyle=":", color="#666",
                   linewidth=1, alpha=0.7,
                   label=f"SVM raw error ({raw_risks['svm']:.3f})")

    # Operating-point note.
    ax.text(
        0.98, 0.02,
        f"○  operating point  τ = {operating_tau}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#aaa", alpha=0.9),
    )

    ax.set_xlabel("Coverage (fraction of inputs accepted)", fontsize=11)
    ax.set_ylabel("Selective risk (error rate on accepted)", fontsize=11)
    ax.set_title("Risk–Coverage curves on SafeSOS test set", fontsize=12)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_coverage_vs_threshold(
    curves: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    """Plot coverage and selective accuracy versus threshold."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, curve in curves.items():
        axes[0].plot(curve["thresholds"], curve["coverage"], label=name)
        axes[1].plot(curve["thresholds"], curve["selective_accuracy"], label=name)
    for ax in axes:
        ax.set_xlabel("Confidence threshold τ")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[0].set_ylabel("Coverage")
    axes[0].set_title("Coverage vs τ")
    axes[1].set_ylabel("Selective accuracy")
    axes[1].set_title("Selective accuracy vs τ")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_per_class_abstention(
    per_class_stats: dict[str, dict[str, np.ndarray]],
    class_names: list[str],
    tau: float,
    out_path: Path,
) -> None:
    """Grouped bar of per-class abstention rate at a fixed tau."""
    import matplotlib.pyplot as plt
    K = len(class_names)
    width = 0.8 / len(per_class_stats)
    x = np.arange(K)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, stats) in enumerate(per_class_stats.items()):
        offset = (i - (len(per_class_stats) - 1) / 2) * width
        ax.bar(x + offset, stats["abstention_rate"], width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Abstention rate")
    ax.set_title(f"Per-class abstention at τ = {tau:.2f}")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# Entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-dir", default="data/outputs")
    p.add_argument("--figures-dir", default="reports/figures")
    p.add_argument("--keypoint-dir", default="data/processed/keypoints")
    p.add_argument(
        "--operating-tau", type=float, default=0.75,
        help="threshold for per-class abstention analysis; default 0.75",
    )
    p.add_argument(
        "--n-thresholds", type=int, default=101,
        help="number of taus to sweep for risk-coverage curve",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)
    kp_dir = Path(args.keypoint_dir)

    # Class names from keypoint metadata.
    test_npz = np.load(kp_dir / "test.npz", allow_pickle=True)
    class_names = list(test_npz["class_names"])
    K = len(class_names)
    print(f"[init] classes ({K}): {class_names}")

    # Models with probability outputs.
    candidate_models = ["svm", "rf", "mobilenet"]
    thresholds = np.linspace(0.0, 1.0, args.n_thresholds)

    curves: dict[str, dict[str, np.ndarray]] = {}
    aurcs: dict[str, float] = {}
    per_class_stats: dict[str, dict[str, np.ndarray]] = {}

    for name in candidate_models:
        loaded = load_predictions(outputs_dir / f"predictions_{name}.npz")
        if loaded is None:
            print(f"[skip] no usable predictions for {name}")
            continue
        probs, y_true = loaded
        print(f"[curve] {name}: {len(y_true)} samples, K={probs.shape[1]}")

        curve = risk_coverage_curve(probs, y_true, thresholds)
        curves[name] = curve
        aurcs[name] = aurc(curve["coverage"], curve["selective_risk"])

        per_class_stats[name] = per_class_abstention(
            probs, y_true, args.operating_tau, K,
        )

    if not curves:
        print("[error] no models had probability outputs to analyse.")
        sys.exit(1)

    # Print summary.
    print("\n" + "=" * 60)
    print(f"  {'model':<12s} {'AURC':>10s}  (lower = better calibration)")
    print("  " + "-" * 30)
    for name, val in aurcs.items():
        print(f"  {name:<12s} {val:>10.5f}")
    print()

    print(f"At τ = {args.operating_tau:.2f}:")
    print(f"  {'model':<12s} {'coverage':>10s} {'sel.acc':>10s}")
    print("  " + "-" * 36)
    for name, c in curves.items():
        # Find row with threshold closest to operating_tau
        idx = int(np.argmin(np.abs(c["thresholds"] - args.operating_tau)))
        print(f"  {name:<12s} {c['coverage'][idx]:>10.4f} {c['selective_accuracy'][idx]:>10.4f}")
    print("=" * 60)

    # Save figures and metrics.
    plot_risk_coverage(
        curves, figures_dir / "risk_coverage.png",
        operating_tau=args.operating_tau,
    )
    plot_coverage_vs_threshold(curves, figures_dir / "coverage_vs_tau.png")
    plot_per_class_abstention(
        per_class_stats, class_names, args.operating_tau,
        figures_dir / "per_class_abstention.png",
    )

    summary = {
        "operating_tau": args.operating_tau,
        "class_names": class_names,
        "aurc": aurcs,
        "curves": {
            name: {k: v.tolist() for k, v in c.items()}
            for name, c in curves.items()
        },
        "per_class_at_operating_tau": {
            name: {k: v.tolist() for k, v in s.items()}
            for name, s in per_class_stats.items()
        },
    }
    summary_path = outputs_dir / "selective_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {summary_path}")


if __name__ == "__main__":
    main()
