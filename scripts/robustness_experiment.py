"""
Robustness experiment: evaluate all three SafeSOS models under realistic
input perturbations.

For each perturbation type and severity, we apply the perturbation to every
test image, re-run all three models, and record:
  - accuracy
  - macro-F1
  - abstention rate (for models that produce calibrated probabilities)
  - selective accuracy at tau = 0.75
  - MediaPipe detection rate (for the classical pipeline)

The central narrative this experiment supports is that MobileNetV3 degrades
gracefully under distribution shift while the keypoint-based pipeline collapses
when MediaPipe stops detecting hands. This is the inverse of the clean-data
finding (where SVM dominated) and is the third leg of the SafeSOS paper.

Reads:
    models/checkpoints/*           (all three trained models)
    data/processed/test/<class>/*.jpg

Writes:
    data/outputs/robustness_metrics.json
    reports/figures/robustness_<perturbation>.png   (one per perturbation type)
    reports/figures/robustness_summary.png          (4-panel summary)

This script is slower than evaluate.py because it re-runs MediaPipe on
perturbed images. Budget ~10-20 minutes on a Mac M-series CPU.

Run:
    python scripts/robustness_experiment.py
    python scripts/robustness_experiment.py --max-test-samples 1000  # quicker debug run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Make project root importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


# ============================================================
# Perturbation primitives
# ============================================================

def perturb_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Gaussian blur — simulates lens defocus or motion blur."""
    if sigma <= 0:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def perturb_low_light(img: Image.Image, gamma: float) -> Image.Image:
    """Brightness reduction — simulates low-light camera capture.

    gamma=1.0 leaves image unchanged; smaller gamma -> darker.
    We multiply pixel values by gamma in linear RGB.
    """
    if gamma >= 1.0:
        return img
    arr = np.asarray(img).astype(np.float32) * gamma
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def perturb_rotation(img: Image.Image, angle_deg: float) -> Image.Image:
    """In-plane rotation — simulates off-axis camera."""
    if angle_deg == 0:
        return img
    return img.rotate(angle_deg, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def perturb_occlusion(img: Image.Image, mask_ratio: float, seed: int = 0) -> Image.Image:
    """Random rectangular black mask — simulates partial hand occlusion."""
    if mask_ratio <= 0:
        return img
    rng = np.random.default_rng(seed)
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    mh = int(h * mask_ratio)
    mw = int(w * mask_ratio)
    if mh < 1 or mw < 1:
        return img
    # Random top-left, ensure mask stays inside the image
    y0 = int(rng.integers(0, max(1, h - mh)))
    x0 = int(rng.integers(0, max(1, w - mw)))
    arr[y0:y0 + mh, x0:x0 + mw] = 0
    return Image.fromarray(arr)


# Perturbation registry: name -> (function, list of severity values, severity label)
PERTURBATIONS: dict[str, tuple[Callable, list[float], str]] = {
    "blur":      (perturb_blur,      [0.0, 1.0, 2.0, 4.0, 8.0],      "σ (Gaussian radius)"),
    "low_light": (perturb_low_light, [1.0, 0.7, 0.5, 0.3, 0.15],     "γ (brightness)"),
    "rotation":  (perturb_rotation,  [0.0, 10.0, 20.0, 30.0, 45.0],  "rotation angle (°)"),
    "occlusion": (perturb_occlusion, [0.0, 0.1, 0.2, 0.3, 0.4],      "mask ratio"),
}


# ============================================================
# Test-set inventory
# ============================================================

@dataclass
class TestImage:
    path: Path
    label: int


def collect_test_images(test_dir: Path) -> tuple[list[TestImage], list[str]]:
    """Walk test_dir/<class>/*.jpg in deterministic order."""
    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    items: list[TestImage] = []
    for label, cdir in enumerate(class_dirs):
        for p in sorted(cdir.iterdir()):
            name = p.name
            if name.startswith(".") or p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            items.append(TestImage(path=p, label=label))
    return items, class_names


# ============================================================
# Predictor wrappers (one per model family)
# ============================================================

class ClassicalGroup:
    """Multiple keypoint-based classifiers sharing ONE MediaPipe pass.

    Major optimisation: running MediaPipe is by far the slowest stage
    (~30 ms / image vs <1 ms for SVM/RF classification on 63-d vectors).
    Previously SVM and RF were separate ClassicalPredictor instances and
    each cell re-ran MediaPipe twice. This class extracts keypoints once
    and runs every wrapped classifier on the shared feature matrix.
    """

    def __init__(self, classifier_ckpts: dict[str, Path]) -> None:
        from models.classical_ml import HandKeypointExtractor, KeypointClassifier
        self.extractor = HandKeypointExtractor()
        self.classifiers = {
            name: KeypointClassifier.load(path)
            for name, path in classifier_ckpts.items()
        }
        self.extractor._ensure_initialised()

    def predict_batch(
        self, images: list[Image.Image],
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Returns {classifier_name: (y_pred, max_conf, detected_mask)}."""
        n = len(images)
        feature_buffer: list[np.ndarray] = []
        feature_indices: list[int] = []
        detected = np.zeros(n, dtype=bool)

        # Stage 1: MediaPipe once
        for i, img in enumerate(images):
            arr = np.asarray(img.convert("RGB"))
            vec = self.extractor.extract(arr)
            if vec is None:
                continue
            detected[i] = True
            feature_buffer.append(vec)
            feature_indices.append(i)

        # Stage 2: each classifier on the same feature matrix
        results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if feature_buffer:
            X = np.stack(feature_buffer, axis=0)
            for name, clf in self.classifiers.items():
                probs = clf.predict_proba(X)
                preds = probs.argmax(axis=1)
                maxes = probs.max(axis=1)
                y_pred = np.full(n, -1, dtype=np.int64)
                max_conf = np.zeros(n, dtype=np.float32)
                for k, idx in enumerate(feature_indices):
                    y_pred[idx] = int(preds[k])
                    max_conf[idx] = float(maxes[k])
                results[name] = (y_pred, max_conf, detected.copy())
        else:
            # No detections at all: produce empty results for every classifier
            for name in self.classifiers:
                results[name] = (
                    np.full(n, -1, dtype=np.int64),
                    np.zeros(n, dtype=np.float32),
                    detected.copy(),
                )
        return results

    def close(self) -> None:
        self.extractor.close()


class ClassicalPredictor:
    """MediaPipe -> SVM (or RF) inference on perturbed images."""

    def __init__(self, ckpt_path: Path) -> None:
        from models.classical_ml import HandKeypointExtractor, KeypointClassifier
        self.extractor = HandKeypointExtractor()
        self.classifier = KeypointClassifier.load(ckpt_path)
        # Force-initialise to surface any errors early
        self.extractor._ensure_initialised()

    def predict_batch(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (y_pred, max_conf, detected_mask).

        detected_mask[i] = True iff MediaPipe successfully extracted keypoints.
        Samples with no detection are assigned y_pred=-1, max_conf=0.

        Performance note: keypoint extraction is the slow stage and is
        inherently per-image (MediaPipe has no batch API). We extract all
        keypoints first, then call predict_proba once on the stacked matrix,
        which makes the classifier stage ~30x faster than per-image calls.
        """
        n = len(images)
        y_pred = np.full(n, -1, dtype=np.int64)
        max_conf = np.zeros(n, dtype=np.float32)
        detected = np.zeros(n, dtype=bool)

        # Stage 1: MediaPipe keypoint extraction (per-image, unavoidable)
        feature_buffer: list[np.ndarray] = []
        feature_indices: list[int] = []
        for i, img in enumerate(images):
            arr = np.asarray(img.convert("RGB"))
            vec = self.extractor.extract(arr)
            if vec is None:
                continue
            detected[i] = True
            feature_buffer.append(vec)
            feature_indices.append(i)

        # Stage 2: batched classifier inference on the detected subset
        if feature_buffer:
            X = np.stack(feature_buffer, axis=0)
            probs = self.classifier.predict_proba(X)   # batched, big speedup
            preds = probs.argmax(axis=1)
            maxes = probs.max(axis=1)
            for k, original_idx in enumerate(feature_indices):
                y_pred[original_idx] = int(preds[k])
                max_conf[original_idx] = float(maxes[k])

        return y_pred, max_conf, detected

    def close(self) -> None:
        self.extractor.close()


class DeepPredictor:
    """MobileNetV3 (with temperature scaling) on perturbed images."""

    def __init__(self, ckpt_path: Path, meta_path: Path) -> None:
        import torch
        from torchvision import transforms
        from models.deep_learning import SelectiveMobileNet

        with open(meta_path) as f:
            meta = json.load(f)
        temperature = float(meta["temperature"])

        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.device = device
        self.temperature = temperature

        self.sm = SelectiveMobileNet(
            num_classes=10,
            confidence_threshold=0.0,
            temperature=temperature,
            device=device,
        )
        self.sm.load(str(ckpt_path))
        self.sm.model.eval()

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._torch = torch
        self._F = __import__("torch.nn.functional", fromlist=["softmax"])

    def predict_batch(
        self, images: list[Image.Image], batch_size: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (y_pred, max_conf, detected_mask) — detected always True."""
        torch = self._torch
        F = self._F

        all_pred, all_conf = [], []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            x = torch.stack([self.tf(img.convert("RGB")) for img in batch]).to(self.device)
            with torch.no_grad():
                logits = self.sm.model(x)
                probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()
            all_pred.append(probs.argmax(axis=1))
            all_conf.append(probs.max(axis=1))

        y_pred = np.concatenate(all_pred, axis=0)
        max_conf = np.concatenate(all_conf, axis=0).astype(np.float32)
        detected = np.ones(len(images), dtype=bool)
        return y_pred, max_conf, detected


# ============================================================
# Metrics per (model, perturbation, severity)
# ============================================================

@dataclass
class CellMetrics:
    """One row of the robustness results table."""
    model: str
    perturbation: str
    severity: float
    n_total: int
    detection_rate: float       # fraction of samples MediaPipe handled (1.0 for DL)
    accuracy: float             # overall accuracy treating non-detections as wrong
    accuracy_when_detected: float  # accuracy among detected samples
    macro_f1: float
    abstention_rate_at_075: float
    selective_acc_at_075: float


def evaluate_cell(
    model_name: str,
    pert_name: str,
    severity: float,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    max_conf: np.ndarray,
    detected: np.ndarray,
    tau: float = 0.75,
) -> CellMetrics:
    """Aggregate per-sample arrays into a single CellMetrics row."""
    n = len(y_true)

    # Overall accuracy: treats non-detections as wrong (-1 != any true label)
    overall_acc = float((y_pred == y_true).mean())

    # Accuracy among detected only
    if detected.any():
        acc_when_det = float((y_pred[detected] == y_true[detected]).mean())
    else:
        acc_when_det = 0.0

    # macro-F1 across all samples (non-detections count as errors)
    # We replace -1 with a sentinel that's not a valid class to ensure they count as misses.
    y_pred_safe = np.where(y_pred < 0, -1, y_pred)
    try:
        f1 = float(f1_score(y_true, y_pred_safe, average="macro",
                            labels=list(range(int(y_true.max()) + 1)),
                            zero_division=0))
    except Exception:
        f1 = 0.0

    # Selective behaviour at tau among detected samples
    accepted = detected & (max_conf >= tau)
    if accepted.any():
        sel_acc = float((y_pred[accepted] == y_true[accepted]).mean())
    else:
        sel_acc = 0.0
    abstention = 1.0 - float(accepted.mean())  # fraction of n that we did NOT accept

    return CellMetrics(
        model=model_name,
        perturbation=pert_name,
        severity=float(severity),
        n_total=n,
        detection_rate=float(detected.mean()),
        accuracy=overall_acc,
        accuracy_when_detected=acc_when_det,
        macro_f1=f1,
        abstention_rate_at_075=abstention,
        selective_acc_at_075=sel_acc,
    )


# ============================================================
# Plotting
# ============================================================

def _x_label_for(pert_name: str) -> str:
    return PERTURBATIONS[pert_name][2]


def plot_perturbation(
    pert_name: str,
    cells: list[CellMetrics],
    out_path: Path,
) -> None:
    """For one perturbation, plot accuracy vs severity for all models."""
    import matplotlib.pyplot as plt

    colours = {"svm": "#1f77b4", "rf": "#ff7f0e", "mobilenet": "#2ca02c"}
    markers = {"svm": "o", "rf": "s", "mobilenet": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)

    # Group cells by model
    models = sorted({c.model for c in cells})
    for m in models:
        rows = sorted([c for c in cells if c.model == m], key=lambda c: c.severity)
        sev = [c.severity for c in rows]
        acc = [c.accuracy for c in rows]
        det = [c.detection_rate for c in rows]
        axes[0].plot(sev, acc, marker=markers.get(m, "o"), color=colours.get(m),
                     label=m, linewidth=1.8)
        axes[1].plot(sev, det, marker=markers.get(m, "o"), color=colours.get(m),
                     label=m, linewidth=1.8)

    axes[0].set_ylabel("Overall accuracy", fontsize=11)
    axes[0].set_title(f"{pert_name}: accuracy vs severity")
    axes[1].set_ylabel("Detection / coverage", fontsize=11)
    axes[1].set_title(f"{pert_name}: pipeline availability")
    for ax in axes:
        ax.set_xlabel(_x_label_for(pert_name))
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_summary(
    all_cells: list[CellMetrics],
    out_path: Path,
) -> None:
    """4-panel grid: one panel per perturbation, accuracy vs severity."""
    import matplotlib.pyplot as plt

    colours = {"svm": "#1f77b4", "rf": "#ff7f0e", "mobilenet": "#2ca02c"}
    markers = {"svm": "o", "rf": "s", "mobilenet": "D"}

    pert_order = list(PERTURBATIONS.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, pert_name in zip(axes, pert_order):
        cells = [c for c in all_cells if c.perturbation == pert_name]
        models = sorted({c.model for c in cells})
        for m in models:
            rows = sorted([c for c in cells if c.model == m], key=lambda c: c.severity)
            sev = [c.severity for c in rows]
            acc = [c.accuracy for c in rows]
            ax.plot(sev, acc, marker=markers.get(m, "o"), color=colours.get(m),
                    label=m, linewidth=1.8)
        ax.set_title(pert_name)
        ax.set_xlabel(_x_label_for(pert_name))
        ax.set_ylabel("Overall accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle("Robustness under perturbation (overall accuracy)", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ============================================================
# Main experiment loop
# ============================================================

def run_experiment(
    test_items: list[TestImage],
    predictors: dict[str, object],
    perturbations: dict[str, tuple[Callable, list[float], str]],
    progress: bool = True,
) -> list[CellMetrics]:
    """Outer loop over (perturbation, severity); inner loop loads images.

    `predictors` may contain a mix of:
      - single predictors (have predict_batch returning a triple)
      - group predictors  (have predict_batch returning a dict of triples)

    Group predictors are detected by their class name 'ClassicalGroup'.
    """
    all_cells: list[CellMetrics] = []
    y_true = np.array([t.label for t in test_items], dtype=np.int64)

    for pert_name, (pert_fn, severities, _) in perturbations.items():
        for severity in severities:
            label = f"{pert_name} @ {severity}"
            print(f"\n[run] {label}")

            # Load + perturb everything in memory once per (pert, severity) cell
            perturbed: list[Image.Image] = []
            iterator = test_items
            if progress:
                iterator = tqdm(test_items, desc=f"perturb {label}", leave=False)
            for item in iterator:
                with Image.open(item.path) as raw:
                    raw_rgb = raw.convert("RGB")
                    if pert_name == "occlusion":
                        img = pert_fn(raw_rgb, severity, seed=hash(str(item.path)) & 0xFFFF)
                    else:
                        img = pert_fn(raw_rgb, severity)
                    perturbed.append(img.copy())

            # Run each predictor on the same perturbed batch
            for key, predictor in predictors.items():
                t0 = __import__("time").time()
                out = predictor.predict_batch(perturbed)
                elapsed = __import__("time").time() - t0

                # Group predictor: dict of model_name -> triple
                if isinstance(out, dict):
                    for mname, (y_pred, max_conf, detected) in out.items():
                        cell = evaluate_cell(
                            model_name=mname, pert_name=pert_name, severity=severity,
                            y_pred=y_pred, y_true=y_true,
                            max_conf=max_conf, detected=detected,
                        )
                        all_cells.append(cell)
                        print(f"  {mname:<10s} acc={cell.accuracy:.4f}  "
                              f"det={cell.detection_rate:.3f}  "
                              f"sel_acc={cell.selective_acc_at_075:.4f}  "
                              f"abst={cell.abstention_rate_at_075:.3f}")
                    print(f"  (group inference {elapsed:.1f}s, MediaPipe shared)")
                else:
                    # Single predictor
                    y_pred, max_conf, detected = out
                    cell = evaluate_cell(
                        model_name=key, pert_name=pert_name, severity=severity,
                        y_pred=y_pred, y_true=y_true,
                        max_conf=max_conf, detected=detected,
                    )
                    all_cells.append(cell)
                    print(f"  {key:<10s} acc={cell.accuracy:.4f}  "
                          f"det={cell.detection_rate:.3f}  "
                          f"sel_acc={cell.selective_acc_at_075:.4f}  "
                          f"abst={cell.abstention_rate_at_075:.3f}  "
                          f"({elapsed:.1f}s)")
            # release memory for this cell
            del perturbed

    return all_cells


# ============================================================
# Entry point
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--checkpoint-dir", default="models/checkpoints")
    p.add_argument("--outputs-dir", default="data/outputs")
    p.add_argument("--figures-dir", default="reports/figures")
    p.add_argument(
        "--max-test-samples", type=int, default=None,
        help="if set, subsample test set for a quicker debug run "
             "(stratified would be ideal; this is just a head-cut by ordering)",
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=list(PERTURBATIONS.keys()),
        help="perturbations to skip (faster debug runs)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    outputs_dir = Path(args.outputs_dir)
    figures_dir = Path(args.figures_dir)

    # ---- Test set ----
    test_items, class_names = collect_test_images(data_dir / "test")
    print(f"[init] test set: {len(test_items)} images, {len(class_names)} classes")
    if args.max_test_samples is not None and args.max_test_samples < len(test_items):
        # Take every Nth item to retain rough class balance
        step = len(test_items) // args.max_test_samples
        test_items = test_items[::step][:args.max_test_samples]
        print(f"[init] subsampled to {len(test_items)} for debug")

    # ---- Predictors ----
    # Use ClassicalGroup so SVM and RF share a single MediaPipe pass.
    print("[init] loading classical group (SVM + RF, shared MediaPipe)...")
    classical_group = ClassicalGroup({
        "svm": ckpt_dir / "keypoint_svm.joblib",
        "rf":  ckpt_dir / "keypoint_rf.joblib",
    })

    print("[init] loading MobileNet...")
    mn_pred = DeepPredictor(
        ckpt_dir / "mobilenet_best.pth",
        ckpt_dir / "mobilenet_meta.json",
    )

    predictors: dict[str, object] = {
        "classical": classical_group,   # produces both svm + rf rows per cell
        "mobilenet": mn_pred,
    }

    # ---- Perturbations to run ----
    active_perturbations = {
        k: v for k, v in PERTURBATIONS.items() if k not in args.skip
    }
    print(f"[init] perturbations: {list(active_perturbations.keys())}")

    # ---- Run ----
    cells = run_experiment(test_items, predictors, active_perturbations)

    # ---- Save ----
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = outputs_dir / "robustness_metrics.json"
    metrics_path.write_text(json.dumps({
        "class_names": class_names,
        "n_test_samples": len(test_items),
        "results": [c.__dict__ for c in cells],
    }, indent=2))
    print(f"\n[save] {metrics_path}")

    # ---- Plots ----
    for pert_name in active_perturbations:
        pert_cells = [c for c in cells if c.perturbation == pert_name]
        plot_perturbation(pert_name, pert_cells,
                          figures_dir / f"robustness_{pert_name}.png")

    plot_summary(cells, figures_dir / "robustness_summary.png")

    # ---- Cleanup ----
    classical_group.close()
    print("\n[done] robustness experiment complete.")


if __name__ == "__main__":
    main()
