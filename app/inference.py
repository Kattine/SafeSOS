"""
Inference pipeline for SafeSOS.

Wraps the deployed MobileNetV3 model with selective prediction logic and
maps predictions to spoken emergency messages.

The predictor is the single source of truth for "what the user is shown".
The Gradio interface delegates all decision logic here.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Make project root importable so `from models.deep_learning import ...` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# Output structure
# ============================================================

@dataclass
class PredictionResult:
    """Structured prediction output for the UI."""
    message: str            # spoken emergency message
    status: str             # "OK" | "ABSTAINED" | "IDLE" | "ERROR"
    confidence: float       # max softmax probability [0, 1]
    class_name: Optional[str] = None   # gesture class label if not abstained
    detail: str = ""        # human-readable explanation (e.g. error reason)


# ============================================================
# Predictor
# ============================================================

class GesturePredictor:
    """Loads the trained MobileNetV3 with temperature scaling.

    Thread-safety:
        Not thread-safe. Gradio's default streaming setup is single-threaded
        per session, so this is fine in practice.
    """

    # Maps trained class name -> spoken emergency message.
    # Keep these short and unambiguous: TTS engines and stressed users
    # both prefer terse phrases.
    GESTURE_TO_MESSAGE: dict[str, str] = {
        "call":       "I need to make a call. Please help.",
        "stop":       "STOP. Do not approach.",
        "palm":       "I need help.",
        "fist":       "ALERT. I am in danger.",
        "ok":         "I am OK.",
        "peace":      "Confirmed. I am safe.",
        "one":        "One.",
        "two_up":     "Two.",
        "three":      "Three.",
        "no_gesture": "",  # silent — model decided there's no signal
    }

    DEFAULT_CHECKPOINT = "models/checkpoints/mobilenet_best.pth"
    DEFAULT_META       = "models/checkpoints/mobilenet_meta.json"

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        meta_path: str | Path = DEFAULT_META,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.meta_path = Path(meta_path)
        self.confidence_threshold = confidence_threshold

        # Load metadata (class names and temperature for calibrated softmax)
        if not self.meta_path.is_file():
            raise FileNotFoundError(
                f"Model metadata not found at {self.meta_path}. "
                "Did you train the model and copy the checkpoint here?"
            )
        with open(self.meta_path) as f:
            meta = json.load(f)
        self.class_names: list[str] = list(meta["class_names"])
        self.temperature: float = float(meta["temperature"])
        self.num_classes: int = len(self.class_names)

        # Lazy-load torch/model so that just importing this module doesn't
        # pull in torch on the worker that's only serving static pages.
        self._sm = None
        self._tf = None
        self._torch = None
        self._F = None

    # ---- Lazy init ---------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._sm is not None:
            return
        import torch
        import torch.nn.functional as F
        from torchvision import transforms
        from models.deep_learning import SelectiveMobileNet

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.checkpoint_path}. "
                "Did you train the model and copy the .pth here?"
            )

        # Auto-select device: CUDA > MPS > CPU
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        sm = SelectiveMobileNet(
            num_classes=self.num_classes,
            confidence_threshold=self.confidence_threshold,
            temperature=self.temperature,
            device=device,
        )
        sm.load(str(self.checkpoint_path))
        sm.model.eval()

        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

        # Stash references
        self._sm = sm
        self._tf = tf
        self._torch = torch
        self._F = F
        self._device = device
        print(f"[predictor] loaded MobileNetV3 on {device}; "
              f"T={self.temperature:.3f}, threshold={self.confidence_threshold}")

    # ---- Public API --------------------------------------------------

    def predict(self, image: Image.Image | np.ndarray | None) -> PredictionResult:
        """Single-frame prediction with selective abstention.

        Args:
            image: PIL Image, numpy RGB array, or None (no camera input yet).

        Returns:
            PredictionResult with status in {IDLE, OK, ABSTAINED, ERROR}.
        """
        if image is None:
            return PredictionResult(
                message="Waiting for camera…",
                status="IDLE",
                confidence=0.0,
            )

        # Normalise input
        try:
            if isinstance(image, np.ndarray):
                pil = Image.fromarray(image).convert("RGB")
            elif isinstance(image, Image.Image):
                pil = image.convert("RGB")
            else:
                return PredictionResult(
                    message="Unsupported input.",
                    status="ERROR",
                    confidence=0.0,
                    detail=f"Got {type(image).__name__}",
                )
        except Exception as e:
            return PredictionResult(
                message="Could not read image.",
                status="ERROR",
                confidence=0.0,
                detail=str(e),
            )

        try:
            self._ensure_loaded()
        except Exception as e:
            return PredictionResult(
                message="Model could not be loaded.",
                status="ERROR",
                confidence=0.0,
                detail=str(e),
            )

        # Run inference
        torch = self._torch
        F = self._F
        x = self._tf(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._sm.model(x)
            probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()[0]

        idx = int(probs.argmax())
        confidence = float(probs[idx])
        class_name = self.class_names[idx]

        # Selective decision: abstain on low confidence
        if confidence < self.confidence_threshold:
            return PredictionResult(
                message="I'm not sure. Please try again.",
                status="ABSTAINED",
                confidence=confidence,
                detail=f"Top guess: {class_name} ({confidence:.0%})",
            )

        # Special case: high-confidence no_gesture means "no signal"
        if class_name == "no_gesture":
            return PredictionResult(
                message="No gesture detected.",
                status="IDLE",
                confidence=confidence,
                class_name=class_name,
            )

        # Confident, meaningful prediction
        message = self.GESTURE_TO_MESSAGE.get(class_name, class_name)
        return PredictionResult(
            message=message,
            status="OK",
            confidence=confidence,
            class_name=class_name,
        )

    def predict_with_full_probs(
        self, image: Image.Image | np.ndarray,
    ) -> tuple[PredictionResult, dict[str, float]]:
        """Same as predict(), but also returns the full probability dict.

        Used by the UI's "top-3" diagnostic panel.
        """
        result = self.predict(image)
        if image is None or result.status == "ERROR":
            return result, {}

        # Re-run to get probabilities (cheap; already cached model)
        torch = self._torch
        F = self._F
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image).convert("RGB")
        else:
            pil = image.convert("RGB")
        x = self._tf(pil).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._sm.model(x)
            probs = F.softmax(logits / self.temperature, dim=-1).cpu().numpy()[0]

        prob_dict = {name: float(p) for name, p in zip(self.class_names, probs)}
        return result, prob_dict
