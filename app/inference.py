"""
Inference pipeline for Silent-SOS.

Wraps the deployed deep learning model with selective prediction logic.
TODO: implement actual model loading + prediction.
"""

from dataclasses import dataclass


@dataclass
class PredictionResult:
    """Structured prediction output for the UI."""
    message: str
    status: str          # "OK" | "ABSTAINED" | "ERROR"
    confidence: float    # [0, 1]
    class_name: str | None = None


class GesturePredictor:
    """Loads the trained model and exposes a predict() method."""

    # Maps HaGRID class name -> spoken emergency message
    GESTURE_TO_MESSAGE = {
        "call": "I need to make a call. Please help.",
        "stop": "STOP. Do not approach.",
        "palm": "I need help.",
        "fist": "ALERT. I am in danger.",
        "ok": "I am OK.",
        "peace": "Confirmed. I am safe.",
        "one": "One.",
        "two_up": "Two.",
        "three": "Three.",
        "no_gesture": "",  # silent
    }

    def __init__(
        self,
        checkpoint_path: str = "models/checkpoints/mobilenet_best.pth",
        confidence_threshold: float = 0.75,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        self._model = None  # lazy loaded
        # TODO: load model from checkpoint_path

    def predict(self, image) -> PredictionResult:
        """Run a single image through the model with selective prediction."""
        # TODO: implement actual inference
        # Placeholder so the UI runs end-to-end during development
        return PredictionResult(
            message="(model not loaded yet)",
            status="IDLE",
            confidence=0.0,
        )
