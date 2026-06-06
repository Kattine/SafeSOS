"""
MobileNetV3-Small fine-tuned on 10-class gesture problem,
with selective prediction via softmax thresholding + temperature scaling.

This is the model deployed in the live app.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision import models


@dataclass
class SelectivePrediction:
    class_idx: int | None  # None when abstained
    confidence: float
    abstained: bool


class SelectiveMobileNet:
    """MobileNetV3-Small with built-in selective prediction."""

    def __init__(
        self,
        num_classes: int = 10,
        confidence_threshold: float = 0.75,
        temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.threshold = confidence_threshold
        self.temperature = temperature
        self.device = device
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Replace the classifier head for our 10 classes
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, self.num_classes)
        return net.to(self.device)

    def predict_with_abstention(self, x: torch.Tensor) -> SelectivePrediction:
        """Single-sample prediction with selective prediction logic.

        Returns SelectivePrediction.abstained=True when max-softmax < threshold,
        which is the core safety feature of the system.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x.to(self.device))
            # Temperature scaling for better-calibrated softmax
            scaled = logits / self.temperature
            probs = torch.softmax(scaled, dim=-1)
            max_prob, pred_idx = probs.max(dim=-1)
            confidence = float(max_prob.item())
            if confidence < self.threshold:
                return SelectivePrediction(
                    class_idx=None,
                    confidence=confidence,
                    abstained=True,
                )
            return SelectivePrediction(
                class_idx=int(pred_idx.item()),
                confidence=confidence,
                abstained=False,
            )

    def load(self, checkpoint_path: str) -> None:
        """Load trained weights."""
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)

    def save(self, checkpoint_path: str) -> None:
        torch.save(self.model.state_dict(), checkpoint_path)
