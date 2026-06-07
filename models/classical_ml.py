"""Classical ML backend for SafeSOS.

Uses MediaPipe hand keypoints (63-d) with SVM/RF classifiers and calibrated
probabilities for selective prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# Import MediaPipe lazily where needed.


# Keypoint extraction

@dataclass
class KeypointExtractorConfig:
    """Configuration for MediaPipe keypoint extraction."""
    num_hands: int = 1                          # one dominant hand
    min_detection_confidence: float = 0.3       # permissive on detection
    min_presence_confidence: float = 0.3        # permissive on presence
    min_tracking_confidence: float = 0.3        # only used in video mode
    model_filename: str = "hand_landmarker.task"  # downloaded on first use


# Public URL for MediaPipe hand landmark model.
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandKeypointExtractor:
    """Extract 21 hand landmarks per image using MediaPipe Tasks API."""

    NUM_LANDMARKS = 21
    FEATURE_DIM = NUM_LANDMARKS * 3  # x, y, z per landmark -> 63

    def __init__(
        self,
        config: KeypointExtractorConfig | None = None,
        model_dir: str | Path = "models/checkpoints",
    ) -> None:
        self.config = config or KeypointExtractorConfig()
        self.model_dir = Path(model_dir)
        self._detector = None  # lazy init: avoids importing mediapipe for unit tests

    # Model file management

    def _ensure_model_file(self) -> Path:
        """Download the .task model bundle on first use, then cache locally."""
        model_path = self.model_dir / self.config.model_filename
        if model_path.is_file() and model_path.stat().st_size > 0:
            return model_path

        from urllib.request import urlretrieve
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[mediapipe] downloading hand landmark model -> {model_path}")
        urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"[mediapipe] model ready ({size_mb:.1f} MB)")
        return model_path

    # Detector lifecycle

    def _ensure_initialised(self) -> None:
        if self._detector is not None:
            return
        # Use MediaPipe Tasks API (works on Python 3.12+).
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = self._ensure_model_file()
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract(self, image_rgb: np.ndarray) -> np.ndarray | None:
        """Extract a 63-d feature vector from a single RGB image.

        Args:
            image_rgb: H x W x 3 uint8 RGB image.

        Returns:
            (63,) float32 array of flattened (x, y, z) for 21 landmarks,
            or None if MediaPipe failed to detect a hand.
        """
        self._ensure_initialised()
        # MediaPipe task API uses its own Image wrapper.
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        # Take the first detected hand.
        landmarks = result.hand_landmarks[0]
        if len(landmarks) != self.NUM_LANDMARKS:
            # Defensive check in case API output changes.
            return None
        vec = np.empty(self.FEATURE_DIM, dtype=np.float32)
        for i, lm in enumerate(landmarks):
            vec[3 * i + 0] = lm.x
            vec[3 * i + 1] = lm.y
            vec[3 * i + 2] = lm.z
        return vec

    def extract_batch(
        self,
        image_paths: list[Path],
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract keypoints for a list of image paths.

        Returns:
            X: (N_detected, 63) feature matrix.
            valid_mask: (N_input,) boolean array, True if MediaPipe detected
                        a hand. Callers should align labels with this mask.
        """
        from PIL import Image
        from tqdm import tqdm

        valid_mask = np.zeros(len(image_paths), dtype=bool)
        features: list[np.ndarray] = []

        iterator = tqdm(image_paths, desc="MediaPipe", disable=not verbose)
        for i, path in enumerate(iterator):
            try:
                img = np.array(Image.open(path).convert("RGB"))
            except Exception:
                continue
            vec = self.extract(img)
            if vec is not None:
                features.append(vec)
                valid_mask[i] = True

        if not features:
            return np.empty((0, self.FEATURE_DIM), dtype=np.float32), valid_mask
        X = np.stack(features, axis=0)
        return X, valid_mask

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


# Selective prediction output

@dataclass
class SelectivePrediction:
    """Same structure as the deep-learning model's output."""
    class_idx: int | None
    confidence: float
    abstained: bool


# Classifier with calibration + selective prediction

class KeypointClassifier:
    """Train SVM or Random Forest on keypoints with calibrated probabilities."""

    def __init__(
        self,
        backend: Literal["svm", "rf"] = "svm",
        random_state: int = 42,
    ) -> None:
        self.backend = backend
        self.random_state = random_state
        self.model = None
        self.classes_: np.ndarray | None = None

    def _make_base_model(self):
        """Build the base estimator used before calibration."""
        if self.backend == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False,
                random_state=self.random_state,
            )
        if self.backend == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                n_jobs=-1,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown backend: {self.backend}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "KeypointClassifier":
        """Fit model and calibrate probabilities with sigmoid scaling."""
        from sklearn.calibration import CalibratedClassifierCV

        base = self._make_base_model()
        # Use internal CV calibration for more stable probabilities.
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        if X_val is not None and y_val is not None:
            val_acc = float((self.model.predict(X_val) == y_val).mean())
            print(f"[classical_ml] validation accuracy: {val_acc:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.model.predict_proba(X)

    def predict_with_abstention(
        self,
        x: np.ndarray,
        threshold: float = 0.75,
    ) -> SelectivePrediction:
        """Single-sample selective prediction."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        proba = self.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        if confidence < threshold:
            return SelectivePrediction(class_idx=None,
                                       confidence=confidence,
                                       abstained=True)
        # Map probability index back to class label.
        class_label = int(self.classes_[idx])
        return SelectivePrediction(class_idx=class_label,
                                   confidence=confidence,
                                   abstained=False)

    # Persistence

    def save(self, path: str | Path) -> None:
        """Save model bundle with joblib."""
        import joblib
        joblib.dump({"model": self.model,
                     "backend": self.backend,
                     "classes_": self.classes_},
                    str(path))

    @classmethod
    def load(cls, path: str | Path) -> "KeypointClassifier":
        import joblib
        bundle = joblib.load(str(path))
        obj = cls(backend=bundle["backend"])
        obj.model = bundle["model"]
        obj.classes_ = bundle["classes_"]
        return obj
