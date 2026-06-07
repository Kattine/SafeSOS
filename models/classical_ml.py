"""
Classical ML pipeline for SafeSOS.

Pipeline:
    Image -> MediaPipe Hands -> 21 (x, y, z) keypoints (63-d feature) -> SVM/RF
                                                                          |
                                                                          v
                                                              Platt scaling for
                                                              calibrated probabilities
                                                              (needed for selective
                                                               prediction)

Why this design:
- MediaPipe provides a strong, hand-specific feature extractor that compresses
  a 224x224x3 image (~150k floats) into 63 floats, drastically reducing the
  amount of data the classical model has to learn from.
- SVM with Platt scaling produces probabilities that can be thresholded for
  selective prediction, matching what we do for the deep model.
- All keypoint extraction is done once and cached to .npz so we never re-run
  MediaPipe during training or evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

# MediaPipe is heavy and CPU-only; import lazily where used.


# ============================================================
# Keypoint extraction
# ============================================================

@dataclass
class KeypointExtractorConfig:
    """Configuration for MediaPipe-based keypoint extraction.

    Uses the new `mp.tasks.vision.HandLandmarker` API. The older
    `mp.solutions.hands` API is broken on Python 3.12+ (mediapipe issues
    #6200, #6204, #6261), so we use the modern task-based API which
    Google now recommends for all new projects.
    """
    num_hands: int = 1                          # one dominant hand
    min_detection_confidence: float = 0.3       # permissive on detection
    min_presence_confidence: float = 0.3        # permissive on presence
    min_tracking_confidence: float = 0.3        # only used in video mode
    model_filename: str = "hand_landmarker.task"  # downloaded on first use


# Public download URL for the off-the-shelf hand landmark model bundle.
# This is the float16 variant published by Google AI Edge; ~7MB, no auth.
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class HandKeypointExtractor:
    """Extracts 21 hand landmarks per image using MediaPipe's task API.

    Designed as a drop-in replacement for the legacy `mp.solutions.hands`
    wrapper: `extract()` returns either a (63,) float32 vector or None,
    and `extract_batch()` returns (X, valid_mask) just like before.
    """

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

    # ---- Model file management --------------------------------------

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

    # ---- Detector lifecycle -----------------------------------------

    def _ensure_initialised(self) -> None:
        if self._detector is not None:
            return
        # Use the modern task API (mediapipe.tasks.vision.HandLandmarker).
        # The legacy `mp.solutions.hands` API is broken on Python 3.12+.
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
        # mediapipe's task API expects its own Image wrapper.
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._detector.detect(mp_image)
        if not result.hand_landmarks:
            return None
        # `result.hand_landmarks` is a list of hands; each is a list of 21 NormalizedLandmark
        landmarks = result.hand_landmarks[0]
        if len(landmarks) != self.NUM_LANDMARKS:
            # Defensive: in theory always 21, but bail if the API changes
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


# ============================================================
# Selective classification result
# ============================================================

@dataclass
class SelectivePrediction:
    """Same structure as the deep-learning model's output."""
    class_idx: int | None
    confidence: float
    abstained: bool


# ============================================================
# Keypoint classifier with calibration + selective prediction
# ============================================================

class KeypointClassifier:
    """Trains SVM or Random Forest on MediaPipe keypoints, with calibrated
    probabilities for selective prediction.

    Use:
        clf = KeypointClassifier(backend="svm").fit(X_train, y_train, X_val, y_val)
        y_pred = clf.predict(X_test)
        sel = clf.predict_with_abstention(x_single, threshold=0.75)
    """

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
        """Build the underlying estimator. Wrapped in CalibratedClassifierCV
        downstream so probabilities are reliable."""
        if self.backend == "svm":
            from sklearn.svm import SVC
            return SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=False,  # we will calibrate explicitly via CalibratedClassifierCV
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
        """Fit the model with Platt-scaled probabilities.

        We use CalibratedClassifierCV with method='sigmoid' (Platt scaling)
        and an internal 5-fold split, so X_val/y_val are not strictly
        required, but if provided we can report validation metrics here.
        """
        from sklearn.calibration import CalibratedClassifierCV

        base = self._make_base_model()
        # cv=5 trains 5 base models on rotating splits; final probabilities
        # are averaged. This is more reliable than probability=True in SVC.
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
        """Single-sample selective prediction matching the deep-learning API.

        Returns abstained=True when max calibrated probability < threshold.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        proba = self.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        if confidence < threshold:
            return SelectivePrediction(class_idx=None,
                                       confidence=confidence,
                                       abstained=True)
        # Map array-index back to original class label
        class_label = int(self.classes_[idx])
        return SelectivePrediction(class_idx=class_label,
                                   confidence=confidence,
                                   abstained=False)

    # ---- Persistence -------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save with joblib (handles sklearn objects properly)."""
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
