# SafeSOS: Selective Static Gesture Recognition for Emergency Communication

> A computer-vision system for silent emergency communication, with built-in
> **selective prediction** to abstain when uncertain — designed for high-stakes
> accessibility scenarios.

## 🎯 What it does

Recognises 9 static hand gestures (plus a no-gesture class) from a webcam
and translates them into spoken emergency messages. When the model's
confidence falls below the threshold (τ = 0.75) it abstains rather than
guessing — critical for safety-sensitive use.

🔗 **Live demo:** https://huggingface.co/spaces/zkmine/safesos

## 🏗️ Project structure

```
.
├── app/                  # Gradio Blocks UI + inference glue
│   ├── inference.py      # Predictor wrapping the MobileNetV3 checkpoint
│   ├── interface.py      # Gradio Blocks layout
│   └── styles.py         # Custom CSS for the dark "mission control" UI
├── scripts/              # Data, training, evaluation pipelines
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── extract_keypoints.py
│   ├── train_classical.py        # SVM + Random Forest
│   ├── train_dl.py               # MobileNetV3 fine-tuning
│   ├── evaluate.py
│   ├── selective_prediction.py
│   └── robustness_experiment.py
├── models/               # Model implementations + checkpoints
│   ├── naive_baseline.py
│   ├── classical_ml.py          # MediaPipe keypoints + SVM/RF wrappers
│   ├── deep_learning.py         # MobileNetV3-Small + selective prediction
│   └── checkpoints/             # .pth / .joblib files (gitignored except small ones)
├── data/                 # raw/, processed/, outputs/ (mostly gitignored)
├── notebooks/            # Exploration only — not graded
├── reports/              # Final report + figures (risk-coverage, robustness)
└── main.py               # Launch the Gradio app
```

## 🚀 Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download HaGRID 10-class subset (~3.8 GB; uses cj-mills' classification format)
python scripts/download_data.py

# 3. Prepare 70/15/15 train/val/test splits (caps each class at 3000 images)
python scripts/prepare_data.py

# 4. Extract MediaPipe hand keypoints for the classical pipeline
python scripts/extract_keypoints.py

# 5. Train each model family
python scripts/train_classical.py     # SVM + Random Forest
python scripts/train_dl.py            # MobileNetV3-Small (Colab T4 recommended)

# 6. Evaluate + risk-coverage analysis
python scripts/evaluate.py
python scripts/selective_prediction.py

# 7. Robustness experiment under 4 perturbation types
python scripts/robustness_experiment.py

# 8. Run the demo app locally
python main.py
```

## 📊 Models

| Model | Approach | Test accuracy | AURC ↓ |
|---|---|---|---|
| Naive baseline | Majority-class predictor | 0.1034 | — |
| MediaPipe + Random Forest (300 trees) | 21 keypoints → RF | 0.9274 | 0.00886 |
| MediaPipe + SVM (Platt) | 21 keypoints → SVM | **0.9680** | 0.01393 |
| MobileNetV3-Small (T = 1.10) | End-to-end CNN | 0.9271 | **0.00702** |

**Key finding:** the SVM dominates raw accuracy, but MobileNet has roughly
half the AURC. At deployment threshold τ = 0.75, MobileNet abstains on
11.8% of inputs and reaches **97.94% selective accuracy** — exceeding the
SVM's raw accuracy ceiling.

## 🧪 Robustness experiment

We evaluate all three models under 4 perturbations × 5 severities:
Gaussian blur, brightness reduction, in-plane rotation, and random
rectangular occlusion. **Finding:** an asymmetric robustness pattern —
MobileNet dominates under photometric degradation (low light, blur,
occlusion), while the classical keypoint pipeline retains an advantage
under extreme geometric variation (45° rotation). Full numbers in the
report.

## 📄 Report

The full technical write-up (LaTeX + figures) is in `reports/`. Includes
the risk-coverage analysis, robustness experiment, error analysis with
five canonical failure modes, and discussion of deployment / ethics.

## ⚖️ Ethics

This system is a research prototype and **must not** be deployed as a
medical device or substitute for professional emergency response. The
selective-prediction design is intended to mitigate — but cannot
eliminate — the risk of misrecognition in safety-critical settings.
The HaGRID dataset has a primarily Eastern European subject pool and we
did not audit subgroup performance; the emergency vocabulary was curated
by the authors without participatory input from target communities.

## 🙏 Acknowledgements

- Dataset: **HaGRID** (Kapitanov et al., WACV 2024)
- Classification-formatted subset: `cj-mills/hagrid-classification-512p-no-gesture-150k-zip`
- Hand landmarks: **MediaPipe Hands** (Google)
- Backbone: **MobileNetV3-Small** (Howard et al., ICCV 2019)
