# Silent-SOS: Reliable Static Gesture Recognition for Emergency Communication

> A computer-vision system for silent emergency communication, with built-in
> **selective prediction** to abstain when uncertain — designed for high-stakes
> accessibility scenarios.

## 🎯 What it does
Recognizes 9 static hand gestures (plus a no-gesture class) from a webcam and
translates them into spoken emergency messages. When the model is not confident,
it abstains rather than guessing — critical for safety-sensitive use.

## 🏗️ Project Structure
```
.
├── app/                  # Gradio Blocks UI + inference glue
├── scripts/              # Data, training, evaluation pipelines
├── models/               # Three model implementations + checkpoints
│   ├── naive_baseline.py
│   ├── classical_ml.py   # MediaPipe keypoints + SVM
│   └── deep_learning.py  # MobileNetV3 + selective prediction
├── data/                 # raw/, processed/, outputs/
├── notebooks/            # Exploration only — not graded
├── reports/              # Final report + figures
└── main.py               # Launch the app
```

## 🚀 Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download a subset of HaGRID (only 9 classes, ~1-2 GB)
python scripts/download_data.py

# 3. Train all three models
python scripts/train_pipeline.py

# 4. Run the app locally
python main.py
```

Live demo: https://huggingface.co/spaces/<your-username>/silent-sos

## 📊 Models Implemented
| Model | Approach | Where |
|---|---|---|
| Naive Baseline | Majority-class predictor | `models/naive_baseline.py` |
| Classical ML | MediaPipe 21 keypoints → SVM | `models/classical_ml.py` |
| Deep Learning | MobileNetV3-Small fine-tuned + selective prediction | `models/deep_learning.py` |

## 🧪 Experiment
We evaluate **risk-coverage trade-offs**: as we tighten the rejection
threshold, coverage drops but accuracy on accepted predictions rises.
See `reports/` for the full Risk-Coverage curve and robustness analysis
under blur / low-light / occlusion perturbations.

## 📄 Report
See `reports/silent_sos_report.pdf` for the full technical write-up.

## ⚖️ Ethics
This system is a research prototype. It is NOT a substitute for human
emergency response. The selective-prediction design is intended to
mitigate — but cannot eliminate — the risk of mis-recognition in
safety-critical settings. See the Ethics Statement in the report.

## 🙏 Acknowledgements
Dataset: HaGRID (Kapitanov et al., WACV 2024).
