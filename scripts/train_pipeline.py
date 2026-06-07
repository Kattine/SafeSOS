"""Run model training in sequence: naive, classical, then deep learning."""

from scripts import train_classical, train_dl


def main() -> None:
    print("[1/3] Naive baseline: no training step (computed at evaluation).")
    print("[2/3] Classical ML (SVM on MediaPipe keypoints)")
    train_classical.main()
    print("[3/3] Deep learning (MobileNetV3)")
    train_dl.main()


if __name__ == "__main__":
    main()
