"""Train all three models in sequence: naive -> classical -> deep learning."""

from scripts import train_classical, train_dl


def main() -> None:
    print("[1/3] Naive baseline — nothing to train (computed at eval time).")
    print("[2/3] Classical ML (SVM on MediaPipe keypoints)")
    train_classical.main()
    print("[3/3] Deep learning (MobileNetV3)")
    train_dl.main()


if __name__ == "__main__":
    main()
