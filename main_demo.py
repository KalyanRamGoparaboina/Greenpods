import argparse
import sys
import os
import random
import time
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    missing = []
    for pkg in ["cv2", "torch", "sklearn", "skimage", "PIL", "tqdm", "joblib"]:
        try:
            __import__(pkg if pkg != "cv2" else "cv2")
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing libraries: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)


def load_models(models_dir: str):
    from src.classical_model import load_model
    from src.dl_model import load_dl_model, get_device

    models_dir = Path(models_dir)
    class_names = ["underripe", "breaking", "ripe", "overripe"]
    device = get_device()

    svm_model, rf_model, dl_model = None, None, None

    svm_path = models_dir / "svm_model.joblib"
    rf_path = models_dir / "rf_model.joblib"
    dl_path = models_dir / "mobilenetv2_best.pth"

    if svm_path.exists():
        svm_model = load_model(str(svm_path))

    if rf_path.exists():
        rf_model = load_model(str(rf_path))

    if dl_path.exists():
        dl_model = load_dl_model(str(dl_path), num_classes=len(class_names), device=device)

    return svm_model, rf_model, dl_model, class_names, device


def predict_all(image_path: str, svm_model, rf_model, dl_model, class_names, device):
    results = {}

    if svm_model is not None:
        from src.classical_model import predict_single_image
        t0 = time.perf_counter()
        svm_class, svm_conf = predict_single_image(svm_model, image_path, class_names)
        svm_ms = (time.perf_counter() - t0) * 1000
        results["SVM"] = {"class": svm_class, "confidence": svm_conf, "ms": svm_ms}

    if rf_model is not None:
        from src.classical_model import predict_single_image
        t0 = time.perf_counter()
        rf_class, rf_conf = predict_single_image(rf_model, image_path, class_names)
        rf_ms = (time.perf_counter() - t0) * 1000
        results["Random Forest"] = {"class": rf_class, "confidence": rf_conf, "ms": rf_ms}

    if dl_model is not None:
        from src.dl_model import predict_single_image_dl
        t0 = time.perf_counter()
        dl_class, dl_conf, dl_probs = predict_single_image_dl(dl_model, image_path, class_names, device=device)
        dl_ms = (time.perf_counter() - t0) * 1000
        results["MobileNetV2"] = {
            "class": dl_class, "confidence": dl_conf,
            "probs": dl_probs, "ms": dl_ms
        }

    return results


def print_results(image_path: str, results: dict, class_names: list):
    width = 65
    print("\n" + "=" * width)
    print(f"Prediction Results: {image_path}")
    print("=" * width)
    for model_name, res in results.items():
        print(f"{model_name:<16} -> {res['class'].upper():<12} (confidence: {res['confidence']:.1%}, latency: {res['ms']:.1f}ms)")
    print("=" * width)

    if "MobileNetV2" in results and "probs" in results["MobileNetV2"]:
        print("\nMobileNetV2 Class Probabilities:")
        dl_class = results["MobileNetV2"]["class"]
        for cls, prob in zip(class_names, results["MobileNetV2"]["probs"]):
            bar = "#" * int(prob * 30)
            marker = " *" if cls == dl_class else ""
            print(f"  {cls:<12}: {bar:<30} {prob:.1%}{marker}")
    print()


def visualise_results(image_path: str, results: dict, class_names: list, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(15, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(img_rgb)
    ax_img.set_title("Input Image")
    ax_img.axis("off")

    ax_cls = fig.add_subplot(gs[1])
    cls_models = [k for k in results if k != "MobileNetV2"]
    if cls_models:
        cls_confs = [results[k]["confidence"] for k in cls_models]
        cls_labels = [f"{results[k]['class'].capitalize()}" for k in cls_models]
        colors_cls = ["#4CAF50" if c > 0.7 else "#FF9800" if c > 0.5 else "#F44336" for c in cls_confs]
        bars = ax_cls.barh(cls_models, cls_confs, color=colors_cls, height=0.5)
        for bar, label, conf in zip(bars, cls_labels, cls_confs):
            ax_cls.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{label} ({conf:.1%})", va="center", fontsize=9)
        ax_cls.set_xlim(0, 1.3)
        ax_cls.set_title("Classical ML Predictions")
        ax_cls.set_xlabel("Confidence")
        ax_cls.grid(axis="x", alpha=0.3)

    if "MobileNetV2" in results and "probs" in results["MobileNetV2"]:
        ax_dl = fig.add_subplot(gs[2])
        dl_class = results["MobileNetV2"]["class"]
        dl_probs = results["MobileNetV2"]["probs"]
        bar_colors = ["#4CAF50" if cls == dl_class else "#B0BEC5" for cls in class_names]
        bars_dl = ax_dl.barh(class_names, dl_probs, color=bar_colors, height=0.5)
        for bar, prob in zip(bars_dl, dl_probs):
            ax_dl.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1%}", va="center", fontsize=9)
        ax_dl.set_xlim(0, 1.3)
        ax_dl.set_title(f"MobileNetV2 -> {dl_class.upper()} ({results['MobileNetV2']['confidence']:.1%})")
        ax_dl.set_xlabel("Probability")
        ax_dl.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def get_random_test_image(data_dir: str) -> str:
    data_dir = Path(data_dir)
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = []
    for cls_dir in data_dir.iterdir():
        if cls_dir.is_dir():
            for f in cls_dir.iterdir():
                if f.suffix.lower() in valid_exts:
                    all_images.append(str(f))
    if not all_images:
        raise FileNotFoundError(f"No images found in {data_dir}")
    return random.choice(all_images)


def parse_args():
    parser = argparse.ArgumentParser(description="Avocado Ripening Stage Classification Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file")
    parser.add_argument("--demo", action="store_true", help="Auto-pick random image from data/raw/")
    parser.add_argument("--models-dir", type=str, default="models/", help="Saved models directory")
    parser.add_argument("--data-dir", type=str, default="data/raw/", help="Dataset directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip visualization plot")
    parser.add_argument("--save-plot", type=str, default=None, help="Save visualization to file")
    return parser.parse_args()


def main():
    args = parse_args()
    check_dependencies()

    if args.demo:
        image_path = get_random_test_image(args.data_dir)
    elif args.image:
        image_path = args.image
        if not Path(image_path).exists():
            print(f"Error: image not found at {image_path}")
            sys.exit(1)
    else:
        print("Please provide --image <path> or use --demo")
        sys.exit(1)

    svm_model, rf_model, dl_model, class_names, device = load_models(args.models_dir)

    if not any([svm_model, rf_model, dl_model]):
        print(f"No trained model checkpoints found in {args.models_dir}")
        sys.exit(1)

    results = predict_all(image_path, svm_model, rf_model, dl_model, class_names, device)
    print_results(image_path, results, class_names)

    if not args.no_plot:
        visualise_results(image_path, results, class_names, save_path=args.save_plot)


if __name__ == "__main__":
    main()
