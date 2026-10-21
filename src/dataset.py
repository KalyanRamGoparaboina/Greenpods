import os
import random
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import local_binary_pattern

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES: List[str] = ["underripe", "breaking", "ripe", "overripe"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_SIZE: int = 224
CLASSICAL_SIZE: int = 128
SEED: int = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image_paths_and_labels(
    data_dir: str,
    class_names: Optional[List[str]] = None,
    valid_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> Tuple[List[str], List[int], List[str]]:
    data_dir = Path(data_dir)
    if class_names is None:
        class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])

    c2i = {c: i for i, c in enumerate(class_names)}
    paths, labels = [], []

    for cls in class_names:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in valid_exts:
                paths.append(str(f))
                labels.append(c2i[cls])

    return paths, labels, class_names


def stratified_split(
    paths: List[str],
    labels: List[int],
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = SEED,
) -> Tuple[List, List, List, List, List, List]:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=relative_val,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_transforms(mode: str = "train", img_size: int = IMG_SIZE) -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


class AvocadoDataset(Dataset):
    def __init__(
        self,
        paths: List[str],
        labels: List[int],
        transform: Optional[transforms.Compose] = None,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform or get_transforms("val")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        img_tensor = self.transform(img)
        return img_tensor, label


def extract_features_single(
    image_path: str,
    img_size: int = CLASSICAL_SIZE,
    n_bins_color: int = 16,
    lbp_points: int = 8,
    lbp_radius: int = 1,
) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        total = 5 * n_bins_color + (lbp_points + 2)
        return np.zeros(total, dtype=np.float32)

    img_bgr = cv2.resize(img_bgr, (img_size, img_size))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hist_l, _ = np.histogram(lab[:, :, 0], bins=n_bins_color, range=(0, 256))
    hist_a, _ = np.histogram(lab[:, :, 1], bins=n_bins_color, range=(0, 256))
    hist_b, _ = np.histogram(lab[:, :, 2], bins=n_bins_color, range=(0, 256))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist_h, _ = np.histogram(hsv[:, :, 0], bins=n_bins_color, range=(0, 180))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=n_bins_color, range=(0, 256))

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=lbp_points, R=lbp_radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=lbp_points + 2, range=(0, lbp_points + 2))

    feature_vec = np.concatenate([hist_l, hist_a, hist_b, hist_h, hist_s, lbp_hist]).astype(np.float32)
    norm = feature_vec.sum()
    if norm > 0:
        feature_vec /= norm

    return feature_vec


def build_feature_matrix(
    paths: List[str],
    labels: List[int],
    verbose: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Extracting features", unit="img") if verbose else paths
    except ImportError:
        iterator = paths

    features = [extract_features_single(p, **kwargs) for p in iterator]
    X = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    return X, y


def download_dataset(
    file_id: str = "1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET",
    dest_dir: str = "data/raw",
    zip_name: str = "avocado_dataset.zip",
) -> None:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir.parent / zip_name

    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(zip_path), quiet=False)
    except ImportError:
        raise ImportError("Install gdown: pip install gdown")

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        zip_path.unlink()
