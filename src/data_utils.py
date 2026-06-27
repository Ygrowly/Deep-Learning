from pathlib import Path
import shutil

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from PIL import ImageOps

from .config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    NUM_WORKERS,
    RAW_DATA_DIR,
    SEED,
)


def count_images_by_class(root: Path = DATA_DIR):
    counts = {}
    if not root.exists():
        return counts
    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        counts[class_dir.name] = sum(
            1 for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def ensure_processed_data(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = DATA_DIR):
    processed_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in CLASS_NAMES if not (processed_dir / name).exists()]
    if missing:
        for class_name in CLASS_NAMES:
            src = raw_dir / class_name
            dst = processed_dir / class_name
            if not src.exists():
                raise FileNotFoundError(f"Missing raw class folder: {src}")
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return count_images_by_class(processed_dir)


def build_transforms(img_size: int = IMG_SIZE):
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_finetune_transforms(img_size: int = IMG_SIZE):
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


class FundusPreprocess:
    def __init__(self, crop_black=True, enhance_contrast=True, black_threshold=12, margin=8):
        self.crop_black = crop_black
        self.enhance_contrast = enhance_contrast
        self.black_threshold = black_threshold
        self.margin = margin

    def __call__(self, img):
        img = img.convert("RGB")
        if self.crop_black:
            arr = np.asarray(img)
            mask = arr.max(axis=2) > self.black_threshold
            if mask.any():
                ys, xs = np.where(mask)
                left = max(int(xs.min()) - self.margin, 0)
                upper = max(int(ys.min()) - self.margin, 0)
                right = min(int(xs.max()) + self.margin + 1, img.width)
                lower = min(int(ys.max()) + self.margin + 1, img.height)
                if right > left and lower > upper:
                    img = img.crop((left, upper, right, lower))

        if self.enhance_contrast:
            # Lightweight dependency-free contrast enhancement. This is intentionally
            # milder than full CLAHE to avoid amplifying artifacts in low-quality images.
            img = ImageOps.autocontrast(img, cutoff=1)
        return img


def build_preprocess_finetune_transforms(img_size: int = IMG_SIZE, enhance_contrast=True):
    train_transform = transforms.Compose(
        [
            FundusPreprocess(crop_black=True, enhance_contrast=enhance_contrast),
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            FundusPreprocess(crop_black=True, enhance_contrast=enhance_contrast),
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def stratified_split_indices(targets, train_ratio=0.70, val_ratio=0.15, seed: int = SEED):
    targets = np.array(targets)
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_id in sorted(set(targets.tolist())):
        idx = np.where(targets == class_id)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def split_distribution(indices, targets, class_names):
    counts = {name: 0 for name in class_names}
    for index in indices:
        counts[class_names[targets[index]]] += 1
    return counts


def create_dataloaders(data_dir: Path = DATA_DIR, batch_size: int = BATCH_SIZE):
    train_transform, eval_transform = build_transforms()
    base_dataset = datasets.ImageFolder(data_dir)
    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_dataset_full = datasets.ImageFolder(data_dir, transform=eval_transform)

    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split_indices(targets)

    train_ds = Subset(train_dataset_full, train_idx)
    val_ds = Subset(eval_dataset_full, val_idx)
    test_ds = Subset(eval_dataset_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    metadata = {
        "class_names": base_dataset.classes,
        "class_to_idx": base_dataset.class_to_idx,
        "targets": targets,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_distribution": split_distribution(train_idx, targets, base_dataset.classes),
        "val_distribution": split_distribution(val_idx, targets, base_dataset.classes),
        "test_distribution": split_distribution(test_idx, targets, base_dataset.classes),
    }
    return train_loader, val_loader, test_loader, metadata


def create_finetune_dataloaders(data_dir: Path = DATA_DIR, batch_size: int = BATCH_SIZE):
    train_transform, eval_transform = build_finetune_transforms()
    base_dataset = datasets.ImageFolder(data_dir)
    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_dataset_full = datasets.ImageFolder(data_dir, transform=eval_transform)

    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split_indices(targets)

    train_ds = Subset(train_dataset_full, train_idx)
    val_ds = Subset(eval_dataset_full, val_idx)
    test_ds = Subset(eval_dataset_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    metadata = {
        "class_names": base_dataset.classes,
        "class_to_idx": base_dataset.class_to_idx,
        "targets": targets,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_distribution": split_distribution(train_idx, targets, base_dataset.classes),
        "val_distribution": split_distribution(val_idx, targets, base_dataset.classes),
        "test_distribution": split_distribution(test_idx, targets, base_dataset.classes),
    }
    return train_loader, val_loader, test_loader, metadata


def create_preprocess_finetune_dataloaders(data_dir: Path = DATA_DIR, batch_size: int = BATCH_SIZE):
    train_transform, eval_transform = build_preprocess_finetune_transforms()
    base_dataset = datasets.ImageFolder(data_dir)
    train_dataset_full = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_dataset_full = datasets.ImageFolder(data_dir, transform=eval_transform)

    targets = np.array(base_dataset.targets)
    train_idx, val_idx, test_idx = stratified_split_indices(targets)

    train_ds = Subset(train_dataset_full, train_idx)
    val_ds = Subset(eval_dataset_full, val_idx)
    test_ds = Subset(eval_dataset_full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    metadata = {
        "class_names": base_dataset.classes,
        "class_to_idx": base_dataset.class_to_idx,
        "targets": targets,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "train_distribution": split_distribution(train_idx, targets, base_dataset.classes),
        "val_distribution": split_distribution(val_idx, targets, base_dataset.classes),
        "test_distribution": split_distribution(test_idx, targets, base_dataset.classes),
    }
    return train_loader, val_loader, test_loader, metadata
