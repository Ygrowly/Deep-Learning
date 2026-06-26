import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import CLASS_NAMES, DATA_DIR, IMAGE_EXTENSIONS, OUTPUT_DIR


def plot_class_distribution(class_counts, save_path: Path | None = None):
    names = list(class_counts.keys())
    values = [class_counts[name] for name in names]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=20)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 10, str(value), ha="center", va="bottom")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def show_sample_images(data_dir: Path = DATA_DIR, class_names=None, n_per_class=4, save_path: Path | None = None):
    class_names = class_names or CLASS_NAMES
    fig, axes = plt.subplots(len(class_names), n_per_class, figsize=(3 * n_per_class, 2.5 * len(class_names)))
    axes = np.array(axes).reshape(len(class_names), n_per_class)

    for row, class_name in enumerate(class_names):
        paths = sorted([p for p in (data_dir / class_name).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
        chosen = random.sample(paths, min(n_per_class, len(paths)))
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                img = Image.open(chosen[col]).convert("RGB")
                ax.imshow(img)
                ax.set_title(class_name)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_training_history(history, title, save_path: Path | None = None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="val")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="train")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="val")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path: Path | None = None):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > threshold else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def show_wrong_predictions(model, loader, class_names, device, save_path: Path | None = None, max_items=12):
    wrong_items = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1).cpu()
            preds = outputs.argmax(dim=1).cpu()
            for img, label, pred, prob in zip(images, labels, preds, probs):
                if int(label) != int(pred):
                    wrong_items.append((img, int(label), int(pred), float(prob[pred])))
                if len(wrong_items) >= max_items:
                    break
            if len(wrong_items) >= max_items:
                break

    if wrong_items:
        cols = 4
        rows = math.ceil(len(wrong_items) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, rows * 3.2))
        axes = np.array(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, (img, label, pred, conf) in zip(axes, wrong_items):
            ax.imshow(denormalize(img).permute(1, 2, 0))
            ax.set_title(f"T: {class_names[label]}\nP: {class_names[pred]} ({conf:.2f})", fontsize=9)
            ax.axis("off")
    else:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No wrong predictions on test set.", ha="center", va="center")
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def show_saved_image(filename, title=None, output_dir: Path = OUTPUT_DIR):
    path = output_dir / filename
    img = Image.open(path)
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

