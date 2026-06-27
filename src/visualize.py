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


def _short_model_name(name):
    mapping = {
        "Baseline CNN": "Baseline",
        "ResNet18 frozen fc": "ResNet18 frozen",
        "ResNet18 fine-tune layer4": "ResNet18 layer4",
        "ResNet18 layer4 + preprocessing": "ResNet18 prep",
        "DenseNet121 fine-tune": "DenseNet121",
        "EfficientNet-B0 fine-tune": "EfficientNet-B0",
    }
    return mapping.get(name, name)


def plot_overall_metric_comparison(rows, save_path: Path | None = None):
    names = [_short_model_name(row["model"]) for row in rows]
    acc = [row["accuracy"] for row in rows]
    f1 = [row["macro_f1"] for row in rows]

    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    acc_bars = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4C78A8")
    f1_bars = ax.bar(x + width / 2, f1, width, label="Macro F1", color="#F58518")

    ax.set_title("Overall Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for bars in (acc_bars, f1_bars):
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_metric_delta_vs_reference(rows, reference_model="ResNet18 fine-tune layer4", save_path: Path | None = None):
    reference = next(row for row in rows if row["model"] == reference_model)
    names = [_short_model_name(row["model"]) for row in rows if row["model"] != reference_model]
    acc_delta = [row["accuracy"] - reference["accuracy"] for row in rows if row["model"] != reference_model]
    f1_delta = [row["macro_f1"] - reference["macro_f1"] for row in rows if row["model"] != reference_model]

    y = np.arange(len(names))
    height = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(y - height / 2, acc_delta, height, label="Accuracy Δ", color="#54A24B")
    ax.barh(y + height / 2, f1_delta, height, label="Macro F1 Δ", color="#B279A2")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Metric Difference Compared with ResNet18 Layer4")
    ax.set_xlabel("Score Difference")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def _parse_class_f1(report_text, class_names, section_title=None):
    segment = report_text
    if section_title and section_title in report_text:
        segment = report_text[report_text.index(section_title) :]

    scores = {}
    for line in segment.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] in class_names:
            scores[parts[0]] = float(parts[3])
            if len(scores) == len(class_names):
                break
    return scores


def load_class_f1_table(output_dir: Path = OUTPUT_DIR, class_names=None):
    class_names = class_names or CLASS_NAMES
    report_specs = [
        ("Baseline CNN", "classification_report.txt", "Baseline CNN 测试集分类报告"),
        ("ResNet18 frozen fc", "classification_report.txt", "ResNet18 测试集分类报告"),
        ("ResNet18 fine-tune layer4", "resnet18_finetune_classification_report.txt", None),
        ("ResNet18 layer4 + preprocessing", "resnet18_preprocess_finetune_classification_report.txt", None),
        ("DenseNet121 fine-tune", "densenet121_finetune_classification_report.txt", None),
        ("EfficientNet-B0 fine-tune", "efficientnet_b0_finetune_classification_report.txt", None),
    ]

    rows = []
    for model_name, filename, section_title in report_specs:
        path = output_dir / filename
        if not path.exists():
            continue
        report_text = path.read_text(encoding="utf-8")
        scores = _parse_class_f1(report_text, class_names, section_title=section_title)
        if scores:
            rows.append({"model": model_name, **scores})
    return rows


def plot_class_f1_heatmap(class_f1_rows, class_names=None, save_path: Path | None = None):
    class_names = class_names or CLASS_NAMES
    model_names = [_short_model_name(row["model"]) for row in class_f1_rows]
    matrix = np.array([[row[class_name] for class_name in class_names] for row in class_f1_rows])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.35, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, label="F1-score")
    ax.set_title("Class-wise F1-score Heatmap")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value > 0.78 else "black", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()
