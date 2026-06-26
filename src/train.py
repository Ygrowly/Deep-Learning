import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, OUTPUT_DIR


def run_one_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def fit_model(model, train_loader, val_loader, num_epochs, lr=1e-3, save_path: Path | None = None, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            if save_path is not None:
                torch.save(best_state, save_path)

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={time.time() - start:.1f}s"
        )

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict_all(model, loader, device=DEVICE):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true), int(pred)] += 1
    return cm


def build_classification_report(y_true, y_pred, class_names):
    lines = []
    lines.append(f"{'class':24s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    lines.append("-" * 70)

    precisions, recalls, f1s, supports = [], [], [], []
    for class_id, name in enumerate(class_names):
        tp = int(((y_true == class_id) & (y_pred == class_id)).sum())
        fp = int(((y_true != class_id) & (y_pred == class_id)).sum())
        fn = int(((y_true == class_id) & (y_pred != class_id)).sum())
        support = int((y_true == class_id).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        lines.append(f"{name:24s} {precision:10.4f} {recall:10.4f} {f1:10.4f} {support:10d}")

    total = len(y_true)
    accuracy = float((y_true == y_pred).sum() / total) if total else 0.0
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))
    weighted_f1 = float(np.average(f1s, weights=supports)) if sum(supports) else 0.0

    lines.append("-" * 70)
    lines.append(f"{'accuracy':24s} {'':10s} {'':10s} {accuracy:10.4f} {total:10d}")
    lines.append(f"{'macro avg':24s} {macro_precision:10.4f} {macro_recall:10.4f} {macro_f1:10.4f} {total:10d}")
    lines.append(f"{'weighted avg':24s} {'':10s} {'':10s} {weighted_f1:10.4f} {total:10d}")
    return "\n".join(lines), accuracy


def evaluate_model(model, loader, class_names, device=DEVICE):
    y_true, y_pred, probs = predict_all(model, loader, device=device)
    report, accuracy = build_classification_report(y_true, y_pred, class_names)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "report": report,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }


def most_confused_pairs(cm, class_names, top_k=3):
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    return pairs[:top_k]


def save_reports(baseline_metrics, resnet_metrics, resnet_note, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(exist_ok=True)
    report_text = "\n".join(
        [
            "Baseline CNN 测试集分类报告",
            baseline_metrics["report"],
            "",
            "ResNet18 测试集分类报告",
            resnet_metrics["report"],
            "",
            "ResNet18 权重说明",
            resnet_note,
        ]
    )
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")

    summary = {
        "baseline_test_acc": baseline_metrics["accuracy"],
        "resnet18_test_acc": resnet_metrics["accuracy"],
        "resnet_note": resnet_note,
        "confusion_matrix": resnet_metrics["confusion_matrix"].tolist(),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_text, summary


def load_metrics_summary(output_dir: Path = OUTPUT_DIR):
    path = output_dir / "metrics_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

