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


def fit_resnet18_layer4_finetune(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    layer4_lr=3e-5,
    fc_lr=3e-4,
    weight_decay=1e-4,
    patience=3,
    save_path: Path | None = None,
    device=DEVICE,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        [
            {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": layer4_lr},
            {"params": [p for p in model.fc.parameters() if p.requires_grad], "lr": fc_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device=device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            bad_epochs += 1

        lrs = [group["lr"] for group in optimizer.param_groups]
        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr_layer4={lrs[0]:.2e} lr_fc={lrs[1]:.2e} "
            f"time={time.time() - start:.1f}s"
        )

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}; best_val_acc={best_acc:.4f}")
            break

    model.load_state_dict(best_state)
    return model, history


def fit_param_group_finetune(
    model,
    train_loader,
    val_loader,
    param_groups,
    num_epochs=10,
    weight_decay=1e-4,
    patience=3,
    save_path: Path | None = None,
    device=DEVICE,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device=device)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, device=device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
            if save_path is not None:
                torch.save(best_state, save_path)
        else:
            bad_epochs += 1

        lr_text = " ".join([f"lr{idx}={group['lr']:.2e}" for idx, group in enumerate(optimizer.param_groups)])
        print(
            f"Epoch {epoch:02d}/{num_epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"{lr_text} time={time.time() - start:.1f}s"
        )

        if bad_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}; best_val_acc={best_acc:.4f}")
            break

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
    metrics = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }
    return "\n".join(lines), accuracy, metrics


def evaluate_model(model, loader, class_names, device=DEVICE):
    y_true, y_pred, probs = predict_all(model, loader, device=device)
    report, accuracy, summary_metrics = build_classification_report(y_true, y_pred, class_names)
    cm = compute_confusion_matrix(y_true, y_pred, len(class_names))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "report": report,
        "accuracy": accuracy,
        "macro_f1": summary_metrics["macro_f1"],
        "summary_metrics": summary_metrics,
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
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "resnet18_test_acc": resnet_metrics["accuracy"],
        "resnet18_macro_f1": resnet_metrics["macro_f1"],
        "resnet_note": resnet_note,
        "confusion_matrix": resnet_metrics["confusion_matrix"].tolist(),
    }
    (output_dir / "metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_text, summary


def save_finetune_report(finetune_metrics, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(exist_ok=True)
    report_text = "\n".join(
        [
            "ResNet18 Fine-tune Layer4 测试集分类报告",
            finetune_metrics["report"],
        ]
    )
    (output_dir / "resnet18_finetune_classification_report.txt").write_text(report_text, encoding="utf-8")

    summary_path = output_dir / "metrics_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update(
        {
            "resnet18_finetune_test_acc": finetune_metrics["accuracy"],
            "resnet18_finetune_macro_f1": finetune_metrics["macro_f1"],
            "resnet18_finetune_confusion_matrix": finetune_metrics["confusion_matrix"].tolist(),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_text, summary


def save_named_experiment_report(
    metrics,
    report_title,
    report_filename,
    summary_prefix,
    output_dir: Path = OUTPUT_DIR,
):
    output_dir.mkdir(exist_ok=True)
    report_text = "\n".join([report_title, metrics["report"]])
    (output_dir / report_filename).write_text(report_text, encoding="utf-8")

    summary_path = output_dir / "metrics_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update(
        {
            f"{summary_prefix}_test_acc": metrics["accuracy"],
            f"{summary_prefix}_macro_f1": metrics["macro_f1"],
            f"{summary_prefix}_confusion_matrix": metrics["confusion_matrix"].tolist(),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_text, summary


def extract_accuracy_and_macro_f1(report_text, section_title):
    lines = report_text.splitlines()
    start = 0
    for idx, line in enumerate(lines):
        if section_title in line:
            start = idx
            break

    accuracy = None
    macro_f1 = None
    for line in lines[start:]:
        parts = line.split()
        if parts and parts[0] == "accuracy":
            accuracy = float(parts[-2])
        if len(parts) >= 6 and parts[0] == "macro" and parts[1] == "avg":
            macro_f1 = float(parts[4])
        if accuracy is not None and macro_f1 is not None:
            break
    return accuracy, macro_f1


def build_model_comparison_rows(output_dir: Path = OUTPUT_DIR):
    main_report = (output_dir / "classification_report.txt").read_text(encoding="utf-8")
    finetune_report = (output_dir / "resnet18_finetune_classification_report.txt").read_text(encoding="utf-8")

    baseline_acc, baseline_macro_f1 = extract_accuracy_and_macro_f1(main_report, "Baseline CNN")
    frozen_acc, frozen_macro_f1 = extract_accuracy_and_macro_f1(main_report, "ResNet18 测试集")
    finetune_acc, finetune_macro_f1 = extract_accuracy_and_macro_f1(finetune_report, "ResNet18 Fine-tune Layer4")

    return [
        {"model": "Baseline CNN", "accuracy": baseline_acc, "macro_f1": baseline_macro_f1},
        {"model": "ResNet18 frozen fc", "accuracy": frozen_acc, "macro_f1": frozen_macro_f1},
        {"model": "ResNet18 fine-tune layer4", "accuracy": finetune_acc, "macro_f1": finetune_macro_f1},
    ]


def build_final_comparison_rows(output_dir: Path = OUTPUT_DIR):
    rows = build_model_comparison_rows(output_dir)
    extra_reports = [
        (
            "ResNet18 layer4 + preprocessing",
            "resnet18_preprocess_finetune_classification_report.txt",
            "ResNet18 Layer4 + Preprocess",
        ),
        ("DenseNet121 fine-tune", "densenet121_finetune_classification_report.txt", "DenseNet121"),
        ("EfficientNet-B0 fine-tune", "efficientnet_b0_finetune_classification_report.txt", "EfficientNet-B0"),
    ]
    for model_name, filename, section in extra_reports:
        path = output_dir / filename
        if path.exists():
            report_text = path.read_text(encoding="utf-8")
            acc, macro_f1 = extract_accuracy_and_macro_f1(report_text, section)
            rows.append({"model": model_name, "accuracy": acc, "macro_f1": macro_f1})
    return rows


def load_metrics_summary(output_dir: Path = OUTPUT_DIR):
    path = output_dir / "metrics_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
