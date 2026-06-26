import json
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR
from .train import most_confused_pairs


def build_conclusion(total_images, class_names, baseline_acc, resnet_acc, confusion_matrix):
    cm = np.array(confusion_matrix)
    pairs = most_confused_pairs(cm, class_names)
    confused_text = "；".join([f"{true} 被预测为 {pred}（{count} 次）" for count, true, pred in pairs])
    if not confused_text:
        confused_text = "测试集中未出现明显混淆。"

    winner = "ResNet18 优于 Baseline CNN" if resnet_acc >= baseline_acc else "Baseline CNN 暂时优于 ResNet18"
    return f"""
本实验共使用 {total_images} 张眼科图像，分为 {', '.join(class_names)} 四个类别。

Baseline CNN 在测试集上的准确率为 {baseline_acc:.4f}，ResNet18 在测试集上的准确率为 {resnet_acc:.4f}。
从测试结果看，{winner}。

根据 ResNet18 混淆矩阵，较容易混淆的类别包括：{confused_text}

本实验的不足包括：训练轮数较少；眼底图像可能存在拍摄设备、光照、视野范围和图像质量差异；当前模型只做图像级分类，没有定位病灶区域。若更换环境运行时无法读取 ResNet18 预训练权重，模型效果会明显下降。
后续可改进方向包括：增加训练轮数和学习率调度，解冻 ResNet18 的 layer4 进行微调，尝试 EfficientNet、DenseNet 等更强模型，并结合 Grad-CAM 做可解释性分析。
""".strip()


def save_conclusion(conclusion, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(exist_ok=True)
    (output_dir / "experiment_conclusion.txt").write_text(conclusion, encoding="utf-8")


def load_summary_or_default(output_dir: Path = OUTPUT_DIR):
    path = output_dir / "metrics_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
