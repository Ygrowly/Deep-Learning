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


def build_finetune_conclusion(
    total_images,
    class_names,
    baseline_acc,
    frozen_acc,
    finetune_acc,
    finetune_confusion_matrix,
):
    cm = np.array(finetune_confusion_matrix)
    pairs = most_confused_pairs(cm, class_names)
    confused_text = "；".join([f"{true} 被预测为 {pred}（{count} 次）" for count, true, pred in pairs])
    if not confused_text:
        confused_text = "测试集中未出现明显混淆。"

    return f"""
本实验共使用 {total_images} 张眼科图像，分为 {', '.join(class_names)} 四个类别。

Baseline CNN 在测试集上的准确率为 {baseline_acc:.4f}；ResNet18 frozen 只训练 fc 分类头，测试准确率为 {frozen_acc:.4f}；ResNet18 Fine-tune Layer4 的测试准确率为 {finetune_acc:.4f}。

从模型结构看，frozen ResNet18 只训练最后分类头，主干特征仍主要来自 ImageNet 自然图像，对眼底医学图像的适配能力有限。解冻 layer4 后，模型的高层语义特征可以进一步适配眼底图像中的视盘、血管、出血点、渗出和整体颜色纹理等信息，因此通常能获得更好的分类能力。

根据 ResNet18 Fine-tune Layer4 的混淆矩阵，较容易混淆的类别包括：{confused_text}

青光眼、正常眼底、糖尿病视网膜病变之间仍容易混淆，原因可能是病灶区域较小，全图分类模型难以稳定聚焦视盘、血管、出血点等关键区域；同时不同图像存在光照、清晰度、视野范围和拍摄设备差异。

后续可加入眼底黑边裁剪、CLAHE、Grad-CAM 可解释性分析、EfficientNet 等更适合图像分类的模型来进一步提升效果。
""".strip()


def build_extended_conclusion(total_images, class_names, comparison_rows, summary):
    best = max(comparison_rows, key=lambda row: row["accuracy"] if row["accuracy"] is not None else -1)
    base = {row["model"]: row for row in comparison_rows}
    frozen = base.get("ResNet18 frozen fc")
    finetune = base.get("ResNet18 fine-tune layer4")
    preprocess = base.get("ResNet18 layer4 + preprocessing")
    densenet = base.get("DenseNet121 fine-tune")
    efficientnet = base.get("EfficientNet-B0 fine-tune")

    confusion = summary.get("resnet18_preprocess_finetune_confusion_matrix") or summary.get(
        "resnet18_finetune_confusion_matrix"
    )
    confused_text = "测试集中未出现明显混淆。"
    if confusion is not None:
        pairs = most_confused_pairs(np.array(confusion), class_names)
        if pairs:
            confused_text = "；".join([f"{true} 被预测为 {pred}（{count} 次）" for count, true, pred in pairs])

    preprocess_text = "未运行预处理增强实验。"
    if preprocess and finetune:
        delta = preprocess["accuracy"] - finetune["accuracy"]
        preprocess_text = (
            f"黑边裁剪和轻量对比度增强后的 ResNet18 layer4 微调准确率为 {preprocess['accuracy']:.4f}，"
            f"相比普通 layer4 微调变化 {delta:+.4f}。"
            + ("说明预处理带来了提升。" if delta > 0 else "说明当前预处理没有带来额外提升，可能与裁剪阈值、对比度增强方式或数据本身质量有关。")
        )

    dense_text = "未运行 DenseNet121 实验。"
    if densenet and finetune:
        dense_delta = densenet["accuracy"] - finetune["accuracy"]
        dense_text = f"DenseNet121 fine-tune 准确率为 {densenet['accuracy']:.4f}，相比 ResNet18 layer4 微调变化 {dense_delta:+.4f}。"

    eff_text = "未运行 EfficientNet-B0 实验。"
    if efficientnet and finetune:
        eff_delta = efficientnet["accuracy"] - finetune["accuracy"]
        eff_text = f"EfficientNet-B0 fine-tune 准确率为 {efficientnet['accuracy']:.4f}，相比 ResNet18 layer4 微调变化 {eff_delta:+.4f}。"

    return f"""
本实验共使用 {total_images} 张眼科图像，分为 {', '.join(class_names)} 四个类别。

综合所有实验，当前效果最好的模型是 {best['model']}，测试集 Accuracy 为 {best['accuracy']:.4f}，Macro F1 为 {best['macro_f1']:.4f}。

frozen ResNet18 只训练最后分类头，主干特征主要来自 ImageNet 自然图像，对眼底医学图像的适配能力有限；解冻 layer4 后，高层语义特征可以进一步适配视盘、血管、出血点、渗出和整体颜色纹理等眼底图像特征，因此相比 frozen fc 有明显提升。

{preprocess_text}

{dense_text}
{eff_text}

从混淆情况看，青光眼、正常眼底、糖尿病视网膜病变之间仍容易混淆，主要混淆包括：{confused_text}

这说明全图分类模型仍可能难以稳定聚焦视盘、血管、出血点等关键区域；图像中的黑边、光照、清晰度和视野范围差异也会影响分类。后续可继续优化黑边裁剪参数，尝试真正的 CLAHE、Grad-CAM 可解释性分析，以及更强的 EfficientNet / ConvNeXt / DenseNet 配置。
""".strip()


def save_conclusion(conclusion, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(exist_ok=True)
    (output_dir / "experiment_conclusion.txt").write_text(conclusion, encoding="utf-8")


def load_summary_or_default(output_dir: Path = OUTPUT_DIR):
    path = output_dir / "metrics_summary.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None
