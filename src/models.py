import torch.nn as nn
from torchvision import models

from .config import ALLOW_DOWNLOAD_WEIGHTS, DEVICE

try:
    from torchvision.models import ResNet18_Weights
except Exception:
    ResNet18_Weights = None


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_baseline_cnn(num_classes: int, device=DEVICE):
    return BaselineCNN(num_classes).to(device)


def build_resnet18(
    num_classes: int,
    device=DEVICE,
    allow_download: bool = ALLOW_DOWNLOAD_WEIGHTS,
    unfreeze_layer4: bool = False,
):
    weights = ResNet18_Weights.DEFAULT if ResNet18_Weights is not None else None
    note = ""

    try:
        if weights is not None:
            model = models.resnet18(weights=weights if allow_download else None)
            note = (
                "使用 torchvision.models.resnet18 的 ImageNet 预训练权重。"
                if allow_download
                else "当前为报告展示模式，未重新加载 ResNet18 预训练权重；已保存训练结果使用 ImageNet 预训练权重。"
            )
        else:
            model = models.resnet18(pretrained=allow_download)
            note = (
                "使用 torchvision.models.resnet18 的 ImageNet 预训练权重。"
                if allow_download
                else "当前为报告展示模式，未重新加载 ResNet18 预训练权重；已保存训练结果使用 ImageNet 预训练权重。"
            )
    except Exception as exc:
        model = models.resnet18(weights=None) if weights is not None else models.resnet18(pretrained=False)
        note = (
            "预训练权重加载失败，使用随机初始化 ResNet18。"
            f"原因：{exc}。若网络可连接 download.pytorch.org，重新运行 notebook 可自动下载权重。"
        )

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device), note
