我已经把 Kaggle 眼科疾病分类数据集下载并放到当前项目目录中了。

请你现在完成以下任务：

1. 在项目中查找数据集目录，确认图片所在路径和类别文件夹。
2. 如果数据集不是 ImageFolder 格式，请整理成：
   data/processed/
   ├── cataract/
   ├── diabetic_retinopathy/
   ├── glaucoma/
   └── normal/

3. 统计每个类别的图片数量，输出类别分布。
4. 基于 data/processed/ 编写一个完整 Jupyter Notebook：
   notebook 文件名：DL_眼科疾病分类课程设计.ipynb

5. Notebook 内容包括：
   - 课程设计题目
   - 项目背景
   - 数据集说明
   - 数据预处理
   - 类别分布可视化
   - 样本图片展示
   - train/val/test 划分
   - 数据增强
   - Baseline CNN 模型
   - ResNet18 迁移学习模型
   - 模型训练
   - loss 和 accuracy 曲线
   - 测试集分类报告
   - 混淆矩阵
   - 错误预测样本展示
   - 实验结果分析
   - 总结与展望

6. 模型设置：
   - 输入尺寸：224x224
   - batch_size：16；如果内存不够改 8
   - Baseline CNN：3 epoch
   - ResNet18：5 epoch
   - 优先使用 GPU，没有 GPU 就使用 CPU
   - 使用 torchvision.models.resnet18 预训练模型
   - 只训练最后分类层，必要时再解冻 layer4

7. 输出文件保存到 outputs/：
   - class_distribution.png
   - sample_images.png
   - baseline_curves.png
   - resnet18_curves.png
   - confusion_matrix.png
   - wrong_predictions.png
   - classification_report.txt
   - best_resnet18.pth

8. 最后生成一段实验结论文字，说明：
   - 数据集共有多少张图片
   - 分为哪几个类别
   - Baseline CNN 和 ResNet18 的测试准确率分别是多少
   - ResNet18 是否优于 Baseline
   - 哪些类别容易混淆
   - 本实验的不足和改进方向

注意：
- 不要做 Web 系统。
- 不要做后端接口。
- 不要接入原来的比赛项目。
- 目标是快速完成一个可提交的深度学习课程设计。