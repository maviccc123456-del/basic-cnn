# 基础 CNN：钢材表面缺陷分类

使用 PyTorch 逐层搭建的三层卷积神经网络，对 NEU-CLS 钢材表面图像进行六分类。模型从随机权重开始训练。

## 网络结构

输入为单通道灰度图，原始数据尺寸为 200 × 200。

| 阶段 | 操作 | 输出尺寸（省略 batch 维） |
| --- | --- | --- |
| 输入 | 灰度图转为张量 | 1 × 200 × 200 |
| 第一组 | 3×3 卷积 + ReLU + 2×2 最大池化 | 32 × 100 × 100 |
| 第二组 | 3×3 卷积 + ReLU + 2×2 最大池化 | 64 × 50 × 50 |
| 第三组 | 3×3 卷积 + ReLU + 2×2 最大池化 | 128 × 25 × 25 |
| 全局平均池化 | AdaptiveAvgPool2d(1) | 128 × 1 × 1 |
| 分类层 | 展平 + 全连接层 | 6 个类别的 logits |

模型定义见 `models/simple_cnn.py`。本仓库只包含基础 CNN，不包含 BatchNorm 增强版本或 ResNet18。

## 安装

建议使用 Python 3.10，在本仓库目录打开终端：

```console
python -m venv .venv
python -m pip install -r requirements.txt
```

安装依赖前请先激活虚拟环境：Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1`，macOS/Linux 使用 `source .venv/bin/activate`。

## 准备数据

自行取得 NEU-CLS 数据集，将 1800 张原始 BMP 图片直接放到 `data/NEU-CLS/` 下，不要按类别再分子文件夹。例如：

```text
data/
  NEU-CLS/
    Cr_1.bmp
    In_1.bmp
    Pa_1.bmp
    PS_1.bmp
    RS_1.bmp
    Sc_1.bmp
    ...
  splits/
    NEU-CLS_deduplicated_v1.json
```

类别由文件名前缀读取：Cr（裂纹）、In（夹杂）、Pa（斑块）、PS（麻点表面）、RS（轧入氧化皮）、Sc（划痕）。原始图片和训练权重不随仓库分发。

仓库保留原实验的数据划分清单，加载时会核验图片内容。原实验的 1800 张图片中有一组像素完全相同的重复图，去重后为 1799 张：训练集 1259 张、验证集 270 张、测试集 270 张。划分采用随机种子 42，按类别分层，比例约为 70% / 15% / 15%。

如果你的数据与清单不一致，程序会报错。请先检查是否使用了相同原始数据；确需使用不同版本时，备份并移走旧清单，再重新训练。不同划分的权重与结果不能混用。像素去重不保证排除裁剪后的近似图片或同一物理样本的不同照片。

## 训练和评估

```console
python data_split.py
python train.py
python evaluate.py
```

默认训练 10 轮，batch size 为 16，优化器为 Adam，学习率为 0.001，损失函数为交叉熵，训练随机种子为 42。程序自动选择可用的 CUDA 或 CPU。基础版仅将灰度图片转为张量，不进行数据增强。

只按验证集准确率选择最佳模型，同分保留较早轮次，训练过程中不使用测试集选模。评估加载最近完成的训练，并校验权重与数据划分是否匹配。

结果存放在 `outputs/leak_free/simple_cnn/run_001/` 等目录，每次训练创建新目录，包含训练记录、最佳权重、数据划分快照；评估后生成分类报告、逐图预测及混淆矩阵。

原项目已记录的基础 CNN 测试结果为 **226 / 270，准确率 83.70%**（10 轮训练，最佳验证轮次为第 10 轮）。这是原实验记录，不是本仓库在所有设备和依赖组合上的保证；本次整理没有重新训练模型。

## 检查

```console
python models/simple_cnn.py
python -m unittest discover -s . -p test_data_split.py -v
```

第一条检查输入、输出尺寸；第二条检查像素去重、划分持久化、数据变动检测以及权重与划分的匹配。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `models/simple_cnn.py` | 基础 CNN 网络结构 |
| `data_split.py` | 图片扫描、像素去重和持久化划分 |
| `dataset.py` | 灰度图片加载和张量转换 |
| `experiments.py` | 训练、验证、模型保存和测试评估 |
| `train.py` / `evaluate.py` | 训练与评估入口 |
| `test_data_split.py` | 数据与权重检查 |

“自行搭建”指使用 PyTorch 基础层组合网络结构，不代表提出了新的原创网络架构。
