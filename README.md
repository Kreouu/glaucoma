# 青光眼图像二分类

当前实现：数据审计与最小图像训练基线。训练、验证使用官方划分；不使用年龄，不读取测试集，不自动下载预训练权重。

## 完整一轮训练（带进度）

在 WSL 项目目录执行 `bash scripts/run_baseline.sh`，自动使用新的带时间戳输出目录及日志。默认 batch 16、workers 4，启用 `--jpeg-draft`。训练与验证在首批、每 25 批及结束时打印百分比、平均 loss、图片/秒和当前阶段 ETA；最初 ETA 包含启动开销，不能作为最终耗时。

当前一轮基线已在完整验证集复现通过，见 [实验记录](docs/基线实验记录.md)。保持相同配置从预训练权重重新训练 5 轮：

```bash
bash scripts/run_baseline.sh 5
```

默认参数仍为 1 轮。脚本不续接旧 checkpoint，每轮以完整验证集 AUROC 选择最佳模型。完成后可执行 `python validate.py --run-dir outputs/对应实验目录` 重新加载最佳模型并检查数据清单、checkpoint 哈希、每图概率和验证指标；验证时须使用该实验对应的 `dataset.py` 和 `model.py`。

`--jpeg-draft` 在大 JPEG 转 RGB 前使用 Pillow 缩小解码，减少 CPU 解码、缩放和临时内存开销；PNG 等格式维持原路径。该选项会改变像素值，不是与完整解码逐像素等价的优化，因此训练、验证和最终推理必须采用一致设置，并依据完整验证指标评估效果。配置和 checkpoint 均保存该开关。直接调用 `train.py` 时默认仍使用完整解码。

当前已运行的进程不会因更新文件自动获得进度或新的解码模式。若要切换，在原训练终端 Ctrl+C 后重新执行脚本；旧目录保留，新运行从预训练权重重新开始，不会接续未保存的轮内进度。

## 环境

在 WSL 使用 `glaucoma` Conda 环境。当前已验收 Python 3.12、PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128；实际依赖见 `glaucoma-environment-freeze.txt`。代码还使用 NumPy、Pillow 和 scikit-learn。

## 数据

当前 WSL 数据根目录：

```text
/home/ng/workspace/glaucoma/datasets/glaucoma_first/
├── train.csv
├── valid.csv
├── train/
└── valid/
```

CSV 使用 `.jpg`、实际图片为 `.JPG`，读取器在唯一匹配时处理大小写差异，不重命名原始文件。数据审计用法见 [数据审计](docs/数据审计.md)。

## 少量样本验收

在项目根目录执行：

```bash
conda activate glaucoma
python train.py \
  --data-root datasets/glaucoma_first \
  --output-dir outputs/smoke_001 \
  --epochs 2 --batch-size 8 --workers 4 \
  --train-limit 128 --valid-limit 64
```

限量模式按类别比例抽样，保证包含正负样本，仅检查工程流程。其 AUROC/AUPRC 不作为正式效果依据。未指定预训练权重时为随机初始化。

## 正式基线

本次已整理 torchvision EfficientNet-B0 权重的来源、具体上游 Apache-2.0 模型卡及张量对应证据，见 [权重核对说明](licenses/efficientnet_b0/核对说明.md)。`--pretrained-path` 接受 torchvision 原始 EfficientNet-B0 的 1000 类 ImageNet state_dict；不要传入医学预训练权重或本项目的训练 checkpoint。

```bash
python train.py \
  --data-root datasets/glaucoma_first \
  --output-dir outputs/baseline_001 \
  --pretrained-path pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth \
  --epochs 5 --batch-size 16 --workers 4
```

原始预训练权重已下载并通过 GPU 加载检查；完整数据预训练基线尚未启动。正式效果必须在完整验证集评估，不能带 `--train-limit` 或 `--valid-limit`。已有输出目录会拒绝覆盖，请每次使用新的实验名。当前没有断点续训、早停、测试集推理或官方计时实现。

训练采用 EfficientNet-B0 单 logit、普通 BCE、AdamW、AMP，默认输入 256×256。图片转 RGB、保持宽高比缩放后黑色补边、采用固定 ImageNet 归一化；训练仅增加水平翻转。验证不做随机增强，不更新梯度或 BN。固定种子便于对照，但不承诺跨硬件或软件版本逐位一致。

## 实验产物

每个输出目录包含：

- `config.json`：参数、软件版本、GPU、数据与源代码哈希、Git 状态及原始预训练权重哈希。
- `train_manifest.csv`、`valid_manifest.csv`：本次实际使用样本及文件映射。
- `metrics.jsonl`：逐轮 loss、验证 AUROC/AUPRC、耗时与峰值显存。
- `best.pt`：验证 AUROC 最高轮的模型、优化器和 AMP scaler 状态。
- `best_valid_predictions.csv`：该 checkpoint 的每图验证概率。
- `best.sha256`：最佳 checkpoint 哈希。

`best.pt` 可用 `torch.load(path, map_location='cpu', weights_only=True)` 加载，再将其中的 `model` 传入 `build_model().load_state_dict(...)`；推理使用 `eval()`、相同预处理及 AMP 精度。

数据、权重、运行日志及逐图审计报告留在已忽略目录，不能公开上传比赛数据。训练脚本及说明尚需提交 Git 后再按正常流程 push/pull 同步。
