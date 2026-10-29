# 本轮选定模型：ConvNeXt-Tiny（2026-10-29）

> 2026-10-29 17:45:30更新：512新提交24639已评分98.08，成为当前最高分候选。本文仍保留为384/97.13版本的复现档案，请勿把下文384权重或材料包当作新的512版本；512路径和哈希以主记录末节为准。


这是本轮有限搜索后的交接说明。主要进度和历史仍统一记录在 [基线实验记录](基线实验记录.md)。原 [384 B0复现指南](384基线复现与协作指南.md) 保留用于复现旧对照。

## 选定结果与边界

选定 `convnext_tiny.fb_in1k`：384输入、FP16、AdamW、恒定学习率3e-5、训练正类权重3、5轮。三种子验证AUROC均值0.984241066，样本标准差0.002359886；B0为0.982407186 ± 0.000637991。选择的是该家族seed43第4轮，单模型、无TTA、无年龄输入，验证AUROC 0.986849922、AP 0.788000929。

seed43参与超参数选择，三种子均值仍有选择偏差。独立于初筛的seed42/44对B0仅平均提升约0.000737，且ConvNeXt波动更大；这是当前有限范围内的实用选择，不代表架构上限已被证明。

导出权重111,341,599 bytes（约111 MB），只保留model/config/metrics，去掉优化器和scaler。原训练checkpoint保持不变；独立审计已逐项验证导出参数与原参数完全相同。导出文件哈希因序列化和内容精简而不同于原训练文件。

本机RTX5060Ti、100张验证图/5张预热/3轮、batch1端到端均时16.4048ms；概率、标签、排序一致，最大概率差0。**尚未在官方T4与指定图片上核验。** 用户已于2026-10-29 11:43:39提交本轮CSV，ID24294，平台“已正确评分”97.13分；较最早256普通BCE提交96.73提高0.40分。平台评分与正式模型核验分开记录。

## Mac与WSL文件位置

两端项目根目录下的 `outputs/model_selection_20261029_v1/`：

- `final/submission.csv`：已生成的5855行新预测，用户已通过平台CSV入口提交此文件并反馈97.13分；助手没有自动上传。
- `final/best.pt`：选定的推理权重。
- `final/config.json`、`best.sha256`、`reload_check.json`、`submission.manifest.json`：配置、哈希、完整验证和预测来源证据。
- `final/source/`、`final/licenses/`、`final/environment-freeze.txt`：冻结源码、许可来源和本次实际环境快照。这个快照在实验产物内部，不是第三个conda环境。
- `selection.json`、`experiments.json`、`groups.json`、`plan*.json`、`independent_audit.json`：完整决策与审计记录。
- `records/`：24项尝试的小型原始记录，包括配置、指标、验证预测、样本清单、重载报告或失败日志。其他完整训练checkpoint仍留在WSL原目录，Mac没有声称全部备份。

Mac根目录为 `/Users/Ng/workspace/glaucoma`，WSL为 `/home/ng/workspace/glaucoma`。Mac另有 `outputs/glaucoma-selection-final-20261029.tar`，包含最终模型和上述实验记录，331个文件已逐项核对大小与SHA256。它是队内交接归档，不是已确认符合主办方结构的模型上传包。

文件校验：

```text
final/best.pt
cc3a5d9bb169a0da8cca8404aab4119e7ae7adaa9d4cc5ef32d3320115296e5e
final/submission.csv
4460a754be6300103ba85a235ff1890aa54de123daa5a1e6287e6dce77b548c8
```

## 快速复核现成权重

队友拉取最新仓库，并另外接收上述归档，解包到 `outputs/model_selection_20261029_v1/`。数据仍使用主办方原始数据，结构和CSV哈希按旧复现指南第2节。以下在Linux/WSL项目根目录和可用CUDA环境执行，Mac用于管理材料，不能按此复现CUDA训练。

已有原环境直接激活即可。新机器创建Python3.12环境后，可安装产物内的实际环境快照：

```bash
conda activate glaucoma
python -m pip install -r outputs/model_selection_20261029_v1/final/environment-freeze.txt --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip check
sha256sum outputs/model_selection_20261029_v1/final/best.pt
python validate.py --run-dir outputs/model_selection_20261029_v1/final
```

参考环境Python3.12.13、torch2.10.0+cu128、torchvision0.25.0+cu128、timm1.0.22。验收应为5849张、191阳性、`status=PASS`，原环境概率差0；跨硬件若出现差异，保留日志核对环境、源码和数据，不擅自修改参考预测或放宽阈值。

## 从公开ImageNet原权重重新训练

无需使用他人微调模型作为训练起点。公开原权重和许可见 [来源核对](../licenses/convnext_tiny/核对说明.md)，固定下载版本与SHA256如下。不要用其他同名权重替换。

```bash
mkdir -p pretrain_model
curl -fL --retry 3 'https://huggingface.co/timm/convnext_tiny.fb_in1k/resolve/b43a6303c9fcf176d2d707478a128c2c91e93528/pytorch_model.bin' -o pretrain_model/convnext_tiny_fb_in1k.bin
printf '%s\n' '4bee155ff60a2ee5415f777a381068ebb2a658fc6c7cbb6a33aca13426fb4105  pretrain_model/convnext_tiny_fb_in1k.bin' | sha256sum -c -
python -u train.py --model convnext_tiny --precision fp16 \
  --data-root datasets/glaucoma_first --output-dir outputs/reproduce_convnext_seed43 \
  --pretrained-path pretrain_model/convnext_tiny_fb_in1k.bin \
  --epochs 5 --batch-size 16 --size 384 --workers 4 --jpeg-draft \
  --lr 0.00003 --lr-schedule constant --pos-weight 3 --seed 43
python validate.py --run-dir outputs/reproduce_convnext_seed43
```

训练源码在 `0e263af724bf8ef11ac30f61f90833401e4139a1` 留有Git记录，当前训练源码与之相同，冻结副本也在final/source。比较三种子时，将seed分别设42/43/44并使用不同输出目录，不能覆盖原实验。输出目录、Git元数据和序列化可使文件哈希不同；新训练验收比较配置、验证指标和预测，不要求新文件哈希等于归档权重。不同GPU/软件环境也不保证逐位一致。

## 固定权重生成新CSV

已经有经过核验的CSV，正常交接直接使用它。如需在Linux/WSL复算，选择一个尚不存在的输出文件名：

```bash
python predict.py --checkpoint outputs/model_selection_20261029_v1/final/best.pt \
  --input-csv datasets/glaucoma_first/test.csv --images-dir datasets/glaucoma_first/test \
  --output outputs/reproduced_convnext_submission.csv
```

输出应5855行，沿原test.csv顺序，列为Image_name、Probability、Pred_Label。固定threshold=0.5；不修改CSV概率或凭线上分数反向调参。正式模型核验仍需依主办方要求在T4/指定图片上运行cost.py，并准备相应上传结构和声明。
