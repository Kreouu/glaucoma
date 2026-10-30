# ConvNeXt512 EMA/logit/TTA 候选

2026-10-30，本轮新候选的验证AUROC0.9878132062、AP0.8298282927；相比已评分98.15的概率集成0.9872450443/0.8260003565，分别增加0.0005681618/0.0038279362。本机100图/5预热/3轮约49.31ms，权重334,033,245 bytes。完整5849张独立重载差0、标签差0；T4未验证。

推理：三种子EMA0.999最佳权重，各对原图与水平翻转图输出原始logit，将六个logit等权平均后再sigmoid，阈值0.5。没有温度拟合、混合权重搜索或测试集拟合。EMA是训练过程中成功优化步的参数平均，不是冻结训练骨干。

最终组合涉及EMA、聚合和TTA三项，不能把组合增益全归因于EMA。三EMA直接概率平均、单成员替换及其他固定聚合结果均保留；只有logit+TTA满足预设AUROC至少+0.0005且AP不下降的筛选条件。bootstrap两个指标区间均跨0，反复使用验证集选型可能乐观，不能承诺线上提高。官方最高仍98.15。

## 文件

候选目录：outputs/convnext_512_ema999_inference_v1/。

- logit_tta.json是选定推理配方；weights.json为兼容旧冻结加载器的权重清单，实际聚合方式以配方和ema_variant_inference.py为准。
- seed42.pt/seed43.pt/seed44.pt三份EMA模型，均与已审计原EMA checkpoint字节相同；不含优化器，不用于恢复训练。
- 其他配方及验证预测保留为实验记录，不表示均采用。
- 新提交CSV在outputs/convnext_ema999_logit_tta_submission/submission.csv，5855行，SHA256 855831fe645e121907c6ab65d575ffcf9cc033355bc10c77952dd45e8cc0d051。已核验并同步Mac，尚未上传。

在WSL仓库及原glaucoma环境运行；没有安装新依赖，没有新增conda环境。完整重载：

```bash
~/miniconda3/envs/glaucoma/bin/python scripts/reload_ema_variant.py \
  --recipe outputs/convnext_512_ema999_inference_v1/logit_tta.json
```

需要重新生成CSV时须选择新的输出目录，避免覆盖：

```bash
~/miniconda3/envs/glaucoma/bin/python scripts/predict_ema_variant.py \
  --recipe outputs/convnext_512_ema999_inference_v1/logit_tta.json \
  --names-csv datasets/glaucoma_first/test.csv \
  --images datasets/glaucoma_first/test \
  --output outputs/convnext_ema999_logit_tta_submission_v2/submission.csv
```

source_snapshot/用于还原对应源码。来源与许可沿用ConvNeXt公开ImageNet-1k留档；无外部医学数据。已评分98.15模型、CSV、材料包和旧98.08版本均保持原样。
