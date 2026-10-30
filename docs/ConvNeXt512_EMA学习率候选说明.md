# ConvNeXt512 EMA学习率候选

当前线上最高仍为ID25760/98.16。本文件说明尚未提交的新验证候选；不覆盖旧权重、CSV或材料包。

| 项目 | 新候选 |
|---|---|
| 训练 | ImageNet-1k初始化、512、batch16、w3、AdamW/wd0.01、固定LR6e-5、FP16、EMA0.999，种子42/43/44各5轮 |
| 最佳EMA轮 | 5/3/5 |
| 实际推理 | 三EMA原图各一次，FP32原始logit等权平均后sigmoid，无TTA，阈值0.5，不使用年龄 |
| 完整验证 | 5849图/191阳性，batch1 AUROC0.9899576932、AP0.8376815675，独立重载差0 |
| 本机耗时 | RTX5060Ti，同100图/5预热/3轮31.7202ms；不是T4 |
| 模型体积 | 334033437 bytes |
| CSV | outputs/convnext_ema_lr6e5_logit_submission/submission.csv，5855行，SHA065e6774aa0cf2721fd11e84bb457fdcdd0443369db0158557cfba13fc2ca546 |
| 完整候选备份 | outputs/convnext_512_ema_lr_inference_v1.tar，341585920 bytes，67文件逐哈希核验 |

冻结推理入口：`outputs/convnext_512_ema_lr_inference_v1/logit.json`。`weights.json`只兼容加载三模型，其方法字段不是实际聚合定义；实际由recipe和ema_variant_inference.py控制。

在仓库根目录重新生成独立CSV：

```bash
python scripts/predict_ema_variant.py --recipe outputs/convnext_512_ema_lr_inference_v1/logit.json --names-csv datasets/glaucoma_first/test.csv --images datasets/glaucoma_first/test --output outputs/reproduce_ema_lr/submission.csv
```

解包独立复现及从公开原始权重重训的命令见候选目录README.md，冻结源码、训练配置和许可均随备份保留。依赖使用timm1.0.22阶段的同一glaucoma环境。

TTA的AUROC只额外提高0.00002915，耗时56.2475ms，按预设门槛选无TTA。相对98.16的AUROC差配对bootstrap区间为[0.000469,0.004115]；这是反复选型后的验证样本区间，不校正选择偏差，也不是线上成绩保证。测试集仅生成固定方案CSV，尚未上传。T4及官方指定图片仍待核验。
