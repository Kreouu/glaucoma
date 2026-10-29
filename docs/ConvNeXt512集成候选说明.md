# ConvNeXt-512 集成候选

2026-10-30验证；沿用已训练的seed42/43/44三个模型，概率等权平均，不重新训练或搜索混合权重。

| batch1方案 | 验证AUROC | 验证AP | RTX5060Ti均时 |
|---|---:|---:|---:|
|原seed43|0.986103|0.814204|20.87ms|
|三种子等权集成|0.987245|0.826000|33.84ms|
|集成+水平翻转TTA|0.987463|0.829444|54.31ms|

实用候选：`ensemble.json`；最高验证指标候选：`ensemble_tta.json`。两者均使用相同三份精简模型，总计334,041,477 bytes（十进制334.04MB）。增加TTA的AUROC差小于事先约定的0.0005取舍阈值，因此默认选耗时更低的纯集成，同时保留TTA版本。

两方案分别独立重新加载，完整5849张/191阳性batch1概率差0、标签差0。计时同100验证图、5预热、三轮，包含JPEG读取、预处理、传输、前向、概率平均及CSV写入/flush；同一图片只预处理一次。完整记录见候选目录的verification.json及两个full_reload.json。

原单模型batch1和历史batch16有FP16舍入差，AUROC约差0.00000139，不能把跨batch概率差写成0；本表统一batch1。bootstrap差值区间跨0，属于当前验证集点估计改善，并非已经证明泛化或官方分数提升。T4仍未验证。当前官方最高仍为98.08，原CSV/模型和材料包保持原样。

## 文件与复用

候选目录（Mac/WSL同相对路径）：`outputs/convnext_512_inference_candidates_v1/`。

- `seed42.pt`、`seed43.pt`、`seed44.pt`：只保留model/config/metrics，逐张量核对相等；不含优化器，不能用于原地续训。
- `ensemble.json`、`ensemble_tta.json`：方法、模型SHA和源码SHA；两个方案共用三份模型，不重复计算模型大小。
- `*_valid_predictions.csv`：完整验证概率，不是测试提交文件。
- `source_snapshot/`：推理源码留档。使用对应Git代码和现有同一glaucoma环境，timm1.0.22；不需要新环境清单。

在WSL仓库根目录核验实用候选：

```bash
~/miniconda3/envs/glaucoma/bin/python scripts/reload_convnext_candidate.py \
  --manifest outputs/convnext_512_inference_candidates_v1/ensemble.json
```

准备正式预测时，可按官方CSV顺序生成一个新文件（本轮未执行测试预测）：

```bash
~/miniconda3/envs/glaucoma/bin/python scripts/predict_convnext_ensemble.py \
  --manifest outputs/convnext_512_inference_candidates_v1/ensemble.json \
  --names-csv datasets/glaucoma_first/test.csv \
  --images datasets/glaucoma_first/test \
  --output outputs/convnext_ensemble_submission/submission.csv
```

如选最高验证指标版本，只把manifest换为ensemble_tta.json。脚本拒绝覆盖已存在的输出；两方案都必须按实际采用的manifest记录版本，不与旧98.08文件混用。许可与初始化来源沿用ConvNeXt原始ImageNet-1k留档，未引入外部医学数据。上述命令不是T4验收或平台提交。
