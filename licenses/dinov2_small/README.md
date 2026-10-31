# DINOv2 ViT-S/14 后备候选

仅准备官方通用 LVD-142M 自监督 `dinov2_vits14`，不是 XRay-DINO、Cell-DINO 或其他医学权重。官方固定版本 README 的标准 DINOv2 License 段明确代码与权重 Apache-2.0；医学模型的单独非商业条款不属于此候选。保留完整上游声明，避免断章取义。

HF/timm 固定版本模型卡标 Apache，但 config 仍写 CC-BY-NC；该镜像未选作训练初始化。改用官方 dl.fbaipublicfiles.com 原始文件，URL、SHA256 和原始大小见 manifest.json。所有原文件仅保存在被忽略的 pretrain_model。

当前仅完成来源准备；尚未通过CPU适配、GPU训练、验证或推理耗时核验，不是已选定的新主力模型。后续将先测试官方权重到 timm 的严格转换，使用独立源码/输出，保持当前运行任务不变。
