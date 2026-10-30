# ConvNeXt512 后续优化

当前已评分参照：纯三种子等权集成25528/98.15，CSV SHA35a061478c49014184506193ad92178d06e8f2b9966c0c723dfec6273bfd9cc5。用户授权继续到下一次提升；不自动上传平台，不能保证官方分数必增。长训练约每小时检查实际进程，结束集中验收，记录主要写基线实验记录.md。

第一轮：EMA0.999单因素，512、固定LR3e-5、w3、batch16、5轮×42/43/44，其余不变。普通模型与EMA共享同一次训练，验证时同一批图同时前向，不多迭代DataLoader，避免额外消耗随机数；EMA只在GradScaler成功更新优化器后更新，首次成功步直接复制，之后指数平均。ConvNeXt无BN/缓存buffer，本轮直接断言buffer为空。EMA是训练过程参数平均，不需要额外推理模型。

先两个64/64两轮检查：EMA开启/关闭的普通权重、指标、预测必须完全相同；EMA更新计数=总步数-AMP跳步。通过后启动全量队列。各轮普通/EMA分别取最佳AUROC并保存；EMA checkpoint仅含model/config/metrics，不提供断点续训。

验收：每组5轮、完整5849/191阳性重载PASS、源码/CSV/预训练/模型SHA通过。普通轨迹还应与已有相同种子512运行一致。比较3个配对种子与固定等权集成（同为batch16初筛）；优先寻找集成AUROC至少+0.0005且AP不下降，再做batch1实际输出和同图测速。阈值只是工程取舍，不能称统计显著。不得挑单个最高种子就宣称总体胜出。

若EMA无收益，保留98.15，下一轮依证据选择有限的学习率或正则化单因素适配；避免盲目扩大模型/分辨率，FasterViT当前NC权重因规则不接受而排除比赛候选。整个过程不将测试图像用于训练、拟合或模型选择。

PyTorch依据：https://docs.pytorch.org/docs/2.10/generated/torch.optim.swa_utils.AveragedModel.html 。配置和源码哈希见evidence/convnext_512_ema999_v1/plan.json。
