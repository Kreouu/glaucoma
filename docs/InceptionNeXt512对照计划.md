# InceptionNeXt-Tiny 512 对照

用户授权写脚本并直接启动。保持ConvNeXt-512配方，仅更换架构及对应公开ImageNet-1k初始化：512、batch16、workers4、FP16、AdamW/wd0.01、固定LR3e-5、BCE正类权重3、5轮、seed42/43/44；训练只用水平翻转，保留JPEG draft和letterbox。全量微调，骨干不冻结，不叠加EMA、旋转或TTA。

先运行64训练/64验证smoke并重载，再顺序执行三个完整种子，每组结束运行validate.py验证checkpoint和概率重现。失败即停，不自动重启或降级参数。原始权重版本、许可及哈希见licenses/inception_next_tiny/manifest.json；源码和CSV冻结清单见evidence/inceptionnext_512_w3_v1/plan.json。没有新建conda环境。

与outputs/convnext_512_w3_v1对应三个种子比较最佳验证AUROC、对应AP、耗时及显存。均值改善小于0.0005暂视不确定，此为工程判断而非统计显著性。初轮配方落后不等于模型上限低；接近时可讨论有限学习率适配。当前98.08模型、CSV和材料保留；本队列不生成测试预测、不提交平台。官方T4核验仍暂缓。

运行：bash scripts/run_inceptionnext_512.sh
预览：bash scripts/run_inceptionnext_512.sh --dry-run
输出：outputs/inceptionnext_512_w3_v1/
总日志：outputs/inceptionnext_512_w3_v1.supervisor.log
退出标记：outputs/inceptionnext_512_w3_v1.exit_code

状态：代码准备完成，等待运行检查后记录实际启动状态。
