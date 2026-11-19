# ICAR-UIE × TC-USOD

本实验在 USOD10K 官方 Test 上比较 Raw RGB 与 LSUI-trained ICAR-UIE V16 增强 RGB。每组使用同一个冻结的作者 TC-USOD 完整预训练权重、同一份原始 Depth 和同一评价协议。P1 可加入有正式权重的 FX 方法，P2 可加入 UIEB-trained V16。

| 阶段 | 条件与验收 |
| --- | --- |
| P0 | Raw、ICAR-UIE 各覆盖全部 1,026 个官方 ID。通过相同资产的 CUDA smoke 后生成逐图预测、六项指标、逐图配对变化和 95% 配对 bootstrap 区间。 |
| P1 | 在 P0 完整后，按配置逐个验证 WaterNet、Ucolor、FDCE-Net、MARINE-Net、SWLGM。纳入全量完成的 baseline，逐一记录缺失资产或推理失败。 |
| P2 | 在 P0 完整后，用冻结的 UIEB 正式权重及独立 smoke 增加条件。汇总与原 Raw 和 LSUI 主条件配对。 |

运行配置在 config 目录。默认用 1024 tile、128 overlap、FP32、批大小 1。超过 tile 的输入使用重叠拼接，小图直接做原尺寸前向。USOD10K 这版官方 Test 全部为 640×480，因此默认条件对全 Test 使用整图。需要原尺寸模式时把 tile_size 和 overlap 都设为 0。批处理只把原尺寸相同的增强样本放进同一个 batch。不同推理配置使用不同输出目录。

作者 TC-USOD 的两路输入均解码为 RGB，Pillow 双线性缩放至 224×224，再做 ImageNet 归一化。采用已通过 sigmoid 的 d1，先量化为灰度 PNG8，再以 Pillow 双线性插值恢复到原尺寸。增强输出为原尺寸 RGB PNG8。测试 GT 用于清单几何审核和后续评价，模型前向只取 RGB 及固定 Depth。

MaxF、MeanF、MaxEm、MeanEm 使用官方 255 阈值曲线。总体结果先按图平均曲线，再取最大值或均值。F、E、S 的预测图按图做 min-max 归一化，MAE 使用原始灰度强度。F、E 使用灰度 GT，S 的结构计算用 0.5 二值化。每次配对 bootstrap 共享按图抽样权重，重新聚合两组数据集曲线。默认 5,000 次，seed 为 3520，报告百分位 95% CI。逐图改善率使用逐图指标和各自方向。示例依据 Raw MAE 减 ICAR-UIE MAE 的最大值、中位附近和最小值排序，样本 ID 处理并列。

开发和运行说明见 [AutoDL 部署](docs/downstream_usod/DEPLOY_AUTODL.md)、[资产与来源核查](docs/downstream_usod/PREDEPLOY_AUDIT.md) 和 [本地验证](docs/downstream_usod/LOCAL_VALIDATION.md)。

| 本次修改 | 文件 |
| --- | --- |
| 数据、输入和运行状态 | src/downstream_usod/config.py、dataset.py、storage.py、preprocessing.py、inference.py |
| 模型接入 | src/downstream_usod/v16_adapter.py、tc_usod_adapter.py、fx_adapter.py |
| 指标与出图 | src/downstream_usod/evaluation.py、statistics.py、visualization.py |
| 阶段与命令 | src/downstream_usod/__init__.py、pipeline.py、cli.py、export.py |
| 运行配置与依赖 | configs/downstream_usod/tc_usod_usod10k.yaml、paths.example.yaml、requirements.txt，根目录 .gitignore |
| 部署与结果归档 | scripts/downstream_usod/deploy_autodl.sh、check_autodl_assets.sh、run_p0.sh、run_p1.sh、export_results.sh、download_results.sh |
| 测试和说明 | tests/downstream_usod/test_dataset_inference.py、test_metrics.py、test_delivery.py、本文及 docs/downstream_usod 下三份文档 |
