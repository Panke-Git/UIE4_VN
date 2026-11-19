# 本地验证记录

验证地点为 Mac。本地使用 Python 3.12.11、PyTorch 2.8.0、torchvision 0.23.0、timm 1.0.29、NumPy 2.2.6、Pillow 12.0.0、PyYAML 6.0.2。已使用现有 Pytorch 环境运行 CPU 验证。

| 检查 | 实际结果 |
| --- | --- |
| Python AST、import、Shell 语法、CLI help、YAML | 检查了 USOD 新增模块、全部 CLI 子命令、脚本和 YAML 模板。 |
| 正式数据成员和格式 | 全 1,026 个官方 Test ID 匹配 RGB、Depth、GT，每组 640×480。已解码并逐文件记录 SHA-256。 |
| 正式 TC-USOD 加载 | 用作者的完整 TCUSOD.pth 在 CPU 严格加载并冻结，125,957,743 个模型参数。 |
| TC-USOD 三图真实 Raw 前向 | ID 00002、05132、10252，使用对应的官方原始 Depth。输出均为原尺寸、灰度 PNG8，GT 只在评价阶段读取。00002、05132 与作者发布图逐像素一致，10252 有 5 个像素相差 1 个灰度级。 |
| 三模态匹配、指标及统计单测 | 测试缺文件、重复 ID、错尺寸、路径迁移、输入哈希变化、原版评估器的 F/E 曲线和六项指标数值、按重采样重新聚合曲线。 |
| 合成 CPU 端到端 | 用小尺寸 V16 网络和模拟双模态 TC-USOD 跑 native 与 tiled、保持原尺寸、分批预测、锁定同一 Depth、模拟中断后续跑、拒绝损坏输出并生成可视化。另用 mock 跑 P0/P1/P2 门槛和汇总覆盖。 |
| 部署保护及归档 | 在本地临时 Git 仓库验证完全匹配的干净检出通过，有未提交变更的检出被拒绝且文件内容保留。归档测试验证只收集指定摘要，已有目标包会被拒绝。 |
| FX 导入 | WaterNet、Ucolor、FDCE-Net、SWLGM 的既有适配模块在本地导入通过。MARINE-Net 在当前 Mac 解释器缺 pytorch-msssim，需用其复现环境。FX 的真实权重加载和前向待相应资产到位后验证。 |
| 合并验证计数 | 新模块 12 项通过。合并既有 DUO 和 UICF 回归，共 61 项通过，1 项既有 DUO 可选集成 smoke 跳过。 |

额外使用作者仓库保存的全部 1,026 张 TC-USOD Raw 预测图，与原作者 Test GT 重新评分。这是指标对照测试。

| 指标 | 本项目复算 | 作者 result.txt |
| --- | ---: | ---: |
| MaxF | 0.923593584 | 0.9236 |
| MeanF | 0.909835625 | 0.9098 |
| S-measure | 0.921454743 | 0.9215 |
| MaxEm | 0.968271978 | 0.9683 |
| MeanEm | 0.963379642 | 0.9634 |
| MAE | 0.020084741 | 0.0201 |

六项指标均与发布值匹配至四位小数。NumPy 按像素计数等价重建官方阈值集合，并保留官方 float32 阈值和灰度输入。数据集及 bootstrap 聚合使用 float64。S-measure 的空象限贡献零权重，单像素样本标准差按 0 处理。 [作者 result.txt](https://github.com/LinHong-HIT/USOD10K/blob/fcdd53926054c79c20a64d1c69621316c5562929/TC_USOD/result.txt)

从 feature checkout 复跑本地回归。

```bash
export USOD_OFFICIAL_REPO=/path/to/pinned/USOD10K-source
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -B -m pytest -q \
  tests/downstream_usod tests/test_downstream_duo.py tests/test_uicf_unet.py
```

USOD_OFFICIAL_REPO 指向含 Evaluation/evaluator.py 的固定作者 checkout。此文件在 Mac 的 publicdata/USOD10K_assets/USOD10K-source 中保留。源码和测试不自动下载第三方文件。

具备本地 paths.local.yaml 时可以用 preflight --load-models 检查实际资产。此次 CPU preflight 核查通过数据、作者源码和 TC-USOD 严格加载，并因缺正式 LSUI V16 的 config_resolved.yaml 和最佳权重报告整体阻塞。当前 AutoDL 的 CUDA 权重加载、显存、正式 V16 前向、各 FX 真实 smoke 以及正式全量 P0/P1/P2，须在资产迁移后按部署文档验证。
