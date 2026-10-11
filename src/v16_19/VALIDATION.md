# V16-19 本地验证记录

日期：2026-10-11。环境：macOS、CPU、Python `/opt/anaconda3/envs/Pytorch/bin/python`、PyTorch 2.8.0、scikit-image 0.25.2。CUDA 训练及正式 200 epoch 实验未执行。

## 代码与回归

执行：

```bash
python -m pytest -q \
  tests/test_v16_19.py \
  tests/test_configs.py \
  tests/test_uieb_protocol.py \
  tests/test_e00.py \
  tests/test_color_query_unet.py \
  tests/test_uicf.py \
  tests/test_training_integration.py \
  tests/test_experiment_status.py
```

最终结果：**133 passed**，无失败、无跳过。

新版本检查包括：

- 完整默认模型的全部 state_dict key 和初始化张量与 V16 完全一致；两个正式配置的 model/loss/optimizer/scheduler/training/checkpoint/evaluation/metrics 均保持原 V16 配方。
- 缩小通道数的同构模型在非零 UICF 修正条件下，非整除空间尺寸的前向、UICF 输出、输入梯度和参数梯度与旧 V16 逐项完全一致。
- 新运行模块没有旧版本或 shared 的运行时导入；没有独立 test 模块。
- 两套 synthetic 4-train/2-Val 配对分别完成一轮真实优化、最佳 checkpoint 重载、CSV/JSON 汇总、完整 PNG 导出、固定样本拼图和曲线导出。这些合成检查不是正式数据分数。
- 一次连续两轮训练与一轮后续训到第二轮的最终全部模型张量逐项相等；历史只有 epoch 1/2，无重复。已完成 run 续训仅重做最终 Val 导出。
- 原分辨率评估、数据目录迁移、run 快照优先于后来修改的源清单、旧版本 checkpoint 拒绝、配置/recipe 换用拒绝均通过。
- 模拟 checkpoint 写入失败后原 checkpoint 保留；数据异常导致重新导出失败时既有成功 Val 报告保留。

AST/编译及新增文件空白检查通过。将实施前的 Git 已跟踪文件逐个 SHA256 与完成后的文件比较：**340 个原文件均未改变或删除**。旧 V16 源码、共享模块、旧配置和旧 split 清单均保留。

## 真实本地数据

两套完整协议由当前 `publicdata` 目录实际文件生成，并再次使用 `prepare --verify-only` 核对全部 input/GT 文件字节及成员。

| 数据集 | train | validation | 跨 Train/Val 精确重复输入组 | 协议 fingerprint |
| --- | ---: | ---: | ---: | --- |
| LSUI19 | 3,851 | 428 | 9 | d87af13185bbef9b22ea11c10fbb7530d6f862b27d6daacaddd4aa7780fb51f8 |
| UIEB19 | 801 | 89 | 0 | 4fd9c002ef5c5823644bba3b02ef177966762251e8b4f4e5657106b5134bb124 |

没有重新抽样、移动、去重或重新编码图像。两份协议均 `has_test_split=false`，没有 test.tsv。重复及真实编码信息见各自 `data_quality.json`。

执行两套正式配置的 `preflight`：

- 完整 V16-19 模型均为 **38,878,217** 个参数，与原模型一致。
- LSUI19 真实样本 `1004`，UIEB19 真实样本 `116_img_`。
- 两者均以 `[1,3,256,256]` 输入完成 CPU 前向，输出 shape 正确，输出及 PSNR/SSIM/ΔE00 计算有限。
- preflight 状态均为 `CONFIG_DATA_AND_SAMPLE_FORWARD_PASS`；不代表已训练或完成全量 Val 评测。

## 使用边界

正式默认输出使用 best_psnr，并将最终结果写到 `<run>/result/val/best_psnr/`。分数是参与 checkpoint 选择的 Val 成绩，不能标成独立 test 成绩。两个数据集分别训练、分别选模型。

本地环境不提供 CUDA。本次已经完成代码、数据协议、CPU 数值等价及训练/续训/导出流程验证；目标 GPU 环境仍应先运行 preflight，再进行正式训练。
