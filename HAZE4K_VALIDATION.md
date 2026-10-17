# Haze4K 工程交付验证记录

日期：2026-10-17（Asia/Shanghai）。本记录只涉及软件测试，**没有真实 Haze4K 实验数值**。

实际执行环境：macOS ARM64，Python 3.13，PyTorch 2.14.1，CPU，CUDA unavailable。依赖安装在仓库外 `/tmp/haze4k-codex-venv`；未修改旧 requirements。

实际测试命令（从 UIE4_VN 根运行）：

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg /tmp/haze4k-codex-venv/bin/python -m pytest \
  tests/test_haze4k_transfer.py -q --basetemp=/tmp/haze4k-release-tests --durations=10
```

结果：**12 passed, 1 skipped in 15.57s**。跳过项为 `test_cuda_enabled_amp_resume`，原因是没有 CUDA。该测试会在可用 CUDA 的 AutoDL 环境实际检查启用的 GradScaler 状态恢复；本地未声称该项已通过。

已实际完成：

- 完整扫描合成 3000/1000 对图像，得到 train=2700、validation=300、test=1000；同名数值 ID 不误报跨官方 split 泄漏。
- 合法多 haze 对一 GT、缺失/多重 GT、未使用 GT、损坏像素内容、尺寸不符、错误 manifest 配对、重复 input、目录别名歧义、文件内容变动、split 防覆盖、元数据/trans 隔离。
- 确定性 TSV 字节与 SHA256；跨官方 split 文件 hash 和标准化 RGB hash 重复报告。
- 原 Full/V15 模型构造、初始骨干全张量相等、两组奇数尺寸前向及有效 backward。
- 两组各两轮合成训练，第一轮停止，加载 checkpoint 恢复第二轮；优化器 step、RNG、DataLoader generator、禁用的 AMP scaler 状态保存/恢复。
- 额外连续训练 w/o UICF 两轮，与恢复训练后的全部 model state 张量逐项精确相等（CPU、workers0、deterministic=true）。
- 两组实际 test_256、Full test_native、mean-of-per-image 指标核对、配对 clustered bootstrap、拒绝 ID/协议/checkpoint 不一致、生成带 SYNTHETIC/SMOKE 标识的 paper_table.md 和 Hazy|w/o|Full|GT PNG。
- 中断、尚未达到目标 epoch 的运行禁止执行最终 test。

模型实际计数：Full=38,878,217 参数；w/o UICF=38,740,483 参数；287 个骨干 state_dict 张量完全一致。参数差 137,734 正好属于 UICF。

另已通过 Python compileall、全部新增 shell 的 `bash -n`、`git diff --check`。最终受保护目录及旧配置/旧 split/旧 experiments 的 `git diff` 为空。原有未跟踪 `haze4k_preflight.py` 未改动。测试仅删除自身创建的临时 run；大型 checkpoint 没有加入仓库。

真实限制：目标 `/root/autodl-tmp/pro/publicdata/Haze4K` 在本机不存在。真实数据全量解码、尺寸统计、实际 SHA256 审计和两组主训练均**待 AutoDL 执行**。真实数据 prepare 通过前不得将用户此前的文件名检查当作完整预检；本记录不证明 CUDA 性能、实际数据零重叠或真实去雾效果。


## AMP 修复追加验证（2026-10-17）

服务器实际日志报告 Full 在 epoch1 出现 `backbone.output_conv.weight` 非有限梯度。本机检查发现独立训练入口在 scaled backward 前先乘 batch size、反向后才除回，增加了FP16中间梯度溢出风险。仅修改隔离工程的训练数值处理，改成反向前按真实累积组样本数归一化；没有改动原模型或loss。

实际执行：

```bash
OMP_NUM_THREADS=1 MPLBACKEND=Agg /tmp/haze4k-codex-venv/bin/python -m pytest tests/test_haze4k_transfer.py -q --durations=5
```

结果：**15 passed, 1 skipped in 17.22s**。额外测试覆盖：旧计算顺序的FP16 Inf可复现、修正后同例梯度有限；不同累积步数/尾部batch/截断smoke均等价于实际整组均值梯度；非法AMP scale被拒绝；注入Inf梯度后不更新参数、不保存checkpoint并写失败诊断。原合成训练/恢复/eval/compare全链测试再次通过。

新增显式 `--amp-init-scale` 和仅限诊断的 `--fp32-smoke`；默认AMP scale没有静默降低。服务器可尝试两组一致的初始scale1024并先smoke。当前仍没有CUDA，不能声称已在用户GPU上修复或已确定此次日志的全部根因。真实数据与主实验数值仍以AutoDL实际运行结果为准。
