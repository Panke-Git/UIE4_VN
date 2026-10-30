# UICF 参数化消融：本地测试报告

验证日期：2026-10-30；时区：Asia/Shanghai。地点：MacBook 本地 UIE4_workspace/UIE4_VN。

## 已实际执行的最终检查

| 检查 | 结果 | 证据 |
|---|---:|---|
| 新增自动化测试 | 37 通过，0 失败/错误/跳过 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/new_tests.xml` |
| 原有模型/指标/状态回归 | 72 通过，0 失败/错误/跳过 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/regressions.xml` |
| 实际独立 CLI smoke | 三组均通过训练、保存、恢复和测试 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/local_tests/cli_smoke.log` |
| 标准维度参数/输出图检查 | 四组均通过，三变体各少 3331 参数 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/analysis/uicf_parameterization/parameter_counts.json` |
| 运行指南命令语法 | 17 个 Bash 代码块通过 bash -n，内嵌 Python 编译通过 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/docs/UICF_PARAMETERIZATION_ABLATION_GUIDE_CN.md` |
| Shell 脚本 | 3 个脚本 bash -n 通过 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/scripts/uicf_parameterization_ablations` |
| 原有文件保护 | 545 个已跟踪文件 SHA256 全部不变 | `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN/src/uicf_parameterization_ablations/protection_snapshot.json` |

运行的命令：

```bash
cd /Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN
PYTHON=/opt/anaconda3/envs/Pytorch/bin/python bash scripts/uicf_parameterization_ablations/verify_local.sh
/opt/anaconda3/envs/Pytorch/bin/python -m src.uicf_parameterization_ablations.smoke --output-dir analysis/uicf_parameterization/cli_smoke
/opt/anaconda3/envs/Pytorch/bin/python -m src.uicf_parameterization_ablations.capacity
/opt/anaconda3/envs/Pytorch/bin/python tools/uicf_parameterization_report.py --use-provided-baseline
```

自动化测试覆盖：

- 三个真实数学公式的确定性逐元素数值，Direct Residual 诊断语义，固定 0.5 与无锚点区分。
- 三组 UICF 初始输出逐项等于输入，奇数图像尺寸的 Backbone 接口和数值有限性。
- 两步 forward/backward；第一步零初始化允许上游零梯度，第二步检查 Encoder、MLP、Backbone、token refinement、guidance 和 local refinement 的非零梯度，无死参数。
- 标准维度下三组共有 state（包括 Backbone/QGRD/Color Query、Encoder、MLP、BN 缓冲）与同 seed 原 V16 逐项 torch.equal；仅移除 anchor_mlp。
- 标准维度 Full=38878217，三组各 38874886，总量/可训练/实际输出图可达数一致，无注册但不参与输出的参数。
- 原始 V16 仍为 I + R*(I-b)，初始化与原工厂保持不变；全部既有文件逐字节保护。
- 每组小模型 checkpoint 的保存前/加载后 forward 逐项一致；真实入口的 model/optimizer/AMP/RNG/generator 保存及严格恢复。
- 每组连续两轮与第 1 轮后中断再恢复到第 2 轮：model state、optimizer state、Torch RNG、DataLoader generator 完全一致。
- 错误 variant、未完成训练的 test、被修改的冻结清单、错误 recipe 均拒绝；拒绝错误变体时不改变原运行状态。
- 三个正式配置的 dry-run 均确认固定 3466/385/428，不创建训练目录；四个模块的 --help 可执行。
- 汇总只读原生测试证据，拒绝验证集/人工 smoke、重复或不完整样本、均值冲突；兼容旧 V16 缺少 dataset/expected_counts 的配置。用于校验读取器的人工元数据仅位于 pytest 临时目录，未写入正式 results.md/results.csv。

原有回归文件是 tests/test_uicf.py、tests/test_uicf_unet.py、tests/test_color_query_unet.py、tests/test_e00.py、tests/test_experiment_status.py。

## 测试环境与初次设置问题

Python：`/opt/anaconda3/envs/Pytorch/bin/python` / 3.12.11；PyTorch 2.8.0；NumPy 2.2.6；Pillow 12.0.0；PyYAML 6.0.2；scikit-image 0.25.2；pytest 9.1.1。CUDA available=False。

系统 Python 和当前 Codex 捆绑 Python 缺少 PyTorch，改用本机已有 Pytorch 环境，没有安装或修改任何环境依赖。该环境缺少 Matplotlib，本轮未运行整个仓库的可视化测试，未将其计为通过。

首轮新增测试为 22 通过、8 个 setup 错误，原因是指定的 pytest basetemp 父目录尚未创建，工作流测试未执行。验证脚本增加 mkdir -p 后消除该设置问题；最终完整结果如上。最终没有遗留失败、跳过或未解决的实现错误。

## 没有执行的检查

- 没有读取或修改 WSL、AutoDL、GitHub，未运行任何 SSH 命令；WSL 基线文件、环境、GPU、数据路径均待你以后验证。
- 未读取原 Full checkpoint；其指标仅采用用户明确提供的六位小数，生成表中标注待 WSL 验证。
- 未执行 CUDA AMP、标准 batch 16 / 256×256 显存测试、200 epoch 正式训练、真实 LSUI 图像测试或真实基线再评估。没有真实消融分数。
- 本地 smoke 明确使用 2/1/1 人工数据、微型模型、CPU、2 epoch；输出保存在 analysis/uicf_parameterization 下的独立临时目录，不是正式实验。
- 源码哈希固定与 epoch 边界 RNG 恢复可帮助复现；不承诺不同 CUDA/PyTorch/设备/worker 或非确定性内核之间位级一致。

现有原 V16 和其他实验没有改变；没有 git add、commit、push、pull、fetch、checkout、reset 或工作区切换。
