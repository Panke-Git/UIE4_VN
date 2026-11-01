# 任务②：UICF 证据链工具本地开发与测试报告

日期：2026-11-01；时区：Asia/Shanghai。地点：MacBook 本地 UIE4_workspace/UIE4_VN。

本次已完成独立只读核验工具、自动化测试、中文操作指南和六文件/ZIP 输出。
这里记录的是软件验证，不是 WSL 真实实验证据已经核验通过的声明。
未连接 WSL 或其他服务器，未执行 SSH、训练、正式图像推理或远程 Git 操作。

## 实际执行的检查

| 检查 | 结果 | 证据 |
|---|---|---|
| 新增核验工具测试 | 45 通过，0 失败/错误/跳过 | analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/pytest_delivery.xml |
| 原有相关回归测试 | 82 通过，0 失败/错误/跳过 | analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/original_regression.xml |
| 默认 CLI 缺失证据路径 | 实际生成六份文件及 ZIP；状态 UNVERIFIED，退出码 2 | analysis/uicf_evidence_chain_audit/mac_missing_evidence_preview/task2_20261101_172418_358385/ |
| 人工完整数字产物的集成路径 | Spatial/Intervention 各子项数字可通过；整体明确标识人工证据、论文待核验 | 新增测试中的 SYNTHETIC_EVIDENCE_NOT_EXPERIMENT 临时目录 |
| 不导入模型/图像框架 | import guard 下默认核验及可选文件 SHA256 测试通过 | 新增测试 |
| 原始输入不变 | 人工源文件前后 SHA256 逐项一致 | 新增集成测试 |
| 开发起点保护 | 原有 573 个已跟踪文件 SHA256 全部不变 | analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/tracked_protection_check.json |
| 指南语法 | 11 个 Bash 块 bash -n 通过；1 个内嵌 Python 编译通过 | 同目录 guide_syntax_check.json |

合计 127 项测试通过。这不包含任何真实 LSUI 数据/模型评估。
指南中的 Git、WSL 和导出说明命令只做语法检查，没有被当成远程操作执行。

## 测试覆盖

- 从现有 CSV 数字独立重算图像等权均值、有效/缺失/非法计数，区分缺失与零。
- 原统计源码中的平均并列秩 Spearman、常量图未定义处理、精确 Top-20% 的 52/256 块及稳定并列处理。
- 先求 RGB 范数再池化与先池化再求范数不等价的人工反例。
- 大循环位移零对照的非零、位移阈值、数量、种子可复现性。
- 独立实现的图像 Bootstrap 与已审阅原纯统计函数一致；冻结 checkpoint 的图像重采样不作跨种子解释。
- Spatial 全部主要表示、控制、实际校正的保存均值、差值字段、Bootstrap、引用数值容差。
- Functional 的八种变体、逐图配对、三指标方向、保存配对差、CI 和诊断不变量。
- 缺失 CSV 不被制造为零观测，缺失变体不被假设完整；重复 ID、非有限值和明确冲突不能 PASS。
- checkpoint 路径迁移、错误 selector/哈希、缺失哈希、仅文件流式 SHA256、不加载权重。
- 官方 Original 差异单独报告；干预配对差始终使用内部 Original。
- 未保存历史源码时保留 UNVERIFIED，不用当前源码替代历史执行证明。
- 原始 manifest、冻结样本索引与文件名、伪造额外测试样本的检查。
- 输出目录限制、符号链接拒绝、时间戳隔离、重复 ZIP 拒绝覆盖、来源文件不变。
- 原文缺失时 PENDING_MANUSCRIPT；提供表格 JSON 时数值冲突保留真实输入，不编造论文段落。

完整数字夹具使用固定清单的 ID 和人工构造的数字，目的是验证解析器对 428 行/八变体原生格式的处理，
没有读取对应图像或真实权重，没有执行 428 图像的机制分析。
这些夹具只保存在独立 pytest 临时目录，并有 SYNTHETIC 标记，不能作为论文实验结果。
新工具复核均值用独立的 math.fsum；测试仅提取原程序的纯统计函数构造原生格式及交叉验证，
不导入或改写原分析程序、不运行它们的 main，也不做运行时 monkey-patching。

## 实际命令与环境

新增测试使用现有环境：

~~~bash
cd /Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN
/opt/anaconda3/envs/Pytorch/bin/python -m pytest tests/test_uicf_evidence_chain_audit.py -q \
  --basetemp=analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/pytest_delivery \
  --junitxml=analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/pytest_delivery.xml
~~~

Python 3.12.11、NumPy 2.2.6、PyYAML 6.0.2、pytest 9.1.1。
该环境缺少 Matplotlib，因此原有相关回归改用工程已有的完整环境，没有安装/升级依赖：

~~~bash
MPLBACKEND=Agg \
MPLCONFIGDIR=analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/mpl_cache \
experiments/downstream_duo/.venv/bin/python -m pytest \
  tests/test_uicf_alignment.py tests/test_uicf_intervention.py \
  tests/test_uicf.py tests/test_uicf_unet.py -q \
  --basetemp=analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/original_regression \
  --junitxml=analysis/uicf_evidence_chain_audit/local_development_20261101_165940_334397/original_regression.xml
~~~

完整环境为 Python 3.12.11、PyTorch 2.8.0、NumPy 2.2.6、PyYAML 6.0.2、pytest 9.1.1、
Matplotlib 3.10.8、scikit-image 0.25.2；CUDA available=False。
原回归涉及的模型运算仅为原测试中的微型人工 CPU 用例。

首轮新增测试为 31 通过、5 失败：一个断言未容忍浮点相关系数的机器误差，
一处局部同名导入遮蔽导致多个集成用例失败，以及测试在目录/ZIP 同名 glob 中误选 ZIP。
修复后 36 项通过；随后增加主 CSV 缺失、错误配对方向、历史来源、冻结清单等检查，
最终扩展至上述 45 项全部通过，没有遗留失败/错误/跳过。

## 新增文件清单

均相对于当前 UIE4_VN：

~~~text
tools/audit_uicf_evidence_chain.py
tools/uicf_evidence_audit/__init__.py
tools/uicf_evidence_audit/io.py
tools/uicf_evidence_audit/statistics.py
tools/uicf_evidence_audit/provenance.py
tools/uicf_evidence_audit/alignment.py
tools/uicf_evidence_audit/intervention.py
tools/uicf_evidence_audit/report.py
tools/uicf_evidence_audit/source_contract.json
tests/test_uicf_evidence_chain_audit.py
docs/UICF_EVIDENCE_CHAIN_AUDIT_GUIDE_CN.md
docs/UICF_EVIDENCE_CHAIN_LOCAL_TEST_REPORT_CN.md
~~~

原 src/v16、src/v15、共享 UICF、两个既有分析入口、tools/uicf_intervention、
configs/config_v16.yaml 以及原始实验产物没有修改。
工作区仍为 UIE4_workspace；没有暂存、提交或推送。
当前交付为上述新增未跟踪文件；analysis 下本地检查输出沿用原 .gitignore，不纳入提交。

## 尚未执行与仍需核实

- 没有访问指定 WSL 基线、真实 summary/protocol/per-sample CSV 或真实 checkpoint。
- 没有执行真实 Table 4 数字重聚合或冻结模型干预产物的正式核验；任务给出的数值只用于比较目标。
- 未读取当前论文 Section 4.4 / Supplementary Table 4 原文；状态 PENDING_MANUSCRIPT。
- 没有完整仓库无关测试、GPU 训练、CUDA 推理或重新计算机制分析特征图。
- 历史源码、命令、配置及二进制身份缺失时，工具会保留 UNVERIFIED，不补造证据。

当前可以保留第 4.4 节 Spatial Correspondence / Functional Intervention 的组织，
不能在没有真实数值与原文核验的情况下宣布论文无需实质修改。
实际校正量与系数表示分别核验，不自动要求 Table 4 新增 Actual Correction 一行。
本地开发已完成；WSL 正式核验由你手动同步和执行，命令见中文指南。
