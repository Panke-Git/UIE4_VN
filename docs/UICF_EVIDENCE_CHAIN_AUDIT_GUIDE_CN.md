# UICF 实验证据链核验：Mac 开发与 WSL 只读运行指南

本工具服务任务②，不重新训练、不运行模型推理、不重新生成 Spatial Correspondence 或 Functional Intervention。
当前 Codex 工作区始终保持 UIE4_workspace；新增文件都在其中的 UIE4_VN 工程内。
2026-11-01 的开发与测试仅在 Mac 完成，本文 WSL 命令由你后续手动执行。

## 核验范围与状态

唯一基线运行名为 v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528。
目标协议是 LSUI19 的 3466 / 385 / 428，seed 3520，200 epochs，Best Validation PSNR。
用户提供的性能和 Table 4 数字仅用于核对，缺少 CSV 时绝不代入为观测结果。

工具默认只读已有 JSON、CSV、TSV、YAML、文本与源码，甚至不需要导入 PyTorch。
唯一运行依赖是 PyYAML 和 NumPy；NumPy 用于复核已保存数字的 Bootstrap，不运行网络。
默认只检查 checkpoint 的存在性与大小；可选 --hash-checkpoint 只按块读取文件 SHA256，不反序列化、不加载模型。

- PASS：该项所需证据充分且一致。
- FAIL：存在明确数据、统计或身份冲突，详见逐项检查。
- UNVERIFIED：缺失或不足，不能作为已核验通过。
- PENDING_MANUSCRIPT：缺论文原文；在总状态中按 UNVERIFIED 处理。

程序退出码：0=PASS，1=FAIL，2=UNVERIFIED；参数使用错误也可能返回 2，此时应查看终端报错。
不能把“六份报告已生成”理解为真实数据已经通过核验。

## 第一阶段：MacBook 检查与手动 GitHub 同步

进入实际工程，不切换 Codex 工作区：

~~~bash
cd /Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN
git status --short --branch
git diff --stat
git diff -- src/v16 src/v15 src/shared/uicf_inr.py src/shared/uicf_models.py \
  tools/analyze_uicf_alignment.py tools/analyze_uicf_intervention.py \
  tools/uicf_intervention configs/config_v16.yaml
~~~

最后一条应无输出。新增工具的源码审阅契约还会核验这些关键算法的源码/AST 指纹。
契约不能自动改写为“当前哈希”：指纹不符时先审阅实际差异，不能为了得到 PASS 而更新预期值。
本次开发起点全部 573 个已跟踪文件的保护结果与本地测试证据见本地测试报告。

使用 Mac 已有 Pytorch 环境运行独立工具测试；每次新建测试目录，保留既有记录：

~~~bash
export AUDIT_PYTHON=/opt/anaconda3/envs/Pytorch/bin/python
"$AUDIT_PYTHON" -c 'import sys,numpy,yaml,pytest; print(sys.executable,numpy.__version__,yaml.__version__,pytest.__version__)'
export AUDIT_TEST_DIR="analysis/uicf_evidence_chain_audit/mac_tests_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$AUDIT_TEST_DIR"
"$AUDIT_PYTHON" -m pytest tests/test_uicf_evidence_chain_audit.py -q \
  --basetemp="$AUDIT_TEST_DIR/tmp" --junitxml="$AUDIT_TEST_DIR/tests.xml"
"$AUDIT_PYTHON" tools/audit_uicf_evidence_chain.py --help
~~~

测试只构造人工数字、少量标识和假 checkpoint 字节，不需要真实 LSUI、GPU 或真实权重。
这些人工元数据不是论文证据，位于 analysis 下独立的 pytest 目录并明确标识为 SYNTHETIC。

如果想检查“缺失证据”的报告路径，可以在 Mac 执行下面的入口；缺少 WSL 真实产物应返回 UNVERIFIED：

~~~bash
audit_rc=0
"$AUDIT_PYTHON" tools/audit_uicf_evidence_chain.py \
  --run-dir experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528 \
  --output-dir analysis/uicf_evidence_chain_audit/mac_missing_evidence_preview \
  --export-zip || audit_rc=$?
printf '审计退出码：%s\n' "$audit_rc"
~~~

确认只包含预期新增文件后，由你手动选择性提交。以下命令没有由 Codex 自动执行：

~~~bash
git status --short --branch
git diff
git add tools/audit_uicf_evidence_chain.py \
  tools/uicf_evidence_audit/ \
  tests/test_uicf_evidence_chain_audit.py \
  docs/UICF_EVIDENCE_CHAIN_AUDIT_GUIDE_CN.md \
  docs/UICF_EVIDENCE_CHAIN_LOCAL_TEST_REPORT_CN.md
git diff --cached --stat
git diff --cached
git commit -m "Add read-only UICF evidence chain audit"
git push
~~~

暂存列表不包含 analysis、实验目录、权重或原始分析脚本。若当前分支与预期不一致，先人工核对。

## 第二阶段：WSL 安全同步

本阶段由你手动在 WSL 执行；Codex 本次没有连接 WSL。

~~~bash
cd /home/paxton/pro/UIE4_VN
git status --short --branch
git branch --show-current
git diff --stat
git diff --cached --stat
~~~

先确认当前分支是你在 Mac 提交的分支。下面只在工作区没有未提交修改或未跟踪文件时允许快进更新：

~~~bash
if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
  echo "存在未提交或未跟踪文件，请先人工保存并核对，暂不拉取。"
else
  git pull --ff-only
fi
~~~

不自动 stash、不使用 reset --hard、不清理实验目录。快进更新失败时保留现场，人工处理分支关系。
原有被忽略的 experiments/result/checkpoint 文件无需加入本次提交。

## 第三阶段：执行只读证据核验

激活你实际安装 NumPy、PyYAML 的 Python 环境；不需要 CUDA，也不需要 LSUI 图像目录。

~~~bash
cd /home/paxton/pro/UIE4_VN
command -v python
python --version
python -c 'import numpy,yaml; print("NumPy",numpy.__version__,"PyYAML",yaml.__version__)'
python tools/audit_uicf_evidence_chain.py --help
export FULL_RUN=/home/paxton/pro/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528
test -d "$FULL_RUN"
~~~

若缺少依赖，在你选定的环境人工安装 NumPy/PyYAML 即可，不要为了此工具更换原训练环境或安装 GPU 框架。

默认自动扫描该运行 result 下一级目录，读取 protocol.json 的 script 字段分类，不按目录名猜测分析内容。
已知候选为 uicf_alignment、uicf_representation_alignment、uicf_field_intervention。
同一类发现多个产物时全部列出并检查一致性，不静默选择相关性最高的一份。
不递归扫描/解压大型文件，不读取 PNG、NPY 或图像特征。

正式只读命令如下：

~~~bash
audit_rc=0
python tools/audit_uicf_evidence_chain.py \
  --run-dir "$FULL_RUN" \
  --output-dir analysis/uicf_evidence_chain_audit \
  --export-zip || audit_rc=$?
printf '审计退出码：%s\n' "$audit_rc"
~~~

终端输出 AUDIT_STATUS、OUTPUT_DIR 和 ZIP_FILE。每次都新建 task2_时间戳 子目录；已有父目录可以使用，
已有审计子目录及 ZIP 不会被替换。输出必须在本工程 analysis 内，不允许位于数据、源实验目录或其符号链接别名中。

如果历史工具保存了 checkpoint 哈希，但基线摘要没有保存该字段，默认不能独立确证二进制同一性。
此时可人工执行下面的可选检查；只按块读取指定 best/best_psnr.pt 的 SHA256，无反序列化或推理：

~~~bash
audit_rc=0
python tools/audit_uicf_evidence_chain.py \
  --run-dir "$FULL_RUN" \
  --output-dir analysis/uicf_evidence_chain_audit \
  --hash-checkpoint --export-zip || audit_rc=$?
printf '审计退出码：%s\n' "$audit_rc"
~~~

路径迁移本身不是错误：历史 AutoDL 绝对路径与 WSL 路径可以不同。
工具分别核对运行 basename、best_psnr 文件名、epoch、冻结配置和 manifest 哈希；有可信哈希时再核对内容身份。
只有名称/epoch 相同而缺少内容身份依据时，明确保留 UNVERIFIED。
历史源码指纹与当前审阅版本不符或未保存，也会单独标记；不会自动改写历史记录或要求重跑 GPU 实验。

如果自动识别没有命中实际文件位置，可显式指定真实产物目录，参数可重复：

~~~bash
audit_rc=0
python tools/audit_uicf_evidence_chain.py \
  --run-dir "$FULL_RUN" \
  --alignment-dir "$FULL_RUN/result/uicf_representation_alignment" \
  --intervention-dir "$FULL_RUN/result/uicf_field_intervention" \
  --output-dir analysis/uicf_evidence_chain_audit \
  --export-zip || audit_rc=$?
printf '审计退出码：%s\n' "$audit_rc"
~~~

只在已确认这些目录真实存在且属于指定基线时使用显式参数；缺失时保留缺失清单，不能填入其他模型的结果。
工具按原生文件名读取 per_sample_metrics.csv、summary.json、protocol.json、failed_samples.csv；
功能干预另核对 paired_differences.csv、diagnostics.csv、COMPLETED.json。
可选次级 CSV 仅用来与主 CSV 交叉核对，不用于替代缺失的逐图来源。
未知/旧格式会标记缺口，不假定所有干预都已完成。

### 论文原文输入

本地目前没有可确认的当前 Section 4.4 和 Supplementary Table 4 原文。
默认会输出 PENDING_MANUSCRIPT，不声称已经完成论文文字核验，也不会自动要求扩充 Table 4。

后续可把实际原文保存在工程内部，再给 --paper-claims-file 指定 .md、.txt、.tex 或 JSON。
参数只记录你提供的原文与 SHA256；科学语义仍需结合报告人工核查。
JSON 支持 section_4_4、table_4_caption，以及 table_4_rows 列表，
每行使用 response（anchor_deviation / image_gradient / shifted_raw_field / raw_field）、spearman、iou。
仅对你提供的表格数字做自动交叉核对，不能用缺少关键词代替语义审阅。

### 报告如何解释

输出目录含六个文件：

~~~text
EVIDENCE_CHAIN_AUDIT_CN.md
spatial_correspondence_audit.csv
functional_intervention_audit.csv
evidence_claim_matrix.csv
audit_summary.json
source_manifest.json
~~~

报告保留不同层面的状态。numeric_status=PASS 只说明该数值重算一致，不能替代来源、协议和论文审阅的 audit_status。
均值核对绝对容差为 1e-9；任务提供的 Table 4 三位小数参考记录使用 0.0005 容差；
已知六位小数参考使用 0.0000005 容差。所有实际数值在 JSON/CSV 中保留原始浮点精度，不靠显示舍入掩盖差异。
缺失值保留空值，NaN/Inf、重复 ID、错误配对、明确哈希冲突会被显式报告。

重点检查：

- Spatial：先 RGB L2 再 patch mean；Spearman 平均并列秩；Top-20% 精确选 52/256 块；图像等权宏平均。
- 表示零对照：对池化标量幅度图作多次确定性大循环位移；与功能干预的原始 RGB R 四分之一图像位移不同。
- Functional：只干预 R，复用原 b，eval/inference-mode，无中间 Ic clamp；每种变体是否完整评估分别核验。
- 差值：PSNR/SSIM 为内部 Original−Intervention；ΔE00 为 Intervention−内部 Original。正数统一表示 Original 更好。
- CI：固定 checkpoint 的图像级配对 percentile Bootstrap，不是跨训练随机种子的稳定性证据。
- 官方测试与内部 Original 的差异单独列出，不用不同协议的官方均值替代干预内部基线。
- R 与实际校正 Delta I 的数值、定义和解释分别保留；不从相关性推导物理因果性或普遍不可替代性。

先补已有日志/协议/历史代码等缺口；不会因 UNVERIFIED 自动调用原分析脚本。任务①只作证据 A 引用，不重算其训练结果。

## 第四阶段：导出 ZIP

推荐直接在审计命令添加 --export-zip，终端的 ZIP_FILE 就是本次可提交的压缩包，
其中只有上述六份文件，不含 checkpoint、数据、图像或大型可视化。

若首次没有使用该开关，可粘贴刚才 OUTPUT_DIR 的完整值后单独导出：

~~~bash
read -r -p '粘贴本次 OUTPUT_DIR 的完整路径：' UICF_AUDIT_DIR
export UICF_AUDIT_DIR
python - <<'PY'
import os
from pathlib import Path
from tools.uicf_evidence_audit.io import ROOT, export_zip
output = Path(os.environ['UICF_AUDIT_DIR']).resolve()
assert output.is_relative_to(ROOT/'analysis'), '仅导出当前工程 analysis 下的审计'
assert output.name.startswith('task2_') and output.is_dir()
print(export_zip(output))
PY
~~~

同名 ZIP 已存在时拒绝覆盖；已经导出成功就直接使用已有 ZIP，不删除旧档。
把真实 WSL 运行生成的 ZIP 提交给 ChatGPT 后，再结合论文原文完成最终科学表述核验。
