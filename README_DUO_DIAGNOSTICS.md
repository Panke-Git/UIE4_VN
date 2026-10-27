# DUO / ICAR-UIE P0 下降诊断：原实验只读

P0入口调查固定 Raw-domain YOLO11s 下的检测下降，先排除工程/I/O/标注/评测协议问题，再测独立分块推理与 full-native 的差异。**P0不训练、不微调、不改阈值、不改原数据或旧实验。** 可选P1有独立的显式启动入口（第8节），不会被P0自动执行。

## 已给定真实结果：始终保留

| 整个官方 test，1111图 / 10517框 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Raw | 0.890866 | 0.768893 | 0.867808 | 0.698035 |
| LSUI V16 Tiled Full | 0.750440 | 0.500597 | 0.582162 | 0.415379 |

这些是用户已提供的主实验结果，不是诊断程序新测量值。所有报告分别保存 `given_negative_results` 和实际读取的 `measured_existing_results`。子集 AP、Identity AP、合成 smoke 指标不会替换主实验结果。

## 1. 当前能审计到什么

开发环境实际是 macOS 本地仓库 `/Users/paxton/Project/PythonProject/02_CL_Papers/UIE4_workspace/UIE4_VN`，并非 AutoDL 的 `/root/...` 执行环境。本地存在真实 DUO 原图，但缺少云端实际 trained YOLO、真实 LSUI checkpoint、1111张增强输出和主实验评估目录。不能声称本地已经实测真实 Identity mAP 或真实 native OOM。云端执行后才能完成这些诊断。

程序在每次运行时读取**实际当前源码**，保存完整文本至 `source_snapshot/` 并记录绝对路径/SHA256，涵盖现存 `src/downstream_duo/` 核心、`scripts/downstream_duo/`、`configs/downstream_duo/`，以及 V16 factory、UICF wrapper/INR、Color-Query、V4 padding、V16 paired test 和工具。新诊断代码另记录 `diagnostic_source_hashes.json`。以云端新生成的源码快照为准，不以旧上传日期推测代码内容。

当前源码已确认的处理链：

1. 原 DUO 经 Pillow RGB 解码，无损保存到 prepared Raw PNG。
2. DUO 包装器从 Raw PNG 读取 RGB → float32 /255 → BCHW。
3. 每个 tile 独立对完整 V16 执行 right/bottom factor16 padding（reflect 或 replicate）→ `model.uicf` → `model.backbone` → 原 tile 精确裁回。
4. 原分块使用二维 Hann 权重、每轴下限0.01、逐像素归一化，CPU累积；不缩放内容。
5. 输出 clamp[0,1] → round(255*x) → uint8 RGB PNG，原尺寸；标签逐字节复制。
6. 当前 DUO wrapper 没有 autocast；V16 paired test 的 AMP 配置不等于 DUO 的实际 AMP。原 manifest 未显式记录的历史 dtype/AMP/source hash 会标为 unknown；新推理实际记录 FP32/no AMP。原始 tiled 重放用于检查当前源代码/环境是否能复现旧图。

UICF 源码中的 normalized coordinates 来自每次调用的 H/W 上的 `linspace(-1,1)`；GAP `g` 和 chromatic anchor `b` 每次调用重算。因此独立 tile 的条件化与整图不同，CQ context 也随 tile 改变。**这只是源码层面的非等价性，不能直接证明是约28pp下降的主因，更不能凭理论宣布已经存在接缝。**

## 2. 环境检查、上传和单测

只新增：

```text
src/downstream_duo/diagnostics/
  __init__.py
  common.py
  audit_protocol.py
  evaluator.py
  identity_control.py
  compare_full_vs_tiled.py
  report.py
  run_p0.py
scripts/downstream_duo/diagnostics/run_p0.sh
tests/test_duo_diagnostics.py
README_DUO_DIAGNOSTICS.md
```

原 `.gitignore` 中已有不带根路径限定的 `diagnostics/`，所以它也会忽略上面两个新源码目录。**本任务不改旧 .gitignore，使用 Git 同步源码。** 首次纳入版本管理时，仅对新增源码文件执行强制添加；文件一旦被 Git 跟踪，后续修改会正常显示，不需要再次强制添加。实验输出仍然忽略，不提交数据或模型。

```bash
git add -f -- src/downstream_duo/diagnostics/*.py \
  scripts/downstream_duo/diagnostics/run_p0.sh
git add -- tests/test_duo_diagnostics.py README_DUO_DIAGNOSTICS.md
git diff --cached --stat
```

确认共11个新增文件后，按现有分支流程提交并推送，再在AutoDL同一仓库执行`git pull`。若诊断已在运行，等待运行结束后再同步源码，以免运行中版本发生变化。

以下命令从真实 AutoDL 仓库根目录执行；保留现有环境，不需要安装/升级 torch/numpy：

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
conda activate duo_yolo
export PYTHONDONTWRITEBYTECODE=1

CHECK="experiments/downstream_duo/diagnostics/environment_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECK/ultralytics/Ultralytics" "$CHECK/matplotlib"
export YOLO_CONFIG_DIR="$PWD/$CHECK/ultralytics"
export MPLCONFIGDIR="$PWD/$CHECK/matplotlib"
export YOLO_OFFLINE=true

python -m src.downstream_duo.preflight --role detector \
  --data-root /root/autodl-tmp/pro/publicdata/DUO
python -m pip check

python -m pytest -q tests/test_duo_diagnostics.py \
  -o cache_dir="$CHECK/pytest-cache"

/root/miniconda3/envs/pax/bin/python -m src.downstream_duo.preflight --role uie
bash scripts/downstream_duo/diagnostics/run_p0.sh --help
```

训练/原评测用的 detector 环境必须保持一致（已知 Ultralytics8.3.253、torch2.5.1+cu121）；native/replay 由原增强使用的 pax 执行，检查 torch/numpy/Pillow 与旧 enhancement manifest 一致，避免将环境差异当成 tiling 差异。`UIE_PYTHON` 可设置路径默认值，`UIE_RUN_DIR` 可导出作为 run-dir 默认值；以下明确给参数。

上述环境目录将原 preflight 导入 Ultralytics 时可能生成的配置与绘图缓存也放在新的诊断目录，避免写入原实验配置目录。单测需要已有 `pytest`，不要求安装或升级任何检测/增强模型依赖。

## 3. P0 一键命令

输出必须是新目录，**不要提前创建 RUN 目录**。所有参数和测试阈值固定，sample-count 默认32，可预先选择64；不看检测表现重抽样。

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
conda activate duo_yolo
export PYTHONDONTWRITEBYTECODE=1

RUN="experiments/downstream_duo/diagnostics/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"

bash scripts/downstream_duo/diagnostics/run_p0.sh \
  experiments/downstream_duo/prepared_duo \
  experiments/downstream_duo/yolo_raw_seed3520/run/weights/best.pt \
  experiments/downstream_duo/enhanced_full_tiled \
  "$RUN" \
  --data-root /root/autodl-tmp/pro/publicdata/DUO \
  --evaluation-root experiments/downstream_duo/evaluation \
  --uie-run-dir /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528 \
  --uie-python /root/miniconda3/envs/pax/bin/python \
  --gpu 0

cat "$RUN/diagnosis_summary.md"
less "$RUN/audit/audit_protocol.md"
cat "$RUN/runner_status.json"
cat "$RUN/input_integrity_after.json"
```

停止关卡会返回退出码2，仍然落盘报告；这代表必须先处理失败/未知项，不应删掉错误或改 test 参数继续。输入缺失/环境不一致同样会报告。即使 native 全部 OOM，脚本也会保留 OOM 证据和计划子集的 Raw/Tiled AP，明确 native 未评估。

原始输入目录与模型会运行前后作 SHA256 检查，包括发现原目录出现新缓存文件。哈希整树可能耗时；读取不修改输入。YOLO、Matplotlib 配置、字体/绘图缓存及验证数据 `.cache` 均定位到新诊断目录。评估使用私有图像/标签副本，不对原数据建立可能被写回的 hardlink/symlink。需要额外磁盘存放1111张 Identity PNG及验证副本、固定子集的重放/原生图和检测副本。

## 4. P0-A 协议、数据和数值检查

输出 `audit/audit_protocol.{json,md}`：

- 独立解析原 COCO test；检查 image/annotation/category IDs，显式使用已记录的 clipping/drop 策略，重建期望 bbox，再与每图 YOLO 坐标作同类别、一一匹配比较（绝对误差<=1e-5原图像素，不依赖框的排列顺序）。校验实际更改的 annotation ID 列表与准备阶段更改日志一致。
- 每张图验证源路径唯一对应、原始 file_name、prepared stem、官方 split、Raw/Full ID集合、manifest记录、真实文件集合、尺寸、RGB PNG模式、图像和标签 hash。逐图比较原始 RGB 解码与 prepared Raw PNG；允许的工程解码差异上限为1个uint8级，并完整记录非零差异，不武断认定是库版本造成。Full label与Raw label字节一致。不同 split 的 numeric ID重用不单独判泄漏。
- 原始 COCO JSON与准备快照逐条核对；两个 prediction 导出检查 ID、类别和图像几何；空预测图依然必须出现在逐图导出中。COCO validator导出允许省略空预测图。
- 读取原 Raw/Full metrics JSON，核对同一 YOLO path/hash、参数、版本、整集 identity/GT hash。原始训练登记的 evaluation参数也要匹配，直接读取实际 best.pt，记录 detector task/classes/yaml，不下载替代模型。
- 读取真实 V16 checkpoint/config hash，和原增强 manifest核对。当前源码事实与历史 manifest没记录的字段分开，未知字段不填默认值伪装为事实。
- 全部增强图解码，记录 RGB均值/标准差、范围、黑白饱和比例、纹理梯度、像素内容重复组、Raw/Full RGB差异和 BGR替代差异。保留每图记录、像素加权整体统计、极端样本表；这些提示不自动成为故障结论。
- 固定 seed3520 抽30张，在 Raw与Full副本上各画GT框，共最多60张。**人工查看是否错位仍是必要核验，自动尺寸/标签通过不等于已完成语义对齐的人工审查。**

## 5. P0-B Identity（归因前置关卡）

用 `torch.nn.Identity().eval()` 替换增强调用，直接复用原 `enhance_test()` 的 PNG读取、float32归一化、原 tile/overlap、padding、融合和保存逻辑。**不加载/调用ICAR-UIE**。Identity在CPU执行；原 B同样先读 prepared Raw PNG，它已与原始官方图像解码像素逐图核验一致。

输出 `identity/test_identity/images/test_full/`、对应 labels、每图与原始/Raw PNG 的 MAE/max-error、`identity/metrics_identity.json`。使用原冻结 YOLO、原参数、相同1111图ID进行独立评价，完整有效参数保存在 `effective_validator_args`。

以**原检测器实际评估的 prepared Raw PNG**为严格像素参考，Identity decoded RGB应完全一致，PNG文件压缩字节不要求一致。同时逐图比较直接解码的官方源JPEG；两个环境Pillow/torch版本及任何微小解码差异分别记录，不将其自动归为错误或作为下降原因。

若像素出现非预期差异，或 `abs(Identity−Raw mAP50)>0.005` / `abs(Identity−Raw mAP50-95)>0.005`，停止P0-C与模型归因，保存结果并检查I/O/映射/评测。0.005是工程告警门槛，不是统计显著性判断。异常本身也不自动证明具体原因。

## 6. P0-C 固定子集 native / tiled

抽样先考虑类别出现覆盖与实际宽高覆盖，再以seed3520补足32个，完全不用检测预测。名单只写一次至 `sample_ids.json`，并保存SHA256。它用于诊断而非整个测试集的代表性估计。预先选择64可加 `--sample-count 64`。

对每张未缩放原图：

1. 重放原 tiled（同checkpoint、原tile/overlap、FP32、padding/crop），与旧增强PNG做像素对比。旧图不能复现到1个uint8级别内时先调查环境/源码历史，不能把差异直接归为分块条件化。
2. 真正整图一次推理，保持原宽高，不resize。若OOM，记录完整错误、图像尺寸、GPU型号、总显存、peak allocated/reserved；标记native未评估，没有替代缩放图。逐图清理缓存但不自动减小图像或换precision。
3. hooks观测每次UICF输入尺寸、normalized coordinate H/W、GAP g和anchor b；不改forward结果或模型权重。记录每张图tile个数、每块anchor范围。
4. 计算native/tiled预测之间的MAE、PSNR（非GT恢复PSNR）、RGB统计/梯度差异；tile开始/结束边界附近梯度变化以原始天然边缘和非边界区域作对照。统计不会自动宣称存在可视接缝。
5. 保存所有固定样本的Raw/Tiled/Native overview、Raw/Tiled半透明叠图、固定tile边界裁剪。Overview/crop为展示缩放，**模型输入始终不缩放**。人工审图状态默认pending。
6. YOLO分别评估计划子集Raw/Tiled，以及**同一 native-success ID交集**的Raw/Tiled/Native。完整32图若有OOM，不伪造Native AP；成功交集属于硬件筛选，不能外推整个1111图。若只有单tile小图可native，报告明确不能据此排除大图tiling问题。

报告预设描述性工程门槛：mean native/tiled MAE>1 uint8级，或同子集AP绝对差>0.005时提示优先调查tiling；这些不是统计显著性或28pp因果贡献的证明。

## 7. 输出和决策

```text
<RUN>/
  source_snapshot/                       # 精确原源码文本
  source_inventory.json
  diagnostic_source_hashes.json
  availability.json
  input_integrity_before.json
  input_integrity_after.json
  runner_status.json
  audit/
    audit_protocol.json
    audit_protocol.md
    extreme_samples.csv
    gt_previews/
    source_snapshot/
  identity/
    identity_control.json
    identity_control.md
    metrics_identity.json
    test_identity/{images,labels}/test_full/
    evaluation/{dataset,validator}/
  full_vs_tiled/
    sample_ids.json
    sample_ids.sha256
    full_vs_tiled.json
    full_vs_tiled.md
    subset_metrics.json
    native/
    recomputed_tiled/
    figures/
    evaluation/{planned,native_common}/
  full_vs_tiled_inference.log
  diagnosis_summary.json
  diagnosis_summary.md
```

在前置关卡停止时，后续目录/指标不会生成，summary明确标为not_run/not_evaluated；不能把缺失解释成0。保存原真实负面主结果不变。

决策码包括：

- `stop_protocol_or_availability`：对齐/协议/权重未知或失败，先解决明确问题。
- `stop_identity_pipeline`：Identity异常，停止其它归因。
- `original_tiled_replay_mismatch`：旧分块图未重现，先查环境/历史源码。
- `native_unavailable_cause_undetermined`：native不可评估，原因未确定。
- `native_evidence_only_single_tile_images`：仅单tile小图可native，无法排除大图分块问题。
- `tiling_difference_supported_on_subset`：同子集确有差异，优先查分块，但不量化成整集28pp的主因。
- `consistent_with_domain_or_task_information_change_on_subset`：Identity通过、native/tiled接近、native仍低于Raw，和域/任务信息变化一致；两种机制尚不能区分。
- `cause_undetermined`：现有检查无法定位，诚实保留不确定性。
- `stop_input_integrity_failure`：发现受保护输入变化，诊断结果需调查，不会自动还原/覆盖源文件。

查看：

```bash
cat "$RUN/diagnosis_summary.md"
less "$RUN/audit/audit_protocol.md"
cat "$RUN/identity/identity_control.json"
less "$RUN/full_vs_tiled/full_vs_tiled.md"
cat "$RUN/full_vs_tiled/subset_metrics.json"
tail -n 30 "$RUN/full_vs_tiled_inference.log"
```

## 8. 可选P1：显式启动增强域匹配训练

P1回答：检测下降是否与YOLO只在Raw域训练、不能适应增强域有关。固定LSUI V16，不训练增强器；沿用已经冻结的增强checkpoint、tile/overlap、FP32和原尺寸规则，只新生成官方train内原有6004/667划分的增强图。已有1111张`enhanced_full_tiled`测试图直接复用，不重新生成或选参数。

新YOLO从**与Raw基线完全相同SHA256的COCO预训练文件**初始化，使用原登记训练参数（80 epochs、imgsz640、seed3520、原batch/optimizer/增强默认项等），在增强val上选best.pt，然后冻结同一个新best.pt评价Raw test与Enhanced test。保留原有R→R、R→E负面结果，得到：

| 检测器训练域 | Raw test | Enhanced test |
|---|---|---|
| Raw | 原R→R，保留 | 原R→E，保留 |
| Enhanced | 新E→R | 新E→E |

E→E恢复支持域失配是重要因素，但不能证明没有任何信息损失。E→E仍低也不能直接证明增强损坏信息，还需检查训练/优化。一组seed不提供统计显著性结论。P1改变了检测器训练域，论文应与原“冻结Raw检测器”协议分开解释。

新增实现：`src/downstream_duo/diagnostics/p1.py`、`scripts/downstream_duo/diagnostics/run_p1.sh`。**本地仅执行了合成软件测试，真实DUO P1未启动。** 真实启动要求用户在AutoDL命令中显式加入`--start-training`；P0脚本不会调用P1。不会重写原`prepared_duo/detector_training_lock.json`。

### 8.1 Git同步与生成计划（不推理、不训练）

P1源码目录仍被原`.gitignore`忽略，首次同步需明确加入新增文件：

```bash
git add -f -- src/downstream_duo/diagnostics/p1.py scripts/downstream_duo/diagnostics/run_p1.sh
git add -- README_DUO_DIAGNOSTICS.md tests/test_duo_diagnostics.py
git diff --cached --stat
# 确认后按现有分支流程commit/push；AutoDL执行以下命令。
```

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
git pull
conda activate duo_yolo
export PYTHONDONTWRITEBYTECODE=1

df -h .
P1="experiments/downstream_duo/diagnostics/p1_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"

python -m src.downstream_duo.diagnostics.p1 plan \
  --prepared experiments/downstream_duo/prepared_duo \
  --raw-weights experiments/downstream_duo/yolo_raw_seed3520/run/weights/best.pt \
  --enhanced-test experiments/downstream_duo/enhanced_full_tiled \
  --p0-run experiments/downstream_duo/diagnostics/20261025_110654 \
  --uie-run-dir /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN/experiments/v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528 \
  --output "$P1"

cat "$P1/training_config_snapshot.yaml"
cat "$P1/p1_summary.md"
```

不要预先创建`$P1`。计划会核对已完成P0的协议/Identity/完整性、原train/val清单SHA256、真实Raw best.pt、原COCO初始化文件、LSUI checkpoint、版本、测试增强和固定参数，并保存原输入树哈希。这一步可能耗时，但不加载模型做推理或训练。若原COCO初始化文件已移动，添加`--initial-weights <实际文件路径>`；仍必须与Raw登记的初始文件SHA256一致，不自动下载替代文件。

无需创建新环境，也不升级torch/numpy。检测训练/评估使用原`duo_yolo`；增强使用原`pax`。原始DUO/LSUI训练图不需要重新上传；P1读取prepared Raw PNG和已有LSUI权重。需要额外空间保存6671张原尺寸增强train/val PNG、检测训练产物及两组私有test评估副本；不自动缩图节省空间。

### 8.2 用户显式启动P1（会新增一次YOLO训练）

在生成计划的同一终端执行；这一命令明确进入P1，不需要再次运行Raw训练或P0。

```bash
nohup bash scripts/downstream_duo/diagnostics/run_p1.sh \
  "$P1" \
  --uie-python /root/miniconda3/envs/pax/bin/python \
  --gpu 0 \
  --start-training \
  >> "$P1/launch.log" 2>&1 &

echo "P1 PID=$!"
echo "P1目录=$P1"
tail -f "$P1/launch.log"
```

`nohup`后台运行，退出`tail -f`或关闭该终端不终止已启动的后台任务。中途不要更新Git源码或修改计划；运行各阶段会核对源码/计划哈希。脚本依次检查→原pax增强train/val→原duo_yolo训练新检测器→同新权重评估E→R/E→E→生成报告。V16始终eval/frozen/inference_mode，不用test选epoch、阈值或增强参数。

完成训练的同一个P1目录再次调用不会重新训练；若训练已启动却没有成功完成，持久训练锁会拒绝自动重训/resume，先检查错误。增强阶段可复用已逐图记录且哈希正确的成功输出；残留active锁或未登记文件会要求人工核对，不自动覆盖未知产物。测试图与原标签都保持只读；新训练锁、缓存和日志只在P1目录。

### 8.3 找回终端与查看结果

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
P1=$(ls -d experiments/downstream_duo/diagnostics/p1_20??????_??????/ | sort | tail -n 1)
echo "$P1"
tail -n 60 "$P1/launch.log"
cat "$P1/p1_summary.md"
cat "$P1/input_integrity_after.json"
```

最终输出：

```text
<P1>/plan.json + plan.sha256
<P1>/training_config_snapshot.yaml
<P1>/launch.log
<P1>/enhanced/{images,labels}/{train_enhanced,val_enhanced}/
<P1>/enhanced/enhancement_progress.jsonl
<P1>/enhanced/enhancement_manifest.json
<P1>/detector/registry.json
<P1>/detector/run/weights/best.pt
<P1>/detector/run/results.csv
<P1>/evaluation/raw/metrics_raw.json       # E→R
<P1>/evaluation/full/metrics_full.json     # E→E
<P1>/evaluation/{raw,full}/per_image_predictions.json
<P1>/p1_summary.{json,md}
<P1>/input_integrity_{before,after}.json
```

`p1_summary.md`计划阶段就存在，新结果未运行时标为not evaluated；最终`p1_summary.json`的status=complete且输入完整性pass才表示整个P1完成。细分阶段可以手动调用`python -m src.downstream_duo.diagnostics.p1 <check|enhance|train|evaluate|report> --help`查看真实参数，训练子命令同样要求`--start-training`。

## 9. 本次开发验证状态

本地已实际执行38项自动测试（其中1项核验已运行的合成smoke收据），以及5个Python入口的`--help`、shell语法与帮助检查。普通云端单测没有本地合成收据时为37项通过、1项跳过。另用3张合成图、已有的合成YOLO权重和随机小型V16，在CPU上跑通协议审计、Identity、tiled重放、full-native、三条件子集评估和最终报告；本次没有训练模型。合成指标不用于论文或真实下降归因。

软件smoke前后126个受保护文件的SHA256一致，未发现新增源目录缓存。`git diff --exit-code`确认原有受版本控制文件无变更。真实1111张Identity评估、真实LSUI V16 native GPU推理、实际OOM证据及真实子集AP仍待AutoDL运行；本地报告没有将这些未运行项伪装成测量结果。

### 2026-10-25：修正诊断器的框排序误报

云端首轮`20261025_001808`在26张图上停止，原输入完整性检查通过（25,695文件未改变）。本地用真实官方test COCO的10,517个框按原prepare的12位有效数字转换，再还原像素坐标，复现了完全相同的26个失败ID。逐框实际往返误差最大约1.92e-9像素；旧诊断器对两组框各自排序后逐行比较，相同x坐标受浮点尾差影响会交换排序，造成错误配对。

修正仅影响新诊断模块：同类别bbox改用容差内的一一二部图匹配，仍保持原来的1e-5像素绝对容差，不复用同一参考框，不更改原标注或旧实验。新增排序、重复匹配、歧义候选、无效输入回归测试。新audit JSON记录`bbox_comparison.method`及容差。旧云端审计目录保留，更新Git代码后用新的timestamp目录重跑P0；真实云端标签仍需重审，不能把本地标注往返复现说成云端已通过，也不能将此诊断器误报解释为原mAP下降的原因。

修正后实际运行42项单测通过（含本地合成smoke收据核验）；云端默认运行应为41项通过、1项收据检查跳过。本地10,517个真实COCO框的标注往返模拟中，修正后1111张全部匹配通过，原始JSON的SHA256前后不变。

### 2026-10-26：P1软件验证

新增P1后实际执行56项测试通过，包含P0与P1两份合成集成收据；AutoDL默认没有本地收据时为54项通过、2项跳过。合成P1在CPU执行6张train/2张val的固定V16增强、新YOLO 1 epoch训练、同best.pt对3张Raw/Enhanced test的两次评价及四组汇总。合成初始化为本地随机YAML配置，不是COCO预训练，因此明确标为software-only，不能解释真实P1效果；真实CLI拒绝synthetic数据且强制原COCO初始化SHA256。

合成P1输入保护检查通过，额外核验336个原有合成测试文件无改变/新增文件；再次调用已完成训练未重训。真实DUO的6671张train/val增强与80 epoch P1尚未运行。只增加P1专用模块/脚本，并更新本README和诊断测试，未修改原Raw训练入口或模型代码。
