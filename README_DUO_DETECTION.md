# DUO 检测：LSUI ICAR-UIE V16 / frozen Raw-trained YOLO11s

研究问题：在**原始 DUO train 域训练并冻结**的同一个 YOLO11s 上，LSUI 训练的固定 V16 预处理是否改变官方 DUO test 的检测性能？A=Raw，B=ICAR-UIE Full。结果可以提升、持平或下降；这是 downstream utility under a frozen-detector protocol，不代表增强器跨退化泛化。此模块不训练增强器，不实现 SIFT/分割/pose，不自动扩展 matched-domain detector 实验。

所有新增生成文件均进入 `experiments/downstream_duo/`（仓库现有 `.gitignore` 已忽略）。模块不会移动、覆盖或修改 DUO 源图/JSON、原有源码、旧实验或 checkpoint。输出目录必须新建，已有目录会报错；重跑用新名称，避免覆盖。

## 1. 已核实的数据与需要显式处理的边界问题

原样上传以下目录；**不需要在 macOS 上先转标签，不需要手工画框**：

```text
/root/autodl-tmp/pro/publicdata/DUO/
  annotations/instances_train.json
  annotations/instances_test.json
  images/train/
  images/test/
  README.txt
```

本地真实数据初次只读审计（2026-10-22，完整解码和尺寸核验）得到：

| 官方 split | JSON images / 真实文件 | 原始 annotations | 越界 bbox |
|---|---:|---:|---:|
| train | 6671 / 6671 | 63998 | 847 |
| test | 1111 / 1111 | 10517 | 106 |

953 个框均在右/下边界超出 **1 像素**。未发现坏图、文件缺失或完全相同 RGB 的跨 split 重复。原始 categories 为 `1: holothurian, 2: echinus, 3: scallop, 4: starfish`。这些是本地审计事实；云端仍会重新审计，**没有在本地训练真实 DUO detector，也没有真实 DUO mAP**。

`inspect_duo` 始终严格报告，不静默裁剪。此份数据会返回退出码 2，并留下 `dataset_audit.json`。`--bbox-policy clip` 仅将与图像有正面积交集的越界框裁到 `[0,W]×[0,H]`；每一处更改写入 `bbox_corrections.json`（split、annotation_id、image_id、category_id、原框、派生框、action）。原始 JSON 保持不变。论文应披露这一边界约定；A/B 共用完全相同的派生 GT。省略该选项则严格报错停止。

其中还有一个特殊 train 框：`image_id=3783, annotation_id=36144, category_id=2, bbox=[2951,2160,130,1]`，图像高2160，裁剪后无正面积交集。必须额外显式指定 `--empty-after-clip drop` 才会将这一框从派生标签中删除，并记录 `action=drop_empty_after_clip, derived_bbox=null`；**图像保留**。默认 `error` 会中止。零/负面积原框、NaN/Inf、重复框、缺文件、歧义路径仍中止。当前数据使用两个显式选项后应为：952 个框正常裁剪，1 个 train 框丢弃，train 63997 个派生标注，test 10517 个派生标注；完整官方图片数仍为6671/1111。

类映射由 JSON 原始 category ID 排序生成，再映射到 contiguous YOLO `[0..3]`，保存双向映射；不硬编码名称/顺序。用 JSON 文件名与真实路径唯一匹配，支持 basename、split 前缀、`images/split` 前缀；不按排序配图。重复 basename 或文件对应歧义报错。重复 numeric image ID / basename 跨官方 split 仅报告，**不会单独判定为泄漏**；内容相同的 RGB hash 才报告为内容重复。跨官方 train/test 内容重复会阻止转换，避免污染协议。

## 2. 环境：先检查 pax，避免破坏已有训练环境

先在 AutoDL 查看现有环境，不自动安装：

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
conda env list
conda activate pax
python -m src.downstream_duo.preflight --role uie \
  --data-root /root/autodl-tmp/pro/publicdata/DUO \
  --output experiments/downstream_duo/preflight_pax_uie.json
python -m pip check
```

`preflight --role uie` 验证 torch/numpy 互转和 V16 factory 可导入，记录版本/CUDA/GPU；`--role detector` 还检查固定 Ultralytics 版本和 torchvision NMS。没有 GPU 时会如实记录，CPU 可用于软件 smoke。`pax` 不存在时不要假设已有；使用实际 V16 工作环境。开发机实际没有 pax，已有 Pytorch 为 torch 2.8.0 / torchvision 0.23.0 / numpy 2.2.6，无 CUDA。

**不要在现有 pax 中直接 `pip install -U ultralytics`。** 推荐新建 detector 环境。以下是一套固定的 CUDA 12.1 依赖安装命令（需 AutoDL 驱动支持；其他 CUDA 平台选择匹配的官方 PyTorch wheel，仍保留 torch/torchvision 成对版本）：

```bash
conda create -n duo_yolo python=3.11 -y
conda activate duo_yolo
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r configs/downstream_duo/requirements-detector.txt
python -m pip check
python -m src.downstream_duo.preflight --role detector \
  --data-root /root/autodl-tmp/pro/publicdata/DUO \
  --output experiments/downstream_duo/preflight_detector.json
```

固定 `ultralytics==8.3.253`；包 API 以此版本为准。训练和 A/B 评估要求**完全相同的软件版本**，会登记和校验 torch、torchvision、numpy、Pillow、PyYAML、Ultralytics、OpenCV、SciPy、Python/平台。全部实际 YOLO 默认参数另保存在 Ultralytics `run/args.yaml`，`pip freeze` 可另存于实验目录。

若要借用已兼容的 torch 环境而不修改它，可在新目录创建 `python -m venv --system-site-packages experiments/downstream_duo/detector_env`，只向这个 venv 安装缺失的依赖；安装前后核对 torch/torchvision/numpy 版本并运行 preflight 和 `pip check`。开发验证采用这一隔离方式，实测版本见本 README 最后。

V16 推理可继续在 pax，YOLO 在 duo_yolo，通过 PNG/JSON 交换，二者无需共用 Python 环境。所有 CLI 均从仓库根运行；`--data-root` 没有硬编码 macOS 路径。

## 3. 云端审计与转换

```bash
cd /root/autodl-tmp/pro/UIE4_workspace/UIE4_VN
conda activate duo_yolo
ls /root/autodl-tmp/pro/publicdata/DUO/annotations
python -m src.downstream_duo.inspect_duo \
  --data-root /root/autodl-tmp/pro/publicdata/DUO
# 这份源数据严格审计返回 2 是已知边界问题；先查看报告，再显式转换。
python -m src.downstream_duo.prepare \
  --data-root /root/autodl-tmp/pro/publicdata/DUO \
  --seed 3520 --bbox-policy clip --empty-after-clip drop
```

审计输出为 `experiments/downstream_duo/audit/{dataset_audit.json,categories_mapping.json,gt_previews/}`。转换会重新独立审计，不依赖以前的审计缓存。准备输出：

```text
experiments/downstream_duo/prepared_duo/
  images/{train_raw,val_raw,test_raw}/<image_id>/<original_stem>.png
  labels/{train_raw,val_raw,test_raw}/<image_id>/<original_stem>.txt
  duo_raw.yaml
  manifests/{train,val,test}.{tsv,json}
  split_metadata.json
  split_metadata.sha256
  dataset_snapshot.json
  category_mapping.json
  dataset_audit.json
  bbox_corrections.json
  gt_previews/
```

训练/验证只取官方 6671 train。固定 seed=3520，约 10% 为 val；无分组时为 train=6004、val=667。test 始终是完整官方 1111 张，绝不参与训练、选择 epoch、阈值或推理超参。固定排序后以 Python 独立 Random(seed) 打乱分组，按与目标 val 数量的距离贪心选择；实际数量保存在 metadata。不能看过 test mAP 后重选划分。

同一解码 RGB 的 train 图像强制放在同组。当前数据没有可直接证明场景归属的元数据，不根据文件名猜测场景，也没有声称已排除所有近重复。若能预先确认同场景/近重复，在**看 test 指标之前**提供 train-only TSV：

```text
image_id	group
123	scene_a
124	scene_a
```

使用 `prepare ... --groups-tsv /path/train_scene_groups.tsv`。不允许 test ID 专属记录，连接的内容/场景组不可拆分；输入分组文件 SHA256 写入 metadata。相同 numeric ID 在两个官方 split 重用时，该 TSV 的 ID 明确只解释为 train ID。

所有图像用 Pillow 解码为 RGB，再无损保存为 8-bit PNG；逐张验证 PNG 解码 RGB 与源解码 RGB 完全一致。保留宽高和内容坐标，不有损重编码、不改 EXIF 几何（发现非默认 EXIF orientation 会中止）。Raw 和 Full 同为 PNG，避免 JPEG/PNG 保存格式混淆；原始 JPEG 解码本身采用统一的 Pillow RGB 约定。PNG 可能占用较多磁盘，请预留完整 Raw PNG 和 test Full PNG 的空间。

图像置于 image_id 子目录，保留原 stem，避免不同来源重名。YOLO labels 为 `class cx/W cy/H w/W h/H`，12 位有效数字，0-based 可追溯；无目标图片仍保留并生成空 txt。JSON 中其他 segmentation 字段不用于此实验；iscrowd/ignore 非零会阻止直接转换，避免悄悄改变 detection 语义。GT 框只绘制到预览副本。源图/JSON SHA256、解码 RGB hash、派生图片/label hash、JSON/TSV/YAML 快照 hash 均保存；`dataset_digest` 与逻辑 JSON/TSV 快照可复现，YAML 中绝对路径会随机器/输出目录改变。

## 4. 只训练一次 Raw detector

在任何 test 评估前固定 `configs/downstream_duo/detection.yaml`。默认 YOLO11s、640、80 epochs、batch 16、seed3520、device0；使用 Raw val 选择 `best.pt`。训练 YAML **只含 train/val，不含 test**。

```bash
conda activate duo_yolo
python -m src.downstream_duo.train_detector \
  --prepared experiments/downstream_duo/prepared_duo \
  --output experiments/downstream_duo/yolo_raw_seed3520 \
  --model yolo11s.pt --device 0 --batch 16
```

`yolo11s.pt` 是 COCO 预训练初始化；在 DUO Raw train 上的这一次训练才是下游训练。只有默认官方资产名允许首次自动下载到 `experiments/downstream_duo/pretrained/yolo11s.pt`（遵循 Ultralytics 权重许可）；离线使用 `--model /absolute/offline/yolo11s.pt`。生产模式拒绝从 YAML 随机初始化，验证 YOLO11s 架构，保存初始化文件 hash。请自行保证离线文件确为 COCO 预训练来源。固定 `train.amp=false`，拒绝启用会触发额外测试权重下载的 AMP self-check。

输出：

```text
experiments/downstream_duo/yolo_raw_seed3520/
  train_config.yaml
  train.log
  protocol.json
  raw_train_val.yaml
  run/args.yaml
  run/results.csv
  run/results.png
  run/weights/best.pt
  run/weights/last.pt
  ... Ultralytics loss / val curves and plots
```

`prepared_duo/detector_training_lock.json` 以排他方式创建，完成后登记 `best.pt` 绝对路径和 SHA256；同一 prepared 数据禁止第二次启动训练，包括换输出目录。后续评估必须指向这同一路径、同一 hash。OOM 时**不自动降低 batch 并重新训练**；根据预先显存检查在第一次训练前选好 `--batch`。若首次启动失败，保留失败日志；这次协议须人工判定无有效训练后，归档失败 run/lock，再以明确记录的新尝试继续，不能偷偷删除成功训练登记来挑 test 结果。

## 5. 固定 LSUI V16，增强整个官方 test

本地没有真实 LSUI V16 checkpoint；历史示例路径不视为存在。运行时**必须传入实际 run-dir**，代码先检查以下两文件：

```text
<UIE_RUN_DIR>/config_resolved.yaml
<UIE_RUN_DIR>/best/best_psnr.pt
```

```bash
conda activate pax
# 将下面路径替换为云端真实、已训练完成的 LSUI V16 run-dir。
UIE_RUN_DIR=/absolute/path/to/your/actual/LSUI_V16_run
test -f "$UIE_RUN_DIR/config_resolved.yaml"
test -f "$UIE_RUN_DIR/best/best_psnr.pt"
python -m src.downstream_duo.model_inference_v16 \
  --prepared experiments/downstream_duo/prepared_duo \
  --uie-run-dir "$UIE_RUN_DIR" --uie-checkpoint best_psnr \
  --device cuda:0 \
  --output experiments/downstream_duo/enhanced_full
```

独立包装器直接复用 `src.v16.models.network.build_model`，**不调用** LSUI 配对 test 入口。验证 run/checkpoint version=v16、`resolved_config.model == config.model`、sigmoid、run 与 checkpoint 数据集都标识 LSUI 且无 Haze4K 数据标识；然后 `load_state_dict(..., strict=True)`。provenance 来自可信 run 配置及 checkpoint，不声称从权重数值反推出真实训练集。仅加载可信本地 PyTorch checkpoint。`eval()`、所有参数 `requires_grad=False`、`torch.inference_mode()`，不计算或更新增强器梯度。

兼容早期缺少 `data.dataset` 的 V16 run：要求 `data.root` 和 train/validation/test 三份 manifest 路径均有明确的 LSUI 路径组件（例如 `LSUI19_dup_train`、`split/lsui19/train.tsv`），并核对 run 配置与 checkpoint 中这些来源字段一致。任一来源字段指向 Haze4K、信息不足或相互冲突均拒绝；不会通过实验目录名字猜测来源。核验方法和原始字段保存在 enhancement manifest 的 `provenance.training_provenance` 中。历史 LSUI 训练集本身不必重新上传；无需修改旧 run 配置或 checkpoint。

源码审计：V16 → shared UICFPreBackbone → V15 Color-Query U-Net → V4 PlainUNet，四次 pooling，`padding_factor=16`。包装器以原生 RGB `[0,1]` 输入，在右/下补到16倍数（reflect，不满足小图约束时 replicate），准确裁回；不进行256缩小/放大。padding 在整条 UICF+backbone 前应用，固定边界语义并记录。输出 sigmoid `[0,1]`，检查有限数值、原尺寸，`round(255*x)` 保存 RGB PNG。

默认 `--tile-size 0` 为原生全图模式。OOM 会失败并写日志，**不会隐式切换推理协议**。如果在预先显存试运行中确定需要 tiled 推理，在独立的新输出目录显式指定，例如：

```bash
python -m src.downstream_duo.model_inference_v16 \
  --prepared experiments/downstream_duo/prepared_duo \
  --uie-run-dir "$UIE_RUN_DIR" --uie-checkpoint best_psnr --device cuda:0 \
  --tile-size 1024 --overlap 128 \
  --output experiments/downstream_duo/enhanced_full_tiled
```

tile 与 overlap 需在查看 test mAP 前固定，不能用 test mAP 选择。重叠 Hann 权重加正下限，逐像素归一化，CPU 累积；覆盖边缘且不移动坐标。**UICF 空间编码、全局 anchor 和 CQ attention 在 tile 内重新计算，因此 tiled V16 与全图 V16 不是数值等价算法**；必须报告实际推理协议。单测中的 identity tile 回归只验证拼接没有坐标/亮度错误，不证明 V16 tile/full 等价。

输出 `enhanced_full/images/test_full/<image_id>/<stem>.png`、`labels/test_full/`（与 Raw test 标签逐字节一致）、`enhancement_progress.jsonl`、`enhancement_manifest.json`。记录每图 ID、尺寸、耗时、成功/失败、输出/标签 hash、checkpoint hash、版本、padding/tile 参数。任一失败会中止且 manifest `complete=false`，后续 Full 评估拒绝继续；不允许少图评估。可用新目录从头重试，不覆盖旧失败产物。

本轮没有可用的同协议 LSUI w/o UICF 权重，因此不包含 C。不接受用 Haze4K 权重代替 C，也不为此重训增强器。

## 6. 同一冻结 detector 的 A/B 评估、表格、Figure

```bash
conda activate duo_yolo
bash scripts/downstream_duo/eval_compare.sh \
  experiments/downstream_duo/prepared_duo \
  experiments/downstream_duo/yolo_raw_seed3520/run/weights/best.pt \
  experiments/downstream_duo/enhanced_full \
  experiments/downstream_duo/evaluation 0
```

若采用 tiled 版本，将第三参数替换为 `enhanced_full_tiled`。脚本执行以下真实 CLI（也可逐条运行）：

```bash
python -m src.downstream_duo.evaluate --condition raw \
  --prepared experiments/downstream_duo/prepared_duo \
  --weights experiments/downstream_duo/yolo_raw_seed3520/run/weights/best.pt \
  --output experiments/downstream_duo/evaluation/raw --device 0
python -m src.downstream_duo.evaluate --condition full \
  --prepared experiments/downstream_duo/prepared_duo \
  --weights experiments/downstream_duo/yolo_raw_seed3520/run/weights/best.pt \
  --enhanced experiments/downstream_duo/enhanced_full \
  --output experiments/downstream_duo/evaluation/full --device 0
python -m src.downstream_duo.compare \
  --raw experiments/downstream_duo/evaluation/raw/metrics_raw.json \
  --full experiments/downstream_duo/evaluation/full/metrics_full.json \
  --output experiments/downstream_duo/evaluation/comparison
python -m src.downstream_duo.visualize \
  --raw experiments/downstream_duo/evaluation/raw/metrics_raw.json \
  --full experiments/downstream_duo/evaluation/full/metrics_full.json \
  --output experiments/downstream_duo/evaluation/figures \
  --conf 0.25 --rows 3 --cell-width 800 --gt
```

固定 Ultralytics val、`split=test`、imgsz640、conf0.001、NMS IoU0.7、max_det300、batch1、rect=true、half=false、无 TTA、class-aware NMS；相同 resize/letterbox/GT/classes/test IDs。训练登记中锁住评估配置；A/B 比较还会核对全部参数、软件版本、路径/hash、图像列表/尺寸/GT hash，拒绝不公平比较。设备可通过 CLI 显式指定，但 A/B 必须相同。

指标均为整集 Ultralytics detection val 的 0–1 值：Precision/Recall 为类别平均、在内部平滑 mean-F1 最大工作点读取；AP50 为 IoU=.5，AP50:95 为 `.50:.05:.95` 的平均，再按有 GT 的类别平均。AP 对整集置信度排序和 PR 曲线积分，**不是每图 AP 的平均**。Precision/Recall 的内部工作点可能因条件不同而不同，这是库的报告约定，不能写成固定 `conf=.25` 的 P/R；Figure 的 .25 仅用于显示。没有自动做 bootstrap CI；若将来增加，必须按测试图像重采样并重新汇总整集 AP。

输出位置：

- `evaluation/raw/metrics_raw.json`、`evaluation/full/metrics_full.json`：整集 P/R/mAP、逐类指标、协议、权重/预测文件 hash。
- `evaluation/{raw,full}/predictions_{raw,full}.json`：**每张图都有记录，包括空预测**；保留官方 image_id、stem路径、原图宽高，检测框为原图坐标 xyxy，同时保存 YOLO class_id、原 COCO category_id、confidence。
- `evaluation/{raw,full}/validator/predictions.json`：原 COCO category_id 和 image_id 的 xywh 预测，由相同 metric pass 的 post-NMS 预测导出；无需第二次 predict 导致不一致。validator 另保存 PR 曲线、混淆矩阵和验证预览。
- `evaluation/comparison/comparison.{json,csv,md}`：总体和分类别 A/B 与 B−A；支持负值。
- `evaluation/figures/detection_{improved,unchanged,worsened}.{png,pdf}`、各图 native PNG、可选 GT 图和 `figure_selection.json`。两列标题固定为 `(a) Detection on Raw`、`(b) Detection on ICAR-UIE`，相同类别颜色、框宽2、confidence小数2位、显示阈值.25。没有该类型样本时不制造示例，selection 中明确空列表。

Figure 默认按 display conf 下 class-aware IoU=.5 贪心匹配的逐图 F1 差值选改善/不变/变差各至多3行（并保存所有候选排名）；**此 F1 只用于案例定位，不是定量 AP，也不以框更多判断改善**。可用 `--image-ids 1 20 300` 指定另外一张 manual figure，所有 ID 必须来自官方 test。定量结论始终来自完整1111张。native 图绘框后按同一比例缩放用于版面，不改模型输入。DUO 只有 JSON 中的四类，不添加潜水员或虚构框；此节不额外制作 Raw/Enhanced 无框两列。

## 7. 自动测试与开发验证

```bash
conda activate duo_yolo
python -m pytest -q tests/test_downstream_duo.py
# 显式执行1 epoch YOLO11s随机初始化 + 随机tiny V16 + A/B eval + 比较/图：
YOLO_OFFLINE=true python -m src.downstream_duo.smoke \
  --output experiments/downstream_duo/smoke_cloud
# 或在 pytest 中一并执行集成测试和重复训练/hash/公平比较保护：
DUO_RUN_SMOKE=1 YOLO_OFFLINE=true python -m pytest -q tests/test_downstream_duo.py
```

测试覆盖：非连续 COCO IDs/映射、非法/越界/重复标注、路径歧义/坏图/缺图、显式裁剪/空交集显式丢弃与原文件不变、固定划分及快照 hash、空标签、GT副本、真实 factory strict checkpoint 载入及拒绝错误版本/架构/Haze4K/缺 key、任意宽高与极小图、tile identity 拼接及 tiny V16 tile 输出、完整增强计数和失败门禁、一次性训练、同路径/hash 冻结权重、逐图 test ID/GT 一致、结果表再现。

软件 smoke 采用自己生成的 COCO 数据、随机初始化 YOLO11s、缩小通道的随机 V16 checkpoint，所有结果带 `synthetic=true`，Markdown 表显著标记 **NOT PAPER RESULTS**；其 checkpoint 的 LSUI_SYNTHETIC_SOFTWARE_TEST 只是测试 fixture，不代表真实 LSUI 训练。真实 DUO 生产模式不接受 synthetic prepared 标记作为训练输入。

开发验证在 macOS Apple M4 CPU 的隔离 venv 中完成，Python3.12.11 / torch2.8.0 / torchvision0.23.0 / numpy2.2.6 / Ultralytics8.3.253。上面的 CUDA2.5.1 固定环境配方需在 AutoDL 实际 preflight；未在此开发机声称通过 GPU 验证。真实 LSUI best_psnr 与云端 GPU 实验仍需用户提供路径后执行。最终测试结果和开发产物见本次交付说明。

参考：[固定 Ultralytics 源码 tag](https://github.com/ultralytics/ultralytics/tree/v8.3.253)、[DetectionValidator](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/models/yolo/detect/val.py)、[AP实现](https://github.com/ultralytics/ultralytics/blob/v8.3.253/ultralytics/utils/metrics.py)、[YOLO11官方说明](https://docs.ultralytics.com/models/yolo11/)。

## 8. 新增文件清单

```text
src/downstream_duo/
  __init__.py
  inspect_duo.py
  prepare.py
  model_inference_v16.py
  train_detector.py
  evaluate.py
  compare.py
  visualize.py
  utils.py
  preflight.py
  smoke.py
configs/downstream_duo/
  detection.yaml
  requirements-detector.txt
scripts/downstream_duo/
  preflight.sh
  train_detector.sh
  enhance_test.sh
  eval_compare.sh
tests/test_downstream_duo.py
README_DUO_DETECTION.md
```

以上19个文件均为新增。生成数据、synthetic 权重、venv 和审计产物未纳入 Git。
