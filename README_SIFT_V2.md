# SIFT 参考特征一致性 V2（WSL）

独立入口 `tools/sift_reference_consistency_v2.py`，辅助代码与测试在 `tools/sift_ref_v2/`。只读取保存的 RGB 图片；不导入 PyTorch、不读取网络权重、不训练或推理。V1 的训练、测试、SIFT 代码和 `sift_gt_saved` 结果不被修改。

当前配置 `configs/sift/reference_v2.home_wsl.json` 已按来源审计填入 WSL 路径，包括 Raw、MARINE-Net、SWLGM、ICAR-UIE 和额外 IMIF 对照。IMIF 为 Python compatibility port，原作者核心是加密 MATLAB P-code，报告会保留来源/实现限制，不能宣称等同作者原程序。没有可靠的 w/o UICF LSUI 428 张输出，因此未加入该条件。

## 环境与执行顺序

在 Mac 用 `ssh home-wsl` 连接后：

```bash
cd /home/paxton/pro/UIE4_VN
conda activate duo_yolo
python -m pytest -q tools/sift_ref_v2/test_reference.py
bash scripts/run_sift_reference_v2_wsl.sh dry-run
bash scripts/run_sift_reference_v2_wsl.sh smoke
bash scripts/run_sift_reference_v2_wsl.sh full
```

脚本默认使用 `/home/paxton/miniconda3/envs/duo_yolo/bin/python`，可用 `SIFT_V2_PYTHON` 指定另一解释器。现有环境已具备依赖；`requirements-sift-v2.txt` 记录测试版本，无需重新安装。OpenCV 线程设置必须实际报告为 1，不能只调用设置函数后假定成功。

每次输出到新的 `experiments/eval_sift_reference_v2/<stage>_<北京时间>/`，也可把新目录作为脚本第二个参数。**已有目录绝不覆盖，不提供 --overwrite。** `dry-run` 只解码、核对和哈希全部 428 张输入；`smoke` 固定 10 个 ID（四个指定异常案例加六个预先抽查的 ID），清楚标为部分结果；`full` 使用全部 428 张。

完整运行内部分两阶段：先只提取 GT/Raw/ICAR 特征并逐图复现历史 unlimited；回归失败则保存差异清单并停止，不运行 SOTA 或固定 K。通过后才开始其余方法和固定预算。也可仅运行 `bash scripts/run_sift_reference_v2_wsl.sh regression`。

完成后将完整输出目录写入环境变量，可运行第 8 项真实数据回归测试（普通软件测试明确跳过此项）：

```bash
SIFT_V2_RESULT_DIR=/home/paxton/pro/UIE4_VN/experiments/eval_sift_reference_v2/实际完整结果目录 \
  python -m pytest -q tools/sift_ref_v2/test_reference.py
```

## CLI

```bash
python tools/sift_reference_consistency_v2.py \
  --config configs/sift/reference_v2.home_wsl.json \
  --output-dir experiments/eval_sift_reference_v2/my_new_smoke \
  --only-ids 1930 220 895 1757 1544 2753 3035 2010 2519 1812 \
  --K unlimited 128 256 512
```

- `--config`：JSON/YAML，指定真实方法名称、`prediction_dir`、文件名模板和来源记录。按图像 ID 配对，不能按排序位置配对。
- `--dry-run`：写入来源审计与配置，不提取任何 SIFT 特征。
- `--only-ids`：空格或逗号分隔的精确 ID；未知/重复 ID 报错。统计明确标为 partial。
- `--K`：`unlimited` 或正整数，默认 unlimited、128、256、512。为保证回归，unlimited 总会加入。只有 K=256 存在时生成主指标 bootstrap；论文图只在完整 428 张、K=256 和可比较 SOTA 都存在时生成。
- `--regression-only`：只做 Raw/ICAR unlimited 历史回归。

路径迁移时复制并编辑配置，保存为新文件；同步迁移的 Step 1 审计路径/哈希也需要重新核查。本工具遇到路径、ID、编码、尺寸、来源哈希或回归异常会报错并保留审计，绝不跳过图片补出“完整”结果。

## 冻结协议

Raw/GT 使用历史 V16 测试的 Pillow BILINEAR 整图 256×256，增强图不重采样；OpenCV RGB2GRAY on RGB uint8，无直方图均衡化。SIFT 全量检测：nfeatures=0、nOctaveLayers=3（旧版默认）、contrastThreshold=0.04、edgeThreshold=10、sigma=1.6。BF L2 k=2，严格 `d1 < 0.75*d2` 双向 ratio 交集，不等价于单独 `crossCheck=True`。恒等几何欧氏距离 ≤3px；OpenCV 单线程、种子 3407、OpenCL 关闭。配置中的协议字段必须与冻结值一致，不能在本工具中调参。

unlimited 保留 OpenCV 原始特征顺序，先逐图核对 Raw/ICAR 关键点、前向 ratio、双向匹配、几何一致匹配、比例和状态，再允许后续实验。K 有限时按响应降序，依次用 y/x/size/angle/octave/class_id/原描述子索引打破平局，描述子同步选择。每图只缓存一份完整 GT 特征库，所有方法/K 共享；不会把 GT 裁到 K。

K 是上限，实际保留 `min(K,N)`。低于 K、零点和零匹配都参与 macro 统计；ID 1757 的 Raw 原有 46 个点时只能保留 46，记 budget_shortage。按 SIFT 特征实例索引验证一对一，同一 GT 实例不重复计数；不同方向的 SIFT 实例与历史规则相同。

## 输出和解读

| 文件 | 内容 |
|---|---|
| `config_resolved.json` | 实际路径、范围、版本、参数、排序和 bootstrap 协议 |
| `audit_sources.csv` / `audit_documents.csv` | 全部输入图、配置、历史、代码和来源文档 SHA256 |
| `invalid_missing_sources.csv` | 无效、缺失、额外和来源变化；成功时仅表头 |
| `legacy_regression*.csv/json` | 逐图回归值、差异与最大误差；部分 PASS 不冒充 428 张 PASS |
| `baseline_unlimited_per_image.csv` / `baseline_unlimited_summary.csv` | 全量关键点和匹配的逐图/宏均值、中位数、零案例 |
| `fixed_budget_per_image.csv` / `fixed_budget_summary.csv` | 每 K 的 detected/retained/shortage、tentative/verified、precision、coverage、诊断比例 |
| `paired_differences_per_image.csv` / `paired_differences_summary.csv` | 相对 Raw 和 ICAR 相对所有纳入 SOTA 的绝对与相对差异 |
| `paired_bootstrap_ci.csv` | verified@256，图像级 paired bootstrap 5000 次、95% percentile CI、增/平/减数量 |
| `strongest_sota.json` | 预先指定按 macro mean verified@256 选择的最强方法和所有并列项 |
| `features/gt/` | 每图完整公共 GT 关键点/描述子缓存；其余 features 子目录保留各方法全量缓存 |
| `correspondences/` | 各方法/图/K 的原始关键点索引、GT 索引、描述子距离、一致掩码与恒等矩阵 |
| `validation.json` | 用独立聚合方法重读 CSV、预算校验、抽查坐标掩码与来源保持性 |
| `qualitative_2x3.png/.svg` | 完整统计后生成；固定 1930/895，K=256，真实计数；300 dpi、3.5 英寸宽 |
| `report.md` / `run_status.json` | 客观研究说明和完成状态 |

逐图表中的 `tentative` 是双向 ratio 后计数；`verified` 是 Identity@3px 参考几何一致计数，另有直观别名 `identity_consistent_matches_at_3px` 或 `verified_at_K`。`reference_coverage` 分母是完整 GT 特征数。`consistency_precision=verified/tentative`，`verified_per_retained=verified/retained` 只作诊断；分母为零记 0 并报告零案例。逐图相对变化基线为零留空，汇总的相对变化是均值差除以基线均值。

这些是 **reference-geometrically consistent correspondences**，没有独立人工匹配真值标签，不能称为 ground-truth matching accuracy 或真实跨视角任务准确率。近饱和的条件 precision 不等于模型准确率。最强 SOTA 根据当前评价集选出，固定对比的 bootstrap 区间不校正选择偏差或多重比较。IMIF 实现差异、模型原训练方式及源图配准假设应与数字一起保留。

## 测试

测试覆盖恒等坐标及阈值边界、邻居不足、46 点/零点仍纳入、稳定排序与描述子对应、重复 GT 拒绝计数、重复运行指标文件逐字节一致、公共 GT 只提取一次、逐图/汇总 CSV 独立复核、缺失不跳过、历史差异清单及 unlimited 的旧实现回归。完整 428 张数值测试在实际 WSL 结果可用后通过 `SIFT_V2_RESULT_DIR` 启用。
